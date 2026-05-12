"""
Submit a Gemini batch job for sliding-puzzle step reasoning.

Walks an output directory produced by `main.py`, bootstraps a
cot_reasoning.json scaffold from metadata.json if one doesn't exist,
then for every step missing reasoning/move submits a batch request with
two images: the current tile state and the next tile state (hint).

The batch job name is written to `<output_dir>/.batch_job.txt` for
later harvest by `batch_harvest.py`.

Layout handled:
  <output_dir>/puzzle_*    (flat)

Usage:
    python batch_submit.py --output-dir ../output_sft_100
"""
import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

from google import genai
from google.genai import types


PROMPT_TEMPLATE = """
You are provided with two images of a sliding tile puzzle.
1. The first image is the current puzzle state (tiles scrambled, one black blank tile).
2. The second image is the next puzzle state, acting as a visual hint for which single tile was just moved.

<<<SEMANTIC_LINE>>>

Ground-truth for this step:
- The blank space moves <<<GT_MOVE>>>.

Convention: the direction always refers to which way the BLANK moves.
For example "up" means the blank slides upward, pulling the tile above it downward.

Your task:
Write reasoning that justifies why the blank should move <<<GT_MOVE>>> now.
Your move in the JSON response MUST exactly match the ground-truth value above.

Your reasoning MUST explain why moving the blank <<<GT_MOVE>>> now helps unscramble the puzzle toward the solved image.

CRITICAL CONSTRAINTS:
- Write the reasoning as if you logically deduced this move PURELY from the current state. You must NEVER mention the second image, the "next state", or any "hint".
- Keep the reasoning to 2–5 short sentences. No informal blocks, no roleplay.

Respond EXACTLY with this JSON format:
{
  "reasoning": "[Brief logical explanation of why moving the blank this way helps solve the puzzle.]",
  "move": "<DIRECTION>"
}
where <DIRECTION> is exactly one of: up, down, left, right — the direction the BLANK moves.
"""

MODEL = "gemini-3.1-pro-preview"


def _load_dotenv() -> None:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / ".env"
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return


def _png_part(path: Path) -> dict:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"inline_data": {"mime_type": "image/png", "data": data}}


def _iter_puzzle_dirs(output_dir: Path):
    """Yield (group_name, puzzle_dir) for flat puzzle_* layout."""
    for puzzle_dir in sorted(output_dir.glob("puzzle_*")):
        if puzzle_dir.is_dir():
            yield "root", puzzle_dir


def _bootstrap_cot(puzzle_dir: Path) -> bool:
    """
    Create cot_reasoning.json scaffold from metadata.json if it doesn't exist.
    Returns True if the file exists (or was created), False if metadata is missing.
    """
    cot_file = puzzle_dir / "cot_reasoning.json"
    if cot_file.exists():
        return True
    meta_file = puzzle_dir / "metadata.json"
    if not meta_file.exists():
        return False
    meta = json.loads(meta_file.read_text())
    steps = [
        {"step": i, "image": f"cot_{i:02d}.png"}
        for i in range(len(meta.get("solution_moves", [])))
    ]
    cot_file.write_text(json.dumps(steps, indent=2))
    return True


def _load_semantic(puzzle_dir: Path) -> str:
    """Return semantic description from semantic.json, or empty string if not present."""
    semantic_path = puzzle_dir / "semantic.json"
    if semantic_path.exists():
        try:
            data = json.loads(semantic_path.read_text())
            return data.get("target_semantic", "")
        except (json.JSONDecodeError, KeyError):
            pass
    return ""


def collect_requests(output_dir: Path) -> List[Tuple[str, Path, Path, str, str]]:
    """
    Return list of (key, current_image, next_image, gt_move, semantic)
    for every step missing reasoning+move in its cot_reasoning.json.

    key format: "{group}__{puzzle_dir_name}__step_{NN}"
    """
    jobs: List[Tuple[str, Path, Path, str, str]] = []
    for group, puzzle_dir in _iter_puzzle_dirs(output_dir):
        if not _bootstrap_cot(puzzle_dir):
            continue
        cot_file = puzzle_dir / "cot_reasoning.json"
        initial = puzzle_dir / "initial.png"
        if not initial.exists():
            continue
        try:
            steps = json.loads(cot_file.read_text())
        except json.JSONDecodeError:
            print(f"  skip (bad json): {cot_file}", file=sys.stderr)
            continue
        if not isinstance(steps, list):
            continue

        # Load ground-truth solution_moves from metadata once per puzzle
        meta_file = puzzle_dir / "metadata.json"
        gt_moves: list = []
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
                gt_moves = meta.get("solution_moves", [])
            except json.JSONDecodeError:
                pass

        # Load semantic description once per puzzle
        semantic = _load_semantic(puzzle_dir)

        for step in steps:
            if not isinstance(step, dict):
                continue
            if step.get("reasoning") and step.get("move"):
                continue  # already done
            step_idx = step.get("step")
            if step_idx is None:
                continue
            cur = initial if step_idx == 0 else puzzle_dir / f"cot_{step_idx - 1:02d}.png"
            nxt = puzzle_dir / f"cot_{step_idx:02d}.png"
            if not cur.exists() or not nxt.exists():
                print(f"  skip (missing image): {puzzle_dir} step {step_idx}", file=sys.stderr)
                continue

            gt_move = gt_moves[step_idx] if step_idx < len(gt_moves) else "?"

            key = f"{group}__{puzzle_dir.name}__step_{step_idx:02d}"
            jobs.append((key, cur, nxt, gt_move, semantic))
    return jobs


def build_jsonl(jobs: List[Tuple[str, Path, Path, str, str]], out_path: Path) -> None:
    with out_path.open("w") as f:
        for key, cur, nxt, gt_move, semantic in jobs:
            semantic_line = (
                f"Visual layout of the solved puzzle: {semantic}"
                if semantic else
                "No visual layout description available."
            )
            prompt = (PROMPT_TEMPLATE
                      .replace("<<<SEMANTIC_LINE>>>", semantic_line)
                      .replace("<<<GT_MOVE>>>", gt_move))
            request = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": prompt},
                            _png_part(cur),
                            _png_part(nxt),
                        ],
                    }
                ],
                "generation_config": {"response_mime_type": "application/json"},
            }
            f.write(json.dumps({"key": key, "request": request}) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit Gemini batch job for sliding-puzzle reasoning")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Dataset output dir (from main.py)")
    parser.add_argument("--jsonl", type=str, default=None,
                        help="Path to write the batch JSONL (default: <output-dir>/batch_requests.jsonl)")
    parser.add_argument("--model", type=str, default=MODEL, help=f"Gemini model (default: {MODEL})")
    parser.add_argument("--display-name", type=str, default="slidingpuzzle-reasoning",
                        help="Batch job display name")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.exists():
        sys.exit(f"output-dir does not exist: {output_dir}")

    _load_dotenv()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    print(f"Scanning {output_dir} ...")
    jobs = collect_requests(output_dir)
    if not jobs:
        sys.exit("No pending steps found (all have reasoning+move already, or nothing to process).")
    print(f"Found {len(jobs)} pending step(s).")

    jsonl_path = Path(args.jsonl) if args.jsonl else output_dir / "batch_requests.jsonl"
    print(f"Writing JSONL -> {jsonl_path}")
    build_jsonl(jobs, jsonl_path)
    size_mb = jsonl_path.stat().st_size / (1024 * 1024)
    print(f"JSONL size: {size_mb:.2f} MB")

    print("Uploading file via Files API ...")
    uploaded = client.files.upload(
        file=str(jsonl_path),
        config=types.UploadFileConfig(display_name=args.display_name, mime_type="jsonl"),
    )
    print(f"  file name: {uploaded.name}")

    print(f"Creating batch job (model={args.model}) ...")
    job = client.batches.create(
        model=args.model,
        src=uploaded.name,
        config={"display_name": args.display_name},
    )
    print(f"  job name: {job.name}")
    print(f"  state: {job.state.name if hasattr(job.state, 'name') else job.state}")

    job_record = {
        "job_name": job.name,
        "model": args.model,
        "uploaded_file": uploaded.name,
        "jsonl_path": str(jsonl_path),
        "num_requests": len(jobs),
    }
    record_path = output_dir / ".batch_job.txt"
    record_path.write_text(json.dumps(job_record, indent=2))
    print(f"Saved job record -> {record_path}")
    print()
    print("Next step: python batch_harvest.py --output-dir", args.output_dir)


if __name__ == "__main__":
    main()
