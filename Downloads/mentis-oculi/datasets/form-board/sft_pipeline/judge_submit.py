"""
Submit a Gemini batch job to JUDGE the reasoning+piece pairs produced by
batch_submit/batch_harvest. For each step with (reasoning, piece), sends
the four images plus the student's text to gemini-3-flash-preview and
asks for strict JSON grading.

Usage:
    python judge_submit.py --output-dir ../output_sft_100
    python judge_submit.py --output-dir ../output_sft_100 --dry-run
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


JUDGE_PROMPT = """
You are reviewing one step of a form-board assembly puzzle reasoning trace.

Puzzle setup (same as the student was shown):
You are provided with four images of a form-board assembly puzzle.
1. The first image is the piece legend, showing the target silhouette and all five candidate pieces labeled A, B, C, D, E.
2. The second image is the current assembly state (pieces placed so far; may be empty).
3. The third image is the next assembly state, acting as a visual hint for which single piece was just placed.
4. The fourth image is the fully-solved puzzle.

The student's task was:
Compare the second and third images to identify which single piece (A, B, C, D, or E) was added in this step. Then write reasoning that justifies the choice globally, as if deduced purely from the legend and current state.

CRITICAL CONSTRAINT the student had to obey:
Write the reasoning as if deduced PURELY from the legend and the current state. NEVER mention the third image, the fourth image, the "next state", the "solved puzzle", or any "hint".

You are given:
1. The legend image.
2. The current state image.
3. The next state image (the hint).
4. The solved puzzle image.
5. The student's written reasoning.
6. The student's identified piece label.

Evaluate TWO independent things:
(a) is_correct_reasoning — does the student's reasoning correctly identify which piece fits the open region, and does the stated piece actually match what was placed between the current and next state images?
(b) is_correct_no_hints — does the reasoning obey the critical constraint? It must NOT mention "third image", "fourth image", "next state", "solved puzzle", "hint", "second image", "first image", or any phrase revealing the student was shown future or multiple states. Any mention of the word "image" is a leak.

Respond EXACTLY with this JSON (no prose outside JSON):
{
  "is_correct_reasoning": true | false,
  "is_correct_no_hints": true | false,
  "reasoning": ""
}

If BOTH booleans are true, "reasoning" MUST be the empty string "".
Otherwise, "reasoning" MUST briefly state which check(s) failed and why.

Student reasoning:
<<<REASONING>>>

Student piece:
<<<PIECE>>>
"""

MODEL = "gemini-3-flash-preview"


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
    """Yield (group_name, puzzle_dir) for both level_*/puzzle_* and flat puzzle_* layouts."""
    level_dirs = sorted(output_dir.glob("level_*"))
    if level_dirs:
        for level_dir in level_dirs:
            if not level_dir.is_dir():
                continue
            for puzzle_dir in sorted(level_dir.glob("puzzle_*")):
                yield level_dir.name, puzzle_dir
    else:
        for puzzle_dir in sorted(output_dir.glob("puzzle_*")):
            yield "root", puzzle_dir


def collect_requests(output_dir: Path) -> List[Tuple[str, Path, Path, Path, Path, str, str]]:
    """
    Return list of (key, legend, cur, nxt, final, reasoning, piece) for every
    step with (reasoning, piece) but no judgement yet.

    key format: "{group}__{puzzle_dir_name}__judge_step_{NN}"
    """
    jobs = []
    for group, puzzle_dir in _iter_puzzle_dirs(output_dir):
        cot_file = puzzle_dir / "cot_reasoning.json"
        legend = puzzle_dir / "combined.png"
        silhouette = puzzle_dir / "silhouette.png"
        final = puzzle_dir / "bordered.png"
        if not cot_file.exists() or not legend.exists() or not silhouette.exists() or not final.exists():
            continue
        try:
            steps = json.loads(cot_file.read_text())
        except json.JSONDecodeError:
            print(f"  skip (bad json): {cot_file}", file=sys.stderr)
            continue
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            reasoning = step.get("reasoning")
            piece = step.get("piece")
            if not reasoning or not piece:
                continue
            if "judgement" in step:
                continue  # already judged
            step_idx = step.get("step")
            if step_idx is None:
                continue
            cur = silhouette if step_idx == 0 else puzzle_dir / f"cot_{step_idx - 1:02d}.png"
            nxt = puzzle_dir / f"cot_{step_idx:02d}.png"
            if not cur.exists() or not nxt.exists():
                print(f"  skip (missing image): {puzzle_dir} step {step_idx}", file=sys.stderr)
                continue
            key = f"{group}__{puzzle_dir.name}__judge_step_{step_idx:02d}"
            jobs.append((key, legend, cur, nxt, final, reasoning, piece))
    return jobs


def build_jsonl(jobs: List[Tuple[str, Path, Path, Path, Path, str, str]], out_path: Path) -> None:
    with out_path.open("w") as f:
        for key, legend, cur, nxt, final, reasoning, piece in jobs:
            prompt = JUDGE_PROMPT.replace("<<<REASONING>>>", reasoning).replace("<<<PIECE>>>", piece)
            request = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": prompt},
                            _png_part(legend),
                            _png_part(cur),
                            _png_part(nxt),
                            _png_part(final),
                        ],
                    }
                ],
                "generation_config": {"response_mime_type": "application/json"},
            }
            f.write(json.dumps({"key": key, "request": request}) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit Gemini batch judge for form-board reasoning")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--jsonl", type=str, default=None)
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--display-name", type=str, default="formboard-judge")
    parser.add_argument("--dry-run", action="store_true", help="Just count requests and exit")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.exists():
        sys.exit(f"output-dir does not exist: {output_dir}")

    print(f"Scanning {output_dir} ...")
    jobs = collect_requests(output_dir)
    if not jobs:
        sys.exit("No pending steps to judge.")
    print(f"Found {len(jobs)} step(s) to judge.")
    if args.dry_run:
        return

    _load_dotenv()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    jsonl_path = Path(args.jsonl) if args.jsonl else output_dir / "judge_requests.jsonl"
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
    record_path = output_dir / ".judge_job.txt"
    record_path.write_text(json.dumps(job_record, indent=2))
    print(f"Saved job record -> {record_path}")
    print()
    print("Next step: python judge_harvest.py --output-dir", args.output_dir, "--wait")


if __name__ == "__main__":
    main()
