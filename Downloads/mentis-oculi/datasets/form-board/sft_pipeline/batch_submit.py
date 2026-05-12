"""
Submit a Gemini batch job for form-board puzzle step reasoning.

Walks an output directory produced by `main.py --skip-reasoning`, pairs each
step's (legend=combined.png, current, next) images, and submits one batch
with `response_mime_type = application/json`. The batch job name is written
to `<output_dir>/.batch_job.txt` for later harvest by `batch_harvest.py`.

Layout handled:
  <output_dir>/level_*/puzzle_*    (stratified per-level)
  <output_dir>/puzzle_*            (flat)

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


# Must match the prompt in reasoning_text.py exactly.
STATIC_PROMPT = """
You are provided with four images of a form-board assembly puzzle.
1. The first image is the piece legend, showing the target silhouette and all five candidate pieces labeled A, B, C, D, E.
2. The second image is the current assembly state (pieces placed so far inside the silhouette; may be empty).
3. The third image is the next assembly state, acting as a visual hint for which single piece was just placed.
4. The fourth image is the fully-solved puzzle, showing where every solution piece ends up — use this to make your look-ahead reasoning grounded in the actual final layout.

Your task:
Compare the second and third images to identify which single piece (A, B, C, D, or E) was added in this step. Then write reasoning that justifies the choice GLOBALLY, not just locally.

Your reasoning MUST cover:
1. Why the chosen piece fits the open region right now (edges/angles/size cues from the legend and current state).
2. Look-ahead: what shape/region will remain after this placement, and why that remainder looks compatible with the pieces still available. Skip only if this placement completes the silhouette.

CRITICAL CONSTRAINTS:
- Write the reasoning as if you logically deduced this move PURELY from the legend and the current state. You must NEVER mention the third image, the fourth image, the "next state", the "solved puzzle", or any "hint".
- Use the piece's letter label (A/B/C/D/E) to refer to it.
- Keep the reasoning to 2–5 short sentences. No informal ･･ blocks, no pauses, no roleplay.

Respond EXACTLY with this JSON format:
{
  "reasoning": "[Brief logical explanation of why this piece fits the remaining open region.]",
  "piece": "<LABEL>"
}
where <LABEL> is exactly one of A, B, C, D, E.
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


def collect_requests(output_dir: Path) -> List[Tuple[str, Path, Path, Path, Path]]:
    """
    Return list of (key, legend_image, current_image, next_image, final_image) tuples for every
    step missing reasoning+piece in its cot_reasoning.json.

    key format: "{group}__{puzzle_dir_name}__step_{NN}"
    """
    jobs: List[Tuple[str, Path, Path, Path, Path]] = []
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
            if step.get("reasoning") and step.get("piece"):
                continue  # already done
            step_idx = step.get("step")
            if step_idx is None:
                continue
            cur = silhouette if step_idx == 0 else puzzle_dir / f"cot_{step_idx - 1:02d}.png"
            nxt = puzzle_dir / f"cot_{step_idx:02d}.png"
            if not cur.exists() or not nxt.exists():
                print(f"  skip (missing image): {puzzle_dir} step {step_idx}", file=sys.stderr)
                continue
            key = f"{group}__{puzzle_dir.name}__step_{step_idx:02d}"
            jobs.append((key, legend, cur, nxt, final))
    return jobs


def build_jsonl(jobs: List[Tuple[str, Path, Path, Path, Path]], out_path: Path) -> None:
    with out_path.open("w") as f:
        for key, legend, cur, nxt, final in jobs:
            request = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": STATIC_PROMPT},
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
    parser = argparse.ArgumentParser(description="Submit Gemini batch job for form-board reasoning")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Dataset output dir (e.g. from main.py --skip-reasoning)")
    parser.add_argument("--jsonl", type=str, default=None,
                        help="Path to write the batch JSONL (default: <output-dir>/batch_requests.jsonl)")
    parser.add_argument("--model", type=str, default=MODEL, help=f"Gemini model (default: {MODEL})")
    parser.add_argument("--display-name", type=str, default="formboard-reasoning",
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
        sys.exit("No pending steps found (all have reasoning+piece already, or nothing to process).")
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
