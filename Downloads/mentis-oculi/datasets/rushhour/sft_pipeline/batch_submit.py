"""
Submit a Gemini batch job for rushhour puzzle step reasoning.

Walks an output directory produced by `main.py --skip-reasoning`, pairs each
step's (current, next) images, and submits one batch with `response_mime_type
= application/json`. The batch job name is written to `<output_dir>/.batch_job.txt`
for later harvest by `batch_harvest.py`.

Usage:
    python batch_submit.py --output-dir output_sft_test
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
You are provided with three images of a block sliding puzzle.
1. The first image is the current state.
2. The second image shows the exact next state, acting as a visual hint for which rectangle moved.
3. The third image is the final solved state — the red rectangle (R) has reached the green exit.

Each rectangle has a thin rail overlay with an arrow showing its local drive axis. A rectangle can only translate along that rail — either FORWARD (in the direction the arrow points) or BACKWARD (opposite the arrow). No rotations, no sideways motion. "forward" and "backward" are always relative to that specific rectangle's arrow, NOT relative to the page, compass, or viewer.

Your task:
Compare the first and second images to identify which single rectangle moved and in which direction. Use the third image to understand the goal. Then write reasoning for WHY that move helps clear the path toward the final solved state.

STRICT REASONING RULES:
1. Only reason about what is physically visible in the current state (first image). Do not invent constraints — if you claim a rectangle cannot move in some direction, it must be visually obvious from the first image that another rectangle or the board boundary is blocking it.
2. Only justify THIS move. Do not predict or describe future moves. Do not say what will happen after this move.
3. Do not claim a rectangle is blocked by an obstacle unless that obstacle is clearly overlapping or adjacent to its rail path in the first image.
4. Keep reasoning to 2-3 sentences maximum.

CRITICAL CONSTRAINTS:
- Write the reasoning as if you logically deduced this move PURELY from looking at the first image and the final state. You must NEVER mention the second image, the "next state", or any "hint".
- Describe directions ONLY as "forward" or "backward" relative to each rectangle's own rail arrow. Do NOT use absolute directions like "up", "down", "left", "right".

Respond EXACTLY with this JSON format:
{
  "reasoning": "[2-3 sentences explaining why this move helps clear the path to the final solved state.]",
  "response": "<LABEL> forward"
}
The response field is exactly two tokens: the rectangle label, then either 'forward' or 'backward'. No sentence, no punctuation.
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


def collect_requests(output_dir: Path) -> List[Tuple[str, Path, Path, Path]]:
    """
    Return list of (key, current_image, next_image, final_image) tuples for every step
    missing a reasoning/response field in its cot_reasoning.json.

    key format: "{level_dir_name}__{puzzle_dir_name}__step_{NN}"
    """
    jobs: List[Tuple[str, Path, Path, Path]] = []
    for level_dir in sorted(output_dir.glob("level_*")):
        if not level_dir.is_dir():
            continue
        for puzzle_dir in sorted(level_dir.glob("puzzle_*")):
            cot_file = puzzle_dir / "cot_reasoning.json"
            if not cot_file.exists():
                continue
            try:
                steps = json.loads(cot_file.read_text())
            except json.JSONDecodeError:
                print(f"  skip (bad json): {cot_file}", file=sys.stderr)
                continue
            if not isinstance(steps, list):
                continue

            # final image = last cot step (R at exit)
            valid_steps = sorted(
                (s for s in steps if isinstance(s, dict) and s.get("step") is not None),
                key=lambda s: s["step"]
            )
            if not valid_steps:
                continue
            last_step_idx = valid_steps[-1]["step"]
            final_img = puzzle_dir / f"cot_{last_step_idx:02d}.png"
            if not final_img.exists():
                print(f"  skip (missing final image): {puzzle_dir}", file=sys.stderr)
                continue

            for step in steps:
                if not isinstance(step, dict):
                    continue
                if "reasoning" in step and "response" in step:
                    continue  # already done
                step_idx = step.get("step")
                if step_idx is None:
                    continue
                if step_idx == 0:
                    cur = puzzle_dir / "initial.png"
                else:
                    cur = puzzle_dir / f"cot_{step_idx - 1:02d}.png"
                nxt = puzzle_dir / f"cot_{step_idx:02d}.png"
                if not cur.exists() or not nxt.exists():
                    print(f"  skip (missing image): {puzzle_dir} step {step_idx}", file=sys.stderr)
                    continue
                key = f"{level_dir.name}__{puzzle_dir.name}__step_{step_idx:02d}"
                jobs.append((key, cur, nxt, final_img))
    return jobs


def build_jsonl(jobs: List[Tuple[str, Path, Path, Path]], out_path: Path) -> None:
    with out_path.open("w") as f:
        for key, cur, nxt, final_img in jobs:
            request = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": STATIC_PROMPT},
                            _png_part(cur),
                            _png_part(nxt),
                            _png_part(final_img),
                        ],
                    }
                ],
                "generation_config": {"response_mime_type": "application/json"},
            }
            f.write(json.dumps({"key": key, "request": request}) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit Gemini batch job for rushhour reasoning")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Dataset output dir (e.g. from main.py --skip-reasoning)")
    parser.add_argument("--jsonl", type=str, default=None,
                        help="Path to write the batch JSONL (default: <output-dir>/batch_requests.jsonl)")
    parser.add_argument("--model", type=str, default=MODEL, help=f"Gemini model (default: {MODEL})")
    parser.add_argument("--display-name", type=str, default="rushhour-reasoning",
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
        sys.exit("No pending steps found (all have reasoning+response already, or nothing to process).")
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
