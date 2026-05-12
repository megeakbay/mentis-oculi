"""
Submit a Gemini batch job to JUDGE the reasoning+response pairs produced by
batch_submit/batch_harvest. For each step with (reasoning, response), send
the two state images plus the student's text to gemini-3-flash-preview and
ask for strict JSON grading.

Usage:
    python judge_submit.py --output-dir output_sft_50
    python judge_submit.py --output-dir output_sft_50 --dry-run
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
You are reviewing one step of a block sliding puzzle reasoning trace.

Puzzle setup (same as the student was shown):
You are provided with three images of a block sliding puzzle.
1. The first image is the current state.
2. The second image shows the exact next state, acting as a visual hint.
3. The third image is the final solved state — the red rectangle (R) has reached the green exit.

The student's task was:
Compare the first and second images to see which single rectangle moved and in what direction. Use the third image (final state) to understand the goal. Then write reasoning for WHY that move helps clear the path toward the final solved state.

CRITICAL CONSTRAINT the student had to obey:
The student had to write the reasoning as if they logically deduced the move PURELY from the current state and the final state. They were NEVER allowed to mention the second image, the "next state", or any "hint".

You are given:
1. The current-state image.
2. The next-state image (the hint).
3. The final solved-state image.
4. The student's written reasoning.
5. The student's one-sentence action.

Evaluate TWO independent things:

(a) is_correct_reasoning — does the student's reasoning logically justify, given the current state and final state, why the stated move helps? Check ALL of the following and fail if ANY are violated:
  - Does the stated action match the single rectangle + direction that actually moved between the first and second images?
  - Does the student correctly identify which rectangle is blocking R's path in the current state? If they name the wrong blocker, fail.
  - If the student claims a rectangle CANNOT move in some direction (e.g. "D cannot move backward"), is that constraint actually visible in the current state image — i.e. is another rectangle or wall clearly blocking that direction on its rail? If they invented a false constraint, fail.
  - Does the student ONLY justify this current move? If they predict or describe future moves (e.g. "this will allow C to move backward later"), fail — the reasoning must only cover what this single move achieves right now.

(b) is_correct_no_hints — does the reasoning obey the critical constraint? It must NOT mention "second image", "next state", "hint", or any phrase revealing the student was shown a hint image. Any mention of "second image" or "next state" is a leak. Mentioning the "final state" or "solved state" is fine since that was allowed.

Respond EXACTLY with this JSON (no prose outside JSON):
{
  "is_correct_reasoning": true | false,
  "is_correct_no_hints": true | false,
  "reasoning": ""
}

If BOTH booleans are true, "reasoning" MUST be the empty string "".
Otherwise, "reasoning" MUST briefly state which specific check(s) failed and why (e.g. "wrong blocker identified", "invented false constraint on D", "predicted future move of C").

Student reasoning:
<<<REASONING>>>

Student action:
<<<RESPONSE>>>
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


def collect_requests(output_dir: Path) -> List[Tuple[str, Path, Path, str, str]]:
    """
    Return list of (key, current_image, next_image, reasoning, response) for every
    step with (reasoning, response) but no judgement yet.

    key format: "{level_dir_name}__{puzzle_dir_name}__judge_step_{NN}"
    """
    jobs = []
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
            for step in steps:
                if not isinstance(step, dict):
                    continue
                r = step.get("reasoning")
                resp = step.get("response")
                if not r or not resp:
                    continue
                if "judgement" in step:
                    continue
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
                key = f"{level_dir.name}__{puzzle_dir.name}__judge_step_{step_idx:02d}"
                jobs.append((key, cur, nxt, r, resp))
    return jobs


def build_jsonl(jobs, out_path: Path) -> None:
    with out_path.open("w") as f:
        for key, cur, nxt, reasoning, resp in jobs:
            prompt = JUDGE_PROMPT.replace("<<<REASONING>>>", reasoning).replace("<<<RESPONSE>>>", resp)
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
    parser = argparse.ArgumentParser(description="Submit Gemini batch judge for rushhour reasoning")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--jsonl", type=str, default=None)
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--display-name", type=str, default="rushhour-judge")
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
