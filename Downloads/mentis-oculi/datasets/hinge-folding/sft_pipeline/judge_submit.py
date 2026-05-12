"""
Submit a Gemini batch job to JUDGE the reasoning+hinge_id+angle produced by
batch_submit/batch_harvest. For each step with (reasoning, hinge_id, angle),
sends the three images plus the student's text to gemini-3-flash-preview and
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
You are reviewing one step of a hinge-folding puzzle reasoning trace.

Puzzle setup (same as the student was shown):
You are provided with three images of a hinge-folding puzzle.
1. The first image shows the combined puzzle: the initial chain with labeled hinges (A, B, C, …) on the left, and the target folded configuration on the right.
2. The second image is the current folding state.
3. The third image is the next folding state, acting as a visual hint for which single hinge was just rotated.

The student's task was:
Compare the second and third images to identify which single hinge was rotated and by how many degrees (90, 180, or 270). Then write reasoning that justifies the choice globally, as if deduced purely from the combined image and current state.

CRITICAL CONSTRAINT the student had to obey:
Write the reasoning as if deduced PURELY from the combined puzzle image and the current state. NEVER mention the third image, the "next state", or any "hint".

You are given:
1. The combined puzzle image.
2. The current state image.
3. The next state image (the hint).
4. The student's written reasoning.
5. The student's identified hinge label.
6. The student's identified rotation angle.
7. The ground-truth hinge label for this step.
8. The ground-truth rotation angle (degrees) for this step.

Evaluate TWO independent things:
(a) is_correct_reasoning — does the student's hinge_id and angle exactly match the ground-truth hinge and angle provided below? Use the ground-truth as the authoritative answer; the visual hint (third image) is only a secondary sanity-check.
(b) is_correct_no_hints — does the reasoning obey the critical constraint? It must NOT mention "third image", "next state", "hint", "second image", "first image", or any phrase revealing the student was shown future or multiple states. Any mention of the word "image" is a leak.

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

Student hinge:
<<<HINGE_ID>>>

Student angle:
<<<ANGLE>>>

Ground-truth hinge:
<<<GT_HINGE_ID>>>

Ground-truth angle:
<<<GT_ANGLE>>>
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


def collect_requests(output_dir: Path) -> List[Tuple[str, Path, Path, Path, str, str, str, str, str]]:
    """
    Return (key, combined, cur, nxt, reasoning, hinge_id, angle_str, gt_hinge_id, gt_angle_str)
    for every step with (reasoning, hinge_id, angle) but no judgement yet.

    key format: "{group}__{puzzle_dir_name}__judge_step_{NN}"
    """
    jobs = []
    for group, puzzle_dir in _iter_puzzle_dirs(output_dir):
        cot_file = puzzle_dir / "cot_reasoning.json"
        combined = puzzle_dir / "combined.png"
        initial = puzzle_dir / "initial.png"
        if not cot_file.exists() or not combined.exists() or not initial.exists():
            continue
        try:
            steps = json.loads(cot_file.read_text())
        except json.JSONDecodeError:
            print(f"  skip (bad json): {cot_file}", file=sys.stderr)
            continue
        if not isinstance(steps, list):
            continue

        # Load ground-truth rotation_steps from metadata once per puzzle
        meta_file = puzzle_dir / "metadata.json"
        gt_rotation_steps: list = []
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
                gt_rotation_steps = meta.get("rotation_steps", [])
            except json.JSONDecodeError:
                pass

        for step in steps:
            if not isinstance(step, dict):
                continue
            reasoning = step.get("reasoning")
            hinge_id = step.get("hinge_id")
            angle = step.get("angle")
            if not reasoning or not hinge_id or angle is None:
                continue
            if "judgement" in step:
                continue  # already judged
            step_idx = step.get("step")
            if step_idx is None:
                continue
            cur = initial if step_idx == 0 else puzzle_dir / f"cot_{step_idx - 1:02d}.png"
            nxt = puzzle_dir / f"cot_{step_idx:02d}.png"
            if not cur.exists() or not nxt.exists():
                print(f"  skip (missing image): {puzzle_dir} step {step_idx}", file=sys.stderr)
                continue

            # Ground-truth for this step
            if step_idx < len(gt_rotation_steps):
                gt_step = gt_rotation_steps[step_idx]
                gt_hinge_id = str(gt_step.get("hinge_id", "?"))
                gt_angle_str = str(gt_step.get("angle", "?"))
            else:
                gt_hinge_id = "?"
                gt_angle_str = "?"

            key = f"{group}__{puzzle_dir.name}__judge_step_{step_idx:02d}"
            jobs.append((key, combined, cur, nxt, reasoning, str(hinge_id), str(angle), gt_hinge_id, gt_angle_str))
    return jobs


def build_jsonl(jobs: List[Tuple[str, Path, Path, Path, str, str, str, str, str]], out_path: Path) -> None:
    with out_path.open("w") as f:
        for key, combined, cur, nxt, reasoning, hinge_id, angle_str, gt_hinge_id, gt_angle_str in jobs:
            prompt = (JUDGE_PROMPT
                      .replace("<<<REASONING>>>", reasoning)
                      .replace("<<<HINGE_ID>>>", hinge_id)
                      .replace("<<<ANGLE>>>", angle_str)
                      .replace("<<<GT_HINGE_ID>>>", gt_hinge_id)
                      .replace("<<<GT_ANGLE>>>", gt_angle_str))
            request = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": prompt},
                            _png_part(combined),
                            _png_part(cur),
                            _png_part(nxt),
                        ],
                    }
                ],
                "generation_config": {"response_mime_type": "application/json"},
            }
            f.write(json.dumps({"key": key, "request": request}) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit Gemini batch judge for hinge-folding reasoning")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--jsonl", type=str, default=None)
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--display-name", type=str, default="hingefolding-judge")
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
