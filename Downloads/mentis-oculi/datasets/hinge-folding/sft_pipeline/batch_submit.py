"""
Submit a Gemini batch job for hinge-folding puzzle step reasoning.

Walks an output directory produced by `main.py`, bootstraps a
cot_reasoning.json scaffold from metadata.json if one doesn't exist,
then for every step missing reasoning/hinge_id/angle submits a batch
request with three images: the combined puzzle, the current folding
state, and the next folding state (hint).

The batch job name is written to `<output_dir>/.batch_job.txt` for
later harvest by `batch_harvest.py`.

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


PROMPT_TEMPLATE = """
You are provided with three images of a hinge-folding puzzle.
1. The first image is the combined puzzle view: the initial unfolded chain on the left and the TARGET folded shape on the right. Use this to understand what the final configuration must look like.
2. The second image is the current folding state (chain as folded so far).
3. The third image is the next folding state, acting as a visual hint for which single hinge was just rotated.

Ground-truth for this step:
- Hinge to rotate: <<<GT_HINGE_ID>>>
- Degrees to rotate: <<<GT_ANGLE>>>

Your task:
Write reasoning that justifies rotating hinge <<<GT_HINGE_ID>>> by <<<GT_ANGLE>>> degrees at this step.
Your hinge_id and angle in the JSON response MUST exactly match the ground-truth values above.

Your reasoning MUST explain:
- How the current chain configuration compares to the target shape (from the first image).
- Why rotating hinge <<<GT_HINGE_ID>>> by <<<GT_ANGLE>>> degrees specifically closes the gap toward that target.

CRITICAL CONSTRAINTS:
- Write the reasoning as if you logically deduced this rotation from the current state and the target. You must NEVER mention the third image, the "next state", or any "hint".
- Use the hinge's letter label (A/B/C/…) to refer to it.
- Keep the reasoning to 2–4 short sentences. No informal blocks, no roleplay.

Respond EXACTLY with this JSON format:
{
  "reasoning": "[Explanation grounded in the current state vs. target shape.]",
  "hinge_id": "<LABEL>",
  "angle": <DEGREES>
}
where <LABEL> is one of the hinge letters shown in the puzzle, and <DEGREES> is exactly 90, 180, or 270.
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
        for i in range(len(meta.get("rotation_steps", [])))
    ]
    cot_file.write_text(json.dumps(steps, indent=2))
    return True


def collect_requests(output_dir: Path) -> List[Tuple[str, Path, Path, Path, str, str]]:
    """
    Return list of (key, combined_image, current_image, next_image, gt_hinge_id, gt_angle_str)
    for every step missing reasoning+hinge_id+angle in its cot_reasoning.json.

    key format: "{group}__{puzzle_dir_name}__step_{NN}"
    """
    jobs: List[Tuple[str, Path, Path, Path, str, str]] = []
    for group, puzzle_dir in _iter_puzzle_dirs(output_dir):
        if not _bootstrap_cot(puzzle_dir):
            continue
        cot_file = puzzle_dir / "cot_reasoning.json"
        initial = puzzle_dir / "initial.png"
        combined = puzzle_dir / "combined.png"
        if not initial.exists():
            continue
        if not combined.exists():
            print(f"  skip (missing combined.png): {puzzle_dir}", file=sys.stderr)
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
            if step.get("reasoning") and step.get("hinge_id") and step.get("angle") is not None:
                continue  # already done
            step_idx = step.get("step")
            if step_idx is None:
                continue
            cur = initial if step_idx == 0 else puzzle_dir / f"cot_{step_idx - 1:02d}.png"
            nxt = puzzle_dir / f"cot_{step_idx:02d}.png"
            if not cur.exists() or not nxt.exists():
                print(f"  skip (missing image): {puzzle_dir} step {step_idx}", file=sys.stderr)
                continue

            # Ground-truth for this step and remaining steps
            if step_idx < len(gt_rotation_steps):
                gt_step = gt_rotation_steps[step_idx]
                gt_hinge_id = str(gt_step.get("hinge_id", "?"))
                gt_angle_str = str(gt_step.get("angle", "?"))
            else:
                gt_hinge_id = "?"
                gt_angle_str = "?"

            key = f"{group}__{puzzle_dir.name}__step_{step_idx:02d}"
            jobs.append((key, combined, cur, nxt, gt_hinge_id, gt_angle_str))
    return jobs


def build_jsonl(jobs: List[Tuple[str, Path, Path, Path, str, str]], out_path: Path) -> None:
    with out_path.open("w") as f:
        for key, combined, cur, nxt, gt_hinge_id, gt_angle_str in jobs:
            prompt = (PROMPT_TEMPLATE
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
    parser = argparse.ArgumentParser(description="Submit Gemini batch job for hinge-folding reasoning")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Dataset output dir (from main.py)")
    parser.add_argument("--jsonl", type=str, default=None,
                        help="Path to write the batch JSONL (default: <output-dir>/batch_requests.jsonl)")
    parser.add_argument("--model", type=str, default=MODEL, help=f"Gemini model (default: {MODEL})")
    parser.add_argument("--display-name", type=str, default="hingefolding-reasoning",
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
        sys.exit("No pending steps found (all have reasoning already, or nothing to process).")
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
