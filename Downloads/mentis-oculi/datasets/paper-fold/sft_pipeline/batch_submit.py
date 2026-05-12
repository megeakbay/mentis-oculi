"""
Submit a Gemini batch job for paper-fold puzzle step reasoning.

Walks an output directory produced by `main.py`, bootstraps a
cot_reasoning.json scaffold from metadata.json if one doesn't exist,
then for every unfolding step missing reasoning submits a batch request
with three images: the question (fold sequence + hole), the current
unfolding state, and the next unfolding state (hint).

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
You are provided with three images of a paper-folding puzzle.
1. The first image shows the complete folding sequence: the paper folded step-by-step with a hole punch applied at the end.
2. The second image is the current unfolding state (paper partially unfolded so far).
3. The third image is the next unfolding state, acting as a visual hint for how the paper looks after one more unfold.

Ground-truth for this step:
- Fold type being reversed: <<<GT_FOLD_TYPE>>>
- Remaining folds to reverse after this step: <<<REMAINING_FOLDS>>>

Your task:
Write reasoning that justifies this unfolding step — i.e., why reversing fold <<<GT_FOLD_TYPE>>> at this point propagates the hole correctly. Your reasoning MUST be written as if deduced purely from the question image and the current unfolding state.

Your reasoning MUST cover:
1. How the current hole pattern propagates when this fold is reversed (reflection symmetry).
2. Why the resulting hole positions are geometrically consistent with the prior folds. Skip if this is the final unfold.

CRITICAL CONSTRAINTS:
- Write the reasoning as if you logically deduced this unfolding PURELY from the question image and the current state. You must NEVER mention the third image, the "next state", or any "hint".
- Keep the reasoning to 2–5 short sentences. No informal blocks, no roleplay.

Respond EXACTLY with this JSON format:
{
  "reasoning": "[Brief logical explanation of why this unfold propagates the holes correctly.]",
  "fold_type": "<FOLD_TYPE>"
}
where <FOLD_TYPE> is one of: horizontal, vertical, diag_pos, diag_neg.
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
    level_dirs = sorted(output_dir.glob("level_*"))
    if level_dirs:
        for level_dir in level_dirs:
            if not level_dir.is_dir():
                continue
            for puzzle_dir in sorted(level_dir.glob("puzzle_*")):
                if puzzle_dir.is_dir():
                    yield level_dir.name, puzzle_dir
    else:
        for puzzle_dir in sorted(output_dir.glob("puzzle_*")):
            if puzzle_dir.is_dir():
                yield "root", puzzle_dir


def _bootstrap_cot(puzzle_dir: Path) -> bool:
    cot_file = puzzle_dir / "cot_reasoning.json"
    if cot_file.exists():
        return True
    meta_file = puzzle_dir / "metadata.json"
    if not meta_file.exists():
        return False
    meta = json.loads(meta_file.read_text())
    cot_images = meta.get("cot_images", [])
    steps = [
        {"step": i, "image": img}
        for i, img in enumerate(cot_images)
    ]
    cot_file.write_text(json.dumps(steps, indent=2))
    return True


def collect_requests(output_dir: Path) -> List[Tuple[str, Path, Path, Path, str, str]]:
    """
    Return list of (key, question_image, current_image, next_image, gt_fold_type, remaining_folds_str)
    for every step missing reasoning in its cot_reasoning.json.

    key format: "root__{puzzle_dir_name}__step_{NN}"
    """
    jobs: List[Tuple[str, Path, Path, Path, str, str]] = []
    for group, puzzle_dir in _iter_puzzle_dirs(output_dir):
        if not _bootstrap_cot(puzzle_dir):
            continue
        cot_file = puzzle_dir / "cot_reasoning.json"
        question_img = puzzle_dir / "question.png"
        initial_unfold = puzzle_dir / "silhouette.png"  # fully unfolded = last cot state
        if not question_img.exists():
            continue
        try:
            steps = json.loads(cot_file.read_text())
        except json.JSONDecodeError:
            print(f"  skip (bad json): {cot_file}", file=sys.stderr)
            continue
        if not isinstance(steps, list):
            continue

        meta_file = puzzle_dir / "metadata.json"
        fold_types: list = []
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
                # fold_types are in forward order; CoT unfolds in reverse order
                fold_types = list(reversed(meta.get("fold_types", [])))
            except json.JSONDecodeError:
                pass

        for step in steps:
            if not isinstance(step, dict):
                continue
            if step.get("reasoning") and step.get("fold_type"):
                continue  # already done
            step_idx = step.get("step")
            if step_idx is None:
                continue
            # current state: step 0 is the fully folded+punched state (question image)
            # between steps: cot_{step_idx-1} is current, cot_{step_idx} is next
            img_name = step.get("image") or f"cot_{step_idx:02d}.png"
            nxt = puzzle_dir / img_name
            if step_idx == 0:
                cur = question_img
            else:
                prev_img = steps[step_idx - 1].get("image") or f"cot_{step_idx - 1:02d}.png"
                cur = puzzle_dir / prev_img
            if not cur.exists() or not nxt.exists():
                print(f"  skip (missing image): {puzzle_dir} step {step_idx}", file=sys.stderr)
                continue

            if step_idx < len(fold_types):
                gt_fold_type = fold_types[step_idx]
                remaining = fold_types[step_idx + 1:]
                remaining_str = ", ".join(remaining) if remaining else "none (this is the final unfold)"
            else:
                gt_fold_type = "?"
                remaining_str = "unknown"

            key = f"{group}__{puzzle_dir.name}__step_{step_idx:02d}"
            jobs.append((key, question_img, cur, nxt, gt_fold_type, remaining_str))
    return jobs


def build_jsonl(jobs: List[Tuple[str, Path, Path, Path, str, str]], out_path: Path) -> None:
    with out_path.open("w") as f:
        for key, question_img, cur, nxt, gt_fold_type, remaining_str in jobs:
            prompt = (PROMPT_TEMPLATE
                      .replace("<<<GT_FOLD_TYPE>>>", gt_fold_type)
                      .replace("<<<REMAINING_FOLDS>>>", remaining_str))
            request = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": prompt},
                            _png_part(question_img),
                            _png_part(cur),
                            _png_part(nxt),
                        ],
                    }
                ],
                "generation_config": {"response_mime_type": "application/json"},
            }
            f.write(json.dumps({"key": key, "request": request}) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit Gemini batch job for paper-fold reasoning")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Dataset output dir (from main.py)")
    parser.add_argument("--jsonl", type=str, default=None)
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--display-name", type=str, default="paperfold-reasoning")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.exists():
        sys.exit(f"output-dir does not exist: {output_dir}")

    _load_dotenv()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    print(f"Scanning {output_dir} ...")
    jobs = collect_requests(output_dir)
    if not jobs:
        sys.exit("No pending steps found.")
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
