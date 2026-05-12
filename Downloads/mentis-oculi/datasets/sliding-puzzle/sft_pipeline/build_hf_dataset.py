"""
Convert a sliding-puzzle output dir into a HF DatasetDict matching the schema of
VCoTReasoningKU/TangramData-Instruct-Colored.

Splits: train_colored (90%) + test_colored (10%), shuffled, seed=0.

The `answer` field is synthesized deterministically from
metadata["solution_moves"] — no LLM needed.

Usage:
    python build_hf_dataset.py --output-dir ../output_sft_100 --out ../hf_slidingpuzzle_100
    python build_hf_dataset.py --output-dir ../output_sft_100 --out ../hf_slidingpuzzle_100 --limit 10
    python build_hf_dataset.py ... --push-to-hub VCoTReasoningKU/SlidingPuzzle-Instruct-Colored
"""
import argparse
import json
import random
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value


QUESTION_PROMPT = (
    "Solve this sliding tile puzzle. The image shows scrambled tiles with one black blank tile. "
    "At each step, decide which single tile to slide into the blank space (up, down, left, or right) "
    "to reconstruct the original image. Reason step by step, then state the move direction."
)

INTERLEAVE_ITEM = [{
    "content": Value("string"),
    "index": Value("int64"),
    "type": Value("string"),
}]

FEATURES = Features({
    "id": Value("string"),
    "question_interleave": INTERLEAVE_ITEM,
    "question_images": [Image()],
    "solution_interleave": INTERLEAVE_ITEM,
    "solution_images": [Image()],
    "answer": Value("string"),
    "options": Value("string"),
    "knowledge": Value("string"),
    "subknowledge": Value("string"),
})


def img_entry(path: Path, rel: str) -> dict:
    return {"bytes": path.read_bytes(), "path": rel}


def synth_answer(meta: dict) -> str:
    """Build 'left, down, right, up' from metadata['solution_moves']."""
    return ", ".join(meta.get("solution_moves", []))


def _iter_puzzle_dirs(output_dir: Path):
    for puzzle_dir in sorted(output_dir.glob("puzzle_*")):
        if puzzle_dir.is_dir():
            yield puzzle_dir


def build_row(puzzle_dir: Path) -> dict | None:
    cot_file = puzzle_dir / "cot_reasoning.json"
    meta_file = puzzle_dir / "metadata.json"
    initial = puzzle_dir / "initial.png"
    if not (cot_file.exists() and meta_file.exists() and initial.exists()):
        return None

    steps = json.loads(cot_file.read_text())
    meta = json.loads(meta_file.read_text())
    if not isinstance(steps, list) or not steps:
        return None

    pid = int(meta["puzzle_id"])
    rid = f"slidingpuzzle_{pid:04d}"

    steps_sorted = sorted(
        (s for s in steps if isinstance(s, dict) and "step" in s),
        key=lambda s: s["step"],
    )
    if not steps_sorted:
        return None

    sol_interleave = []
    sol_images = []
    for i, step in enumerate(steps_sorted):
        reasoning = step.get("reasoning", "") or ""
        move = step.get("move", "") or ""
        text = f"<think>{reasoning}</think>\n{move}"
        sol_interleave.append({"content": text, "index": i, "type": "text"})

        img_name = step.get("image") or f"cot_{step['step']:02d}.png"
        img_path = puzzle_dir / img_name
        if not img_path.exists():
            return None
        rel = f"images/{rid}_step_{i + 1}.png"
        sol_interleave.append({"content": rel, "index": i, "type": "image"})
        sol_images.append(img_entry(img_path, rel))

    initial_rel = f"images/{rid}_initial.png"
    q_interleave = [
        {"content": QUESTION_PROMPT, "index": 0, "type": "text"},
        {"content": initial_rel, "index": 0, "type": "image"},
    ]
    q_images = [img_entry(initial, initial_rel)]

    return {
        "id": rid,
        "question_interleave": q_interleave,
        "question_images": q_images,
        "solution_interleave": sol_interleave,
        "solution_images": sol_images,
        "answer": synth_answer(meta),
        "options": None,
        "knowledge": "SlidingPuzzle",
        "subknowledge": "SlidingPuzzle",
    }


def collect_rows(output_dir: Path, limit: int | None) -> list[dict]:
    rows = []
    for puzzle_dir in _iter_puzzle_dirs(output_dir):
        row = build_row(puzzle_dir)
        if row is None:
            print(f"  [skip] {puzzle_dir}")
            continue
        rows.append(row)
        if limit and len(rows) >= limit:
            break
    return rows


def main():
    ap = argparse.ArgumentParser(description="Build HF DatasetDict from sliding-puzzle output dir")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--out", required=True, help="Local save_to_disk path")
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="Cap total puzzles (smoke test)")
    ap.add_argument("--push-to-hub", default=None)
    args = ap.parse_args()

    output_dir = Path(args.output_dir).resolve()
    out_path = Path(args.out).resolve()

    print(f"Scanning {output_dir} ...")
    rows = collect_rows(output_dir, args.limit)
    print(f"  collected {len(rows)} puzzle(s)")

    rng = random.Random(args.seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    n_test = max(1, round(len(shuffled) * args.test_frac)) if shuffled else 0
    test_rows = shuffled[:n_test]
    train_rows = shuffled[n_test:]
    print(f"  split: train={len(train_rows)}  test={len(test_rows)}")

    train_ds = Dataset.from_list(train_rows, features=FEATURES)
    test_ds = Dataset.from_list(test_rows, features=FEATURES)
    ddict = DatasetDict({"train_colored": train_ds, "test_colored": test_ds})

    print(f"Saving to {out_path} ...")
    ddict.save_to_disk(str(out_path))
    print("  done.")

    if args.push_to_hub:
        print(f"Pushing to hub: {args.push_to_hub}")
        ddict.push_to_hub(args.push_to_hub)
        print("  pushed.")


if __name__ == "__main__":
    main()
