"""
Convert output_sft_50/ into a HF DatasetDict matching the schema of
VCoTReasoningKU/TangramData-Instruct-Colored.

Splits: train_colored (45/level) + test_colored (5/level), stratified, seed=0.

Usage:
    python build_hf_dataset.py --output-dir ../output_sft_50 --out ../hf_rushhour
    python build_hf_dataset.py --output-dir ../output_sft_50 --out ../hf_rushhour --limit 10
    python build_hf_dataset.py ... --push-to-hub VCoTReasoningKU/RushhourData-Instruct-Colored
"""
import argparse
import json
import random
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value


QUESTION_PROMPT = (
    "Solve this block-sliding puzzle. The red rectangle (R) must reach the green exit. "
    "At each step, decide which single rectangle to move and in what direction so that "
    "R's path becomes clear. Reason step by step, then state the move."
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
    """Build 'A forward, C backward, R forward' from metadata['actions']."""
    id_to_label = {
        obj["id"]: ("R" if obj["id"] == "red_car" else obj.get("label", obj["id"]))
        for obj in meta.get("objects", [])
    }
    parts = []
    for act in meta.get("actions", []):
        lbl = id_to_label.get(act["object_id"], act["object_id"])
        direction = "forward" if act["direction"] > 0 else "backward"
        parts.append(f"{lbl} {direction}")
    return ", ".join(parts)


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

    level = int(meta["level"])
    pid = int(meta["puzzle_id"])
    rid = f"rushhour_level_{level}_puzzle_{pid:04d}"

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
        response = step.get("response", "") or ""
        text = f"<think>{reasoning}</think>\n{response}"
        sol_interleave.append({"content": text, "index": i, "type": "text"})

        img_name = step.get("image") or f"cot_{step['step']:02d}.png"
        img_path = puzzle_dir / img_name
        if not img_path.exists():
            return None
        rel = f"images/{rid}_step_{i+1}.png"
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
        "knowledge": "Rushhour",
        "subknowledge": "Rushhour",
    }


def collect_rows(output_dir: Path, limit: int | None):
    by_level: dict[int, list[dict]] = {}
    for level_dir in sorted(output_dir.glob("level_*")):
        if not level_dir.is_dir():
            continue
        level = int(level_dir.name.split("_")[1])
        for puzzle_dir in sorted(level_dir.glob("puzzle_*")):
            row = build_row(puzzle_dir)
            if row is None:
                print(f"  [skip] {puzzle_dir}")
                continue
            by_level.setdefault(level, []).append(row)
    if limit:
        for lvl in by_level:
            by_level[lvl] = by_level[lvl][:limit]
    return by_level


def stratified_split(by_level: dict[int, list[dict]], test_frac: float, seed: int):
    rng = random.Random(seed)
    train, test = [], []
    for lvl, rows in sorted(by_level.items()):
        shuffled = rows[:]
        rng.shuffle(shuffled)
        n_test = max(1, round(len(shuffled) * test_frac)) if shuffled else 0
        test.extend(shuffled[:n_test])
        train.extend(shuffled[n_test:])
    return train, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--out", required=True, help="Local save_to_disk path")
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="Per-level cap for smoke tests")
    ap.add_argument("--push-to-hub", default=None)
    args = ap.parse_args()

    output_dir = Path(args.output_dir).resolve()
    out_path = Path(args.out).resolve()

    print(f"Scanning {output_dir} ...")
    by_level = collect_rows(output_dir, args.limit)
    total = sum(len(v) for v in by_level.values())
    print(f"  collected {total} puzzles across {len(by_level)} levels: "
          f"{ {k: len(v) for k, v in sorted(by_level.items())} }")

    train_rows, test_rows = stratified_split(by_level, args.test_frac, args.seed)
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
