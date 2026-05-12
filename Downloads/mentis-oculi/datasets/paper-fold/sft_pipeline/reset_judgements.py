"""
Clear all judgement fields from cot_reasoning.json files so judge_inline
will re-score every step from scratch.

Usage:
    python reset_judgements.py --output-dir ../output_sft_500
    python reset_judgements.py --output-dir ../output_sft_500 --dry-run
"""
import argparse
import json
import os
from pathlib import Path


def _iter_puzzle_dirs(output_dir: Path):
    level_dirs = sorted(output_dir.glob("level_*"))
    if level_dirs:
        for level_dir in level_dirs:
            if not level_dir.is_dir():
                continue
            for puzzle_dir in sorted(level_dir.glob("puzzle_*")):
                if puzzle_dir.is_dir():
                    yield puzzle_dir
    else:
        for puzzle_dir in sorted(output_dir.glob("puzzle_*")):
            if puzzle_dir.is_dir():
                yield puzzle_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    output_dir = Path(args.output_dir).resolve()
    total_steps = 0
    total_cleared = 0

    for puzzle_dir in _iter_puzzle_dirs(output_dir):
        cot_file = puzzle_dir / "cot_reasoning.json"
        if not cot_file.exists():
            continue
        try:
            steps = json.loads(cot_file.read_text())
        except json.JSONDecodeError:
            continue

        cleared = 0
        for step in steps:
            if isinstance(step, dict) and "judgement" in step:
                del step["judgement"]
                cleared += 1
            total_steps += 1

        if cleared and not args.dry_run:
            tmp = cot_file.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(steps, indent=2))
            os.replace(tmp, cot_file)

        if cleared:
            print(f"  {'[dry] ' if args.dry_run else ''}cleared {cleared} judgement(s) in {puzzle_dir.name}")
        total_cleared += cleared

    print(f"\nDone. {total_cleared}/{total_steps} steps cleared.")


if __name__ == "__main__":
    main()
