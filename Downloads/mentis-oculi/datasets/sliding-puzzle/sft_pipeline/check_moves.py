"""
Verify the LLM's identified move per step matches the ground-truth move
recorded in metadata.json["solution_moves"]. Prints a mismatch list.

Usage:
    python check_moves.py --output-dir ../output_sft_100
    python check_moves.py --output-dir ../output_sft_100 --verbose
"""
import argparse
import json
import sys
from pathlib import Path


def _iter_puzzle_dirs(output_dir: Path):
    for puzzle_dir in sorted(output_dir.glob("puzzle_*")):
        if puzzle_dir.is_dir():
            yield puzzle_dir


def main():
    ap = argparse.ArgumentParser(description="Verify predicted moves against ground truth")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--verbose", action="store_true", help="Print every step, not just mismatches")
    args = ap.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.exists():
        sys.exit(f"output-dir does not exist: {output_dir}")

    total = 0
    wrong_move = 0
    missing = 0
    mismatches = []

    for puzzle_dir in _iter_puzzle_dirs(output_dir):
        meta_f = puzzle_dir / "metadata.json"
        cot_f = puzzle_dir / "cot_reasoning.json"
        if not meta_f.exists() or not cot_f.exists():
            continue
        meta = json.loads(meta_f.read_text())
        solution_moves = meta.get("solution_moves", [])
        try:
            steps = json.loads(cot_f.read_text())
        except json.JSONDecodeError:
            print(f"  skip (bad json): {cot_f}", file=sys.stderr)
            continue
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            idx = step.get("step")
            if idx is None or idx >= len(solution_moves):
                continue
            total += 1
            gt_move = solution_moves[idx].strip().lower()
            pred_move = step.get("move")
            if pred_move is None:
                missing += 1
                mismatches.append((puzzle_dir, idx, "MISSING", gt_move, pred_move))
                continue
            pred_move = str(pred_move).strip().lower()
            ok = pred_move == gt_move
            if args.verbose:
                status = "ok" if ok else "MISMATCH"
                print(f"  {puzzle_dir.name} step {idx}: pred={pred_move} gt={gt_move}  [{status}]")
            if not ok:
                wrong_move += 1
                mismatches.append((puzzle_dir, idx, "WRONG", gt_move, pred_move))

    print(f"\ntotal checked: {total}")
    print(f"missing move:  {missing}")
    print(f"wrong move:    {wrong_move}")
    print(f"correct:       {total - missing - wrong_move}")

    if mismatches and not args.verbose:
        print()
        print("=== Mismatches ===")
        for puzzle_dir, idx, kind, gt, pred in mismatches:
            print(f"  {puzzle_dir.name} step {idx}: {kind}  gt={gt}  pred={pred}")


if __name__ == "__main__":
    main()
