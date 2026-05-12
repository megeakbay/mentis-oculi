"""
Verify the LLM's identified piece per step matches the ground-truth piece
recorded in metadata.json["solution_pieces"]. Prints a mismatch list.

Usage:
    python check_pieces.py --output-dir ../output_sft_100
    python check_pieces.py --output-dir ../output_sft_100 --verbose
"""
import argparse
import json
import sys
from pathlib import Path


def _iter_puzzle_dirs(output_dir: Path):
    """Yield puzzle dirs for both level_*/puzzle_* and flat puzzle_* layouts."""
    level_dirs = sorted(output_dir.glob("level_*"))
    if level_dirs:
        for level_dir in level_dirs:
            if not level_dir.is_dir():
                continue
            for puzzle_dir in sorted(level_dir.glob("puzzle_*")):
                yield puzzle_dir
    else:
        for puzzle_dir in sorted(output_dir.glob("puzzle_*")):
            yield puzzle_dir


def main():
    ap = argparse.ArgumentParser(description="Verify predicted pieces against ground truth")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--verbose", action="store_true", help="Print every step, not just mismatches")
    args = ap.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.exists():
        sys.exit(f"output-dir does not exist: {output_dir}")

    total = 0
    wrong_piece = 0
    missing = 0
    mismatches = []

    for puzzle_dir in _iter_puzzle_dirs(output_dir):
        meta_f = puzzle_dir / "metadata.json"
        cot_f = puzzle_dir / "cot_reasoning.json"
        if not meta_f.exists() or not cot_f.exists():
            continue
        meta = json.loads(meta_f.read_text())
        solution_pieces = meta.get("solution_pieces", [])
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
            if idx is None or idx >= len(solution_pieces):
                continue
            total += 1
            gt_piece = solution_pieces[idx]
            pred_piece = step.get("piece")
            if pred_piece is None:
                missing += 1
                mismatches.append((puzzle_dir, idx, "MISSING", gt_piece, pred_piece))
                continue
            pred_piece = pred_piece.strip().upper()
            ok = pred_piece == gt_piece
            if args.verbose:
                status = "ok" if ok else "MISMATCH"
                print(f"  {puzzle_dir.name} step {idx}: pred={pred_piece} gt={gt_piece}  [{status}]")
            if not ok:
                wrong_piece += 1
                mismatches.append((puzzle_dir, idx, "WRONG", gt_piece, pred_piece))

    print(f"\ntotal checked: {total}")
    print(f"missing piece: {missing}")
    print(f"wrong piece:   {wrong_piece}")
    print(f"correct:       {total - missing - wrong_piece}")

    if mismatches and not args.verbose:
        print()
        print("=== Mismatches ===")
        for puzzle_dir, idx, kind, gt, pred in mismatches:
            print(f"  {puzzle_dir.name} step {idx}: {kind}  gt={gt}  pred={pred}")


if __name__ == "__main__":
    main()
