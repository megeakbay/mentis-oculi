"""
Verify the LLM's identified hinge_id+angle per step matches the ground-truth
rotation recorded in metadata.json["rotation_steps"]. Prints a mismatch list.

Usage:
    python check_rotations.py --output-dir ../output_sft_100
    python check_rotations.py --output-dir ../output_sft_100 --verbose
"""
import argparse
import json
import sys
from pathlib import Path


def _iter_puzzle_dirs(output_dir: Path):
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
    ap = argparse.ArgumentParser(description="Verify predicted rotations against ground truth")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--verbose", action="store_true", help="Print every step, not just mismatches")
    args = ap.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.exists():
        sys.exit(f"output-dir does not exist: {output_dir}")

    total = 0
    wrong_hinge = 0
    wrong_angle = 0
    both_wrong = 0
    missing = 0
    mismatches = []

    for puzzle_dir in _iter_puzzle_dirs(output_dir):
        meta_f = puzzle_dir / "metadata.json"
        cot_f = puzzle_dir / "cot_reasoning.json"
        if not meta_f.exists() or not cot_f.exists():
            continue
        meta = json.loads(meta_f.read_text())
        rotation_steps = meta.get("rotation_steps", [])
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
            if idx is None or idx >= len(rotation_steps):
                continue
            total += 1
            gt = rotation_steps[idx]
            gt_hinge = gt["hinge_id"].upper()
            gt_angle = int(gt["angle"])
            pred_hinge = step.get("hinge_id")
            pred_angle = step.get("angle")
            if pred_hinge is None and pred_angle is None:
                missing += 1
                mismatches.append((puzzle_dir, idx, "MISSING", gt_hinge, gt_angle, None, None))
                continue
            pred_hinge = (pred_hinge or "").strip().upper()
            try:
                pred_angle = int(pred_angle) if pred_angle is not None else None
            except (TypeError, ValueError):
                pred_angle = None
            hinge_ok = pred_hinge == gt_hinge
            angle_ok = pred_angle == gt_angle
            if args.verbose:
                status = "ok" if (hinge_ok and angle_ok) else "MISMATCH"
                print(f"  {puzzle_dir.name} step {idx}: pred={pred_hinge} {pred_angle}°  gt={gt_hinge} {gt_angle}°  [{status}]")
            if not hinge_ok and not angle_ok:
                both_wrong += 1
                mismatches.append((puzzle_dir, idx, "BOTH_WRONG", gt_hinge, gt_angle, pred_hinge, pred_angle))
            elif not hinge_ok:
                wrong_hinge += 1
                mismatches.append((puzzle_dir, idx, "WRONG_HINGE", gt_hinge, gt_angle, pred_hinge, pred_angle))
            elif not angle_ok:
                wrong_angle += 1
                mismatches.append((puzzle_dir, idx, "WRONG_ANGLE", gt_hinge, gt_angle, pred_hinge, pred_angle))

    print(f"\ntotal checked:  {total}")
    print(f"missing:        {missing}")
    print(f"wrong hinge:    {wrong_hinge}")
    print(f"wrong angle:    {wrong_angle}")
    print(f"both wrong:     {both_wrong}")
    print(f"correct:        {total - missing - wrong_hinge - wrong_angle - both_wrong}")

    if mismatches and not args.verbose:
        print()
        print("=== Mismatches ===")
        for puzzle_dir, idx, kind, gt_h, gt_a, pred_h, pred_a in mismatches:
            print(f"  {puzzle_dir.name} step {idx}: {kind}  gt={gt_h} {gt_a}°  pred={pred_h} {pred_a}°")


if __name__ == "__main__":
    main()
