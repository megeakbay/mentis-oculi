"""
Inline (non-batch) reasoning fill for form-board puzzles.

Walks an output dir (level_*/puzzle_* or flat puzzle_*), and for every step
in cot_reasoning.json missing reasoning+piece, calls Gemini synchronously
and writes the parsed result back. Intended for small smoke tests only —
use sft_pipeline/batch_submit.py for real runs.

Usage:
    python sft_pipeline/fill_reasoning_inline.py --output-dir output_sft_test
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import reasoning_text


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


def process_puzzle(puzzle_dir: Path) -> tuple[int, int]:
    cot_file = puzzle_dir / "cot_reasoning.json"
    legend = puzzle_dir / "combined.png"
    silhouette = puzzle_dir / "silhouette.png"
    final = puzzle_dir / "bordered.png"
    if not (cot_file.exists() and legend.exists() and silhouette.exists() and final.exists()):
        return 0, 0

    steps = json.loads(cot_file.read_text())
    if not isinstance(steps, list):
        return 0, 0

    filled = 0
    attempted = 0
    for step in steps:
        if not isinstance(step, dict):
            continue
        if step.get("reasoning") and step.get("piece"):
            continue
        step_idx = step.get("step")
        if step_idx is None:
            continue
        cur = silhouette if step_idx == 0 else puzzle_dir / f"cot_{step_idx - 1:02d}.png"
        nxt = puzzle_dir / f"cot_{step_idx:02d}.png"
        if not cur.exists() or not nxt.exists():
            print(f"    skip step {step_idx} (missing image)", file=sys.stderr)
            continue
        attempted += 1
        try:
            info, raw = reasoning_text.generate_step_reasoning(str(legend), str(cur), str(nxt), str(final))
        except Exception as e:
            print(f"    step {step_idx} failed: {e}", file=sys.stderr)
            step["raw"] = str(e)
            continue
        for stale in ("error", "gemini_response", "raw", "reasoning", "piece", "response"):
            step.pop(stale, None)
        step["reasoning"] = info.get("reasoning")
        step["piece"] = info.get("piece")
        step["raw"] = raw
        filled += 1

    cot_file.write_text(json.dumps(steps, indent=2))
    return attempted, filled


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.exists():
        sys.exit(f"output-dir does not exist: {output_dir}")

    total_attempted = 0
    total_filled = 0
    for puzzle_dir in _iter_puzzle_dirs(output_dir):
        print(f"  {puzzle_dir.relative_to(output_dir)}")
        a, f = process_puzzle(puzzle_dir)
        total_attempted += a
        total_filled += f

    print(f"\nattempted: {total_attempted}  filled: {total_filled}")


if __name__ == "__main__":
    main()
