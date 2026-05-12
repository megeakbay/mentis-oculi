"""
Backfill solution_pieces_data into existing puzzle metadata.json files.

For each puzzle, re-runs export() with retry seeds until the solution_mask
matches the one stored in metadata — confirming we found the right seed —
then saves the polygon coords.

Usage:
    python backfill_piece_coords.py --output-dir output_sft_250 --dataset-seed 42
"""
import argparse
import json
import random
from pathlib import Path

from generate import Puzzle

TARGET_SHAPES = {
    "rectangle": "0,1-0,4; 0,4-3,4; 3,4-3,1; 3,1-0,1",
    "square":    "1,1-1,4; 1,4-4,4; 4,4-4,1; 4,1-1,1",
    "L_shape":   "0,0-0,4; 0,4-2,4; 2,4-2,2; 2,2-4,2; 4,2-4,0; 4,0-0,0",
    "T_shape":   "0,3-0,5; 0,5-5,5; 5,5-5,3; 5,3-3,3; 3,3-3,0; 3,0-2,0; 2,0-2,3; 2,3-0,3",
    "cross":     "1,0-1,2; 1,2-0,2; 0,2-0,3; 0,3-1,3; 1,3-1,5; 1,5-2,5; 2,5-2,3; 2,3-3,3; 3,3-3,2; 3,2-2,2; 2,2-2,0; 2,0-1,0",
    "U_shape":   "0,0-0,5; 0,5-1,5; 1,5-1,1; 1,1-4,1; 4,1-4,5; 4,5-5,5; 5,5-5,0; 5,0-0,0",
    "trapezoid": "1,1-2,4; 2,4-4,4; 4,4-5,1; 5,1-1,1",
    "pentagon":  "2,0-0,2; 0,2-1,4; 1,4-4,4; 4,4-5,2; 5,2-2,0",
    "stairs":    "0,0-0,2; 0,2-2,2; 2,2-2,4; 2,4-4,4; 4,4-4,5; 4,5-5,5; 5,5-5,0; 5,0-0,0",
    "arrow":     "0,1-0,4; 0,4-3,4; 3,4-3,5; 3,5-5,3; 5,3-5,2; 5,2-3,0; 3,0-3,1; 3,1-0,1",
    "diamond":   "2,0-0,2; 0,2-2,4; 2,4-4,2; 4,2-2,0",
    "hexagon":   "1,0-0,2; 0,2-1,4; 1,4-4,4; 4,4-5,2; 5,2-4,0; 4,0-1,0",
}


def backfill_puzzle(puzzle_dir: Path, dataset_seed: int) -> bool:
    meta_file = puzzle_dir / "metadata.json"
    if not meta_file.exists():
        return False

    meta = json.loads(meta_file.read_text())

    if "solution_pieces_data" in meta:
        print(f"  [skip] {puzzle_dir.name} already has solution_pieces_data")
        return True

    shape_name = meta["shape_name"]
    puzzle_id = meta["puzzle_id"]
    num_solution_pieces = meta["num_solution_pieces"]
    stored_solution_pieces = meta["solution_pieces"]  # e.g. ["B", "C"]
    stored_mask = "".join(
        "T" if p["is_solution"] else "F" for p in meta["pieces"]
    )

    edges = TARGET_SHAPES[shape_name]
    puzzle = Puzzle.from_edges(edges, grid_size=5)

    # Try each retry seed (mirrors main.py: export_seed = seed + puzzle_id + retry * 10000)
    for retry in range(5):
        export_seed = dataset_seed + puzzle_id + retry * 10000
        random.seed(export_seed)
        try:
            # Call _generate_solution_pieces + _generate_distractors directly
            # to check mask without re-rendering images
            pieces = puzzle._generate_solution_pieces(num_solution_pieces)
            distractors = puzzle._generate_distractors(pieces, 5 - num_solution_pieces)
            options = [(True, p) for p in pieces] + [(False, d) for d in distractors]
            random.shuffle(options)

            candidate_mask = "".join("T" if is_sol else "F" for is_sol, _ in options)
            if candidate_mask != stored_mask:
                continue

            # Mask matches — get colors and piece data
            rng_all = random.Random(export_seed)
            from generate import assign_piece_colors
            all_piece_colors = assign_piece_colors(5, rng_all)

            solution_piece_colors = []
            for sp in pieces:
                for idx, (is_sol, p) in enumerate(options):
                    if p.equals(sp):
                        solution_piece_colors.append(all_piece_colors[idx])
                        break

            solution_pieces_data = [
                {
                    "coords": list(zip(p.exterior.coords.xy[0], p.exterior.coords.xy[1])),
                    "color": solution_piece_colors[i],
                }
                for i, p in enumerate(pieces)
            ]

            meta["solution_pieces_data"] = solution_pieces_data
            meta_file.write_text(json.dumps(meta, indent=2))
            print(f"  [ok]   {puzzle_dir.name} backfilled (retry={retry})")
            return True

        except Exception as e:
            continue

    print(f"  [fail] {puzzle_dir.name} — could not match stored mask after 5 retries")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dataset-seed", type=int, default=42)
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    puzzle_dirs = sorted(output_dir.glob("puzzle_*"))
    print(f"Backfilling {len(puzzle_dirs)} puzzles in {output_dir} (dataset_seed={args.dataset_seed})")

    ok = fail = skip = 0
    for puzzle_dir in puzzle_dirs:
        result = backfill_puzzle(puzzle_dir, args.dataset_seed)
        if result:
            ok += 1
        else:
            fail += 1

    print(f"\nDone: {ok} ok, {fail} failed")


if __name__ == "__main__":
    main()
