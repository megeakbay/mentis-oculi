"""
Render a puzzle from an existing puzzle_XXXX/ directory with a given step ordering.

All games read metadata.json from the puzzle dir — no re-generation from seed.

Usage:
    python render_puzzle.py --game rushhour  --puzzle-dir rushhour/output_sft_100/level_03/puzzle_0042 --ordering 1,0,2 --out /tmp/out
    python render_puzzle.py --game hinge     --puzzle-dir hinge-folding/output_sft_250/puzzle_0001     --ordering 1,0,2 --out /tmp/out
    python render_puzzle.py --game formboard --puzzle-dir form-board/output_sft_250/puzzle_0001        --ordering 2,0,1 --out /tmp/out
    python render_puzzle.py --game sliding   --puzzle-dir sliding-puzzle/output/puzzle_0001            --ordering 2,1,0 --out /tmp/out
    python render_puzzle.py --game paperfold --puzzle-dir paper-fold/output_sft_500/puzzle_0001        --out /tmp/out

--ordering is a comma-separated permutation of step indices (0-based).
Omit --ordering to render the original order.
Paper-fold ordering is ignored (folds are physically sequential).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Rush Hour
# ──────────────────────────────────────────────────────────────────────────────

def render_rushhour(puzzle_dir: Path, ordering: list[int], out: Path):
    sys.path.insert(0, str(Path(__file__).parent / "rushhour"))
    from generator import render_instance, apply_action
    import matplotlib.pyplot as plt

    meta_file = puzzle_dir / "metadata.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"No metadata.json in {puzzle_dir}")

    meta = json.loads(meta_file.read_text())
    actions = meta["actions"]
    instance = {"board": meta["board"], "objects": meta["objects"]}

    if ordering is None:
        ordering = list(range(len(actions)))
    if sorted(ordering) != list(range(len(actions))):
        raise ValueError(f"--ordering must be a permutation of 0..{len(actions)-1}, got {ordering}")

    reordered = [actions[i] for i in ordering]

    fig = render_instance(instance)
    fig.savefig(out / "initial.png", dpi=200)
    plt.close(fig)
    print("  saved initial.png")

    state = instance
    for step_idx, action in enumerate(reordered):
        state = apply_action(state, action)
        fig = render_instance(state)
        fig.savefig(out / f"cot_{step_idx:02d}.png", dpi=200)
        plt.close(fig)
        label = action["object_id"]
        direction = "forward" if action["direction"] > 0 else "backward"
        print(f"  saved cot_{step_idx:02d}.png  ({label} {direction})")

    out_meta = {
        "puzzle_dir": str(puzzle_dir),
        "ordering": ordering,
        "actions": [{"object_id": a["object_id"], "direction": a["direction"]} for a in reordered],
    }
    (out / "meta.json").write_text(json.dumps(out_meta, indent=2))
    print("  saved meta.json")


# ──────────────────────────────────────────────────────────────────────────────
# Hinge-folding
# ──────────────────────────────────────────────────────────────────────────────

def render_hinge(puzzle_dir: Path, ordering: list[int], out: Path):
    # Re-simulate folding with the reordered rotation sequence and re-render each step.
    sys.path.insert(0, str(Path(__file__).parent / "hinge-folding"))
    from generator import (
        _calculate_initial_positions,
        calculate_folded_positions,
        create_initial_image,
        create_target_image,
        create_cot_image,
        create_combined_image,
    )

    meta_file = puzzle_dir / "metadata.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"No metadata.json in {puzzle_dir}")

    meta = json.loads(meta_file.read_text())
    shape_names     = meta["shape_chain"]
    rotation_angles = meta["rotation_angles"]
    piece_colors    = meta["piece_colors"]
    num_hinges      = meta["num_hinges"]
    rotation_steps  = meta["rotation_steps"]

    base_shapes = {
        "square":    {"type": "square",    "size": 1.0},
        "diamond":   {"type": "diamond",   "size": 1.0},
        "triangle":  {"type": "triangle",  "size": 1.0},
        "rectangle": {"type": "rectangle", "width": 1.0, "height": 0.5},
    }
    shape_chain = [base_shapes[n] for n in shape_names]

    if ordering is None:
        ordering = list(range(num_hinges))
    if sorted(ordering) != list(range(num_hinges)):
        raise ValueError(f"--ordering must be a permutation of 0..{num_hinges-1}")

    reordered_angles = [rotation_angles[i] for i in ordering]

    # Compute bounds over all states in the new ordering
    raw_initial = _calculate_initial_positions(shape_chain, scale=1.0, center_offset=(0, 0))
    cx = (min(x for x, y in raw_initial) + max(x for x, y in raw_initial)) / 2
    cy = (min(y for x, y in raw_initial) + max(y for x, y in raw_initial)) / 2
    initial_positions = _calculate_initial_positions(shape_chain, scale=1.0, center_offset=(-cx, -cy))

    all_x, all_y = [], []
    for x, y in initial_positions:
        s = shape_chain[0].get("size", 1.0)
        all_x += [x - s, x + s]; all_y += [y - s, y + s]
    for step in range(len(reordered_angles) + 1):
        pos = calculate_folded_positions(shape_chain, initial_positions, reordered_angles[:step])
        for x, y, _ in pos:
            s = shape_chain[0].get("size", 1.0)
            all_x += [x - s, x + s]; all_y += [y - s, y + s]

    margin = 0.5
    scale = min(4.0 / max(max(all_x) - min(all_x), max(all_y) - min(all_y), 1e-6), 2.0)
    bounds = {
        "xmin": min(all_x) * scale - margin, "xmax": max(all_x) * scale + margin,
        "ymin": min(all_y) * scale - margin, "ymax": max(all_y) * scale + margin,
    }
    scaled_positions = [(x * scale, y * scale) for x, y in initial_positions]

    create_initial_image(out / "initial.png", shape_chain, num_hinges, piece_colors, bounds, scale, scaled_positions)
    print("  saved initial.png")
    create_target_image(out / "target.png", shape_chain, scaled_positions, reordered_angles, bounds, scale)
    print("  saved target.png")
    create_combined_image(out / "combined.png", out / "initial.png", out / "target.png")
    print("  saved combined.png")

    for step_idx, orig_idx in enumerate(ordering):
        create_cot_image(
            out / f"cot_{step_idx:02d}.png",
            shape_chain, scaled_positions,
            reordered_angles[:step_idx + 1],
            piece_colors, bounds, scale,
        )
        step = rotation_steps[orig_idx]
        print(f"  saved cot_{step_idx:02d}.png  (hinge {step['hinge_id']} {step['angle']}°)")

    out_meta = {"puzzle_dir": str(puzzle_dir), "ordering": ordering,
                "reordered_steps": [rotation_steps[i] for i in ordering]}
    (out / "meta.json").write_text(json.dumps(out_meta, indent=2))
    print("  saved meta.json")


# ──────────────────────────────────────────────────────────────────────────────
# Form-board
# ──────────────────────────────────────────────────────────────────────────────

def render_formboard(puzzle_dir: Path, ordering: list[int], out: Path):
    # Reconstruct Shapely polygons from coords stored in metadata.json,
    # then call _render_chain_of_thought with the reordered pieces.
    sys.path.insert(0, str(Path(__file__).parent / "form-board"))
    from generate import Puzzle
    from shapely.geometry import Polygon

    meta_file = puzzle_dir / "metadata.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"No metadata.json in {puzzle_dir}")

    meta = json.loads(meta_file.read_text())

    if "solution_pieces_data" not in meta:
        raise KeyError(
            "metadata.json has no 'solution_pieces_data' — re-generate this puzzle "
            "with the updated main.py that stores polygon coordinates."
        )

    pieces_data = meta["solution_pieces_data"]
    num_solution_pieces = len(pieces_data)

    # Reconstruct Shapely polygons and their colors from stored coords
    solution_pieces = [Polygon(d["coords"]) for d in pieces_data]
    piece_colors    = [d["color"] for d in pieces_data]

    if ordering is None:
        ordering = list(range(num_solution_pieces))
    if sorted(ordering) != list(range(num_solution_pieces)):
        raise ValueError(f"--ordering must be a permutation of 0..{num_solution_pieces-1}")

    reordered_pieces = [solution_pieces[i] for i in ordering]
    reordered_colors = [piece_colors[i]    for i in ordering]

    # Rebuild Puzzle object just for its _render_chain_of_thought method
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
    shape_name = meta["shape_name"]
    puzzle = Puzzle.from_edges(TARGET_SHAPES[shape_name], grid_size=5)
    puzzle._render_chain_of_thought(reordered_pieces, reordered_colors, out)

    solution_labels = meta["solution_pieces"]
    for step_idx, orig_idx in enumerate(ordering):
        print(f"  saved cot_{step_idx:02d}.png  (piece {solution_labels[orig_idx]})")

    shutil.copy(puzzle_dir / "combined.png", out / "combined.png")
    print("  saved combined.png")

    out_meta = {"puzzle_dir": str(puzzle_dir), "shape": shape_name,
                "ordering": ordering,
                "reordered_labels": [solution_labels[i] for i in ordering]}
    (out / "meta.json").write_text(json.dumps(out_meta, indent=2))
    print("  saved meta.json")


# ──────────────────────────────────────────────────────────────────────────────
# Sliding puzzle
# ──────────────────────────────────────────────────────────────────────────────

def render_sliding(puzzle_dir: Path, ordering: list[int], out: Path):
    # Re-simulate moves in the new order from the stored initial_state.
    sys.path.insert(0, str(Path(__file__).parent / "sliding-puzzle"))
    from generate_puzzles import create_tile_images, grid_to_image, apply_move
    from PIL import Image as PILImage

    meta_file = puzzle_dir / "metadata.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"No metadata.json in {puzzle_dir}")

    meta = json.loads(meta_file.read_text())
    solution_moves  = meta["solution_moves"]
    initial_state   = meta["initial_state"]
    blank_position  = tuple(meta["blank_position"])
    source_image_name = meta.get("source_image")
    num_steps = len(solution_moves)

    # locate source image
    src_path = None
    for candidate in ([puzzle_dir / source_image_name, puzzle_dir.parent / source_image_name]
                      if source_image_name else []):
        if candidate.exists():
            src_path = candidate
            break
    if src_path is None:
        img = PILImage.new("RGB", (384, 384), color=(100, 149, 237))
        src_path = out / "_placeholder.png"
        img.save(src_path)
        print("  warning: source image not found, using colour placeholder")

    src_img = PILImage.open(src_path).convert("RGB")
    tile_images = create_tile_images(src_img, len(initial_state), tile_size=128,
                                     blank_position=blank_position)

    if ordering is None:
        ordering = list(range(num_steps))
    if sorted(ordering) != list(range(num_steps)):
        raise ValueError(f"--ordering must be a permutation of 0..{num_steps-1}")

    reordered_moves = [solution_moves[i] for i in ordering]

    grid_to_image(initial_state, tile_images).save(out / "initial.png")
    print("  saved initial.png")

    state = [row[:] for row in initial_state]
    for step_idx, move in enumerate(reordered_moves):
        state = apply_move(state, move)
        grid_to_image(state, tile_images).save(out / f"cot_{step_idx:02d}.png")
        print(f"  saved cot_{step_idx:02d}.png  ({move})")

    out_meta = {"puzzle_dir": str(puzzle_dir), "ordering": ordering, "moves": reordered_moves}
    (out / "meta.json").write_text(json.dumps(out_meta, indent=2))
    print("  saved meta.json")


# ──────────────────────────────────────────────────────────────────────────────
# Paper-fold  (ordering not applicable — folds are physically sequential)
# ──────────────────────────────────────────────────────────────────────────────

def render_paperfold(puzzle_dir: Path, out: Path, seed: int | None = None):
    sys.path.insert(0, str(Path(__file__).parent / "paper-fold"))
    from main import generate_sequence

    if seed is None:
        # fall back to seed stored in metadata
        meta_file = puzzle_dir / "metadata.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"No metadata.json in {puzzle_dir} and no --seed given")
        meta = json.loads(meta_file.read_text())
        seed = meta.get("seed")
        if seed is None:
            raise ValueError("metadata.json has no 'seed' field — pass --seed explicitly")
        grid_size  = meta.get("grid_size", 3)
        num_folds  = meta["num_folds"]
        puzzle_id  = meta["puzzle_id"]
    else:
        # seed supplied directly — use defaults, ignore puzzle_dir metadata
        grid_size = 3
        num_folds = None   # let generate_sequence pick randomly within min/max
        puzzle_id = 1

    result = generate_sequence(
        n=grid_size,
        min_steps=num_folds if num_folds is not None else 3,
        max_steps=num_folds if num_folds is not None else 5,
        seed=seed,
        save_dir=out,
        puzzle_id=puzzle_id,
    )
    print(f"  seed={seed}  folds={result['num_folds']}")
    print(f"  saved {result['num_folds']} fold steps + combined.png")
    (out / "meta.json").write_text(json.dumps(result, indent=2, default=str))
    print("  saved meta.json")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Render a puzzle with a specific step ordering")
    ap.add_argument("--game", required=True,
                    choices=["rushhour", "hinge", "formboard", "sliding", "paperfold"])
    ap.add_argument("--puzzle-dir", type=str, required=True,
                    help="Path to existing puzzle_XXXX/ directory")
    ap.add_argument("--seed", type=int, default=None,
                    help="(paperfold only) override seed instead of reading from metadata.json")
    ap.add_argument("--ordering", type=str, default=None,
                    help="Comma-separated step indices e.g. '1,0,2'. Omit for original order.")
    ap.add_argument("--out", type=str, required=True,
                    help="Output directory for rendered images")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    puzzle_dir = Path(args.puzzle_dir)

    ordering = None
    if args.ordering:
        ordering = [int(x) for x in args.ordering.split(",")]

    print(f"Game:       {args.game}")
    print(f"Puzzle dir: {puzzle_dir.resolve()}")
    print(f"Ordering:   {ordering or 'original'}")
    print(f"Output:     {out.resolve()}")

    if args.game == "rushhour":
        render_rushhour(puzzle_dir, ordering, out)
    elif args.game == "hinge":
        render_hinge(puzzle_dir, ordering, out)
    elif args.game == "formboard":
        render_formboard(puzzle_dir, ordering, out)
    elif args.game == "sliding":
        render_sliding(puzzle_dir, ordering, out)
    elif args.game == "paperfold":
        if ordering is not None:
            print("  note: --ordering ignored for paper-fold (folds are physically sequential)")
        render_paperfold(puzzle_dir, out, seed=args.seed)

    print("Done.")


if __name__ == "__main__":
    main()
