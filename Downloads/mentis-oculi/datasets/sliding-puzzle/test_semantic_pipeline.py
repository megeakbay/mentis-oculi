"""
Small end-to-end test of the semantic reasoning pipeline for sliding puzzle.

Steps:
  1. Pick one puzzle from --output-dir
  2. Generate semantic.json for its target image (what the solved puzzle shows)
  3. Generate step-by-step CoT reasoning for each move, injecting the semantic
     description so the model knows WHAT the puzzle is a picture of
  4. Write cot_reasoning.json and print the result

Run from the sliding-puzzle directory:
    python test_semantic_pipeline.py --output-dir output/level_01 --puzzle puzzle_0001
    python test_semantic_pipeline.py --output-dir output/level_01  # picks first puzzle
"""

import argparse
import json
import os
import sys
from pathlib import Path

import PIL.Image
from google import genai
from google.genai import types


# ── prompts ──────────────────────────────────────────────────────────────────

SEMANTIC_PROMPT = """Divide this image into four quadrants and write one short sentence per quadrant describing what is visually there. Keep each sentence brief and natural — like "blue sky with a sun in the corner" or "grass with a bit of sky at the top". Mention what is present and roughly where, but do not list colors mechanically.

Output exactly four lines:
Top-left: <one sentence>
Top-right: <one sentence>
Bottom-left: <one sentence>
Bottom-right: <one sentence>

Nothing else."""


STEP_PROMPT_TEMPLATE = """You are analyzing a sliding tile puzzle.

Visual layout of the solved puzzle: {semantic}

You are given:
1. The current scrambled puzzle state (one black blank tile).
2. The next state after one tile has moved (use it only to see which tile moved).

Ground-truth for this step:
- The blank space moves {gt_move}.

Convention: the direction always refers to which way the BLANK moves.
For example "up" means the blank slides upward, pulling the tile above it downward.

Your task:
Write reasoning explaining WHY the blank should move {gt_move} now.
Reference what part of the visual layout is being restored or positioned.
Your move in the JSON MUST match the ground truth.

CRITICAL CONSTRAINTS:
- Write as if you deduced the move purely from the current state.
- NEVER mention "second image", "next state", "hint", or any reference to extra views.
- 2-5 sentences max.

Respond EXACTLY with this JSON:
{{
  "reasoning": "<logical explanation referencing the visual layout>",
  "move": "{gt_move}"
}}"""

VALID_MOVES = {"up", "down", "left", "right"}
MODEL = "gemini-3-pro-preview"


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


def generate_semantic(client, puzzle_dir: Path, overwrite: bool = False) -> str:
    semantic_path = puzzle_dir / "semantic.json"
    target_path = puzzle_dir / "target.png"

    if not target_path.exists():
        raise FileNotFoundError(f"No target.png in {puzzle_dir}")

    if semantic_path.exists() and not overwrite:
        data = json.loads(semantic_path.read_text())
        desc = data.get("target_semantic", "")
        print(f"  [semantic] loaded from cache: {desc[:80]}...")
        return desc

    print(f"  [semantic] calling Gemini on target.png ...", end=" ", flush=True)
    img = PIL.Image.open(target_path)
    resp = client.models.generate_content(
        model=MODEL,
        contents=[SEMANTIC_PROMPT, img],
    )
    desc = resp.text.strip()
    semantic_path.write_text(json.dumps({
        "puzzle_id": puzzle_dir.name,
        "target_image": "target.png",
        "target_semantic": desc,
    }, indent=2))
    print("done")
    print(f"    → {desc}")
    return desc


def generate_step_reasoning(client, puzzle_dir: Path, step_idx: int,
                             gt_move: str, semantic: str) -> dict:
    initial = puzzle_dir / "initial.png"
    cur = initial if step_idx == 0 else puzzle_dir / f"cot_{step_idx - 1:02d}.png"
    nxt = puzzle_dir / f"cot_{step_idx:02d}.png"

    if not cur.exists() or not nxt.exists():
        raise FileNotFoundError(f"Missing images for step {step_idx} in {puzzle_dir}")

    prompt = STEP_PROMPT_TEMPLATE.format(
        semantic=semantic,
        gt_move=gt_move,
    )

    img_cur = PIL.Image.open(cur)
    img_nxt = PIL.Image.open(nxt)

    resp = client.models.generate_content(
        model=MODEL,
        contents=[prompt, img_cur, img_nxt],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    data = json.loads(resp.text)
    if isinstance(data, list) and data:
        data = data[0]

    move = str(data.get("move", "")).strip().lower()
    if move not in VALID_MOVES:
        move = gt_move  # fallback to ground truth

    return {
        "reasoning": data.get("reasoning", ""),
        "move": move,
    }


def run_pipeline(client, puzzle_dir: Path, overwrite: bool = False) -> None:
    print(f"\n{'='*60}")
    print(f"Puzzle: {puzzle_dir.name}")
    print(f"{'='*60}")

    meta_file = puzzle_dir / "metadata.json"
    if not meta_file.exists():
        print(f"  [skip] no metadata.json")
        return

    meta = json.loads(meta_file.read_text())
    solution_moves = meta.get("solution_moves", [])
    if not solution_moves:
        print(f"  [skip] no solution_moves in metadata")
        return

    print(f"  Solution: {solution_moves}")

    # Step 1: semantic description
    semantic = generate_semantic(client, puzzle_dir, overwrite=overwrite)

    # Step 2: step-by-step reasoning
    cot_file = puzzle_dir / "cot_reasoning.json"
    steps = []
    for step_idx, gt_move in enumerate(solution_moves):
        print(f"  [step {step_idx}] move={gt_move} ...", end=" ", flush=True)

        try:
            result = generate_step_reasoning(
                client, puzzle_dir, step_idx, gt_move, semantic
            )
            step_entry = {
                "step": step_idx,
                "image": f"cot_{step_idx:02d}.png",
                "move": result["move"],
                "reasoning": result["reasoning"],
            }
            print(f"ok (move={result['move']})")
            print(f"    reasoning: {result['reasoning'][:120]}...")
        except Exception as e:
            print(f"FAILED: {e}")
            step_entry = {
                "step": step_idx,
                "image": f"cot_{step_idx:02d}.png",
                "error": str(e),
            }

        steps.append(step_entry)

    cot_file.write_text(json.dumps(steps, indent=2))
    print(f"\n  Saved cot_reasoning.json ({len(steps)} steps)")


def main():
    ap = argparse.ArgumentParser(description="Test semantic reasoning pipeline on one sliding puzzle")
    ap.add_argument("--output-dir", required=True, help="e.g. output/level_01")
    ap.add_argument("--puzzle", default=None, help="Puzzle dir name e.g. puzzle_0001 (default: first found)")
    ap.add_argument("--overwrite", action="store_true", help="Re-generate even if semantic.json exists")
    args = ap.parse_args()

    _load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY not set — add it to .env or export it")

    client = genai.Client(api_key=api_key)
    output_dir = Path(args.output_dir).resolve()

    if args.puzzle:
        puzzle_dir = output_dir / args.puzzle
        if not puzzle_dir.exists():
            sys.exit(f"Puzzle dir not found: {puzzle_dir}")
    else:
        candidates = sorted(output_dir.glob("puzzle_*"))
        if not candidates:
            sys.exit(f"No puzzle_* dirs found in {output_dir}")
        puzzle_dir = candidates[0]
        print(f"No --puzzle specified, using {puzzle_dir.name}")

    run_pipeline(client, puzzle_dir, overwrite=args.overwrite)

    print(f"\n{'='*60}")
    print("Pipeline complete.")
    print(f"  semantic.json  → {puzzle_dir / 'semantic.json'}")
    print(f"  cot_reasoning.json → {puzzle_dir / 'cot_reasoning.json'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
