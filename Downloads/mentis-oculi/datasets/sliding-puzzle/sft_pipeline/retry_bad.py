"""
Retry puzzle steps whose reasoning is missing, leaks referential phrases,
or (optionally) whose predicted move doesn't match ground truth.
Synchronous Gemini calls, no batch. Writes back into cot_reasoning.json.

Usage:
    python retry_bad.py --output-dir ../output_sft_100
    python retry_bad.py --output-dir ../output_sft_100 --fix-wrong-move
    python retry_bad.py --output-dir ../output_sft_100 --model gemini-3.1-pro-preview
"""
import argparse
import json
import os
import sys
from pathlib import Path

from google import genai
from google.genai import types
import PIL.Image


PROMPT = """
You are analyzing a sliding tile puzzle. You are given:
- The current puzzle state (tiles scrambled, one black blank tile).
- The next puzzle state (use this only to determine which tile moved).

Your task:
Identify which single tile moved and in what direction (up, down, left, or right — the direction the tile moved into the blank space). Then write natural logical reasoning explaining WHY that move helps unscramble the puzzle.

CRITICAL CONSTRAINTS:
- Write as if you deduced the move purely from the current state.
- NEVER use the words "image", "images", "hint", "next state", "second", or any phrase referring to how many views you were given.
- Just reason about the puzzle.

Respond EXACTLY with this JSON:
{
  "reasoning": "<Your natural, logical analysis of why this move helps>",
  "move": "<DIRECTION>"
}
where <DIRECTION> is exactly one of: up, down, left, right.
"""

VALID_MOVES = {"up", "down", "left", "right"}

BAD_PHRASES = [
    "first image", "second image", "next state", "hint", "following image",
]


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


def _iter_puzzle_dirs(output_dir: Path):
    for puzzle_dir in sorted(output_dir.glob("puzzle_*")):
        if puzzle_dir.is_dir():
            yield puzzle_dir


def _is_hint_leak(text: str) -> bool:
    tl = text.lower()
    return any(p in tl for p in BAD_PHRASES)


def find_targets(output_dir: Path, fix_wrong_move: bool):
    targets = []
    for puzzle_dir in _iter_puzzle_dirs(output_dir):
        cot_f = puzzle_dir / "cot_reasoning.json"
        meta_f = puzzle_dir / "metadata.json"
        if not cot_f.exists():
            continue
        solution_moves = []
        if fix_wrong_move and meta_f.exists():
            meta = json.loads(meta_f.read_text())
            solution_moves = meta.get("solution_moves", [])
        try:
            steps = json.loads(cot_f.read_text())
        except json.JSONDecodeError:
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            idx = step.get("step")
            if idx is None:
                continue
            reasoning = step.get("reasoning") or ""
            move = step.get("move")
            if not reasoning or not move:
                targets.append((cot_f, idx, "missing"))
            elif _is_hint_leak(reasoning):
                targets.append((cot_f, idx, "hint-leak"))
            elif fix_wrong_move and idx < len(solution_moves):
                gt = solution_moves[idx].strip().lower()
                pred = str(move).strip().lower()
                if pred != gt:
                    targets.append((cot_f, idx, f"wrong-move(pred={pred} gt={gt})"))
    return targets


def retry_one(client, puzzle_dir: Path, step_idx: int, model: str) -> dict:
    initial = puzzle_dir / "initial.png"
    cur = initial if step_idx == 0 else puzzle_dir / f"cot_{step_idx - 1:02d}.png"
    nxt = puzzle_dir / f"cot_{step_idx:02d}.png"
    img_cur = PIL.Image.open(cur)
    img_nxt = PIL.Image.open(nxt)
    resp = client.models.generate_content(
        model=model,
        contents=[PROMPT, img_cur, img_nxt],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    data = json.loads(resp.text)
    if isinstance(data, list) and data:
        data = data[0]
    move = data.get("move", "")
    if isinstance(move, str):
        move = move.strip().lower()
        if move not in VALID_MOVES:
            move = None
    else:
        move = None
    return {"reasoning": data.get("reasoning"), "move": move}


def main():
    ap = argparse.ArgumentParser(description="Retry bad/missing sliding-puzzle reasoning steps")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--fix-wrong-move", action="store_true",
                    help="Also retry steps where predicted move != ground truth")
    args = ap.parse_args()

    _load_dotenv()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.exists():
        sys.exit(f"output-dir does not exist: {output_dir}")

    targets = find_targets(output_dir, args.fix_wrong_move)
    if not targets:
        print("No targets found.")
        return
    print(f"{len(targets)} target(s) to retry:")
    for f, s, reason in targets:
        print(f"  {f.parent.name} step {s}  [{reason}]")

    for cot_f, step_idx, reason in targets:
        puzzle_dir = cot_f.parent
        print(f"Retrying {puzzle_dir.name} step {step_idx} ({reason}) ...")
        try:
            new_fields = retry_one(client, puzzle_dir, step_idx, args.model)
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            continue
        steps = json.loads(cot_f.read_text())
        for step in steps:
            if isinstance(step, dict) and step.get("step") == step_idx:
                for stale in ("error", "gemini_response", "raw", "reasoning", "move", "judgement"):
                    step.pop(stale, None)
                step.update(new_fields)
                break
        cot_f.write_text(json.dumps(steps, indent=2))
        snippet = (new_fields.get("reasoning") or "")[:80]
        print(f"  ok: move={new_fields.get('move')}  {snippet}...")


if __name__ == "__main__":
    main()
