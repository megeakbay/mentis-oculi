"""
Retry puzzle steps whose reasoning is missing, leaks referential phrases,
or (optionally) whose predicted piece doesn't match ground truth.
Synchronous Gemini calls, no batch. Writes back into cot_reasoning.json.

Usage:
    python retry_bad.py --output-dir ../output_sft_100
    python retry_bad.py --output-dir ../output_sft_100 --fix-wrong-piece
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
You are analyzing a form-board assembly puzzle. You are given:
- The legend image showing the target silhouette and all five candidate pieces (A–E).
- The current assembly state (before this step; may be empty).
- The next assembly state (use this only to determine which piece was placed).
- The fully-solved puzzle (use this only for look-ahead context).

Your task:
Identify which single piece (A, B, C, D, or E) was added in this step. Then write natural logical reasoning explaining WHY that piece fits the open region, covering both the immediate fit and look-ahead compatibility of remaining pieces.

CRITICAL CONSTRAINTS:
- Write as if you deduced the piece purely from the legend and the current state.
- NEVER use the words "image", "images", "hint", "next state", "solved", "third", "fourth",
  or any phrase referring to how many views you were given.
- Just reason about the puzzle.

Respond EXACTLY with this JSON:
{
  "reasoning": "<Your natural, logical analysis of why the piece fits>",
  "piece": "<LABEL>"
}
where <LABEL> is exactly one of A, B, C, D, E.
"""

# Phrases that constitute a hint leak
BAD_PHRASES = [
    "first image", "second image", "third image", "fourth image",
    "next state", "hint", "solved puzzle", "following image",
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


def _is_hint_leak(text: str) -> bool:
    tl = text.lower()
    return any(p in tl for p in BAD_PHRASES)


def find_targets(output_dir: Path, fix_wrong_piece: bool):
    """Return list of (cot_file, step_idx, reason) to retry."""
    targets = []
    for puzzle_dir in _iter_puzzle_dirs(output_dir):
        cot_f = puzzle_dir / "cot_reasoning.json"
        meta_f = puzzle_dir / "metadata.json"
        if not cot_f.exists():
            continue
        solution_pieces = []
        if fix_wrong_piece and meta_f.exists():
            meta = json.loads(meta_f.read_text())
            solution_pieces = meta.get("solution_pieces", [])
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
            piece = step.get("piece")

            # Missing reasoning or piece — always retry
            if not reasoning or not piece:
                targets.append((cot_f, idx, "missing"))
                continue

            # Local hint-leak check (fast, no judge needed)
            if _is_hint_leak(reasoning):
                targets.append((cot_f, idx, "hint-leak"))
                continue

            # Check judge verdict if it exists
            judgement = step.get("judgement")
            if isinstance(judgement, dict) and "error" not in judgement:
                r_ok = judgement.get("is_correct_reasoning", True)
                h_ok = judgement.get("is_correct_no_hints", True)
                if not r_ok and not h_ok:
                    targets.append((cot_f, idx, "judge:[r][h]"))
                    continue
                elif not r_ok:
                    targets.append((cot_f, idx, "judge:[r]"))
                    continue
                elif not h_ok:
                    targets.append((cot_f, idx, "judge:[h]"))
                    continue

            # Wrong-piece check: piece must be one of the valid solution pieces
            if fix_wrong_piece and solution_pieces:
                valid = {p.strip().upper() for p in solution_pieces}
                pred = piece.strip().upper()
                if pred not in valid:
                    targets.append((cot_f, idx, f"wrong-piece(pred={pred} valid={sorted(valid)})"))
    return targets


def retry_one(client, puzzle_dir: Path, step_idx: int, model: str) -> dict:
    legend = puzzle_dir / "combined.png"
    silhouette = puzzle_dir / "silhouette.png"
    final = puzzle_dir / "bordered.png"
    cur = silhouette if step_idx == 0 else puzzle_dir / f"cot_{step_idx - 1:02d}.png"
    nxt = puzzle_dir / f"cot_{step_idx:02d}.png"
    img_legend = PIL.Image.open(legend)
    img_cur = PIL.Image.open(cur)
    img_nxt = PIL.Image.open(nxt)
    img_final = PIL.Image.open(final)
    resp = client.models.generate_content(
        model=model,
        contents=[PROMPT, img_legend, img_cur, img_nxt, img_final],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    data = json.loads(resp.text)
    if isinstance(data, list) and data:
        data = data[0]
    piece = data.get("piece", "")
    if isinstance(piece, str):
        piece = piece.strip().upper()
        if piece not in {"A", "B", "C", "D", "E"}:
            piece = None
    else:
        piece = None
    return {"reasoning": data.get("reasoning"), "piece": piece}


def main():
    ap = argparse.ArgumentParser(description="Retry bad/missing form-board reasoning steps")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--fix-wrong-piece", action="store_true",
                    help="Also retry steps where predicted piece != ground truth")
    args = ap.parse_args()

    _load_dotenv()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.exists():
        sys.exit(f"output-dir does not exist: {output_dir}")

    targets = find_targets(output_dir, args.fix_wrong_piece)
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
                for stale in ("error", "gemini_response", "raw", "reasoning", "piece", "judgement"):
                    step.pop(stale, None)
                step.update(new_fields)
                break
        cot_f.write_text(json.dumps(steps, indent=2))
        snippet = (new_fields.get("reasoning") or "")[:80]
        print(f"  ok: piece={new_fields.get('piece')}  {snippet}...")


if __name__ == "__main__":
    main()
