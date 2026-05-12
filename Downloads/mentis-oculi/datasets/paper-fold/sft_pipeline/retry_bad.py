"""
Retry paper-fold puzzle steps whose reasoning is missing, leaks referential
phrases, or failed the judge (is_correct_reasoning / is_correct_no_hints false).

Synchronous Gemini calls, no batch. Writes back into cot_reasoning.json and
clears the old judgement so judge_inline will re-score the step.

Usage:
    python retry_bad.py --output-dir ../output_sft_100
    python retry_bad.py --output-dir ../output_sft_100 --dry-run
    python retry_bad.py --output-dir ../output_sft_100 --model gemini-3.1-pro-preview
"""
import argparse
import base64
import json
import os
import sys
import time
import traceback
from pathlib import Path

from google import genai
from google.genai import types


PROMPT_TEMPLATE = """
You are analyzing a paper-folding puzzle. You are given:
- The question image: the complete folding sequence with the hole punch applied.
- The current unfolding state (before this step).
- The next unfolding state (use this only to understand which fold is being reversed).

Ground-truth for this step:
- Fold type being reversed: <<<GT_FOLD_TYPE>>>

Your task:
Write reasoning that justifies reversing a <<<GT_FOLD_TYPE>>> fold at this point.

Your reasoning MUST describe the geometric mechanism — specifically:
- Which axis the fold is reversed across (horizontal, vertical, or diagonal)
- How the existing hole(s) reflect or mirror across that axis
- Where the new hole positions appear after unfolding

Do NOT just restate the fold type. Do NOT say "reversing a horizontal fold propagates the hole correctly" — that is not grounded reasoning. You must describe the actual reflection geometry.

CRITICAL CONSTRAINTS:
- Write as if you deduced this purely from the question image and the current unfolding state.
- NEVER use the words "image", "images", "hint", "next state", or any phrase referring to how many views you were given.
- Keep the reasoning to 2–4 short sentences.

Respond EXACTLY with this JSON:
{
  "reasoning": "<geometric explanation describing the reflection axis and hole positions>",
  "fold_type": "<FOLD_TYPE>"
}
where <FOLD_TYPE> is one of: horizontal, vertical, diag_pos, diag_neg.
"""

BAD_PHRASES = [
    "first image", "second image", "third image",
    "next state", "hint", "following image",
]

# Requires genuinely geometric language — not generic words that appear in shallow reasoning too
GROUNDED_KEYWORDS = [
    "reflect", "mirror", "symmetr", "axis",
    "fold line", "fold axis", "across the", "about the",
    "diagonally", "horizontally", "vertically",
    "left half", "right half", "upper half", "lower half",
    "equidistant", "perpendicular",
]


def _is_grounded(text: str) -> bool:
    tl = text.lower()
    return any(kw in tl for kw in GROUNDED_KEYWORDS)

VALID_FOLD_TYPES = {"horizontal", "vertical", "diag_pos", "diag_neg"}

MAX_RETRIES = 3
RETRY_BACKOFF = 4.0


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


def _png_part(path: Path):
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return types.Part(inline_data=types.Blob(mime_type="image/png", data=data))


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


def _is_hint_leak(text: str) -> bool:
    tl = text.lower()
    return any(p in tl for p in BAD_PHRASES)


def _get_gt(puzzle_dir: Path):
    """Return fold_types in reversed (unfolding) order from metadata."""
    meta_f = puzzle_dir / "metadata.json"
    if not meta_f.exists():
        return []
    try:
        meta = json.loads(meta_f.read_text())
        return list(reversed(meta.get("fold_types", [])))
    except json.JSONDecodeError:
        return []


def find_targets(output_dir: Path, fix_ungrounded: bool = False):
    """Return list of (cot_file, step_idx, reason)."""
    targets = []
    for puzzle_dir in _iter_puzzle_dirs(output_dir):
        cot_f = puzzle_dir / "cot_reasoning.json"
        if not cot_f.exists():
            continue
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
            fold_type = step.get("fold_type")

            if not reasoning or not fold_type:
                targets.append((cot_f, idx, "missing"))
                continue

            if _is_hint_leak(reasoning):
                targets.append((cot_f, idx, "hint-leak"))
                continue

            # Grounding check runs first — before judgement — so it catches all
            # steps regardless of whether they were previously judged as passing
            if fix_ungrounded and not _is_grounded(reasoning):
                targets.append((cot_f, idx, "ungrounded"))
                continue

            judgement = step.get("judgement")
            if isinstance(judgement, dict) and "error" not in judgement:
                r_ok = judgement.get("is_correct_reasoning", True)
                h_ok = judgement.get("is_correct_no_hints", True)
                g_ok = judgement.get("is_grounded_reasoning", True)
                flags = []
                if not r_ok: flags.append("[r]")
                if not h_ok: flags.append("[h]")
                if not g_ok: flags.append("[g]")
                if flags:
                    targets.append((cot_f, idx, f"judge:{''.join(flags)}"))
                    continue

            # Wrong fold_type against ground truth
            gt_folds = _get_gt(puzzle_dir)
            if fold_type and idx < len(gt_folds):
                if fold_type.strip().lower() != gt_folds[idx].strip().lower():
                    targets.append((cot_f, idx,
                                    f"wrong-fold(pred={fold_type} gt={gt_folds[idx]})"))

    return targets


def retry_one(client, puzzle_dir: Path, step_idx: int, model: str) -> dict:
    question_img = puzzle_dir / "question.png"
    cot_f = puzzle_dir / "cot_reasoning.json"
    steps = json.loads(cot_f.read_text())

    if step_idx == 0:
        cur = question_img
    else:
        prev = next(
            (s for s in steps if isinstance(s, dict) and s.get("step") == step_idx - 1),
            None,
        )
        prev_img = (prev.get("image") if prev else None) or f"cot_{step_idx - 1:02d}.png"
        cur = puzzle_dir / prev_img

    this_step = next(
        (s for s in steps if isinstance(s, dict) and s.get("step") == step_idx),
        None,
    )
    nxt_img = (this_step.get("image") if this_step else None) or f"cot_{step_idx:02d}.png"
    nxt = puzzle_dir / nxt_img

    gt_folds = _get_gt(puzzle_dir)
    gt_fold_type = gt_folds[step_idx] if step_idx < len(gt_folds) else "?"

    prompt = PROMPT_TEMPLATE.replace("<<<GT_FOLD_TYPE>>>", gt_fold_type)

    last_err = None
    backoff = RETRY_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[
                    types.Content(role="user", parts=[
                        types.Part(text=prompt),
                        _png_part(question_img),
                        _png_part(cur),
                        _png_part(nxt),
                    ])
                ],
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            text = resp.text
            if not text:
                raise RuntimeError("empty response")
            data = json.loads(text)
            if isinstance(data, list) and data:
                data = data[0]
            fold_type = data.get("fold_type", "")
            if isinstance(fold_type, str):
                fold_type = fold_type.strip().lower()
                if fold_type not in VALID_FOLD_TYPES:
                    fold_type = None
            else:
                fold_type = None
            return {"reasoning": data.get("reasoning"), "fold_type": fold_type}, None
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < MAX_RETRIES:
                time.sleep(backoff)
                backoff *= 2
    return None, last_err


def main():
    ap = argparse.ArgumentParser(description="Retry bad/missing paper-fold reasoning steps")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--dry-run", action="store_true", help="List targets and exit")
    ap.add_argument("--fix-ungrounded", action="store_true",
                    help="Also retry steps whose reasoning lacks geometric keywords (no judge needed)")
    args = ap.parse_args()

    _load_dotenv()
    output_dir = Path(args.output_dir).resolve()
    if not output_dir.exists():
        sys.exit(f"output-dir does not exist: {output_dir}")

    targets = find_targets(output_dir, fix_ungrounded=args.fix_ungrounded)
    if not targets:
        print("No targets found.")
        return

    print(f"{len(targets)} target(s) to retry:")
    for f, s, reason in targets:
        print(f"  {f.parent.name} step {s}  [{reason}]")

    if args.dry_run:
        return

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    ok = 0
    failed = 0
    for cot_f, step_idx, reason in targets:
        puzzle_dir = cot_f.parent
        print(f"Retrying {puzzle_dir.name} step {step_idx} ({reason}) ...")
        new_fields, err = retry_one(client, puzzle_dir, step_idx, args.model)
        if err or new_fields is None:
            print(f"  FAILED: {err}", file=sys.stderr)
            failed += 1
            continue
        steps = json.loads(cot_f.read_text())
        for step in steps:
            if isinstance(step, dict) and step.get("step") == step_idx:
                for stale in ("error", "raw", "reasoning", "fold_type", "judgement"):
                    step.pop(stale, None)
                step.update(new_fields)
                break
        tmp = cot_f.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(steps, indent=2))
        os.replace(tmp, cot_f)
        snippet = (new_fields.get("reasoning") or "")[:80]
        print(f"  ok: fold_type={new_fields.get('fold_type')}  {snippet}...")
        ok += 1

    print(f"\nDone. retried={ok} failed={failed}")


if __name__ == "__main__":
    main()
