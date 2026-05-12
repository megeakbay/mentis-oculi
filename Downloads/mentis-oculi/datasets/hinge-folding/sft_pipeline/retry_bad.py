"""
Retry puzzle steps whose reasoning is missing, leaks referential phrases,
or (optionally) whose predicted hinge_id/angle doesn't match ground truth.
Synchronous Gemini calls, no batch. Writes back into cot_reasoning.json.

Usage:
    python retry_bad.py --output-dir ../output_sft_100
    python retry_bad.py --output-dir ../output_sft_100 --fix-wrong-rotation
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


PROMPT_TEMPLATE = """
You are analyzing a hinge-folding puzzle. You are given:
- The combined image showing the initial chain with labeled hinges (A, B, C, …) on the left, and the target folded configuration on the right.
- The current folding state (before this step).
- The next folding state (use this only to determine which hinge was rotated).

Ground-truth for this step:
- Hinge to rotate: <<<GT_HINGE_ID>>>
- Degrees to rotate: <<<GT_ANGLE>>>
- Remaining rotations after this step: <<<REMAINING_STEPS>>>

Your task:
Write natural logical reasoning explaining WHY rotating hinge <<<GT_HINGE_ID>>> by <<<GT_ANGLE>>> degrees brings the chain closer to the target.
Your hinge_id and angle in the JSON response MUST exactly match the ground-truth values above.

CRITICAL CONSTRAINTS:
- Write as if you deduced the rotation purely from the combined image and the current state.
- NEVER use the words "image", "images", "hint", "next state", "third", "second", or any phrase referring to how many views you were given.
- Keep the reasoning to 2–5 short sentences.

Respond EXACTLY with this JSON:
{
  "reasoning": "<Your natural, logical analysis of why this hinge rotation is correct>",
  "hinge_id": "<LABEL>",
  "angle": <DEGREES>
}
where <LABEL> is a hinge letter from the puzzle and <DEGREES> is 90, 180, or 270.
"""

VALID_ANGLES = {90, 180, 270}

BAD_PHRASES = [
    "first image", "second image", "third image",
    "next state", "hint", "following image",
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


def find_targets(output_dir: Path, fix_wrong_rotation: bool):
    targets = []
    for puzzle_dir in _iter_puzzle_dirs(output_dir):
        cot_f = puzzle_dir / "cot_reasoning.json"
        meta_f = puzzle_dir / "metadata.json"
        if not cot_f.exists():
            continue
        rotation_steps = []
        if meta_f.exists():
            try:
                meta = json.loads(meta_f.read_text())
                rotation_steps = meta.get("rotation_steps", [])
            except json.JSONDecodeError:
                pass
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
            hinge_id = step.get("hinge_id")
            angle = step.get("angle")

            # Missing reasoning/hinge/angle
            if not reasoning or hinge_id is None or angle is None:
                targets.append((cot_f, idx, "missing"))
                continue

            # Local hint-leak check
            if _is_hint_leak(reasoning):
                targets.append((cot_f, idx, "hint-leak"))
                continue

            # Judge verdict
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

            # Wrong rotation against ground truth
            if fix_wrong_rotation and idx < len(rotation_steps):
                gt = rotation_steps[idx]
                gt_h = str(gt.get("hinge_id", "")).strip().upper()
                gt_a = int(gt.get("angle", 0))
                pred_h = str(hinge_id).strip().upper()
                try:
                    pred_a = int(angle)
                except (TypeError, ValueError):
                    pred_a = None
                if pred_h != gt_h or pred_a != gt_a:
                    targets.append((cot_f, idx, f"wrong-rotation(pred={pred_h} {pred_a} gt={gt_h} {gt_a})"))
    return targets


def retry_one(client, puzzle_dir: Path, step_idx: int, model: str) -> dict:
    combined = puzzle_dir / "combined.png"
    initial = puzzle_dir / "initial.png"
    cur = initial if step_idx == 0 else puzzle_dir / f"cot_{step_idx - 1:02d}.png"
    nxt = puzzle_dir / f"cot_{step_idx:02d}.png"

    # Load ground-truth for this step
    meta_f = puzzle_dir / "metadata.json"
    gt_hinge_id = "?"
    gt_angle_str = "?"
    remaining_str = "unknown"
    if meta_f.exists():
        try:
            meta = json.loads(meta_f.read_text())
            rotation_steps = meta.get("rotation_steps", [])
            if step_idx < len(rotation_steps):
                gt = rotation_steps[step_idx]
                gt_hinge_id = str(gt.get("hinge_id", "?"))
                gt_angle_str = str(gt.get("angle", "?"))
                remaining = rotation_steps[step_idx + 1:]
                remaining_str = (
                    ", ".join(f"{s.get('hinge_id','?')} {s.get('angle','?')}°" for s in remaining)
                    if remaining else "none (this is the final rotation)"
                )
        except (json.JSONDecodeError, KeyError):
            pass

    prompt = (PROMPT_TEMPLATE
              .replace("<<<GT_HINGE_ID>>>", gt_hinge_id)
              .replace("<<<GT_ANGLE>>>", gt_angle_str)
              .replace("<<<REMAINING_STEPS>>>", remaining_str))

    img_combined = PIL.Image.open(combined)
    img_cur = PIL.Image.open(cur)
    img_nxt = PIL.Image.open(nxt)
    resp = client.models.generate_content(
        model=model,
        contents=[prompt, img_combined, img_cur, img_nxt],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    data = json.loads(resp.text)
    if isinstance(data, list) and data:
        data = data[0]
    hinge_id = data.get("hinge_id", "")
    if isinstance(hinge_id, str):
        hinge_id = hinge_id.strip().upper()
    try:
        angle = int(data.get("angle"))
        if angle not in VALID_ANGLES:
            angle = None
    except (TypeError, ValueError):
        angle = None
    return {"reasoning": data.get("reasoning"), "hinge_id": hinge_id, "angle": angle}


def main():
    ap = argparse.ArgumentParser(description="Retry bad/missing hinge-folding reasoning steps")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--fix-wrong-rotation", action="store_true",
                    help="Also retry steps where predicted hinge/angle != ground truth")
    ap.add_argument("--dry-run", action="store_true",
                    help="List targets and exit without making any Gemini calls")
    args = ap.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.exists():
        sys.exit(f"output-dir does not exist: {output_dir}")

    targets = find_targets(output_dir, args.fix_wrong_rotation)
    if not targets:
        print("No targets found.")
        return
    print(f"{len(targets)} target(s) to retry:")
    for f, s, reason in targets:
        print(f"  {f.parent.name} step {s}  [{reason}]")

    if args.dry_run:
        return

    _load_dotenv()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    ok = 0
    failed = 0
    for cot_f, step_idx, reason in targets:
        puzzle_dir = cot_f.parent
        print(f"Retrying {puzzle_dir.name} step {step_idx} ({reason}) ...")
        try:
            new_fields = retry_one(client, puzzle_dir, step_idx, args.model)
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            failed += 1
            continue
        steps = json.loads(cot_f.read_text())
        for step in steps:
            if isinstance(step, dict) and step.get("step") == step_idx:
                for stale in ("error", "gemini_response", "raw", "reasoning", "hinge_id", "angle", "judgement"):
                    step.pop(stale, None)
                step.update(new_fields)
                break
        tmp = cot_f.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(steps, indent=2))
        os.replace(tmp, cot_f)
        snippet = (new_fields.get("reasoning") or "")[:80]
        print(f"  ok: hinge={new_fields.get('hinge_id')} angle={new_fields.get('angle')}  {snippet}...")
        ok += 1

    print(f"\nDone. retried={ok} failed={failed}")


if __name__ == "__main__":
    main()
