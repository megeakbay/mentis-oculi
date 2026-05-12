"""
Retry Rush Hour puzzle steps whose reasoning is missing, leaks hint phrases,
or failed the judge. Uses the same prompt and images as reasoning_inline.py.

Usage:
    python retry_bad.py --output-dir ../output_sft_test_l45
    python retry_bad.py --output-dir ../output_sft_test_l45 --dry-run
    python retry_bad.py --output-dir ../output_sft_test_l45 --fix-wrong-move
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import PIL.Image
from google import genai
from google.genai import types

from batch_submit import STATIC_PROMPT


MODEL = "gemini-3.1-pro-preview"
MAX_RETRIES = 3
RETRY_BACKOFF = 4.0

BAD_PHRASES = ["first image", "second image", "next state", "hint", "third image"]
VALID_DIRECTIONS = {"forward", "backward"}


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


def _is_hint_leak(text: str) -> bool:
    tl = text.lower()
    return any(p in tl for p in BAD_PHRASES)


def _get_final_img(puzzle_dir: Path, steps: list) -> Path | None:
    valid = sorted(
        (s for s in steps if isinstance(s, dict) and s.get("step") is not None),
        key=lambda s: s["step"],
    )
    if not valid:
        return None
    last_idx = valid[-1]["step"]
    p = puzzle_dir / f"cot_{last_idx:02d}.png"
    return p if p.exists() else None


def find_targets(output_dir: Path, fix_wrong_move: bool = False):
    targets = []
    for cot_f in sorted(output_dir.glob("level_*/puzzle_*/cot_reasoning.json")):
        puzzle_dir = cot_f.parent
        try:
            steps = json.loads(cot_f.read_text())
        except json.JSONDecodeError:
            continue

        meta_f = puzzle_dir / "metadata.json"
        gt_moves = []
        if fix_wrong_move and meta_f.exists():
            try:
                meta = json.loads(meta_f.read_text())
                gt_moves = [
                    f"{('R' if a['object_id'] == 'red_car' else a['object_id'].split('_')[-1].upper())} {'forward' if a['direction'] > 0 else 'backward'}"
                    for a in meta.get("actions", [])
                ]
            except Exception:
                pass

        for step in steps:
            if not isinstance(step, dict):
                continue
            idx = step.get("step")
            if idx is None:
                continue
            reasoning = step.get("reasoning") or ""
            response = step.get("response") or ""

            if not reasoning or not response:
                targets.append((cot_f, idx, "missing"))
                continue

            if _is_hint_leak(reasoning):
                targets.append((cot_f, idx, "hint-leak"))
                continue

            judgement = step.get("judgement")
            if isinstance(judgement, dict) and "error" not in judgement:
                r_ok = judgement.get("is_correct_reasoning", True)
                h_ok = judgement.get("is_correct_no_hints", True)
                flags = []
                if not r_ok: flags.append("[r]")
                if not h_ok: flags.append("[h]")
                if flags:
                    targets.append((cot_f, idx, f"judge:{''.join(flags)}"))
                    continue

            if fix_wrong_move and idx < len(gt_moves):
                if response.strip().lower() != gt_moves[idx].strip().lower():
                    targets.append((cot_f, idx, f"wrong-move(pred={response.strip()} gt={gt_moves[idx]})"))

    return targets


def retry_one(client, puzzle_dir: Path, step_idx: int, steps: list, model: str):
    final_img = _get_final_img(puzzle_dir, steps)
    if final_img is None:
        return None, "missing final image"

    cur = puzzle_dir / ("initial.png" if step_idx == 0 else f"cot_{step_idx - 1:02d}.png")
    nxt = puzzle_dir / f"cot_{step_idx:02d}.png"
    if not cur.exists() or not nxt.exists():
        return None, f"missing cur/nxt image for step {step_idx}"

    last_err = None
    backoff = RETRY_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[
                    STATIC_PROMPT,
                    PIL.Image.open(cur),
                    PIL.Image.open(nxt),
                    PIL.Image.open(final_img),
                ],
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            data = json.loads(resp.text)
            if isinstance(data, list) and data:
                data = data[0]
            if "reasoning" not in data or "response" not in data:
                raise RuntimeError(f"missing fields: {list(data.keys())}")
            return {"reasoning": data["reasoning"], "response": data["response"]}, None
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < MAX_RETRIES:
                time.sleep(backoff)
                backoff *= 2
    return None, last_err


def main():
    ap = argparse.ArgumentParser(description="Retry bad/missing Rush Hour reasoning steps")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--dry-run", action="store_true", help="List targets and exit")
    ap.add_argument("--fix-wrong-move", action="store_true",
                    help="Also retry steps where response doesn't match ground truth")
    args = ap.parse_args()

    _load_dotenv()
    output_dir = Path(args.output_dir).resolve()
    if not output_dir.exists():
        sys.exit(f"output-dir does not exist: {output_dir}")

    targets = find_targets(output_dir, fix_wrong_move=args.fix_wrong_move)
    if not targets:
        print("No targets found.")
        return

    print(f"{len(targets)} target(s) to retry:")
    for f, s, reason in targets:
        print(f"  {f.parent.parent.name}/{f.parent.name} step {s}  [{reason}]")

    if args.dry_run:
        return

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    ok = failed = 0
    for cot_f, step_idx, reason in targets:
        puzzle_dir = cot_f.parent
        print(f"Retrying {puzzle_dir.parent.name}/{puzzle_dir.name} step {step_idx} ({reason}) ...")
        steps = json.loads(cot_f.read_text())
        new_fields, err = retry_one(client, puzzle_dir, step_idx, steps, args.model)
        if err or new_fields is None:
            print(f"  FAILED: {err}", file=sys.stderr)
            failed += 1
            continue
        for step in steps:
            if isinstance(step, dict) and step.get("step") == step_idx:
                for stale in ("error", "raw", "reasoning", "response", "judgement"):
                    step.pop(stale, None)
                step.update(new_fields)
                break
        tmp = cot_f.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(steps, indent=2))
        os.replace(tmp, cot_f)
        print(f"  ok: {new_fields['response']}  {new_fields['reasoning'][:80]}...")
        ok += 1

    print(f"\nDone. retried={ok} failed={failed}")


if __name__ == "__main__":
    main()
