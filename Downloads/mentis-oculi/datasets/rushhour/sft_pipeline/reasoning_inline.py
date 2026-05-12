"""
Generate reasoning for Rush Hour puzzle steps synchronously, one at a time.
Same prompt as batch_submit.py but no batch — useful for small datasets or testing.

Skips steps that already have reasoning+response. Safe to re-run.

Usage:
    python reasoning_inline.py --output-dir ../output_sft_test_l45
    python reasoning_inline.py --output-dir ../output_sft_test_l45 --model gemini-2.0-flash
    python reasoning_inline.py --output-dir ../output_sft_test_l45 --limit 5
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import PIL.Image
from google import genai
from google.genai import types

from batch_submit import STATIC_PROMPT


MODEL = "gemini-3.1-pro-preview"
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


def collect_targets(output_dir: Path):
    """Return list of (cot_file, step_idx, cur, nxt, final_img) for steps missing reasoning."""
    targets = []
    for level_dir in sorted(output_dir.glob("level_*")):
        if not level_dir.is_dir():
            continue
        for puzzle_dir in sorted(level_dir.glob("puzzle_*")):
            cot_file = puzzle_dir / "cot_reasoning.json"
            if not cot_file.exists():
                continue
            try:
                steps = json.loads(cot_file.read_text())
            except json.JSONDecodeError:
                print(f"  [skip bad json] {cot_file}", file=sys.stderr)
                continue
            if not isinstance(steps, list):
                continue

            valid_steps = sorted(
                (s for s in steps if isinstance(s, dict) and s.get("step") is not None),
                key=lambda s: s["step"],
            )
            if not valid_steps:
                continue
            last_idx = valid_steps[-1]["step"]
            final_img = puzzle_dir / f"cot_{last_idx:02d}.png"
            if not final_img.exists():
                print(f"  [skip missing final] {puzzle_dir}", file=sys.stderr)
                continue

            for step in steps:
                if not isinstance(step, dict):
                    continue
                if step.get("reasoning") and step.get("response"):
                    continue  # already done
                idx = step.get("step")
                if idx is None:
                    continue
                cur = puzzle_dir / ("initial.png" if idx == 0 else f"cot_{idx - 1:02d}.png")
                nxt = puzzle_dir / f"cot_{idx:02d}.png"
                if not cur.exists() or not nxt.exists():
                    print(f"  [skip missing image] {puzzle_dir} step {idx}", file=sys.stderr)
                    continue
                targets.append((cot_file, idx, cur, nxt, final_img))
    return targets


def call_reasoning(client, model: str, cur: Path, nxt: Path, final_img: Path):
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
            text = resp.text
            if not text:
                raise RuntimeError("empty response")
            data = json.loads(text)
            if isinstance(data, list) and data:
                data = data[0]
            if not isinstance(data, dict):
                raise RuntimeError(f"unexpected shape: {type(data).__name__}")
            if "reasoning" not in data or "response" not in data:
                raise RuntimeError(f"missing fields: {list(data.keys())}")
            return {
                "reasoning": data["reasoning"],
                "response": data["response"],
            }, None
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < MAX_RETRIES:
                time.sleep(backoff)
                backoff *= 2
    return None, last_err


def write_step(cot_file: Path, step_idx: int, fields: dict) -> None:
    steps = json.loads(cot_file.read_text())
    for step in steps:
        if isinstance(step, dict) and step.get("step") == step_idx:
            for stale in ("raw", "error", "reasoning", "response"):
                step.pop(stale, None)
            step.update(fields)
            break
    tmp = cot_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(steps, indent=2))
    os.replace(tmp, cot_file)


def fmt_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Rush Hour reasoning inline (no batch)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.exists():
        sys.exit(f"output-dir does not exist: {output_dir}")

    _load_dotenv()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    print(f"Scanning {output_dir} ...")
    targets = collect_targets(output_dir)
    print(f"  {len(targets)} step(s) need reasoning.")

    if args.limit:
        targets = targets[: args.limit]
        print(f"  limited to {len(targets)}")

    if not targets:
        return

    ok = errors = 0
    start = time.time()
    total = len(targets)

    for i, (cot_file, idx, cur, nxt, final_img) in enumerate(targets, 1):
        rel = f"{cot_file.parent.parent.name}/{cot_file.parent.name}"
        t0 = time.time()
        result, err = call_reasoning(client, args.model, cur, nxt, final_img)
        dt = time.time() - t0

        elapsed = time.time() - start
        avg = elapsed / i
        eta = avg * (total - i)

        if err:
            errors += 1
            print(f"  [{i:4d}/{total}] ERR {rel} step {idx} ({dt:.1f}s, eta {fmt_eta(eta)})  {err}")
        else:
            try:
                write_step(cot_file, idx, result)
                ok += 1
                snippet = (result.get("response") or "")[:60]
                print(f"  [{i:4d}/{total}] OK  {rel} step {idx} ({dt:.1f}s, eta {fmt_eta(eta)})  {snippet}")
            except Exception as we:
                errors += 1
                print(f"  [{i:4d}/{total}] WRITE FAIL {rel} step {idx}: {we}", file=sys.stderr)
                traceback.print_exc()

    print(f"\nDone. ok={ok} errors={errors}  wall={fmt_eta(time.time() - start)}")


if __name__ == "__main__":
    main()
