"""
Synchronous judge runner. Iterates every step needing judgement, calls
gemini-3-flash-preview one-at-a-time, writes the result into
cot_reasoning.json as it goes. Resumable: already-judged steps are skipped.

Usage:
    python judge_sync.py --output-dir ../output_sft_50
    python judge_sync.py --output-dir ../output_sft_50 --model gemini-3-flash-preview
    python judge_sync.py --output-dir ../output_sft_50 --limit 10     # smoke test
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

from google import genai
from google.genai import types
import PIL.Image

from judge_submit import JUDGE_PROMPT, MODEL, _load_dotenv


MAX_RETRIES = 3
RETRY_BACKOFF = 4.0  # seconds, doubled each retry


def collect_targets(output_dir: Path):
    """Return list of (cot_file, step_idx, cur_path, nxt_path, final_path, reasoning, response)."""
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
                key=lambda s: s["step"]
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
                if "judgement" in step:
                    continue
                r = step.get("reasoning")
                resp = step.get("response")
                if not r or not resp:
                    continue
                idx = step.get("step")
                if idx is None:
                    continue
                cur = puzzle_dir / ("initial.png" if idx == 0 else f"cot_{idx - 1:02d}.png")
                nxt = puzzle_dir / f"cot_{idx:02d}.png"
                if not cur.exists() or not nxt.exists():
                    print(f"  [skip missing image] {puzzle_dir} step {idx}", file=sys.stderr)
                    continue
                targets.append((cot_file, idx, cur, nxt, final_img, r, resp))
    return targets


def call_judge(client, model: str, cur: Path, nxt: Path, final_img: Path, reasoning: str, response: str):
    prompt = JUDGE_PROMPT.replace("<<<REASONING>>>", reasoning).replace("<<<RESPONSE>>>", response)
    last_err = None
    backoff = RETRY_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[prompt, PIL.Image.open(cur), PIL.Image.open(nxt), PIL.Image.open(final_img)],
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            text = resp.text
            if not text:
                raise RuntimeError("empty response text")
            data = json.loads(text)
            if isinstance(data, list) and data:
                data = data[0]
            if not isinstance(data, dict):
                raise RuntimeError(f"unexpected shape: {type(data).__name__}")
            if "is_correct_reasoning" not in data or "is_correct_no_hints" not in data:
                raise RuntimeError(f"missing required fields: {list(data.keys())}")
            return {
                "is_correct_reasoning": bool(data["is_correct_reasoning"]),
                "is_correct_no_hints": bool(data["is_correct_no_hints"]),
                "reasoning": data.get("reasoning", ""),
            }, None
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < MAX_RETRIES:
                time.sleep(backoff)
                backoff *= 2
    return None, last_err


def write_judgement(cot_file: Path, step_idx: int, judgement: dict):
    steps = json.loads(cot_file.read_text())
    for step in steps:
        if isinstance(step, dict) and step.get("step") == step_idx:
            step["judgement"] = judgement
            break
    tmp = cot_file.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(steps, indent=2))
    os.replace(tmp, cot_file)


def fmt_eta(seconds: float) -> str:
    if seconds < 60: return f"{seconds:.0f}s"
    if seconds < 3600: return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"


def main():
    parser = argparse.ArgumentParser(description="Synchronous judge runner")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--limit", type=int, default=None, help="Only process first N targets (smoke test)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.exists():
        sys.exit(f"output-dir does not exist: {output_dir}")

    _load_dotenv()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    print(f"Scanning {output_dir} ...")
    targets = collect_targets(output_dir)
    print(f"  {len(targets)} step(s) need judgement.")
    if args.limit:
        targets = targets[: args.limit]
        print(f"  limited to {len(targets)}")
    if not targets:
        return

    stats = {"both_true": 0, "reasoning_fail": 0, "hint_fail": 0, "both_fail": 0, "errors": 0}
    failures = []
    error_steps = []
    start = time.time()

    total = len(targets)
    for i, (cot_file, idx, cur, nxt, final_img, r, resp) in enumerate(targets, 1):
        rel = f"{cot_file.parent.parent.name}/{cot_file.parent.name}"
        t0 = time.time()
        judgement, err = call_judge(client, args.model, cur, nxt, final_img, r, resp)
        dt = time.time() - t0

        if err is not None:
            stats["errors"] += 1
            error_steps.append((rel, idx, err))
            judgement = {"error": err, "model": args.model}
            try:
                write_judgement(cot_file, idx, judgement)
            except Exception as we:
                print(f"  [{i}/{total}] {rel} step {idx}  WRITE FAIL: {we}", file=sys.stderr)
            mark = "ERR"
            detail = err
        else:
            r_ok = judgement["is_correct_reasoning"]
            h_ok = judgement["is_correct_no_hints"]
            if r_ok and h_ok:
                stats["both_true"] += 1; mark = "OK "
            elif not r_ok and not h_ok:
                stats["both_fail"] += 1; mark = "XX "
                failures.append((rel, idx, "[r][h]", judgement.get("reasoning", "")))
            elif not r_ok:
                stats["reasoning_fail"] += 1; mark = "XR "
                failures.append((rel, idx, "[r]", judgement.get("reasoning", "")))
            else:
                stats["hint_fail"] += 1; mark = "XH "
                failures.append((rel, idx, "[h]", judgement.get("reasoning", "")))
            judgement_out = dict(judgement); judgement_out["model"] = args.model
            try:
                write_judgement(cot_file, idx, judgement_out)
            except Exception as we:
                print(f"  [{i}/{total}] {rel} step {idx}  WRITE FAIL: {we}", file=sys.stderr)
                traceback.print_exc()
            detail = judgement.get("reasoning", "") or ""

        elapsed = time.time() - start
        avg = elapsed / i
        eta = avg * (total - i)
        snippet = (detail or "").replace("\n", " ")[:80]
        print(f"  [{i:4d}/{total}] {mark} {rel} step {idx}  ({dt:.1f}s, avg {avg:.1f}s, eta {fmt_eta(eta)})  {snippet}")

    print()
    print("=== Summary ===")
    print(f"  both true:      {stats['both_true']}")
    print(f"  reasoning fail: {stats['reasoning_fail']}")
    print(f"  hint leak:      {stats['hint_fail']}")
    print(f"  both fail:      {stats['both_fail']}")
    print(f"  errors:         {stats['errors']}")
    print(f"  wall time:      {fmt_eta(time.time() - start)}")

    if failures:
        print()
        print("=== Failures ===")
        for rel, idx, flag, why in failures:
            print(f"  {rel} step {idx} {flag} — {(why or '').replace(chr(10), ' ')[:200]}")
    if error_steps:
        print()
        print("=== Errors (can be retried — these have judgement.error set) ===")
        for rel, idx, err in error_steps:
            print(f"  {rel} step {idx} — {err}")


if __name__ == "__main__":
    main()
