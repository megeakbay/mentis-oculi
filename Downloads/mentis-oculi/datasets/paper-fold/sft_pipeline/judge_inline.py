"""
Run the paper-fold judge synchronously, one request at a time.

Skips steps that already have a `judgement` field; safe to re-run after
partial failures.

Usage:
    python judge_inline.py --output-dir ../output_sft_100
    python judge_inline.py --output-dir ../output_sft_100 --dry-run
    python judge_inline.py --output-dir ../output_sft_100 --limit 10
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


JUDGE_PROMPT = """
You are reviewing one unfolding step of a paper-fold reasoning trace.

Puzzle setup (same as the student was shown):
You are provided with three images of a paper-folding puzzle.
1. The first image shows the complete folding sequence with the hole punch applied.
2. The second image is the current unfolding state.
3. The third image is the next unfolding state, acting as a visual hint.

The student's task was:
Write reasoning that justifies the current unfolding step — how reversing the fold propagates the hole pattern correctly — as if deduced purely from the question image and the current unfolding state.

CRITICAL CONSTRAINT the student had to obey:
NEVER mention the third image, the "next state", or any "hint".

You are given:
1. The question image (full fold sequence + hole punch).
2. The current unfolding state image.
3. The next unfolding state image (the hint).
4. The student's written reasoning.
5. The student's identified fold type being reversed.
6. The ground-truth fold type for this step.

Evaluate THREE independent things:

(a) is_correct_reasoning — does the student's fold_type exactly match the ground-truth fold type provided below?

(b) is_correct_no_hints — does the reasoning obey the critical constraint? It must NOT mention "third image", "next state", "hint", "second image", "first image", or any phrase revealing multiple states. Any mention of the word "image" is a leak.

(c) is_grounded_reasoning — does the reasoning actually describe the geometric hole propagation, or is it just a verbal restatement of the fold type label? To pass this check the reasoning MUST contain at least ONE of the following:
  - A description of which axis the fold is reversed across (e.g. horizontal axis, vertical axis, diagonal)
  - A description of how the hole(s) reflect or mirror across that axis
  - A reference to the symmetry or reflection that produces new hole positions
  - A description of where the holes end up after unfolding relative to where they were
  Fail this check if the reasoning only says something like "reversing a horizontal fold propagates the hole correctly" without explaining the geometric mechanism. Generic statements that just restate the fold type without describing the reflection geometry are NOT grounded.

Respond EXACTLY with this JSON (no prose outside JSON):
{
  "is_correct_reasoning": true | false,
  "is_correct_no_hints": true | false,
  "is_grounded_reasoning": true | false,
  "reasoning": ""
}

If ALL three booleans are true, "reasoning" MUST be the empty string "".
Otherwise, "reasoning" MUST briefly state which specific check(s) failed and why (e.g. "is_grounded_reasoning failed: reasoning only restates fold type without describing reflection geometry").

Student reasoning:
<<<REASONING>>>

Student fold_type:
<<<FOLD_TYPE>>>

Ground-truth fold_type:
<<<GT_FOLD_TYPE>>>
"""

MODEL = "gemini-3-flash-preview"

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


def collect_targets(output_dir: Path):
    """Return list of (cot_file, step_idx, question_img, cur, nxt, reasoning, fold_type, gt_fold_type)."""
    targets = []
    for puzzle_dir in _iter_puzzle_dirs(output_dir):
        cot_file = puzzle_dir / "cot_reasoning.json"
        question_img = puzzle_dir / "question.png"
        if not cot_file.exists() or not question_img.exists():
            continue
        try:
            steps = json.loads(cot_file.read_text())
        except json.JSONDecodeError:
            print(f"  [skip bad json] {cot_file}", file=sys.stderr)
            continue
        if not isinstance(steps, list):
            continue

        meta_file = puzzle_dir / "metadata.json"
        fold_types_reversed: list = []
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
                fold_types_reversed = list(reversed(meta.get("fold_types", [])))
            except json.JSONDecodeError:
                pass

        for step in steps:
            if not isinstance(step, dict):
                continue
            if "judgement" in step:
                continue
            reasoning = step.get("reasoning")
            fold_type = step.get("fold_type")
            if not reasoning or not fold_type:
                continue
            idx = step.get("step")
            if idx is None:
                continue
            img_name = step.get("image") or f"cot_{idx:02d}.png"
            nxt = puzzle_dir / img_name
            if idx == 0:
                cur = question_img
            else:
                prev_img = steps[idx - 1].get("image") or f"cot_{idx - 1:02d}.png"
                cur = puzzle_dir / prev_img
            if not cur.exists() or not nxt.exists():
                print(f"  [skip missing image] {puzzle_dir.name} step {idx}", file=sys.stderr)
                continue
            gt_fold_type = fold_types_reversed[idx] if idx < len(fold_types_reversed) else "?"
            targets.append((cot_file, idx, question_img, cur, nxt, reasoning, str(fold_type), gt_fold_type))
    return targets


def call_judge(client: genai.Client, model: str, question_img: Path, cur: Path, nxt: Path,
               reasoning: str, fold_type: str, gt_fold_type: str = "?"):
    prompt = (JUDGE_PROMPT
              .replace("<<<REASONING>>>", reasoning)
              .replace("<<<FOLD_TYPE>>>", fold_type)
              .replace("<<<GT_FOLD_TYPE>>>", gt_fold_type))
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
                "is_grounded_reasoning": bool(data.get("is_grounded_reasoning", True)),
                "reasoning": data.get("reasoning", ""),
            }, None
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < MAX_RETRIES:
                time.sleep(backoff)
                backoff *= 2
    return None, last_err


def write_judgement(cot_file: Path, step_idx: int, judgement: dict) -> None:
    steps = json.loads(cot_file.read_text())
    for step in steps:
        if isinstance(step, dict) and step.get("step") == step_idx:
            step["judgement"] = judgement
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
    parser = argparse.ArgumentParser(description="Run paper-fold judge inline")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.exists():
        sys.exit(f"output-dir does not exist: {output_dir}")

    _load_dotenv()

    print(f"Scanning {output_dir} ...")
    targets = collect_targets(output_dir)
    print(f"  {len(targets)} step(s) need judgement.")

    if args.dry_run:
        return

    if args.limit:
        targets = targets[:args.limit]
        print(f"  limited to {len(targets)}")

    if not targets:
        return

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    stats = {"all_true": 0, "reasoning_fail": 0, "hint_fail": 0, "grounded_fail": 0, "multi_fail": 0, "errors": 0}
    failures = []
    error_steps = []
    start = time.time()
    total = len(targets)

    for i, (cot_file, idx, question_img, cur, nxt, reasoning, fold_type, gt_fold_type) in enumerate(targets, 1):
        rel = cot_file.parent.name
        t0 = time.time()
        judgement, err = call_judge(client, args.model, question_img, cur, nxt, reasoning, fold_type, gt_fold_type)
        dt = time.time() - t0

        if err is not None:
            stats["errors"] += 1
            error_steps.append((rel, idx, err))
            judgement_out = {"error": err, "model": args.model}
            try:
                write_judgement(cot_file, idx, judgement_out)
            except Exception as we:
                print(f"  [{i}/{total}] {rel} step {idx}  WRITE FAIL: {we}", file=sys.stderr)
            mark = "ERR"
            detail = err
        else:
            r_ok = judgement["is_correct_reasoning"]
            h_ok = judgement["is_correct_no_hints"]
            g_ok = judgement["is_grounded_reasoning"]
            if r_ok and h_ok and g_ok:
                stats["all_true"] += 1
                mark = "OK "
            else:
                flags = []
                if not r_ok:
                    flags.append("[r]")
                    stats["reasoning_fail"] += 1
                if not h_ok:
                    flags.append("[h]")
                    stats["hint_fail"] += 1
                if not g_ok:
                    flags.append("[g]")
                    stats["grounded_fail"] += 1
                if len(flags) > 1:
                    stats["multi_fail"] += 1
                mark = "X" + "".join(f.strip("[]") for f in flags)
                failures.append((rel, idx, " ".join(flags), judgement.get("reasoning", "")))
            judgement_out = dict(judgement)
            judgement_out["model"] = args.model
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
    print(f"  all true:        {stats['all_true']}")
    print(f"  reasoning fail:  {stats['reasoning_fail']}  [r] fold_type mismatch")
    print(f"  hint leak:       {stats['hint_fail']}  [h] mentioned other images")
    print(f"  grounded fail:   {stats['grounded_fail']}  [g] no geometric reflection described")
    print(f"  multiple fails:  {stats['multi_fail']}")
    print(f"  errors:          {stats['errors']}")
    print(f"  wall time:      {fmt_eta(time.time() - start)}")

    if failures:
        print()
        print("=== Failures ===")
        for rel, idx, flag, why in failures:
            print(f"  {rel} step {idx} {flag} — {(why or '').replace(chr(10), ' ')[:200]}")

    if error_steps:
        print()
        print("=== Errors ===")
        for rel, idx, err in error_steps:
            print(f"  {rel} step {idx} — {err}")


if __name__ == "__main__":
    main()
