"""
Run the hinge-folding judge synchronously, one request at a time.
Replaces the judge_submit + judge_harvest batch workflow.

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
You are reviewing one step of a hinge-folding puzzle reasoning trace.

Puzzle setup (same as the student was shown):
You are provided with two images of a hinge-folding puzzle.
1. The first image is the current folding state.
2. The second image is the next folding state, acting as a visual hint for which single hinge was just rotated.

The student's task was:
Compare the two images to identify which single hinge was rotated and by how many degrees (90, 180, or 270). Then write reasoning that justifies the choice as if deduced purely from the current state.

CRITICAL CONSTRAINT the student had to obey:
Write the reasoning as if deduced PURELY from the current folding state. NEVER mention the second image, the "next state", or any "hint".

You are given:
1. The current state image.
2. The next state image (the hint).
3. The student's written reasoning.
4. The student's identified hinge label.
5. The student's identified rotation angle.
6. The ground-truth hinge label for this step.
7. The ground-truth rotation angle for this step.

Evaluate TWO independent things:
(a) is_correct_reasoning — does the student's hinge_id AND angle EXACTLY match the ground-truth hinge and angle provided below? Use ground-truth as the authoritative answer — do NOT rely solely on visual comparison.
(b) is_correct_no_hints — does the reasoning obey the critical constraint? It must NOT mention "second image", "next state", "hint", or any phrase revealing the student was shown future states. Any mention of the word "image" is a leak.

Respond EXACTLY with this JSON (no prose outside JSON):
{
  "is_correct_reasoning": true | false,
  "is_correct_no_hints": true | false,
  "reasoning": ""
}

If BOTH booleans are true, "reasoning" MUST be the empty string "".
Otherwise, "reasoning" MUST briefly state which check(s) failed and why.

Student reasoning:
<<<REASONING>>>

Student hinge:
<<<HINGE_ID>>>

Student angle:
<<<ANGLE>>>

Ground-truth hinge:
<<<GT_HINGE_ID>>>

Ground-truth angle:
<<<GT_ANGLE>>>
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
                yield puzzle_dir
    else:
        for puzzle_dir in sorted(output_dir.glob("puzzle_*")):
            yield puzzle_dir


def collect_targets(output_dir: Path):
    """Return list of (cot_file, step_idx, combined, cur, nxt, reasoning, hinge_id, angle_str, gt_hinge_id, gt_angle_str)."""
    targets = []
    for puzzle_dir in _iter_puzzle_dirs(output_dir):
        cot_file = puzzle_dir / "cot_reasoning.json"
        initial = puzzle_dir / "initial.png"
        if not cot_file.exists() or not initial.exists():
            continue
        try:
            steps = json.loads(cot_file.read_text())
        except json.JSONDecodeError:
            print(f"  [skip bad json] {cot_file}", file=sys.stderr)
            continue
        if not isinstance(steps, list):
            continue

        meta_file = puzzle_dir / "metadata.json"
        rotation_steps: list = []
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
                rotation_steps = meta.get("rotation_steps", [])
            except json.JSONDecodeError:
                pass

        for step in steps:
            if not isinstance(step, dict):
                continue
            if "judgement" in step:
                continue
            reasoning = step.get("reasoning")
            hinge_id = step.get("hinge_id")
            angle = step.get("angle")
            if not reasoning or hinge_id is None or angle is None:
                continue
            idx = step.get("step")
            if idx is None:
                continue
            cur = initial if idx == 0 else puzzle_dir / f"cot_{idx - 1:02d}.png"
            nxt = puzzle_dir / f"cot_{idx:02d}.png"
            if not cur.exists() or not nxt.exists():
                print(f"  [skip missing image] {puzzle_dir.name} step {idx}", file=sys.stderr)
                continue
            if idx < len(rotation_steps):
                gt_hinge_id = str(rotation_steps[idx].get("hinge_id", "?"))
                gt_angle_str = str(rotation_steps[idx].get("angle", "?"))
            else:
                gt_hinge_id = "?"
                gt_angle_str = "?"
            targets.append((cot_file, idx, cur, nxt, reasoning,
                            str(hinge_id), str(angle), gt_hinge_id, gt_angle_str))
    return targets


def call_judge(client: genai.Client, model: str, cur: Path, nxt: Path,
               reasoning: str, hinge_id: str, angle_str: str,
               gt_hinge_id: str = "?", gt_angle_str: str = "?"):
    prompt = (JUDGE_PROMPT
              .replace("<<<REASONING>>>", reasoning)
              .replace("<<<HINGE_ID>>>", hinge_id)
              .replace("<<<ANGLE>>>", angle_str)
              .replace("<<<GT_HINGE_ID>>>", gt_hinge_id)
              .replace("<<<GT_ANGLE>>>", gt_angle_str))
    last_err = None
    backoff = RETRY_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[
                    types.Content(role="user", parts=[
                        types.Part(text=prompt),
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
    parser = argparse.ArgumentParser(description="Run hinge-folding judge inline (one request at a time)")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--limit", type=int, default=None, help="Only process first N targets (smoke test)")
    parser.add_argument("--dry-run", action="store_true", help="Count pending steps and exit")
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
        targets = targets[: args.limit]
        print(f"  limited to {len(targets)}")

    if not targets:
        return

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    stats = {"both_true": 0, "reasoning_fail": 0, "hint_fail": 0, "both_fail": 0, "errors": 0}
    failures = []
    error_steps = []
    start = time.time()
    total = len(targets)

    for i, (cot_file, idx, cur, nxt, reasoning, hinge_id, angle_str, gt_hinge_id, gt_angle_str) in enumerate(targets, 1):
        rel = f"{cot_file.parent.parent.name}/{cot_file.parent.name}"
        t0 = time.time()
        judgement, err = call_judge(client, args.model, cur, nxt, reasoning, hinge_id, angle_str, gt_hinge_id, gt_angle_str)
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
            if r_ok and h_ok:
                stats["both_true"] += 1
                mark = "OK "
            elif not r_ok and not h_ok:
                stats["both_fail"] += 1
                mark = "XX "
                failures.append((rel, idx, "[r][h]", judgement.get("reasoning", "")))
            elif not r_ok:
                stats["reasoning_fail"] += 1
                mark = "XR "
                failures.append((rel, idx, "[r]", judgement.get("reasoning", "")))
            else:
                stats["hint_fail"] += 1
                mark = "XH "
                failures.append((rel, idx, "[h]", judgement.get("reasoning", "")))
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
