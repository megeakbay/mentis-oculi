"""
Harvest the judge batch submitted by judge_submit.py and write results into
each step's `judgement` field in cot_reasoning.json.

Usage:
    python judge_harvest.py --output-dir ../output_sft_100
    python judge_harvest.py --output-dir ../output_sft_100 --wait
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

from google import genai


DONE_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


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


def _state_name(job) -> str:
    state = job.state
    return state.name if hasattr(state, "name") else str(state)


def parse_judge_text(raw: str) -> Dict[str, Any]:
    if raw is None:
        return {"raw": None}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            parsed = parsed[0]
        if isinstance(parsed, dict) and "is_correct_reasoning" in parsed and "is_correct_no_hints" in parsed:
            return {
                "is_correct_reasoning": bool(parsed["is_correct_reasoning"]),
                "is_correct_no_hints": bool(parsed["is_correct_no_hints"]),
                "reasoning": parsed.get("reasoning", ""),
            }
        return {"raw": raw}
    except (json.JSONDecodeError, TypeError):
        return {"raw": raw}


def extract_text(response_obj: Any):
    if response_obj is None:
        return None
    text = getattr(response_obj, "text", None)
    if text:
        return text
    if isinstance(response_obj, dict):
        if "text" in response_obj:
            return response_obj["text"]
        for cand in response_obj.get("candidates", []) or []:
            for part in (cand.get("content", {}) or {}).get("parts", []) or []:
                if "text" in part:
                    return part["text"]
    for cand in getattr(response_obj, "candidates", None) or []:
        content = getattr(cand, "content", None)
        for part in getattr(content, "parts", None) or []:
            t = getattr(part, "text", None)
            if t:
                return t
    return None


def collect_results(client, job) -> Dict[str, str]:
    results: Dict[str, str] = {}
    dest = job.dest

    inlined = getattr(dest, "inlined_responses", None)
    if inlined:
        for i, item in enumerate(inlined):
            resp = getattr(item, "response", None)
            err = getattr(item, "error", None)
            key = getattr(item, "key", f"inline_{i}")
            results[key] = None if err else extract_text(resp)
        return results

    file_name = getattr(dest, "file_name", None)
    if not file_name:
        raise RuntimeError(f"Batch completed but dest has neither inlined_responses nor file_name: {dest}")

    blob = client.files.download(file=file_name)
    if isinstance(blob, bytes):
        blob = blob.decode("utf-8")
    for line in blob.splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        key = row.get("key")
        if not key:
            continue
        if "response" in row:
            results[key] = extract_text(row["response"])
        elif "error" in row:
            results[key] = None
    return results


def apply_results(output_dir: Path, results: Dict[str, str], model_name: str):
    updated = 0
    by_puzzle: Dict[Path, Dict[int, str]] = {}
    for key, raw in results.items():
        try:
            # key: "{group}__{puzzle}__judge_step_{NN}"
            group, puzzle_name, step_part = key.split("__")
            step_idx = int(step_part.replace("judge_step_", ""))
        except ValueError:
            print(f"  skip malformed key: {key}", file=sys.stderr)
            continue
        puzzle_dir = output_dir / puzzle_name if group == "root" else output_dir / group / puzzle_name
        by_puzzle.setdefault(puzzle_dir, {})[step_idx] = raw

    stats = {"both_true": 0, "reasoning_fail": 0, "hint_fail": 0, "both_fail": 0, "unparseable": 0}
    failures = []

    for puzzle_dir, step_map in by_puzzle.items():
        cot_file = puzzle_dir / "cot_reasoning.json"
        if not cot_file.exists():
            print(f"  skip (no cot_reasoning.json): {puzzle_dir}", file=sys.stderr)
            continue
        steps = json.loads(cot_file.read_text())
        for step in steps:
            if not isinstance(step, dict):
                continue
            idx = step.get("step")
            if idx not in step_map:
                continue
            parsed = parse_judge_text(step_map[idx])
            judgement = dict(parsed)
            judgement["model"] = model_name
            step["judgement"] = judgement
            updated += 1

            if "raw" in parsed:
                stats["unparseable"] += 1
                failures.append((puzzle_dir, idx, "unparseable", parsed.get("raw", "")))
            else:
                r_ok = parsed["is_correct_reasoning"]
                h_ok = parsed["is_correct_no_hints"]
                if r_ok and h_ok:
                    stats["both_true"] += 1
                elif not r_ok and not h_ok:
                    stats["both_fail"] += 1
                    failures.append((puzzle_dir, idx, "[r][h]", parsed.get("reasoning", "")))
                elif not r_ok:
                    stats["reasoning_fail"] += 1
                    failures.append((puzzle_dir, idx, "[r]", parsed.get("reasoning", "")))
                else:
                    stats["hint_fail"] += 1
                    failures.append((puzzle_dir, idx, "[h]", parsed.get("reasoning", "")))
        cot_file.write_text(json.dumps(steps, indent=2))

    return updated, stats, failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Harvest Gemini judge batch for hinge-folding")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--job-name", type=str, default=None)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-interval", type=int, default=60)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.exists():
        sys.exit(f"output-dir does not exist: {output_dir}")

    job_name = args.job_name
    model_name = "gemini-3-flash-preview"
    if not job_name:
        record_path = output_dir / ".judge_job.txt"
        if not record_path.exists():
            sys.exit(f"No .judge_job.txt in {output_dir} and no --job-name given.")
        record = json.loads(record_path.read_text())
        job_name = record["job_name"]
        model_name = record.get("model", model_name)
    print(f"Job: {job_name}")

    _load_dotenv()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    while True:
        job = client.batches.get(name=job_name)
        state = _state_name(job)
        print(f"  state: {state}")
        if state in DONE_STATES:
            break
        if not args.wait:
            print("Not done yet. Re-run later, or pass --wait.")
            sys.exit(2)
        time.sleep(args.poll_interval)

    if state != "JOB_STATE_SUCCEEDED":
        err = getattr(job, "error", None)
        sys.exit(f"Job ended in non-success state: {state}  error={err}")

    print("Downloading results ...")
    results = collect_results(client, job)
    print(f"  {len(results)} result row(s).")

    print("Applying to cot_reasoning.json files ...")
    updated, stats, failures = apply_results(output_dir, results, model_name)
    print(f"  updated {updated} step entries.")
    print()
    print("=== Judge summary ===")
    print(f"  both true:       {stats['both_true']}")
    print(f"  reasoning fail:  {stats['reasoning_fail']}")
    print(f"  hint leak:       {stats['hint_fail']}")
    print(f"  both fail:       {stats['both_fail']}")
    print(f"  unparseable:     {stats['unparseable']}")
    if failures:
        print()
        print("=== Failures ===")
        for puzzle_dir, idx, flag, why in failures:
            snippet = (why or "").replace("\n", " ")[:200]
            print(f"  {puzzle_dir.name} step {idx} {flag} — {snippet}")


if __name__ == "__main__":
    main()
