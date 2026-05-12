"""
Harvest a Gemini batch job submitted by `batch_submit.py` and merge results
back into each puzzle's cot_reasoning.json.

Usage:
    # One-shot check (exits nonzero if still running):
    python batch_harvest.py --output-dir ../output_sft_100

    # Or wait until done, polling every 60s:
    python batch_harvest.py --output-dir ../output_sft_100 --wait

    # Or pass an explicit job name to override the recorded one:
    python batch_harvest.py --output-dir ../output_sft_100 --job-name batches/abc123
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


def parse_response_text(raw: str) -> Dict[str, Any]:
    """Extract reasoning+piece or fall back to raw."""
    out: Dict[str, Any] = {}
    if raw is None:
        out["raw"] = None
        return out
    text = raw.strip()
    if text.startswith("```"):
        lines = [l for l in text.splitlines() if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            parsed = parsed[0]
        if isinstance(parsed, dict) and "reasoning" in parsed and "piece" in parsed:
            piece = parsed["piece"]
            if isinstance(piece, str):
                piece = piece.strip().upper()
                if piece not in {"A", "B", "C", "D", "E"}:
                    piece = None
            else:
                piece = None
            out["reasoning"] = parsed["reasoning"]
            out["piece"] = piece
        else:
            out["raw"] = raw
    except (json.JSONDecodeError, TypeError):
        out["raw"] = raw
    return out


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


def collect_results(client: genai.Client, job) -> Dict[str, str]:
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


def apply_results(output_dir: Path, results: Dict[str, str]) -> int:
    """
    For each result key "{group}__{puzzle}__step_{NN}", update the matching
    step entry in its cot_reasoning.json. group is either a level_* name or 'root'.
    """
    updated = 0
    by_puzzle: Dict[Path, Dict[int, str]] = {}
    for key, raw in results.items():
        try:
            group, puzzle_name, step_part = key.split("__")
            step_idx = int(step_part.replace("step_", ""))
        except ValueError:
            print(f"  skip malformed key: {key}", file=sys.stderr)
            continue
        puzzle_dir = output_dir / puzzle_name if group == "root" else output_dir / group / puzzle_name
        by_puzzle.setdefault(puzzle_dir, {})[step_idx] = raw

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
            if idx in step_map:
                for stale in ("error", "gemini_response", "raw", "reasoning", "piece", "response"):
                    step.pop(stale, None)
                parsed = parse_response_text(step_map[idx])
                step.update(parsed)
                updated += 1
        cot_file.write_text(json.dumps(steps, indent=2))
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="Harvest Gemini batch results for form-board reasoning")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--job-name", type=str, default=None,
                        help="Override job name (default: read from <output-dir>/.batch_job.txt)")
    parser.add_argument("--wait", action="store_true",
                        help="Poll until the job is done (default: one-shot check)")
    parser.add_argument("--poll-interval", type=int, default=60,
                        help="Seconds between polls when --wait (default: 60)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    if not output_dir.exists():
        sys.exit(f"output-dir does not exist: {output_dir}")

    job_name = args.job_name
    if not job_name:
        record_path = output_dir / ".batch_job.txt"
        if not record_path.exists():
            sys.exit(f"No .batch_job.txt in {output_dir} and no --job-name given.")
        record = json.loads(record_path.read_text())
        job_name = record["job_name"]
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
    updated = apply_results(output_dir, results)
    print(f"  updated {updated} step entries.")
    print("Done.")


if __name__ == "__main__":
    main()
