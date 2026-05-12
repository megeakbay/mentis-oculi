"""
End-to-end orchestrator for the rushhour SFT pipeline.

Runs sequentially:
  1. reasoning  — batch (default) or inline (--inline-reasoning)
  2. judge      — batch (default) or inline (--inline-judge)

Usage:
    python sft_pipeline/run_pipeline.py --output-dir output_sft_100
    python sft_pipeline/run_pipeline.py --output-dir output_sft_100 --inline-reasoning
    python sft_pipeline/run_pipeline.py --output-dir output_sft_100 --inline-reasoning --inline-judge
    python sft_pipeline/run_pipeline.py --output-dir output_sft_100 --skip reasoning
    python sft_pipeline/run_pipeline.py --output-dir output_sft_100 --skip judge
"""
import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def run(step: str, cmd: list[str]) -> None:
    print(f"\n{'=' * 60}\n[{step}] {' '.join(cmd)}\n{'=' * 60}", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"[{step}] failed with exit code {result.returncode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--poll-interval", type=int, default=60,
                    help="Seconds between batch status polls (default: 60)")
    ap.add_argument("--skip", choices=["reasoning", "judge"], default=None,
                    help="Skip the reasoning or judge half of the pipeline")
    ap.add_argument("--inline-reasoning", action="store_true",
                    help="Use reasoning_inline.py instead of batch_submit/harvest")
    ap.add_argument("--inline-judge", action="store_true",
                    help="Use judge_sync.py instead of judge_submit/harvest")
    args = ap.parse_args()

    python = sys.executable

    if args.skip != "reasoning":
        if args.inline_reasoning:
            run("reasoning-inline", [python, str(HERE / "reasoning_inline.py"),
                                     "--output-dir", args.output_dir])
        else:
            run("reasoning-submit", [python, str(HERE / "batch_submit.py"),
                                     "--output-dir", args.output_dir])
            run("reasoning-harvest", [python, str(HERE / "batch_harvest.py"),
                                      "--output-dir", args.output_dir,
                                      "--wait", "--poll-interval", str(args.poll_interval)])

    if args.skip != "judge":
        if args.inline_judge:
            run("judge-inline", [python, str(HERE / "judge_sync.py"),
                                 "--output-dir", args.output_dir])
        else:
            run("judge-submit", [python, str(HERE / "judge_submit.py"),
                                 "--output-dir", args.output_dir])
            run("judge-harvest", [python, str(HERE / "judge_harvest.py"),
                                  "--output-dir", args.output_dir,
                                  "--wait", "--poll-interval", str(args.poll_interval)])

    print("\nAll steps completed.")


if __name__ == "__main__":
    main()
