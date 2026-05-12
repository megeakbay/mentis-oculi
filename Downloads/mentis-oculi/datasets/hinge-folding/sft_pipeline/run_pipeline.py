"""
End-to-end orchestrator for the hinge-folding SFT pipeline.

Runs sequentially:
  1. batch_submit.py     — bootstrap cot_reasoning.json scaffolds + submit reasoning batch
  2. batch_harvest.py    — wait + apply reasoning results
  3. judge_inline.py     — judge steps one-by-one (no batch job)

Each subprocess inherits stdout/stderr so progress streams live. Stops on
the first non-zero exit. Re-running is idempotent per step (submit scripts
skip already-filled work; judge_inline skips already-judged steps).

Usage:
    python sft_pipeline/run_pipeline.py --output-dir output/level_03
    python sft_pipeline/run_pipeline.py --output-dir output/level_03 --poll-interval 120
    python sft_pipeline/run_pipeline.py --output-dir output/level_03 --skip reasoning
    python sft_pipeline/run_pipeline.py --output-dir output/level_03 --skip judge
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
    ap = argparse.ArgumentParser(description="Run the hinge-folding SFT pipeline end-to-end")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--poll-interval", type=int, default=60,
                    help="Seconds between batch status polls (default: 60)")
    ap.add_argument("--skip", choices=["reasoning", "judge"], default=None,
                    help="Skip the reasoning or judge half of the pipeline")
    args = ap.parse_args()

    python = sys.executable

    if args.skip != "reasoning":
        run("reasoning-submit", [python, str(HERE / "batch_submit.py"),
                                 "--output-dir", args.output_dir])
        run("reasoning-harvest", [python, str(HERE / "batch_harvest.py"),
                                  "--output-dir", args.output_dir,
                                  "--wait", "--poll-interval", str(args.poll_interval)])

    if args.skip != "judge":
        run("judge-inline", [python, str(HERE / "judge_inline.py"),
                             "--output-dir", args.output_dir])

    print("\nAll steps completed.")


if __name__ == "__main__":
    main()
