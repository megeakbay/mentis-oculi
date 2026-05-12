"""
End-to-end orchestrator for the paper-fold SFT pipeline.

Runs sequentially:
  1. batch_submit.py   — bootstrap cot_reasoning.json scaffolds + submit reasoning batch
  2. batch_harvest.py  — wait + apply reasoning results
  3. judge_inline.py   — judge steps one-by-one

Each subprocess inherits stdout/stderr so progress streams live. Stops on
the first non-zero exit. Re-running is idempotent (submit skips already-filled
work; judge_inline skips already-judged steps).

Usage:
    python sft_pipeline/run_pipeline.py --output-dir output_sft_100
    python sft_pipeline/run_pipeline.py --output-dir output_sft_100 --poll-interval 120
    python sft_pipeline/run_pipeline.py --output-dir output_sft_100 --skip reasoning
    python sft_pipeline/run_pipeline.py --output-dir output_sft_100 --skip judge
    python sft_pipeline/run_pipeline.py --output-dir output_sft_100 --retry-rounds 2
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
    ap = argparse.ArgumentParser(description="Run the paper-fold SFT pipeline end-to-end")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--poll-interval", type=int, default=60)
    ap.add_argument("--skip", choices=["reasoning", "judge"], default=None)
    ap.add_argument("--retry-rounds", type=int, default=1,
                    help="How many retry→re-judge cycles to run after the first judge pass (default: 1)")
    ap.add_argument("--retry-model", type=str, default="gemini-3.1-pro-preview",
                    help="Model used by retry_bad.py (default: gemini-3.1-pro-preview)")
    ap.add_argument("--fix-ungrounded", action="store_true",
                    help="Pass --fix-ungrounded to retry_bad.py to catch shallow reasoning without a prior judge run")
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

        for round_num in range(1, args.retry_rounds + 1):
            print(f"\n--- retry round {round_num}/{args.retry_rounds} ---")
            retry_cmd = [python, str(HERE / "retry_bad.py"),
                         "--output-dir", args.output_dir,
                         "--model", args.retry_model]
            if args.fix_ungrounded:
                retry_cmd.append("--fix-ungrounded")
            run(f"retry-{round_num}", retry_cmd)
            run(f"re-judge-{round_num}", [python, str(HERE / "judge_inline.py"),
                                          "--output-dir", args.output_dir])

    print("\nAll steps completed.")


if __name__ == "__main__":
    main()
