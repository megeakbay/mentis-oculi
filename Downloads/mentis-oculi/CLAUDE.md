# Repo context for Claude

Fork of `Jana-Z/mentis-oculi`. We're building SFT data for the Rush Hour task in `datasets/rushhour/`.

## Benchmark contamination — READ FIRST

The official MENTISOCULI Rush Hour benchmark is generated with `main.py --seed 42 --instances 50` (5 levels × 50 = 250 puzzles). Seeds used by the benchmark span **42–857** across levels because rejected seeds still increment `current_seed`.

**Any SFT regeneration MUST use a seed range disjoint from 42–857.** Use `--seed 100000` or higher. A prior SFT run at seed 42 produced a 100% contaminated dataset (all 250 seeds identical to benchmark, verified against upstream repo). That dataset lives in `datasets/rushhour/hf_rushhour/` and `datasets/rushhour/output_sft_50/` and should not be used.

## Running SFT generation on HPC (100 instances, unique seeds)

```bash
cd datasets/rushhour
uv run main.py \
  --instances 100 \
  --seed 100000 \
  --output-dir output_sft_100 \
  --skip-reasoning
```

- `--skip-reasoning` renders images + metadata only; run Gemini reasoning afterwards via `sft_pipeline/batch_submit.py` (batch API is cheaper/faster than inline).
- `output_sft_100/` is gitignored (matches `output_*/`). So is `hf_rushhour*/`.
- Expect ~100× the level-5 generation cost vs. 50 instances since rejection rates dominate at high levels. `--max-attempts` defaults are generous; bump if needed.

## SFT pipeline order

Post-generation pipeline lives in `datasets/rushhour/sft_pipeline/`:

1. `batch_submit.py` → submit per-step reasoning to Gemini batch API
2. `batch_harvest.py` → pull results, write `cot_reasoning.json` per puzzle
3. `judge_submit.py` → LLM-as-judge validation (also batch)
4. `judge_harvest.py` + `judge_sync.py` → apply judgments
5. `retry_bad.py` → regenerate failed steps synchronously
6. `check_actions.py` → sanity check
7. `build_hf_dataset.py --output-dir output_sft_100 --out hf_rushhour_100` → package as HF DatasetDict

All scripts read `GEMINI_API_KEY` from env. Put it in `.env` (gitignored) or export before running.

## Answer format (aligned with evaluator)

Per-step `response` is `"<LABEL> forward"` / `"<LABEL> backward"` (e.g. `"A forward"`, `"R backward"`) — enforced in prompts in both `reasoning_text.py` and `sft_pipeline/batch_submit.py`. Directions are relative to each rectangle's rail arrow, never absolute (no up/down/left/right/down-right).

Final `answer` in `build_hf_dataset.py` is synthesized deterministically from `metadata["actions"]` via `synth_answer()`, producing `"A forward, C backward, R forward"` as the evaluator expects. No LLM needed for the final sequence. `red_car` → `R`; other objects use their `label` field.

## Handoff / reproducibility notes

- **`.env` is gitignored**: each dataset dir (`datasets/rushhour/`, `datasets/form-board/`) needs its own `.env` with `GEMINI_API_KEY=...` before running anything in `sft_pipeline/`.
- **Network flakiness during `--wait` polling**: `batch_harvest.py` / `judge_harvest.py` / `run_pipeline.py` can die with `httpx.RemoteProtocolError` ("Server disconnected without sending a response") while polling. The batch job is still running server-side — job names are persisted in `.batch_job.txt` / `.judge_job.txt`. Just re-run the harvest script (or `run_pipeline.py`, which is idempotent: submit scripts skip already-filled steps and harvests re-read the saved job record).
- **Form-board HF builder is not written yet**: `datasets/form-board/sft_pipeline/` has `batch_submit.py` + `batch_harvest.py` but no `build_hf_dataset.py` equivalent. When added, final `answer` can be synthesized deterministically from `metadata["solution_pieces"]` (e.g. `"A, C, E"`) — no LLM needed, mirrors rushhour's `synth_answer`.

## Generator notes

- Seeds are **not** a contiguous range. `main.py` increments `current_seed` on every attempt (including rejects by obviousness filter or difficulty mismatch). Each accepted puzzle's seed is stored in `metadata.json`.
- `--seed` is the *starting* seed; obviousness filter default is `--margin 2.0` px.
- Generator is deterministic: same seed + same filter settings ⇒ same board. That's why seed disjointness guarantees no overlap.
