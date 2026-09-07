# Metric-Eval

Automated eval on the 45 held-out cards in `Eval-45.jsonl` (15 npc / 15 pippa / 15 rpg).

Three conditions: `no_thinking`, `pre_thinking`, `post_thinking`. Trace window `n=3` on model input only; full transcripts are stored.

## Setup

> [!NOTE]
> We used the preview version of DS-v4-flash, so if you want to reproduce our results properly, you have to use a third-party provider
> We used Nvidia hardware, install the correct version if you using other hardware.

From the repo root:

```bash
uv sync
uv sync --extra eval-gpu
export DS_API="..."
```

Install a CUDA `llama-cpp-python` wheel for Scripts A and B (not the CPU wheel). `MODEL_PATHS` in A and B point at `models/Gemma3-4B-{no,pre,post}-thinking-Q8_0.gguf` from repo-root `setup.sh`. Sampling stays identical across conditions.

## Run

From the repo root:

```bash
uv run python Metric-Eval/a_generate_conversations.py
uv run python Metric-Eval/b_latency_replay.py
```

If latency files were copied in, or an extra replay pass ran, keep reps 0–2 with:

```bash
uv run python Metric-Eval/i_prepare_latency_records.py
```

Then:

```bash
uv run python Metric-Eval/c_metrics_battery.py
uv run python Metric-Eval/d_style_consistency.py
uv run python Metric-Eval/e_nli_contradiction.py
uv run python Metric-Eval/f_aggregate_report.py
```

A is resumable. B replays recorded user turns only (no DeepSeek). D and E need the `eval-gpu` extra.

## Outputs

Gitignored: `runs/`, `results/`, `latency/`.
For actual results, check our paper.
