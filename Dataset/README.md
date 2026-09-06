# Dataset

Scripts used to create the three NPC-RP training mixes (npc 800 + pippa 400 + rpg 800).

Training datasets:

- [`chimbiwide/NPC-RP-No-Thinking`](https://huggingface.co/datasets/chimbiwide/NPC-RP-No-Thinking)
- [`chimbiwide/NPC-RP-Pre-Thinking`](https://huggingface.co/datasets/chimbiwide/NPC-RP-Pre-Thinking)
- [`chimbiwide/NPC-RP-Post-Thinking`](https://huggingface.co/datasets/chimbiwide/NPC-RP-Post-Thinking)

> [!IMPORTANT]
> The DeepSeek-v4-pro used in the paper is the preview version, not DeepSeek-V4-Pro-0813, which is currently used in the official endpoint.
> If you want to reproduce the dataset using the identical model, you have to use a third-party provider (such as Novita).


## Generation workflow

Assume cleaned slices already exist under `dataset/extracted/`.

1. `std_system_prompt.py` — rewrite PIPPA cards (needs `dataset/data/npc-dialogue-info.csv`)
2. `mix.py` — npc 800 + pippa 400 + rpg 800 → `NPC-2k.jsonl`
3. `generate_traces.py` — post-thinking CSVs for those three sources
4. `generate_thinking.py` — pre-thinking traces from `NPC-2k.jsonl` → `Thinking.csv`
5. `patch_empty_traces.py`
6. `combine.py` — post-thinking CSVs → `Post-Thinking.jsonl`
7. `dataset_stats.py`

`prompts.py` / `llm.py` / `data_tools.py` are shared helpers.

Trace generation uses DeepSeek is the result is not deterministic, this is just for reproducibility.
