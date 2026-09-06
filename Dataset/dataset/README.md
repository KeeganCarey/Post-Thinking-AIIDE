# dataset/

Most slices are gitignored. The actual training data is on Hugging Face (`NPC-RP-*`).

### data/

`npc-dialogue-info.csv` — FDG NPC bios used as style exemplars when rewriting PIPPA prompts.

### source/

Original dumps plus the 800-row (and remaining/holdout) splits. Do not train on these.

### cleaned/

Standardized PIPPA prompts (`pippa_system.csv`, `pippa_filtered_prompts_800.jsonl`).

### extracted/

ChatML with a system turn first. Training mix inputs: `npc_dialogue_800.jsonl`, `pippa_400.jsonl`, `rpg_quests_800.jsonl`.

### post-thinking/

Per-source CSVs after trace generation. `combine.py` uses the three training files (not `remaining` or full pippa).
