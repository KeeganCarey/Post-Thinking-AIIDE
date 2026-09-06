# Post-Thinking-AIIDE

Repository for the paper ***Making NPCs Think: Evaluating Post-Thinking for Real-Time Game NPC Dialogue Generation***.

Accepted as poster presentation for [AIIDE 2026](https://sites.google.com/view/aiide2026/home).

By Hexi Wang and Keegan Carey

> [!NOTE]
> If you are using Windows you might have to adopt the shell scripts to PowerShell or use WSL.

---

## Prior Work

Our previous late-breaking/short paper was accepted to FDG 2026. 

You can find our paper here: [Post-Thinking in NPC Dialogue: A Paradigm for Reflective Character Models](https://dl.acm.org/doi/10.1145/3815598.3815681)

Our Github repo: [Post-Thinking-FDG](https://github.com/KeeganCarey/Post-Thinking-FDG)

---

## Repository Map

- `Dataset/`: build the three training mixes (npc 800 + pippa 400 + rpg 800).
- `Training/`: Gemma3-4B LoRA SFT on TPU v6e using Tunix (`No-Thinking.py`, `Pre-Thinking.py`, `Post-Thinking.py`).
- `Metric-Eval/`: automated eval on the 45 held-out cards (`Eval-45.jsonl`): transcripts, latency, style, NLI, aggregate report.
- `Expert-Eval/`: blind Likert form over 10 dialogues, n=32 experts; `analysis.py` scores Prolific + volunteer CSVs.
- `HumanEvalGame/`: Unity WebGL tavern demo and FastAPI proxy (`GameProxy/`). Start at `HumanEvalGame/README.md`.
- `models/`: local Q8 GGUFs, can be downloaded via `setup.sh`

Each subdirectory has its own `README.md` with run details.

---

## Datasets and Models

Collection can be found at [PostThinking-AIIDE](https://huggingface.co/collections/chimbiwide/postthinking-aiide)

Datasets:
    - [`chimbiwide/NPC-RP-No-Thinking`](https://huggingface.co/datasets/chimbiwide/NPC-RP-No-Thinking)
    - [`NPC-RP-Pre-Thinking`](https://huggingface.co/datasets/chimbiwide/NPC-RP-Pre-Thinking)
    - [`NPC-RP-Post-Thinking`](https://huggingface.co/datasets/chimbiwide/NPC-RP-Post-Thinking)

Models:
    - [`chimbiwide/Gemma3NPC-4B-no-thinking`](https://huggingface.co/chimbiwide/Gemma3NPC-4B-no-thinking)
    - [`pre-thinking`](https://huggingface.co/chimbiwide/Gemma3NPC-4B-pre-thinking)
    - [`post-thinking`](https://huggingface.co/chimbiwide/Gemma3NPC-4B-post-thinking)

They can be downloaded using `setup.sh`, just make sure to have the [HF CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) installed.
