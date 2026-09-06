# Training

LoRA SFT of `google/gemma-3-4b-it` with [Tunix](https://tunix.readthedocs.io/en/latest/index.html) on a TPU v6e-8. Checkpoints go to GCS, so you need Google Cloud access.

> [!IMPORTANT]
> Huge thanks to [TPU Research Cloud](https://sites.research.google/trc/about/) for providing us with free TPU compute.

## Data and released models

They are all on HuggingFace


| Condition | Dataset | Model |
|---|---|---|
| no-thinking | [`chimbiwide/NPC-RP-No-Thinking`](https://huggingface.co/datasets/chimbiwide/NPC-RP-No-Thinking) | [`chimbiwide/Gemma3NPC-4B-no-thinking`](https://huggingface.co/chimbiwide/Gemma3NPC-4B-no-thinking) |
| pre-thinking | [`chimbiwide/NPC-RP-Pre-Thinking`](https://huggingface.co/datasets/chimbiwide/NPC-RP-Pre-Thinking) | [`chimbiwide/Gemma3NPC-4B-pre-thinking`](https://huggingface.co/chimbiwide/Gemma3NPC-4B-pre-thinking) |
| post-thinking | [`chimbiwide/NPC-RP-Post-Thinking`](https://huggingface.co/datasets/chimbiwide/NPC-RP-Post-Thinking) | [`chimbiwide/Gemma3NPC-4B-post-thinking`](https://huggingface.co/chimbiwide/Gemma3NPC-4B-post-thinking) |

GGUF quants: `chimbiwide/Gemma3NPC-4B-{no,pre,post}-thinking-GGUF`.

## How to SSH

```
gcloud alpha compute tpus tpu-vm ssh <VM_NAME> \
  --zone=us-east1-d \
  --project=<PROJECT_ID> \
  --tunnel-through-iap
```

## TPU software versions

v6e - `v2-alpha-tpuv6e`

## Setup

From `Training/` on the TPU VM:

1. Run `setup.sh` (uv, Python 3.13, deps, transparent hugepages).
2. Export `HF_TOKEN` and `WANDB_API_KEY`.
3. Edit `GCS_BUCKET` in the training script if you are not using `gs://tpu-aiide`.
4. Run one of the three scripts

```bash
uv run python No-Thinking.py
uv run python Pre-Thinking.py
uv run python Post-Thinking.py
```

## Training scripts

Standard LoRA, plus a custom loss mask so only assistant tokens are trained (Tunix does not ship that, so we built our own version).

Shared settings: LoRA r=64 α=128, batch size 2, max length 4096, 2 epochs, 5% val split (`seed=42`), AdamW peak 2e-4 with 10% warmup.

`plot_training_curves.py` pulls the three W&B runs and writes `figures/`.
