import os

import jax
import qwix
import wandb
from flax import nnx
from huggingface_hub import login, snapshot_download
from safetensors.torch import load_file, save_file
import glob
import torch

# Authenticate
try:
    hf_token = os.environ["HF_TOKEN"]
    login(token=hf_token)
except Exception:
    print("Please add HF_TOKEN to Colab secrets or login manually.")
    login()

if "WANDB_API_KEY" in os.environ and os.environ["WANDB_API_KEY"]:
    wandb.login(key=os.environ["WANDB_API_KEY"])

print(f"JAX Devices: {jax.devices()}")  # Verify TPU access

import shutil

import jax
import numpy as np
import optax
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.models.gemma3 import model as gemma3_model_lib
from tunix.models.gemma3 import params as gemma_params
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib
from tunix.sft import metrics_logger, peft_trainer, utils

MODEL_ID = "google/gemma-3-4b-it"
DATASET_ID = "chimbiwide/NPC-RP-Pre-Thinking"
CONDITION = "pre-thinking"
GCS_BUCKET = "gs://tpu-aiide"
BATCH_SIZE = 2
MAX_LENGTH = 4096
RANK = 64
ALPHA = 128.0
USE_QUANTIZATION = False

NUM_TPUS = len(jax.devices())
if NUM_TPUS == 8:
    MESH_COUNTS = (2, 4)  # 2-way FSDP x 4-way TP
elif NUM_TPUS == 1:
    MESH_COUNTS = (1, 1)
else:
    MESH_COUNTS = (1, NUM_TPUS)

MESH = [
    MESH_COUNTS,
    ("fsdp", "tp"),
]
MESH = jax.make_mesh(*MESH, axis_types=(jax.sharding.AxisType.Auto,) * len(MESH[0]))
print("Mesh configuration created.")

ignore_patterns = ["*.pth"]
print(f"Downloading {MODEL_ID}...")
local_model_path = snapshot_download(repo_id=MODEL_ID, ignore_patterns=ignore_patterns)

model_config = gemma3_model_lib.ModelConfig.gemma3_4b_it(text_only=True)
model_config.num_embed = 262208

# Initialize Base Model on Mesh
base_model = params_safetensors_lib.create_model_from_safe_tensors(
    local_model_path, model_config, MESH
)

tokenizer_path = os.path.join(local_model_path, "tokenizer.model")
try:
    tokenizer = tokenizer_lib.Tokenizer(
        tokenizer_path=tokenizer_path,
        add_bos=False,
        add_eos=False,
    )
except:
    import glob

    t_paths = glob.glob(os.path.join(local_model_path, "*.model"))
    if t_paths:
        tokenizer = tokenizer_lib.Tokenizer(tokenizer_path=t_paths[0])
    else:
        raise FileNotFoundError("Tokenizer model file not found in snapshot.")
BOS_ID = tokenizer.bos_id()
EOS_ID = tokenizer.eos_id()
PAD_ID = tokenizer.pad_id()


def get_lora_model(base_model, mesh, quantize=False):
    target_modules = ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj"

    # Init LoRA/QLoRA
    if quantize:
        # QLoRA Provider
        lora_provider = qwix.LoraProvider(
            module_path=target_modules,
            rank=RANK,
            alpha=ALPHA,
            weight_qtype="nf4",  # 4-bit Normal Float
            tile_size=128,
        )
    else:
        # LoRA Provider
        lora_provider = qwix.LoraProvider(
            module_path=target_modules, rank=RANK, alpha=ALPHA
        )

    # Apply adapters
    model_input = base_model.get_model_input()
    lora_model = qwix.apply_lora_to_model(
        base_model,
        lora_provider,
        **model_input,
        rngs=nnx.Rngs(0),  # rng seed
    )

    # Shard the new LoRA parameters across the TPU mesh
    with jax.set_mesh(mesh):
        state = nnx.state(lora_model)
        pspecs = nnx.get_partition_spec(state)
        sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
        nnx.update(lora_model, sharded_state)

    return lora_model


# Create the wrapped model
lora_model = get_lora_model(base_model, mesh=MESH, quantize=USE_QUANTIZATION)

print(f"Model ready. Mode: {'QLoRA' if USE_QUANTIZATION else 'LoRA'}")


def build_masked_example(example):
    messages = example.get("messages")

    input_ids = [BOS_ID]  # all messages begins with a bos token
    loss_mask = [0]

    for msg in messages:
        role = "model" if msg.get("role") == "assistant" else "user"
        content = msg.get("content")

        header_ids = tokenizer.encode(f"<start_of_turn>{role}\n")
        body_ids = tokenizer.encode(f"{content}<end_of_turn>\n")

        input_ids += header_ids
        loss_mask += [0] * len(
            header_ids
        )  # header ids are never actual targets, so its 0

        input_ids += body_ids
        learn = 1 if role == "model" else 0  # only learn model responses
        loss_mask += [learn] * len(body_ids)

    input_ids.append(EOS_ID)  # append the EOS_ID
    loss_mask.append(0)  # EOS_IDs are not learned

    full_len = len(input_ids)

    # truncate to keep the front
    input_ids = input_ids[:MAX_LENGTH]
    loss_mask = loss_mask[:MAX_LENGTH]
    pad_len = MAX_LENGTH - len(input_ids)
    input_ids += [PAD_ID] * pad_len
    loss_mask += [0] * pad_len  # padding tokens are never learned during training

    return {
        "input_tokens": input_ids,
        "loss_mask": loss_mask,
        "full_length": full_len,
        "has_loss": sum(loss_mask) > 0,
    }


from datasets import load_dataset

raw_dataset = load_dataset(DATASET_ID, split="train").shuffle(seed=42)

processed_ds = raw_dataset.map(
    build_masked_example,
    remove_columns=raw_dataset.column_names,
)

processed_ds = processed_ds.filter(lambda ex: ex["has_loss"])


lengths = processed_ds["full_length"]
avg = round(sum(lengths) / len(lengths))
print(
    f"Avg token length: {avg} | kept {len(processed_ds)} examples | MAX_LENGTH={MAX_LENGTH}"
)
if avg > MAX_LENGTH:
    print(
        "Warning: heavy truncation — fine for a smoke test, raise MAX_LENGTH for the real run."
    )

ex = processed_ds[0]
trained_ids = [t for t, m in zip(ex["input_tokens"], ex["loss_mask"]) if m == 1]

print("Model is trained (mask == 1)")
print(tokenizer.decode(trained_ids))
print("\n Full Sequence (the entire row)")
real_ids = [t for t in ex["input_tokens"] if t != PAD_ID]
print(tokenizer.decode(real_ids))


class MaskedBatches:
    def __init__(self, ds, batch_size, num_passes=1, shuffle=False, seed=42):
        self.ds = ds
        self.bs = batch_size
        self.num_passes = num_passes
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        for p in range(self.num_passes):
            ds = self.ds.shuffle(seed=self.seed + p) if self.shuffle else self.ds
            n_full = (len(ds) // self.bs) * self.bs
            for i in range(0, n_full, self.bs):
                batch = ds[i : i + self.bs]
                input_tokens = np.array(batch.get("input_tokens"), dtype=np.int32)
                loss_mask = np.array(batch.get("loss_mask"), dtype=np.bool_)
                yield peft_trainer.TrainingInput(
                    input_tokens=input_tokens,
                    input_mask=loss_mask,
                )


split = processed_ds.train_test_split(test_size=0.05, seed=42)
NUM_EPOCHS = 2
train_iter = MaskedBatches(
    split["train"], BATCH_SIZE, num_passes=NUM_EPOCHS, shuffle=True
)
val_iter = MaskedBatches(split["test"], BATCH_SIZE, num_passes=1, shuffle=False)

steps_per_epoch = len(split["train"]) // BATCH_SIZE
MAX_STEPS = steps_per_epoch * NUM_EPOCHS

print(MAX_STEPS)


import os


# Define Input Generator
def gen_model_input_fn(x):

    pad_mask = x.input_tokens != PAD_ID
    positions = utils.build_positions_from_mask(pad_mask)
    attention_mask = utils.make_causal_attn_mask(pad_mask)

    # Must return a dictionary of arrays for the model
    return {
        "input_tokens": x.input_tokens,
        "input_mask": x.input_mask,  # the loss mask
        "positions": positions,
        "attention_mask": attention_mask,
    }


# Logging
logging_options = metrics_logger.MetricsLoggerOptions(
    log_dir="/tmp/tunix_logs", flush_every_n_steps=10
)

# Config
train_config = peft_trainer.TrainingConfig(
    eval_every_n_steps=100,
    max_steps=MAX_STEPS,
    metrics_logging_options=logging_options,
    checkpoint_root_directory=f"{GCS_BUCKET}/checkpoints/{CONDITION}",
)


optimizer = optax.chain(
    optax.clip_by_global_norm(max_norm=1.0),
    optax.adamw(
        learning_rate=optax.schedules.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=2e-4,
            warmup_steps=int(0.1 * MAX_STEPS),
            decay_steps=MAX_STEPS,
            end_value=1e-6,
        ),
        b1=0.9,
        b2=0.99,
    ),
)

# Initialize Trainer
trainer = peft_trainer.PeftTrainer(
    lora_model, optimizer, train_config
).with_gen_model_input_fn(gen_model_input_fn)

with jax.set_mesh(MESH):
    trainer.train(train_iter, val_iter)

import importlib
import os

import tunix.models.safetensors_saver as saver_lib
from huggingface_hub import HfApi, create_repo

# import wandb

# Create output directory
output_dir = f"./gemma3-4b-{CONDITION}"

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)


import logging

# The saving process triggers JAX compilation which tries to log metrics.
if wandb.run is None:
    print("Initializing dummy wandb run for JAX logging...")
    wandb.init(mode="disabled")

importlib.reload(saver_lib)
importlib.reload(gemma_params)


shards = sorted(glob.glob(f"{local_model_path}/model-*.safetensors"))
merged = {}
for shard in shards:
    merged.update(load_file(shard))

save_file(merged, f"{local_model_path}/model.safetensors")
print(f"Merged {len(merged)} tensors → {local_model_path}/model.safetensors")

print(f"Saving merged LoRA model to {output_dir}")
print(lora_model.layers[0].attn.q_einsum)

import tunix.models.gemma3.params as gemma_params

def _mm_state_key(lora_name: str) -> str:
    # gemma-3-4b-it is Gemma3ForConditionalGeneration → text weights under language_model.model.*
    return (
        f"language_model.model.{lora_name}.weight".replace(".attn.", ".self_attn.")
        .replace("q_einsum", "q_proj")
        .replace("k_einsum", "k_proj")
        .replace("v_einsum", "v_proj")
        .replace("attn_vec_einsum", "o_proj")
    )

gemma_params._gemma3_state_key_to_safetensors_key = _mm_state_key

# Use the save_lora_merged_model_as_safetensors function
gemma_params.save_lora_merged_model_as_safetensors(
    local_model_path=local_model_path,
    output_dir=output_dir,
    lora_model=lora_model,
    rank=RANK,
    alpha=ALPHA,
)

print("\n" + "=" * 60)
print("Model saved successfully!")
print(f"Output directory: {output_dir}")
print("=" * 60)

print("\nSaved files:")
for f in os.listdir(output_dir):
    size = os.path.getsize(os.path.join(output_dir, f)) / (1024 * 1024)
    print(f"  {f:<30} {size:>10.2f} MB")

idx = os.path.join(output_dir, "model.safetensors.index.json")
if os.path.exists(idx):
    os.remove(idx)

os.system(f"gsutil -m cp -r {output_dir} {GCS_BUCKET}/merged/{CONDITION}/")

# --- Upload to Hugging Face ---
try:
    api = HfApi()
    user_info = api.whoami()
    username = user_info["name"]

    # Define the new repo ID
    NEW_REPO_ID = f"{username}/Gemma3NPC-4B-{CONDITION}"
    print(f"Uploading to Hugging Face Hub: {NEW_REPO_ID}...")

    # Create repo if it doesn't exist
    create_repo(NEW_REPO_ID, exist_ok=True, repo_type="model", private=True)

    # Upload folder
    api.upload_folder(folder_path=output_dir, repo_id=NEW_REPO_ID, repo_type="model")
    print(
        f"\nUpload complete! View your model at: https://huggingface.co/{NEW_REPO_ID}"
    )

except Exception as e:
    print(f"\nUpload failed. Ensure you are logged in with a write token.\nError: {e}")
