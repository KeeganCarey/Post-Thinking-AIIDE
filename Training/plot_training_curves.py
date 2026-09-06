import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import wandb

API_KEY = os.environ["WANDB_API_KEY"]
api = wandb.Api(api_key=API_KEY)
# wandb run
RUN = ""

TARGET_RUNS = {"Pre-Thinking", "Post-Thinking", "No-Thinking"}
COLORS = {
    "No-Thinking": "#4C72B0",
    "Pre-Thinking": "#DD8452",
    "Post-Thinking": "#55A868",
}
LABELS = {
    "No-Thinking": "No-Thinking",
    "Pre-Thinking": "Pre-Thinking",
    "Post-Thinking": "Post-Thinking",
}

print("Fetching runs...")
run_data = {}
for run in api.runs(RUN):
    if run.name not in TARGET_RUNS:
        continue
    print(f"  {run.name} ...")
    rows = list(run.scan_history(keys=["train/loss", "eval/loss", "_step"]))
    hist = pd.DataFrame(rows)
    run_data[run.name] = hist
    print(f"    {len(hist)} rows")

out_dir = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(out_dir, exist_ok=True)


def smooth(values, window=10):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - window // 2 - 1), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
fig.suptitle("Gemma3-4B SFT on TPU v6e-8 (LoRA r64/α128)", fontsize=12)

for metric, ax, title in [
    ("train/loss", axes[0], "Train Loss"),
    ("eval/loss", axes[1], "Validation Loss"),
]:
    for name in ["No-Thinking", "Pre-Thinking", "Post-Thinking"]:
        df = run_data.get(name)
        if df is None:
            continue
        sub = df[["_step", metric]].dropna()
        steps = sub["_step"].values
        vals = sub[metric].values
        smoothed = smooth(vals, window=15)
        ax.plot(steps, smoothed, color=COLORS[name], label=LABELS[name], linewidth=1.8)
        # faint raw trace
        ax.plot(steps, vals, color=COLORS[name], alpha=0.15, linewidth=0.7)

    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[1].legend(frameon=False, fontsize=9)
fig.tight_layout()
out = os.path.join(out_dir, "training_curves.pdf")
fig.savefig(out, bbox_inches="tight")
out_png = os.path.join(out_dir, "training_curves.png")
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
print(f"Saved → {out_png}")
