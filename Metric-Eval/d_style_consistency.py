import csv
import json
import logging
import re
from pathlib import Path

import eval_lib
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

TRANSCRIPTS_DIR = Path(__file__).parent / "runs"
OUT_DIR = Path(__file__).parent / "results"
PER_CONVERSATION_CSV = "style_per_conversation.csv"
SUMMARY_JSON = "style_summary.json"
LOG_FILE = "style_consistency.log"

CONDITIONS = eval_lib.CONDITIONS

STYLE_MODEL_PRIMARY = "StyleDistance/styledistance"
STYLE_MODELS = (STYLE_MODEL_PRIMARY,)

MIN_WORDS_PER_TURN = 3
BATCH_SIZE = 32
DEVICE = "cuda"

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 17
CI_LEVEL = 0.95

_WORD_RE = re.compile(r"\w+")


def extract_embeddable_turns(
    record: dict, min_words: int = MIN_WORDS_PER_TURN
) -> list[str]:
    texts = []
    for turn in record.get("turns", []):
        if turn.get("role") != "model":
            continue
        raw = turn.get("raw")
        text = (
            eval_lib.strip_all_tags(raw)
            if raw is not None
            else (turn.get("dialogue") or "")
        )
        text = eval_lib.strip_stage_directions(text)
        if len(_WORD_RE.findall(text)) < min_words:
            continue
        texts.append(text)
    return texts


def pairwise_mean_cosine(embeddings) -> float:
    e = np.asarray(embeddings, dtype=np.float64)
    n = e.shape[0]
    if n < 2:
        return float("nan")
    norms = np.linalg.norm(e, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    e = e / norms
    g = e @ e.T
    return float((g.sum() - np.trace(g)) / (n * (n - 1)))


def bootstrap_mean_ci(
    values,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    ci: float = CI_LEVEL,
) -> tuple[float, float]:
    vals = np.asarray([v for v in values if v == v], dtype=np.float64)
    if vals.size == 0:
        return (float("nan"), float("nan"))
    if vals.size == 1:
        return (float(vals[0]), float(vals[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(int(n_resamples), vals.size))
    means = vals[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return (float(lo), float(hi))


def _load_style_model(model_name: str, device: str):
    import torch
    from sentence_transformers import SentenceTransformer

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable; falling back to cpu")
        device = "cpu"
    logger.info(f"loading style model {model_name} on {device}")
    return SentenceTransformer(model_name, device=device)


def _encode(model, texts: list[str], batch_size: int) -> np.ndarray:
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(emb, dtype=np.float64)


def _free_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def load_conversations(
    transcripts_dir: Path, min_words: int = MIN_WORDS_PER_TURN
) -> list[dict]:
    conversations = []
    for condition in CONDITIONS:
        path = transcripts_dir / f"transcripts_{condition}.jsonl"
        if not path.exists():
            logger.warning(
                f"missing transcript file {path}; skipping condition {condition}"
            )
            continue
        for rec in eval_lib.read_jsonl(path):
            conversations.append(
                {
                    "condition": rec.get("condition", condition),
                    "character_id": rec.get("character_id", ""),
                    "conversation_id": rec.get("conversation_id", ""),
                    "texts": extract_embeddable_turns(rec, min_words=min_words),
                }
            )
    return conversations


def summarize(
    rows: list[dict],
    models,
    n_resamples: int,
    seed: int,
    ci: float,
    min_words: int = MIN_WORDS_PER_TURN,
) -> dict:
    models = tuple(models)
    summary = {
        "metric": "intra-character style consistency (pairwise-mean cosine, proxy)",
        "style_model_primary": models[0] if models else None,
        "style_model_secondary": models[1] if len(models) > 1 else None,
        "min_words_per_turn": int(min_words),
        "bootstrap": {
            "resamples": int(n_resamples),
            "seed": int(seed),
            "ci_level": float(ci),
            "unit": "characters (one conversation per character per condition)",
        },
        "models": {},
    }
    for model_name in models:
        summary["models"][model_name] = {}
        for condition in CONDITIONS:
            scores = [
                r["pairwise_mean_cosine"]
                for r in rows
                if r["model"] == model_name and r["condition"] == condition
            ]
            scored = [s for s in scores if s == s]
            lo, hi = bootstrap_mean_ci(
                scored, n_resamples=n_resamples, seed=seed, ci=ci
            )
            summary["models"][model_name][condition] = {
                "n_conversations": len(scores),
                "n_scored": len(scored),
                "style_consistency_proxy_mean": float(np.mean(scored))
                if scored
                else None,
                "style_consistency_proxy_median": float(np.median(scored))
                if scored
                else None,
                "style_consistency_proxy_std": float(np.std(scored, ddof=1))
                if len(scored) > 1
                else None,
                "style_consistency_proxy_ci95": [lo, hi] if scored else None,
            }
    return summary


def main(
    transcripts_dir=None,
    out_dir=None,
    models=None,
    min_words=None,
    batch_size=None,
    device=None,
    bootstrap_resamples=None,
    bootstrap_seed=None,
    ci_level=None,
) -> dict:
    transcripts_dir = (
        Path(transcripts_dir) if transcripts_dir is not None else TRANSCRIPTS_DIR
    )
    out_dir = Path(out_dir) if out_dir is not None else OUT_DIR
    if isinstance(models, str):
        models = (models,)
    models = tuple(models) if models is not None else STYLE_MODELS
    min_words = int(min_words) if min_words is not None else MIN_WORDS_PER_TURN
    batch_size = int(batch_size) if batch_size is not None else BATCH_SIZE
    device = device if device is not None else DEVICE
    n_resamples = (
        int(bootstrap_resamples)
        if bootstrap_resamples is not None
        else BOOTSTRAP_RESAMPLES
    )
    seed = int(bootstrap_seed) if bootstrap_seed is not None else BOOTSTRAP_SEED
    ci = float(ci_level) if ci_level is not None else CI_LEVEL

    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out_dir / LOG_FILE, encoding="utf-8"),
        ],
        force=True,
    )

    conversations = load_conversations(transcripts_dir, min_words=min_words)
    if not conversations:
        logger.error(f"no transcripts found under {transcripts_dir}; nothing to do")
        return {}
    logger.info(f"loaded {len(conversations)} conversations from {transcripts_dir}")

    rows = []
    for model_name in models:
        st_model = _load_style_model(model_name, device)
        for conv in tqdm(conversations, desc=model_name):
            texts = conv["texts"]
            if len(texts) >= 2:
                score = pairwise_mean_cosine(_encode(st_model, texts, batch_size))
            else:
                logger.warning(
                    f"{conv['conversation_id']}: only {len(texts)} embeddable turn(s); score is NaN"
                )
                score = float("nan")
            rows.append(
                {
                    "condition": conv["condition"],
                    "character_id": conv["character_id"],
                    "conversation_id": conv["conversation_id"],
                    "model": model_name,
                    "n_turns_embedded": len(texts),
                    "pairwise_mean_cosine": score,
                }
            )
        del st_model
        _free_cuda_cache()

    csv_path = out_dir / PER_CONVERSATION_CSV
    fieldnames = [
        "condition",
        "character_id",
        "conversation_id",
        "model",
        "n_turns_embedded",
        "pairwise_mean_cosine",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            out = dict(r)
            s = r["pairwise_mean_cosine"]
            out["pairwise_mean_cosine"] = f"{s:.6f}" if s == s else ""
            writer.writerow(out)
    logger.info(f"wrote {len(rows)} rows to {csv_path}")

    summary = summarize(
        rows, models, n_resamples=n_resamples, seed=seed, ci=ci, min_words=min_words
    )
    summary_path = out_dir / SUMMARY_JSON
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"wrote summary to {summary_path}")
    return summary


if __name__ == "__main__":
    main()
