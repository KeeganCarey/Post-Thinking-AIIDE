import csv
import json
import logging
import random
import re
from pathlib import Path

import eval_lib
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

CONDITIONS = eval_lib.CONDITIONS

NLI_MODEL = "cross-encoder/nli-deberta-v3-large"
NLI_LABELS = ["contradiction", "entailment", "neutral"]

NLI_BATCH_SIZE = 64
NLI_DEVICE = None

MIN_CLAIM_WORDS = 4
MAX_PAIRS_PER_CONVERSATION = None
SEED = 17
BOOTSTRAP_N = 10_000

PERSONA_ANCHORED = False
CARDS_PATH = Path(__file__).parent / "Eval-45.jsonl"

RUNS_DIR = Path(__file__).parent / "runs"
TRANSCRIPT_TEMPLATE = "transcripts_{condition}.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"
PER_CONVERSATION_CSV = RESULTS_DIR / "nli_per_conversation.csv"
PERSONA_CSV = RESULTS_DIR / "nli_persona_per_conversation.csv"
SUMMARY_JSON = RESULTS_DIR / "nli_summary.json"
LOG_FILE = RESULTS_DIR / "nli_contradiction.log"

_CONFIG_KEYS = (
    "CONDITIONS",
    "NLI_MODEL",
    "NLI_LABELS",
    "NLI_BATCH_SIZE",
    "NLI_DEVICE",
    "MIN_CLAIM_WORDS",
    "MAX_PAIRS_PER_CONVERSATION",
    "SEED",
    "BOOTSTRAP_N",
    "PERSONA_ANCHORED",
    "CARDS_PATH",
    "RUNS_DIR",
    "TRANSCRIPT_TEMPLATE",
    "RESULTS_DIR",
    "PER_CONVERSATION_CSV",
    "PERSONA_CSV",
    "SUMMARY_JSON",
    "LOG_FILE",
)

CSV_COLUMNS = [
    "condition",
    "character_id",
    "conversation_id",
    "n_claims",
    "n_pairs",
    "n_contradictions",
    "contradiction_rate",
]

_WORD_RE = re.compile(r"\w+")
_BACKGROUND_RE = re.compile(
    r"Background:\s*(.*?)\s*(?:Current Location:|Quest:|Roleplaying Instructions:|$)",
    re.DOTALL,
)
_EXCLAMATIVE_RE = re.compile(r"^\W*(?:what|how)\b", re.IGNORECASE)
_TRAILING_CLOSERS = "\"'”’»)]* \t"


def _terminator(sent: str) -> str:
    stripped = sent.rstrip(_TRAILING_CLOSERS)
    return stripped[-1] if stripped else ""


def _is_pure_exclamation(sent: str) -> bool:
    words = _WORD_RE.findall(sent)
    if not any(any(c.isalpha() for c in w) for w in words):
        return True
    return bool(_EXCLAMATIVE_RE.match(sent))


def extract_claims(
    turns: list[dict], min_words: int = MIN_CLAIM_WORDS
) -> list[tuple[int, str]]:
    claims = []
    for turn in turns:
        if turn.get("role") != "model":
            continue
        idx = turn.get("index", 0)
        dialogue = turn.get("dialogue") or eval_lib.strip_all_tags(
            turn.get("raw") or ""
        )
        for sent in eval_lib.sentence_split(eval_lib.strip_stage_directions(dialogue)):
            term = _terminator(sent)
            if term == "?":
                continue
            if term == "!" and _is_pure_exclamation(sent):
                continue
            if len(_WORD_RE.findall(sent)) < min_words:
                continue
            claims.append((idx, sent))
    return claims


def build_pairs(
    claims: list[tuple[int, str]], max_pairs: int, seed_key: str
) -> tuple[list[tuple[str, str]], int]:
    pairs = [(a, b) for ta, a in claims for tb, b in claims if ta < tb]
    n_total = len(pairs)
    if max_pairs is not None and n_total > max_pairs:
        pairs = random.Random(seed_key).sample(pairs, max_pairs)
    return pairs, n_total


def extract_background(card_text: str) -> str:
    m = _BACKGROUND_RE.search(card_text or "")
    if m and m.group(1).strip():
        return m.group(1).strip()
    logger.warning(
        "Background marker not found in card text; using full card text as premise"
    )
    return (card_text or "").strip()


def check_nli_label_order(model, labels) -> None:
    config = getattr(getattr(model, "model", None), "config", None)
    id2label = getattr(config, "id2label", None)
    if not id2label:
        logger.warning("model config has no id2label; cannot verify NLI_LABELS order")
        return
    try:
        model_labels = [str(id2label[k]).lower() for k in sorted(id2label, key=int)]
    except (KeyError, TypeError, ValueError):
        logger.warning(
            f"unparseable id2label {id2label!r}; cannot verify NLI_LABELS order"
        )
        return
    if any(lab.startswith("label_") for lab in model_labels):
        logger.warning(
            f"uninformative id2label {model_labels}; cannot verify NLI_LABELS order"
        )
        return
    if model_labels != [str(lab).lower() for lab in labels]:
        raise ValueError(
            f"NLI_LABELS {list(labels)} does not match the loaded model's id2label "
            f"order {model_labels}; update NLI_LABELS to match NLI_MODEL."
        )
    logger.info(f"NLI label order verified against model config: {model_labels}")


def count_contradictions(
    model, pairs: list[tuple[str, str]], labels: list[str], batch_size: int
) -> int:
    if not pairs:
        return 0
    scores = np.asarray(
        model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    )
    if scores.ndim != 2 or scores.shape[1] != len(labels):
        raise ValueError(
            f"NLI model returned scores of shape {scores.shape}; expected (n, {len(labels)}). "
            "NLI_LABELS does not match the configured NLI_MODEL."
        )
    return int((scores.argmax(axis=1) == labels.index("contradiction")).sum())


def _bootstrap_ci(values: list[float], n_boot: int, seed: int) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def summarize(rows: list[dict], conditions, n_boot: int, seed: int) -> dict:
    out = {}
    for condition in conditions:
        cond_rows = [r for r in rows if r["condition"] == condition]
        scored = [r for r in cond_rows if r["n_pairs"] > 0]
        rates = [r["contradiction_rate"] for r in scored]
        n_pairs_total = sum(r["n_pairs"] for r in cond_rows)
        n_contra_total = sum(r["n_contradictions"] for r in cond_rows)
        ci_low, ci_high = _bootstrap_ci(rates, n_boot, seed)
        out[condition] = {
            "n_conversations": len(cond_rows),
            "n_conversations_scored": len(scored),
            "n_conversations_subsampled": sum(
                1 for r in cond_rows if r.get("subsampled")
            ),
            "n_pairs_total": n_pairs_total,
            "n_contradictions_total": n_contra_total,
            "pooled_contradiction_rate_proxy": (n_contra_total / n_pairs_total)
            if n_pairs_total
            else None,
            "contradiction_rate_proxy": {
                "mean": float(np.mean(rates)) if rates else None,
                "median": float(np.median(rates)) if rates else None,
                "std": float(np.std(rates)) if rates else None,
                "ci95_low": ci_low if rates else None,
                "ci95_high": ci_high if rates else None,
            },
        }
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=CSV_COLUMNS, restval="", extrasaction="ignore"
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )


def _score_conversation(
    model, rec: dict, condition: str, cfg: dict, premise_override: str | None = None
) -> dict:
    conversation_id = rec.get("conversation_id", "")
    claims = extract_claims(rec.get("turns", []), min_words=cfg["MIN_CLAIM_WORDS"])
    if premise_override is None:
        seed_key = f"{cfg['SEED']}:{conversation_id}"
        pairs, n_total = build_pairs(
            claims, cfg["MAX_PAIRS_PER_CONVERSATION"], seed_key
        )
    else:
        all_pairs = [(premise_override, claim) for _idx, claim in claims]
        n_total = len(all_pairs)
        pairs = all_pairs
        max_pairs = cfg["MAX_PAIRS_PER_CONVERSATION"]
        if max_pairs is not None and n_total > max_pairs:
            pairs = random.Random(f"{cfg['SEED']}:{conversation_id}:persona").sample(
                all_pairs, max_pairs
            )
    if n_total > len(pairs):
        logger.info(
            f"{conversation_id}: subsampling {n_total} -> {len(pairs)} pairs (seeded)"
        )
    n_contra = count_contradictions(
        model, pairs, cfg["NLI_LABELS"], cfg["NLI_BATCH_SIZE"]
    )
    return {
        "condition": rec.get("condition", condition),
        "character_id": rec.get("character_id", ""),
        "conversation_id": conversation_id,
        "n_claims": len(claims),
        "n_pairs": len(pairs),
        "n_contradictions": n_contra,
        "contradiction_rate": (n_contra / len(pairs)) if pairs else None,
        "subsampled": n_total > len(pairs),
    }


def main(**overrides):
    cfg = {k: globals()[k] for k in _CONFIG_KEYS}
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise TypeError(f"unknown config overrides: {sorted(unknown)}")
    cfg.update(overrides)

    results_dir = Path(cfg["RESULTS_DIR"])
    results_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(Path(cfg["LOG_FILE"]))

    import torch
    from sentence_transformers import CrossEncoder

    device = cfg["NLI_DEVICE"]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning(
            "running NLI on CPU: functional but slow for a deberta-v3-large cross-encoder"
        )
    logger.info(f"loading NLI model {cfg['NLI_MODEL']} on {device}")
    model = CrossEncoder(cfg["NLI_MODEL"], device=device)
    check_nli_label_order(model, cfg["NLI_LABELS"])

    backgrounds = {}
    if cfg["PERSONA_ANCHORED"]:
        cards = eval_lib.load_character_cards(cfg["CARDS_PATH"])
        backgrounds = {c.character_id: extract_background(c.card_text) for c in cards}
        logger.info(
            f"persona-anchored mode on: {len(backgrounds)} card backgrounds loaded"
        )

    rows = []
    persona_rows = []
    for condition in cfg["CONDITIONS"]:
        path = Path(cfg["RUNS_DIR"]) / cfg["TRANSCRIPT_TEMPLATE"].format(
            condition=condition
        )
        if not path.exists():
            logger.warning(
                f"missing transcripts file {path}; skipping condition {condition}"
            )
            continue
        records = eval_lib.read_jsonl(path)
        logger.info(f"{condition}: {len(records)} conversations from {path}")
        for rec in tqdm(records, desc=f"nli {condition}"):
            rows.append(_score_conversation(model, rec, condition, cfg))
            if cfg["PERSONA_ANCHORED"]:
                background = backgrounds.get(rec.get("character_id", ""))
                if background is None:
                    logger.warning(
                        f"no card background for {rec.get('character_id')!r}; "
                        "skipping persona-anchored scoring for this conversation"
                    )
                    continue
                persona_rows.append(
                    _score_conversation(
                        model, rec, condition, cfg, premise_override=background
                    )
                )

    _write_csv(Path(cfg["PER_CONVERSATION_CSV"]), rows)
    logger.info(f"wrote {len(rows)} rows to {cfg['PER_CONVERSATION_CSV']}")
    if cfg["PERSONA_ANCHORED"]:
        _write_csv(Path(cfg["PERSONA_CSV"]), persona_rows)
        logger.info(f"wrote {len(persona_rows)} rows to {cfg['PERSONA_CSV']}")

    summary = {
        "nli_model": cfg["NLI_MODEL"],
        "nli_labels": list(cfg["NLI_LABELS"]),
        "min_claim_words": cfg["MIN_CLAIM_WORDS"],
        "max_pairs_per_conversation": cfg["MAX_PAIRS_PER_CONVERSATION"],
        "seed": cfg["SEED"],
        "persona_anchored": bool(cfg["PERSONA_ANCHORED"]),
        "per_condition": summarize(
            rows, cfg["CONDITIONS"], cfg["BOOTSTRAP_N"], cfg["SEED"]
        ),
    }
    if cfg["PERSONA_ANCHORED"]:
        summary["persona_anchored_per_condition"] = summarize(
            persona_rows, cfg["CONDITIONS"], cfg["BOOTSTRAP_N"], cfg["SEED"]
        )
    summary_path = Path(cfg["SUMMARY_JSON"])
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"wrote summary to {summary_path}")
    return summary


if __name__ == "__main__":
    main()
