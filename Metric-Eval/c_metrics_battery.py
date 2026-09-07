import json
import logging
import math
import random
import re
from collections import Counter
from pathlib import Path

import eval_lib
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

CONDITIONS = eval_lib.CONDITIONS
FLAGS = eval_lib.FLAGS
WINDOW_N = eval_lib.WINDOW_N

RUNS_DIR = Path(__file__).parent / "runs"
TRANSCRIPT_PATHS = {c: RUNS_DIR / f"transcripts_{c}.jsonl" for c in CONDITIONS}

RESULTS_DIR = Path(__file__).parent / "results"
PER_CONVERSATION_CSV = RESULTS_DIR / "metrics_per_conversation.csv"
SUMMARY_JSON = RESULTS_DIR / "metrics_summary.json"
SUMMARY_CSV = RESULTS_DIR / "metrics_summary.csv"
LOG_FILE = RESULTS_DIR / "metrics_battery.log"

CJK_PATTERN = r"[一-鿿぀-ヿ가-힯]"

PLAYER_ATTRIBUTION_PATTERNS = {
    "second_person": r"\b(you|your)\b",
    "player_action_narration": r"\bthe (player|stranger|traveler) (seems|looks|appears|wants|feels)\b",
    "feeling_attribution": r"\b(he|she|they) (feels?|felt|seems?|seemed)\b",
}

PLACEHOLDER_LITERALS = (
    "[Player Name]",
    "{player}",
    "[player]",
    "{{user}}",
    "{{char}}",
    "[Name]",
)

ANACHRONISM_ENABLED = False
ANACHRONISM_TERMS = (
    "phone",
    "internet",
    "email",
    "computer",
    "tv",
    "car",
    "gun",
    "okay",
    "ok",
    "wifi",
    "app",
    "online",
    "website",
    "robot",
    "electricity",
)

WORD_PATTERN = r"[\w']+"

NEAR_EMPTY_WORDS = 2
DEGENERATE_NGRAM = 4
DEGENERATE_REPEATS = 3
DEGENERATE_SINGLE_WORD_FRAC = 0.30
DEGENERATE_MIN_WORDS = 20

CROSS_CHAR_PAIR_SAMPLES = 500
RNG_SEED = 17


_CJK_RE = re.compile(CJK_PATTERN)
_WORD_RE = re.compile(WORD_PATTERN)
_ATTR_RES = {
    name: re.compile(pat, re.IGNORECASE)
    for name, pat in PLAYER_ATTRIBUTION_PATTERNS.items()
}


def word_tokens(text: str, word_re=None) -> list[str]:
    if not text:
        return []
    return (word_re or _WORD_RE).findall(str(text).lower())


def ngrams(tokens: list[str], n: int) -> list[tuple]:
    if len(tokens) < n:
        return []
    return list(zip(*(tokens[i:] for i in range(n))))


def distinct_n_pooled(token_lists: list, n: int) -> float:
    grams = [g for toks in token_lists for g in ngrams(toks, n)]
    if not grams:
        return float("nan")
    return len(set(grams)) / len(grams)


def trigram_set(tokens: list[str]) -> set:
    return set(ngrams(tokens, 3))


def jaccard(a: set, b: set) -> float:
    union = a | b
    if not union:
        return float("nan")
    return len(a & b) / len(union)


def has_cjk(text: str, cjk_re=None) -> bool:
    return bool((cjk_re or _CJK_RE).search(text)) if text else False


def player_attribution_hits(trace: str, attr_res=None) -> dict:
    return {
        name: bool(rx.search(trace)) for name, rx in (attr_res or _ATTR_RES).items()
    }


def placeholder_counts(text: str, literals) -> Counter:
    low = (text or "").lower()
    counts = Counter()
    for lit in literals:
        k = low.count(lit.lower())
        if k:
            counts[lit] += k
    return counts


def build_term_regex(terms):
    if not terms:
        return None
    return re.compile(r"\b(" + "|".join(re.escape(t.lower()) for t in terms) + r")\b")


def is_degenerate(
    tokens: list[str],
    ngram: int = DEGENERATE_NGRAM,
    repeats: int = DEGENERATE_REPEATS,
    single_word_frac: float = DEGENERATE_SINGLE_WORD_FRAC,
    min_words: int = DEGENERATE_MIN_WORDS,
) -> bool:
    if len(tokens) >= ngram:
        gram_counts = Counter(ngrams(tokens, ngram))
        if max(gram_counts.values()) >= repeats:
            return True
    if len(tokens) > min_words:
        if max(Counter(tokens).values()) / len(tokens) > single_word_frac:
            return True
    return False


def _nanrate(num: int, den: int) -> float:
    return (num / den) if den else float("nan")


def _rate(num: int, den: int):
    return (num / den) if den else None


def _mean(vals: list) -> float:
    return float(np.mean(vals)) if vals else float("nan")


def _median(vals: list) -> float:
    return float(np.median(vals)) if vals else float("nan")


def dist_stats(values) -> dict:
    arr = np.asarray(
        [
            v
            for v in values
            if v is not None and not (isinstance(v, float) and math.isnan(v))
        ],
        dtype=float,
    )
    if arr.size == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "q1": None,
            "q3": None,
            "iqr": None,
            "p95": None,
        }
    q1, med, q3, p95 = np.percentile(arr, [25, 50, 75, 95])
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(med),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "p95": float(p95),
    }


def _jsonsafe(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonsafe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonsafe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        obj = float(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def conversation_metrics(
    rec: dict,
    condition: str,
    literals=PLACEHOLDER_LITERALS,
    anachronism_re=None,
    near_empty_words: int = NEAR_EMPTY_WORDS,
    word_re=None,
    cjk_re=None,
    attr_res=None,
    degenerate_ngram: int = DEGENERATE_NGRAM,
    degenerate_repeats: int = DEGENERATE_REPEATS,
    degenerate_single_word_frac: float = DEGENERATE_SINGLE_WORD_FRAC,
    degenerate_min_words: int = DEGENERATE_MIN_WORDS,
):
    word_re = word_re or _WORD_RE
    cjk_re = cjk_re or _CJK_RE
    attr_res = attr_res or _ATTR_RES
    turns = rec.get("turns") or []
    model_turns = [t for t in turns if t.get("role") == "model"]
    n = len(model_turns)

    flag_counts = Counter()
    dialogue_words, trace_words, ratios, tokens_rec = [], [], [], []
    dlg_token_lists, trace_token_lists = [], []
    cjk_dlg = cjk_trc = 0
    attr_any = 0
    attr_fams = Counter()
    ph_turns = 0
    ph_counter = Counter()
    ana_turns = 0
    ana_counter = Counter()
    near_empty = degenerate = 0
    fr_stop = fr_length = fr_recorded = 0
    n_traces = 0

    for t in model_turns:
        raw = t.get("raw") or ""
        parsed = eval_lib.parse_model_turn(raw, condition)
        flag_counts.update(parsed.flags)

        d_toks = word_tokens(parsed.dialogue, word_re)
        dlg_token_lists.append(d_toks)
        dialogue_words.append(len(d_toks))
        if has_cjk(parsed.dialogue, cjk_re):
            cjk_dlg += 1

        t_toks = []
        if parsed.trace is not None:
            n_traces += 1
            t_toks = word_tokens(parsed.trace, word_re)
            trace_token_lists.append(t_toks)
            trace_words.append(len(t_toks))
            if d_toks:
                ratios.append(len(t_toks) / len(d_toks))
            if has_cjk(parsed.trace, cjk_re):
                cjk_trc += 1
            fams = player_attribution_hits(parsed.trace, attr_res)
            if any(fams.values()):
                attr_any += 1
            for name, hit in fams.items():
                if hit:
                    attr_fams[name] += 1

        ph = placeholder_counts(raw, literals)
        if ph:
            ph_turns += 1
            ph_counter.update(ph)

        if anachronism_re is not None:
            hits = anachronism_re.findall(parsed.dialogue.lower())
            if hits:
                ana_turns += 1
                ana_counter.update(hits)

        tok = t.get("tokens")
        if isinstance(tok, (int, float)) and not isinstance(tok, bool):
            tokens_rec.append(float(tok))

        fr = t.get("finish_reason")
        if fr is not None:
            fr_recorded += 1
            if fr == "stop":
                fr_stop += 1
            elif fr == "length":
                fr_length += 1

        if len(d_toks) < near_empty_words:
            near_empty += 1
        deg_kwargs = dict(
            ngram=degenerate_ngram,
            repeats=degenerate_repeats,
            single_word_frac=degenerate_single_word_frac,
            min_words=degenerate_min_words,
        )
        if is_degenerate(d_toks, **deg_kwargs) or is_degenerate(t_toks, **deg_kwargs):
            degenerate += 1

    dlg_sets = [trigram_set(toks) for toks in dlg_token_lists]
    rep_vals = [
        v for a, b in zip(dlg_sets, dlg_sets[1:]) if not math.isnan(v := jaccard(a, b))
    ]

    trace_sets = [trigram_set(toks) for toks in trace_token_lists]
    collapse_vals = []
    for i in range(len(trace_sets)):
        for j in range(i + 1, len(trace_sets)):
            v = jaccard(trace_sets[i], trace_sets[j])
            if not math.isnan(v):
                collapse_vals.append(v)
    trace_bag = frozenset().union(*trace_sets) if trace_sets else frozenset()

    total_dlg_words = sum(dialogue_words)
    total_trc_words = sum(trace_words)

    row = {
        "conversation_id": rec.get("conversation_id", ""),
        "character_id": rec.get("character_id", ""),
        "condition": condition,
        "n_model_turns": n,
        "n_traces": n_traces,
    }
    for f in FLAGS:
        row[f"flag_{f}"] = _nanrate(flag_counts[f], n)
    row.update(
        {
            "cjk_dialogue_rate": _nanrate(cjk_dlg, n),
            "cjk_trace_rate": _nanrate(cjk_trc, n_traces),
            "player_attr_trace_rate": _nanrate(attr_any, n_traces),
        }
    )
    for fam in attr_res:
        row[f"player_attr_{fam}_count"] = attr_fams[fam]
    row.update(
        {
            "placeholder_turn_rate": _nanrate(ph_turns, n),
            "placeholder_hits": sum(ph_counter.values()),
            "anachronism_turn_rate": _nanrate(ana_turns, n)
            if anachronism_re is not None
            else float("nan"),
            "anachronism_hits": sum(ana_counter.values())
            if anachronism_re is not None
            else float("nan"),
            "dialogue_words_median": _median(dialogue_words),
            "dialogue_words_mean": _mean(dialogue_words),
            "dialogue_words_total": total_dlg_words,
            "trace_words_median": _median(trace_words),
            "trace_words_total": total_trc_words,
            "trace_dialogue_ratio": (total_trc_words / total_dlg_words)
            if (n_traces and total_dlg_words)
            else float("nan"),
            "tokens_recorded_median": _median(tokens_rec),
            "distinct_1": distinct_n_pooled(dlg_token_lists, 1),
            "distinct_2": distinct_n_pooled(dlg_token_lists, 2),
            "distinct_3": distinct_n_pooled(dlg_token_lists, 3),
            "self_repetition_trigram_jaccard": _mean(rep_vals),
            "trace_collapse_within_jaccard": _mean(collapse_vals),
            "finish_stop_rate": _nanrate(fr_stop, fr_recorded),
            "finish_length_rate": _nanrate(fr_length, fr_recorded),
            "finish_recorded_rate": _nanrate(fr_recorded, n),
            "near_empty_dialogue_rate": _nanrate(near_empty, n),
            "degenerate_loop_rate": _nanrate(degenerate, n),
        }
    )

    extras = {
        "n_model_turns": n,
        "n_traces": n_traces,
        "flag_counts": flag_counts,
        "cjk_dialogue_turns": cjk_dlg,
        "cjk_trace_turns": cjk_trc,
        "attr_any": attr_any,
        "attr_family_counts": attr_fams,
        "placeholder_turns": ph_turns,
        "placeholder_counts": ph_counter,
        "anachronism_turns": ana_turns,
        "anachronism_counts": ana_counter,
        "dialogue_word_counts": dialogue_words,
        "trace_word_counts": trace_words,
        "trace_dialogue_ratios": ratios,
        "tokens_recorded": tokens_rec,
        "trace_bag": trace_bag,
        "character_id": rec.get("character_id", ""),
        "finish_stop": fr_stop,
        "finish_length": fr_length,
        "finish_recorded": fr_recorded,
        "near_empty": near_empty,
        "degenerate": degenerate,
    }
    return row, extras


def cross_character_templating(bags: list, max_pairs: int, seed: int) -> dict:
    pairs = [
        (i, j)
        for i in range(len(bags))
        for j in range(i + 1, len(bags))
        if bags[i][0] != bags[j][0]
    ]
    n_total = len(pairs)
    sampled = n_total > max_pairs
    if sampled:
        pairs = random.Random(seed).sample(pairs, max_pairs)
    vals = []
    for i, j in pairs:
        v = jaccard(bags[i][1], bags[j][1])
        if not math.isnan(v):
            vals.append(v)
    return {
        "mean_trigram_jaccard": float(np.mean(vals)) if vals else None,
        "n_pairs_scored": len(vals),
        "n_pairs_total": n_total,
        "n_conversations_with_traces": len(bags),
        "sampled": sampled,
        "seed": seed,
    }


def summarize_condition(condition: str, rows: list, extras: list, cfg: dict) -> dict:
    n_conv = len(rows)
    turns = sum(e["n_model_turns"] for e in extras)
    traces = sum(e["n_traces"] for e in extras)

    flag_counts = Counter()
    attr_fams = Counter()
    ph_counter = Counter()
    ana_counter = Counter()
    for e in extras:
        flag_counts.update(e["flag_counts"])
        attr_fams.update(e["attr_family_counts"])
        ph_counter.update(e["placeholder_counts"])
        ana_counter.update(e["anachronism_counts"])
    flag_rates = {f: _rate(flag_counts[f], turns) for f in FLAGS}

    is_trace_cond = condition in ("pre_thinking", "post_thinking")
    tag_compliance = {
        "n_model_turns": turns,
        "flag_counts": {f: flag_counts[f] for f in FLAGS},
        "flag_rates": flag_rates,
        "false_positive_rate": flag_rates["unexpected_trace"]
        if condition == "no_thinking"
        else None,
        "false_negative_rate": flag_rates["missing_trace"] if is_trace_cond else None,
        "ordering_violation_rate": flag_rates["misordered_trace"]
        if is_trace_cond
        else None,
    }

    language_bleed = {
        "cjk_dialogue_turn_rate": _rate(
            sum(e["cjk_dialogue_turns"] for e in extras), turns
        ),
        "cjk_trace_rate": _rate(sum(e["cjk_trace_turns"] for e in extras), traces),
        "n_traces": traces,
    }

    attr_families = cfg["PLAYER_ATTRIBUTION_PATTERNS"]
    player_attribution = {
        "trace_flag_rate": _rate(sum(e["attr_any"] for e in extras), traces),
        "family_counts": {fam: attr_fams[fam] for fam in attr_families},
        "family_rates": {fam: _rate(attr_fams[fam], traces) for fam in attr_families},
        "n_traces": traces,
    }

    placeholder = {
        "turn_rate": _rate(sum(e["placeholder_turns"] for e in extras), turns),
        "total_hits": sum(ph_counter.values()),
        "per_literal_counts": dict(ph_counter),
    }

    if cfg["ANACHRONISM_ENABLED"]:
        anachronism = {
            "enabled": True,
            "turn_rate": _rate(sum(e["anachronism_turns"] for e in extras), turns),
            "total_hits": sum(ana_counter.values()),
            "per_term_counts": dict(ana_counter),
        }
    else:
        anachronism = {"enabled": False}

    pooled_dlg = [w for e in extras for w in e["dialogue_word_counts"]]
    pooled_trc = [w for e in extras for w in e["trace_word_counts"]]
    pooled_ratio = [r for e in extras for r in e["trace_dialogue_ratios"]]
    pooled_tok = [t for e in extras for t in e["tokens_recorded"]]
    length_verbosity = {
        "dialogue_words_per_turn": dist_stats(pooled_dlg),
        "trace_words_per_turn": dist_stats(pooled_trc),
        "trace_to_dialogue_ratio_per_turn": dist_stats(pooled_ratio),
        "tokens_recorded_per_turn": dist_stats(pooled_tok),
    }

    lexical_diversity = {
        "distinct_1": dist_stats(r["distinct_1"] for r in rows),
        "distinct_2": dist_stats(r["distinct_2"] for r in rows),
        "distinct_3": dist_stats(r["distinct_3"] for r in rows),
        "self_repetition_consecutive_trigram_jaccard": dist_stats(
            r["self_repetition_trigram_jaccard"] for r in rows
        ),
    }

    bags = [(e["character_id"], e["trace_bag"]) for e in extras if e["trace_bag"]]
    trace_mode_collapse = {
        "within_conversation_pairwise_trigram_jaccard": dist_stats(
            r["trace_collapse_within_jaccard"] for r in rows
        ),
        "cross_character_templating": cross_character_templating(
            bags, cfg["CROSS_CHAR_PAIR_SAMPLES"], cfg["RNG_SEED"]
        ),
    }

    recorded = sum(e["finish_recorded"] for e in extras)
    generation_health = {
        "finish_stop_rate": _rate(sum(e["finish_stop"] for e in extras), recorded),
        "finish_length_rate": _rate(sum(e["finish_length"] for e in extras), recorded),
        "finish_recorded_rate": _rate(recorded, turns),
        "near_empty_dialogue_rate": _rate(sum(e["near_empty"] for e in extras), turns),
        "degenerate_loop_rate": _rate(sum(e["degenerate"] for e in extras), turns),
    }

    return {
        "n_conversations": n_conv,
        "n_model_turns": turns,
        "n_traces": traces,
        "rate_weighting": "pooled_over_turns",
        "tag_compliance": tag_compliance,
        "language_bleed": language_bleed,
        "player_attribution_proxy": player_attribution,
        "placeholder_leak": placeholder,
        "anachronism_scan": anachronism,
        "length_verbosity": length_verbosity,
        "lexical_diversity": lexical_diversity,
        "trace_mode_collapse": trace_mode_collapse,
        "generation_health": generation_health,
    }


def _dig(d: dict, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def flatten_scalars(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_scalars(v, key))
        elif v is None or isinstance(v, (bool, int, float, np.integer, np.floating)):
            out[key] = v
    return out


def main(**overrides) -> dict:
    cfg = {
        "TRANSCRIPT_PATHS": dict(TRANSCRIPT_PATHS),
        "PER_CONVERSATION_CSV": PER_CONVERSATION_CSV,
        "SUMMARY_JSON": SUMMARY_JSON,
        "SUMMARY_CSV": SUMMARY_CSV,
        "LOG_FILE": LOG_FILE,
        "CJK_PATTERN": CJK_PATTERN,
        "PLAYER_ATTRIBUTION_PATTERNS": dict(PLAYER_ATTRIBUTION_PATTERNS),
        "WORD_PATTERN": WORD_PATTERN,
        "ANACHRONISM_ENABLED": ANACHRONISM_ENABLED,
        "ANACHRONISM_TERMS": ANACHRONISM_TERMS,
        "PLACEHOLDER_LITERALS": PLACEHOLDER_LITERALS,
        "NEAR_EMPTY_WORDS": NEAR_EMPTY_WORDS,
        "DEGENERATE_NGRAM": DEGENERATE_NGRAM,
        "DEGENERATE_REPEATS": DEGENERATE_REPEATS,
        "DEGENERATE_SINGLE_WORD_FRAC": DEGENERATE_SINGLE_WORD_FRAC,
        "DEGENERATE_MIN_WORDS": DEGENERATE_MIN_WORDS,
        "CROSS_CHAR_PAIR_SAMPLES": CROSS_CHAR_PAIR_SAMPLES,
        "RNG_SEED": RNG_SEED,
    }
    unknown = sorted(set(overrides) - set(cfg))
    if unknown:
        raise TypeError(f"unknown config override(s): {unknown}; known: {sorted(cfg)}")
    cfg.update(overrides)

    log_file = Path(cfg["LOG_FILE"])
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )

    anachronism_re = (
        build_term_regex(cfg["ANACHRONISM_TERMS"])
        if cfg["ANACHRONISM_ENABLED"]
        else None
    )
    word_re = re.compile(cfg["WORD_PATTERN"])
    cjk_re = re.compile(cfg["CJK_PATTERN"])
    attr_res = {
        name: re.compile(pat, re.IGNORECASE)
        for name, pat in cfg["PLAYER_ATTRIBUTION_PATTERNS"].items()
    }

    all_rows = []
    by_condition = {}
    found_any = False
    for file_cond, path in cfg["TRANSCRIPT_PATHS"].items():
        path = Path(path)
        if not path.exists():
            logger.warning(
                f"transcript file missing, skipping condition {file_cond!r}: {path}"
            )
            continue
        found_any = True
        records = eval_lib.read_jsonl(path)
        logger.info(f"{file_cond}: {len(records)} conversations from {path}")
        for rec in tqdm(records, desc=file_cond):
            cond = rec.get("condition") or file_cond
            if cond != file_cond:
                logger.warning(
                    f"{rec.get('conversation_id', '?')}: record condition {cond!r} "
                    f"does not match file condition {file_cond!r}; using record's"
                )
            if cond not in CONDITIONS:
                logger.warning(
                    f"{rec.get('conversation_id', '?')}: unknown condition {cond!r}, skipped"
                )
                continue
            row, extras = conversation_metrics(
                rec,
                cond,
                literals=cfg["PLACEHOLDER_LITERALS"],
                anachronism_re=anachronism_re,
                near_empty_words=cfg["NEAR_EMPTY_WORDS"],
                word_re=word_re,
                cjk_re=cjk_re,
                attr_res=attr_res,
                degenerate_ngram=cfg["DEGENERATE_NGRAM"],
                degenerate_repeats=cfg["DEGENERATE_REPEATS"],
                degenerate_single_word_frac=cfg["DEGENERATE_SINGLE_WORD_FRAC"],
                degenerate_min_words=cfg["DEGENERATE_MIN_WORDS"],
            )
            all_rows.append(row)
            bucket = by_condition.setdefault(cond, {"rows": [], "extras": []})
            bucket["rows"].append(row)
            bucket["extras"].append(extras)

    if not found_any:
        raise FileNotFoundError(
            "no transcript files found; expected "
            + ", ".join(str(p) for p in cfg["TRANSCRIPT_PATHS"].values())
        )
    if not all_rows:
        raise ValueError("transcript files contained no scoreable conversations")

    csv_path = Path(cfg["PER_CONVERSATION_CSV"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows, columns=list(all_rows[0].keys()))
    df.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info(f"wrote {len(df)} per-conversation rows to {csv_path}")

    condition_summaries = {
        cond: summarize_condition(cond, b["rows"], b["extras"], cfg)
        for cond, b in by_condition.items()
    }

    headline = {
        "no_thinking_false_positive_rate": _dig(
            condition_summaries, "no_thinking", "tag_compliance", "false_positive_rate"
        ),
        "pre_thinking_false_negative_rate": _dig(
            condition_summaries, "pre_thinking", "tag_compliance", "false_negative_rate"
        ),
        "post_thinking_false_negative_rate": _dig(
            condition_summaries,
            "post_thinking",
            "tag_compliance",
            "false_negative_rate",
        ),
        "pre_thinking_ordering_violation_rate": _dig(
            condition_summaries,
            "pre_thinking",
            "tag_compliance",
            "ordering_violation_rate",
        ),
        "post_thinking_ordering_violation_rate": _dig(
            condition_summaries,
            "post_thinking",
            "tag_compliance",
            "ordering_violation_rate",
        ),
    }

    summary = {
        "script": "c_metrics_battery",
        "transcripts": {c: str(p) for c, p in cfg["TRANSCRIPT_PATHS"].items()},
        "config": {
            "anachronism_enabled": cfg["ANACHRONISM_ENABLED"],
            "anachronism_terms": list(cfg["ANACHRONISM_TERMS"]),
            "placeholder_literals": list(cfg["PLACEHOLDER_LITERALS"]),
            "player_attribution_patterns": dict(cfg["PLAYER_ATTRIBUTION_PATTERNS"]),
            "cjk_pattern": cfg["CJK_PATTERN"],
            "word_pattern": cfg["WORD_PATTERN"],
            "near_empty_words": cfg["NEAR_EMPTY_WORDS"],
            "degenerate_ngram": cfg["DEGENERATE_NGRAM"],
            "degenerate_repeats": cfg["DEGENERATE_REPEATS"],
            "degenerate_single_word_frac": cfg["DEGENERATE_SINGLE_WORD_FRAC"],
            "degenerate_min_words": cfg["DEGENERATE_MIN_WORDS"],
            "cross_char_pair_samples": cfg["CROSS_CHAR_PAIR_SAMPLES"],
            "rng_seed": cfg["RNG_SEED"],
        },
        "headline": headline,
        "conditions": condition_summaries,
    }

    json_path = Path(cfg["SUMMARY_JSON"])
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_jsonsafe(summary), f, ensure_ascii=False, indent=2)
    logger.info(
        f"wrote summary for {len(condition_summaries)} condition(s) to {json_path}"
    )

    flat_rows = [
        {"condition": cond, "metric": metric, "value": value}
        for cond in CONDITIONS
        if cond in condition_summaries
        for metric, value in flatten_scalars(
            _jsonsafe(condition_summaries[cond])
        ).items()
    ]
    summary_csv_path = Path(cfg["SUMMARY_CSV"])
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(flat_rows, columns=["condition", "metric", "value"]).to_csv(
        summary_csv_path, index=False, encoding="utf-8"
    )
    logger.info(f"wrote {len(flat_rows)} aggregate rows to {summary_csv_path}")

    return summary


if __name__ == "__main__":
    main()
