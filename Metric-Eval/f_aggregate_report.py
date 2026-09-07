import itertools
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import eval_lib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from eval_lib import read_jsonl, slug
from scipy import stats
from tqdm import tqdm

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parent
RUNS_DIR = EVAL_DIR / "runs"
LATENCY_DIR = RUNS_DIR / "latency"
RESULTS_DIR = EVAL_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
LOG_FILE = RESULTS_DIR / "aggregate_report.log"

CONDITIONS = eval_lib.CONDITIONS

LATENCY_RECORDS_TEMPLATE = "latency_records_{condition}.jsonl"
LATENCY_SUMMARY_JSON = LATENCY_DIR / "latency_summary.json"
BATTERY_PER_CONVERSATION_CSV = RESULTS_DIR / "metrics_per_conversation.csv"
BATTERY_SUMMARY_JSON = RESULTS_DIR / "metrics_summary.json"
STYLE_PER_CONVERSATION_CSV = RESULTS_DIR / "style_per_conversation.csv"
STYLE_SUMMARY_JSON = RESULTS_DIR / "style_summary.json"
NLI_PER_CONVERSATION_CSV = RESULTS_DIR / "nli_per_conversation.csv"
NLI_PERSONA_PER_CONVERSATION_CSV = RESULTS_DIR / "nli_persona_per_conversation.csv"
NLI_SUMMARY_JSON = RESULTS_DIR / "nli_summary.json"

HUMAN_SCORES_CSV = RESULTS_DIR / "human_scores.csv"

STYLE_MODEL_PRIMARY = "StyleDistance/styledistance"

LATENCY_METRICS = ("perceived_s", "ttft_s", "prefill_s", "decode_tps", "total_s")
EXCLUDE_THROTTLED = True

ALPHA = 0.05
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 17

SUMMARY_JSON = RESULTS_DIR / "summary.json"
COMPARISON_CSV = RESULTS_DIR / "comparison_table.csv"
COMPARISON_TEX = TABLES_DIR / "comparison_table.tex"
FIG_DPI = 200

_COND_LABELS = {
    "no_thinking": "No-thinking",
    "pre_thinking": "Pre-thinking",
    "post_thinking": "Post-thinking",
}

_RESULTS_CHILDREN = (
    "BATTERY_PER_CONVERSATION_CSV",
    "BATTERY_SUMMARY_JSON",
    "STYLE_PER_CONVERSATION_CSV",
    "STYLE_SUMMARY_JSON",
    "NLI_PER_CONVERSATION_CSV",
    "NLI_PERSONA_PER_CONVERSATION_CSV",
    "NLI_SUMMARY_JSON",
    "HUMAN_SCORES_CSV",
    "SUMMARY_JSON",
    "COMPARISON_CSV",
    "LOG_FILE",
)


def _build_config(overrides: dict) -> dict:
    cfg = {k: v for k, v in globals().items() if k.isupper() and not k.startswith("_")}
    unknown = sorted(set(overrides) - set(cfg))
    if unknown:
        raise ValueError(
            f"unknown config override(s): {unknown}; valid keys: {sorted(cfg)}"
        )
    cfg.update(overrides)
    changed = set(overrides)

    def _rebase(key, value):
        if key not in overrides:
            cfg[key] = value
            changed.add(key)

    if "EVAL_DIR" in changed:
        _rebase("RUNS_DIR", Path(cfg["EVAL_DIR"]) / "runs")
        _rebase("RESULTS_DIR", Path(cfg["EVAL_DIR"]) / "results")
    if "RUNS_DIR" in changed:
        _rebase("LATENCY_DIR", Path(cfg["RUNS_DIR"]) / "latency")
    if "LATENCY_DIR" in changed:
        _rebase(
            "LATENCY_SUMMARY_JSON",
            Path(cfg["LATENCY_DIR"]) / Path(globals()["LATENCY_SUMMARY_JSON"]).name,
        )
    if "RESULTS_DIR" in changed:
        results = Path(cfg["RESULTS_DIR"])
        for key in _RESULTS_CHILDREN:
            _rebase(key, results / Path(globals()[key]).name)
        _rebase("FIGURES_DIR", results / "figures")
        _rebase("TABLES_DIR", results / "tables")
    if "TABLES_DIR" in changed:
        _rebase(
            "COMPARISON_TEX",
            Path(cfg["TABLES_DIR"]) / Path(globals()["COMPARISON_TEX"]).name,
        )
    return cfg


def _setup_logging(log_path: Path) -> None:
    log_path = Path(log_path)
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


def _cond_from_conv(conversation_id) -> str | None:
    if isinstance(conversation_id, str):
        for cond in eval_lib.CONDITIONS:
            if conversation_id.startswith(cond + "__"):
                return cond
    return None


def _char_from_conv(conversation_id, condition) -> str | None:
    if (
        isinstance(conversation_id, str)
        and isinstance(condition, str)
        and conversation_id.startswith(condition + "__")
    ):
        return conversation_id[len(condition) + 2 :]
    return None


def _load_per_conversation_csv(path, label: str):
    path = Path(path)
    if not path.exists():
        logger.warning(f"{label}: input missing, skipped: {path}")
        return None
    df = pd.read_csv(path, encoding="utf-8")
    if "condition" not in df.columns and "conversation_id" in df.columns:
        df["condition"] = [_cond_from_conv(c) for c in df["conversation_id"]]
    if "character_id" not in df.columns and "conversation_id" in df.columns:
        df["character_id"] = [
            _char_from_conv(c, k)
            for c, k in zip(df["conversation_id"], df["condition"])
        ]
    if "condition" not in df.columns or "character_id" not in df.columns:
        logger.warning(
            f"{label}: {path} lacks condition/character_id columns "
            "(and no conversation_id to derive them from); skipped"
        )
        return None
    return df


def collect_latency_metrics(
    latency_dir, template: str, conditions, metric_names, exclude_throttled: bool
):
    latency_dir = Path(latency_dir)
    data = {f"latency_{m}": {} for m in metric_names}
    perceived_raw = {}
    throttled_counts = {}
    found_any = False
    for cond in conditions:
        path = latency_dir / template.format(condition=cond)
        if not path.exists():
            logger.warning(f"Script B latency records: input missing, skipped: {path}")
            continue
        found_any = True
        records = read_jsonl(path)
        n_throttled = 0
        per_conv: dict[str, dict[str, list[float]]] = {}
        for rec in records:
            if not isinstance(rec, dict):
                continue
            if exclude_throttled and rec.get("throttled"):
                n_throttled += 1
                continue
            char = rec.get("character_id") or _char_from_conv(
                rec.get("conversation_id"), cond
            )
            if not char:
                continue
            rec_metrics = (
                rec.get("metrics") if isinstance(rec.get("metrics"), dict) else {}
            )
            bucket = per_conv.setdefault(str(char), {m: [] for m in metric_names})
            for m in metric_names:
                v = rec_metrics.get(m, rec.get(m))
                if (
                    isinstance(v, (int, float))
                    and not isinstance(v, bool)
                    and math.isfinite(v)
                ):
                    bucket[m].append(float(v))
                    if m == "perceived_s":
                        perceived_raw.setdefault(cond, []).append(float(v))
        for char, buckets in per_conv.items():
            for m, vals in buckets.items():
                if vals:
                    data[f"latency_{m}"].setdefault(cond, {})[char] = float(
                        np.median(vals)
                    )
        throttled_counts[cond] = n_throttled
        logger.info(
            f"Script B latency [{cond}]: {len(records)} records, "
            f"{len(per_conv)} conversations, {n_throttled} throttled records excluded"
        )
    if not found_any:
        return {}, {}, {}, {}
    data = {k: v for k, v in data.items() if v}
    defs = {
        f"latency_{m}": (
            f"Script B replay: per-conversation median of per-(turn x rep) '{m}'"
            + (" (throttled records excluded)" if exclude_throttled else "")
        )
        for m in metric_names
        if f"latency_{m}" in data
    }
    return data, defs, perceived_raw, throttled_counts


def collect_battery_metrics(path):
    df = _load_per_conversation_csv(path, "Script C metrics battery")
    if df is None:
        return {}, {}
    id_cols = {
        "condition",
        "character_id",
        "conversation_id",
        "source",
        "name",
        "model",
    }
    data, defs = {}, {}
    for col in df.columns:
        if col in id_cols:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if not series.notna().any():
            continue
        sub = pd.DataFrame(
            {
                "condition": df["condition"],
                "character_id": df["character_id"],
                "value": series,
            }
        ).dropna(subset=["condition", "character_id", "value"])
        per_cond = {}
        for (cond, char), grp in sub.groupby(["condition", "character_id"]):
            per_cond.setdefault(str(cond), {})[str(char)] = float(grp["value"].mean())
        if per_cond:
            name = f"battery_{col}"
            data[name] = per_cond
            defs[name] = f"Script C battery column '{col}' (per-conversation value)"
    logger.info(
        f"Script C metrics battery: {len(data)} metric columns collected from {path}"
    )
    return data, defs


def collect_style_metrics(path):
    df = _load_per_conversation_csv(path, "Script D style consistency")
    if df is None:
        return {}, {}
    if "pairwise_mean_cosine" not in df.columns:
        logger.warning(f"Script D style: {path} lacks 'pairwise_mean_cosine'; skipped")
        return {}, {}
    if "model" not in df.columns:
        df = df.assign(model="style_model")
    sub = pd.DataFrame(
        {
            "model": df["model"],
            "condition": df["condition"],
            "character_id": df["character_id"],
            "value": pd.to_numeric(df["pairwise_mean_cosine"], errors="coerce"),
        }
    ).dropna(subset=["condition", "character_id", "value"])
    data, defs = {}, {}
    for (model, cond, char), grp in sub.groupby(["model", "condition", "character_id"]):
        name = f"style_pairwise_cosine_proxy__{slug(str(model))}"
        data.setdefault(name, {}).setdefault(str(cond), {})[str(char)] = float(
            grp["value"].mean()
        )
        defs[name] = f"intra-character pairwise-mean style cosine, model '{model}'"
    logger.info(f"Script D style: {len(data)} style metric(s) collected from {path}")
    return data, defs


def _nli_rate_metrics(df, path, persona: bool):
    cols = [c for c in df.columns if c.endswith("contradiction_rate")]
    label = "Script E NLI persona-anchored" if persona else "Script E NLI"
    if not cols:
        logger.warning(f"{label}: {path} has no *contradiction_rate column; skipped")
        return {}, {}
    prefix = "nli_persona" if persona else "nli"
    scope = (
        "claims vs the character card's Background (persona-anchored)"
        if persona
        else "within-conversation claim pairs"
    )
    data, defs = {}, {}
    for col in cols:
        sub = pd.DataFrame(
            {
                "condition": df["condition"],
                "character_id": df["character_id"],
                "value": pd.to_numeric(df[col], errors="coerce"),
            }
        ).dropna(subset=["condition", "character_id", "value"])
        per_cond = {}
        for (cond, char), grp in sub.groupby(["condition", "character_id"]):
            per_cond.setdefault(str(cond), {})[str(char)] = float(grp["value"].mean())
        if per_cond:
            name = (
                f"{prefix}_contradiction_rate_proxy"
                if col == "contradiction_rate"
                else f"{prefix}_{col}_proxy"
            )
            data[name] = per_cond
            defs[name] = f"NLI '{col}' over {scope}"
    logger.info(f"{label}: {len(data)} metric(s) collected from {path}")
    return data, defs


def collect_nli_metrics(path, persona_path):
    data, defs = {}, {}
    df = _load_per_conversation_csv(path, "Script E NLI contradiction")
    if df is not None:
        d, dd = _nli_rate_metrics(df, path, persona=False)
        data.update(d)
        defs.update(dd)
    persona_path = Path(persona_path)
    if persona_path.exists():
        pdf = _load_per_conversation_csv(persona_path, "Script E NLI persona-anchored")
        if pdf is not None:
            d, dd = _nli_rate_metrics(pdf, persona_path, persona=True)
            data.update(d)
            defs.update(dd)
    else:
        logger.info(
            f"Script E persona-anchored NLI not present (optional; "
            f"PERSONA_ANCHORED is off by default): {persona_path}"
        )
    return data, defs


def _load_json_passthrough(path, label: str):
    path = Path(path)
    if not path.exists():
        logger.warning(f"{label}: summary not found, skipped: {path}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"{label}: could not read {path}: {e}")
        return None


def holm_bonferroni(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def rank_biserial_paired(diffs) -> float:
    diffs = np.asarray(diffs, dtype=float)
    nz = diffs[diffs != 0]
    if nz.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nz))
    r_pos = float(ranks[nz > 0].sum())
    r_neg = float(ranks[nz < 0].sum())
    total = r_pos + r_neg
    return (r_pos - r_neg) / total if total else 0.0


def aggregate_values(values, n_boot: int, seed: int, alpha: float) -> dict:
    arr = np.asarray(list(values), dtype=float)
    n = int(arr.size)
    if n == 0:
        return {"n": 0}
    agg = {
        "n": n,
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=1)) if n > 1 else 0.0,
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
    }
    if n > 1:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(int(n_boot), n))
        means = arr[idx].mean(axis=1)
        agg["ci95_mean"] = [
            float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))),
        ]
    else:
        agg["ci95_mean"] = [float(arr[0]), float(arr[0])]
    return agg


def compare_metric(metric: str, per_cond: dict, conditions) -> list[dict]:
    tests = []
    available = [c for c in conditions if per_cond.get(c)]
    for a, b in itertools.combinations(available, 2):
        da, db = per_cond[a], per_cond[b]
        entry = {
            "metric": metric,
            "pair": [a, b],
            "test": None,
            "n_pairs": None,
            "n_a": len(da),
            "n_b": len(db),
            "statistic": None,
            "p_value": None,
            "p_holm": None,
            "rank_biserial": None,
            "missing_characters": None,
            "note": None,
        }
        if set(da) == set(db):
            chars = sorted(da)
            x = np.array([da[c] for c in chars], dtype=float)
            y = np.array([db[c] for c in chars], dtype=float)
            entry["test"] = "wilcoxon_signed_rank"
            entry["n_pairs"] = len(chars)
            diffs = x - y
            if len(chars) < 2:
                entry["note"] = "too few pairs for a test"
            elif np.all(diffs == 0):
                entry.update(
                    statistic=0.0,
                    p_value=1.0,
                    rank_biserial=0.0,
                    note="all paired differences zero",
                )
            else:
                try:
                    res = stats.wilcoxon(x, y)
                    entry["statistic"] = float(res.statistic)
                    entry["p_value"] = float(res.pvalue)
                except ValueError as e:
                    entry["note"] = f"wilcoxon failed: {e}"
                entry["rank_biserial"] = rank_biserial_paired(diffs)
        else:
            missing_from_a = sorted(set(db) - set(da))
            missing_from_b = sorted(set(da) - set(db))
            logger.warning(
                f"{metric}: character sets differ for {a} vs {b}; unpaired Mann-Whitney U "
                f"fallback. missing from {a}: {missing_from_a}; missing from {b}: {missing_from_b}"
            )
            entry["test"] = "mann_whitney_u"
            entry["missing_characters"] = {a: missing_from_a, b: missing_from_b}
            x = np.fromiter(da.values(), dtype=float)
            y = np.fromiter(db.values(), dtype=float)
            try:
                res = stats.mannwhitneyu(x, y, alternative="two-sided")
                entry["statistic"] = float(res.statistic)
                entry["p_value"] = float(res.pvalue)
                entry["rank_biserial"] = float(
                    2.0 * res.statistic / (len(x) * len(y)) - 1.0
                )
            except ValueError as e:
                entry["note"] = f"mannwhitneyu failed: {e}"
        tests.append(entry)
    valid = [t for t in tests if t["p_value"] is not None]
    if valid:
        for t, p in zip(valid, holm_bonferroni([t["p_value"] for t in valid])):
            t["p_holm"] = p
    return tests


def proxy_validation(human_csv, metric_data: dict, style_metric_name: str):
    path = Path(human_csv)
    if not path.exists():
        msg = (
            f"NOTICE: human scores not found at {path}; proxy validation skipped "
            "(Spearman rho vs style/NLI proxies)"
        )
        logger.info(msg)
        print(msg)
        return None
    df = pd.read_csv(path, encoding="utf-8")
    required = {"character_id", "condition", "human_consistency"}
    if not required.issubset(df.columns):
        logger.warning(
            f"human scores {path} missing columns {sorted(required - set(df.columns))}; "
            "proxy validation skipped"
        )
        return None
    probes = (
        ("style_proxy", style_metric_name, 1.0),
        ("nli_contradiction_proxy_sign_flipped", "nli_contradiction_rate_proxy", -1.0),
    )
    out = {}
    for label, metric, sign in probes:
        per_cond = metric_data.get(metric)
        if not per_cond:
            logger.warning(
                f"proxy validation: metric '{metric}' unavailable; {label} skipped"
            )
            continue
        human, proxy = [], []
        for _, row in df.iterrows():
            v = per_cond.get(str(row["condition"]), {}).get(str(row["character_id"]))
            try:
                h = float(row["human_consistency"])
            except (TypeError, ValueError):
                continue
            if v is None or not math.isfinite(h):
                continue
            human.append(h)
            proxy.append(sign * float(v))
        entry = {
            "metric": metric,
            "sign_flipped": sign < 0,
            "n": len(human),
            "spearman_rho": None,
            "p_value": None,
        }
        if len(human) >= 3 and len(set(human)) > 1 and len(set(proxy)) > 1:
            rho, p = stats.spearmanr(human, proxy)
            entry["spearman_rho"] = float(rho)
            entry["p_value"] = float(p)
        else:
            entry["note"] = "too few matched rows or constant values"
        out[label] = entry
        logger.info(
            f"proxy validation {label}: rho={entry['spearman_rho']}, "
            f"p={entry['p_value']}, n={entry['n']}"
        )
    return out or None


_LATEX_SPECIALS = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
}


def latex_escape(text) -> str:
    return "".join(_LATEX_SPECIALS.get(ch, ch) for ch in str(text))


def build_comparison_rows(
    metric_order, aggregates: dict, definitions: dict, conditions
) -> list[dict]:
    rows = []
    for metric in metric_order:
        for cond in conditions:
            agg = aggregates.get(metric, {}).get(cond)
            if not agg or agg.get("n", 0) == 0:
                continue
            rows.append(
                {
                    "metric": metric,
                    "condition": cond,
                    "n": agg["n"],
                    "median": agg["median"],
                    "mean": agg["mean"],
                    "std": agg["std"],
                    "q25": agg["q25"],
                    "q75": agg["q75"],
                    "ci95_mean_low": agg["ci95_mean"][0],
                    "ci95_mean_high": agg["ci95_mean"][1],
                    "definition": definitions.get(metric, ""),
                }
            )
    return rows


def build_latex_table(metric_order, aggregates: dict, conditions) -> str:
    conds = [
        c for c in conditions if any(c in aggregates.get(m, {}) for m in metric_order)
    ]
    lines = [
        "% Auto-generated by f_aggregate_report.py - do not edit by hand.",
        "% Cells: median [q25, q75] over per-conversation values (paired by character).",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Per-condition comparison: median [IQR] of per-conversation values.}",
        r"\label{tab:condition_comparison}",
        r"\begin{tabular}{l" + "c" * max(1, len(conds)) + "}",
        r"\toprule",
        "Metric & " + " & ".join(_COND_LABELS.get(c, c) for c in conds) + r" \\",
        r"\midrule",
    ]
    for metric in metric_order:
        cells = []
        for cond in conds:
            agg = aggregates.get(metric, {}).get(cond)
            if agg and agg.get("n", 0) > 0:
                cells.append(
                    f"{agg['median']:.3g} [{agg['q25']:.3g}, {agg['q75']:.3g}]"
                )
            else:
                cells.append("--")
        lines.append(latex_escape(metric) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def _save_fig(fig, figures_dir: Path, name: str, dpi: int) -> None:
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(figures_dir / f"{name}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def fig_box_violin(
    metric: str, per_cond: dict, conditions, figures_dir, dpi: int
) -> None:
    conds = [c for c in conditions if per_cond.get(c)]
    if not conds:
        return
    data = [np.asarray(list(per_cond[c].values()), dtype=float) for c in conds]
    fig, ax = plt.subplots(figsize=(6, 4))
    pos = np.arange(1, len(conds) + 1)
    for p, d in zip(pos, data):
        if d.size >= 2 and np.ptp(d) > 0:
            try:
                vp = ax.violinplot([d], positions=[p], showextrema=False, widths=0.7)
                for body in vp["bodies"]:
                    body.set_alpha(0.3)
            except Exception:
                pass
    ax.boxplot(data, positions=pos, widths=0.25)
    ax.set_xticks(pos)
    ax.set_xticklabels([_COND_LABELS.get(c, c) for c in conds])
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} (per-conversation values)")
    _save_fig(fig, figures_dir, f"box_{slug(metric)}", dpi)


def fig_latency_cdf(perceived_raw: dict, conditions, figures_dir, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    for cond in conditions:
        vals = sorted(perceived_raw.get(cond, []))
        if not vals:
            continue
        ys = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(vals, ys, where="post", label=_COND_LABELS.get(cond, cond))
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Player-perceived latency (s)")
    ax.set_ylabel("Cumulative fraction of turns")
    ax.set_title("Player-perceived latency CDF (per turn x rep record)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(fig, figures_dir, "latency_perceived_cdf", dpi)


def fig_latency_median_bars(per_cond: dict, conditions, figures_dir, dpi: int) -> None:
    conds = [c for c in conditions if per_cond.get(c)]
    if not conds:
        return
    medians, err_lo, err_hi = [], [], []
    for c in conds:
        arr = np.asarray(list(per_cond[c].values()), dtype=float)
        med = float(np.median(arr))
        q25, q75 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
        medians.append(med)
        err_lo.append(max(0.0, med - q25))
        err_hi.append(max(0.0, q75 - med))
    fig, ax = plt.subplots(figsize=(6, 4))
    pos = np.arange(len(conds))
    ax.bar(pos, medians, yerr=[err_lo, err_hi], capsize=5, alpha=0.8)
    ax.set_xticks(pos)
    ax.set_xticklabels([_COND_LABELS.get(c, c) for c in conds])
    ax.set_ylabel("Player-perceived latency (s)")
    ax.set_title("Median perceived latency (IQR whiskers, per-conversation medians)")
    _save_fig(fig, figures_dir, "latency_perceived_median_bars", dpi)


def fig_distribution(
    metric: str, per_cond: dict, conditions, figures_dir, dpi: int
) -> None:
    conds = [c for c in conditions if per_cond.get(c)]
    if not conds:
        return
    pooled = np.concatenate(
        [np.asarray(list(per_cond[c].values()), dtype=float) for c in conds]
    )
    if pooled.size == 0:
        return
    bins = np.histogram_bin_edges(pooled, bins=15) if np.ptp(pooled) > 0 else 10
    fig, ax = plt.subplots(figsize=(6, 4))
    for cond in conds:
        vals = np.asarray(list(per_cond[cond].values()), dtype=float)
        ax.hist(vals, bins=bins, alpha=0.5, label=_COND_LABELS.get(cond, cond))
    ax.set_xlabel(metric)
    ax.set_ylabel("Conversations")
    ax.set_title(f"{metric} distribution (proxy)")
    ax.legend()
    _save_fig(fig, figures_dir, f"dist_{slug(metric)}", dpi)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    return obj


def main(**overrides):
    cfg = _build_config(overrides)
    _setup_logging(Path(cfg["LOG_FILE"]))
    conditions = tuple(cfg["CONDITIONS"])
    results_dir = Path(cfg["RESULTS_DIR"])
    results_dir.mkdir(parents=True, exist_ok=True)

    metric_data: dict[str, dict] = {}
    definitions: dict[str, str] = {}
    inputs_found: dict[str, dict] = {}

    def _note_input(label, path):
        inputs_found[label] = {"path": str(path), "found": Path(path).exists()}

    for cond in conditions:
        _note_input(
            f"latency_records_{cond}",
            Path(cfg["LATENCY_DIR"])
            / cfg["LATENCY_RECORDS_TEMPLATE"].format(condition=cond),
        )
    lat_data, lat_defs, perceived_raw, throttled_counts = collect_latency_metrics(
        cfg["LATENCY_DIR"],
        cfg["LATENCY_RECORDS_TEMPLATE"],
        conditions,
        cfg["LATENCY_METRICS"],
        cfg["EXCLUDE_THROTTLED"],
    )
    metric_data.update(lat_data)
    definitions.update(lat_defs)

    _note_input("battery_per_conversation", cfg["BATTERY_PER_CONVERSATION_CSV"])
    bat_data, bat_defs = collect_battery_metrics(cfg["BATTERY_PER_CONVERSATION_CSV"])
    metric_data.update(bat_data)
    definitions.update(bat_defs)

    _note_input("style_per_conversation", cfg["STYLE_PER_CONVERSATION_CSV"])
    sty_data, sty_defs = collect_style_metrics(cfg["STYLE_PER_CONVERSATION_CSV"])
    metric_data.update(sty_data)
    definitions.update(sty_defs)

    _note_input("nli_per_conversation", cfg["NLI_PER_CONVERSATION_CSV"])
    _note_input("nli_persona_per_conversation", cfg["NLI_PERSONA_PER_CONVERSATION_CSV"])
    nli_data, nli_defs = collect_nli_metrics(
        cfg["NLI_PER_CONVERSATION_CSV"], cfg["NLI_PERSONA_PER_CONVERSATION_CSV"]
    )
    metric_data.update(nli_data)
    definitions.update(nli_defs)

    if not metric_data:
        logger.warning("no inputs found at all; writing empty outputs")

    aggregates: dict[str, dict] = {}
    tests_by_metric: dict[str, list] = {}
    for metric, per_cond in metric_data.items():
        aggregates[metric] = {
            cond: aggregate_values(
                per_cond[cond].values(),
                cfg["BOOTSTRAP_N"],
                cfg["BOOTSTRAP_SEED"],
                cfg["ALPHA"],
            )
            for cond in conditions
            if per_cond.get(cond)
        }
        tests_by_metric[metric] = compare_metric(metric, per_cond, conditions)

    passthrough = {
        "latency": _load_json_passthrough(
            cfg["LATENCY_SUMMARY_JSON"], "Script B latency summary"
        ),
        "battery": _load_json_passthrough(
            cfg["BATTERY_SUMMARY_JSON"], "Script C battery summary"
        ),
        "style": _load_json_passthrough(
            cfg["STYLE_SUMMARY_JSON"], "Script D style summary"
        ),
        "nli": _load_json_passthrough(cfg["NLI_SUMMARY_JSON"], "Script E NLI summary"),
    }

    style_primary_metric = (
        f"style_pairwise_cosine_proxy__{slug(cfg['STYLE_MODEL_PRIMARY'])}"
    )
    _note_input("human_scores", cfg["HUMAN_SCORES_CSV"])
    proxy = proxy_validation(cfg["HUMAN_SCORES_CSV"], metric_data, style_primary_metric)

    metric_order = list(metric_data)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "conditions": list(conditions),
        "alpha": cfg["ALPHA"],
        "bootstrap": {"n_resamples": cfg["BOOTSTRAP_N"], "seed": cfg["BOOTSTRAP_SEED"]},
        "inputs_found": inputs_found,
        "latency_throttled_excluded": throttled_counts,
        "metrics": {
            metric: {
                "definition": definitions.get(metric, ""),
                "per_condition": aggregates.get(metric, {}),
                "tests": tests_by_metric.get(metric, []),
            }
            for metric in metric_order
        },
        "proxy_validation": proxy,
        "passthrough_summaries": passthrough,
    }
    summary_path = Path(cfg["SUMMARY_JSON"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(summary), f, ensure_ascii=False, indent=2)
    logger.info(f"wrote {summary_path}")

    csv_columns = [
        "metric",
        "condition",
        "n",
        "median",
        "mean",
        "std",
        "q25",
        "q75",
        "ci95_mean_low",
        "ci95_mean_high",
        "definition",
    ]
    rows = build_comparison_rows(metric_order, aggregates, definitions, conditions)
    csv_path = Path(cfg["COMPARISON_CSV"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=csv_columns).to_csv(
        csv_path, index=False, encoding="utf-8"
    )
    logger.info(f"wrote {csv_path} ({len(rows)} rows)")

    tex_path = Path(cfg["COMPARISON_TEX"])
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(build_latex_table(metric_order, aggregates, conditions))
    logger.info(f"wrote {tex_path}")

    figures_dir = Path(cfg["FIGURES_DIR"])
    for metric in tqdm(metric_order, desc="figures"):
        fig_box_violin(
            metric, metric_data[metric], conditions, figures_dir, cfg["FIG_DPI"]
        )
    if perceived_raw:
        fig_latency_cdf(perceived_raw, conditions, figures_dir, cfg["FIG_DPI"])
    if "latency_perceived_s" in metric_data:
        fig_latency_median_bars(
            metric_data["latency_perceived_s"], conditions, figures_dir, cfg["FIG_DPI"]
        )
    for metric in metric_order:
        if metric.startswith("style_") or metric.startswith("nli_"):
            fig_distribution(
                metric, metric_data[metric], conditions, figures_dir, cfg["FIG_DPI"]
            )

    logger.info(
        f"done: {len(metric_order)} metrics, "
        f"{sum(len(t) for t in tests_by_metric.values())} pairwise tests"
    )
    return summary


if __name__ == "__main__":
    main()
