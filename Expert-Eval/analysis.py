import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

HERE = Path(__file__).resolve().parent
CONDS = ("no_thinking", "pre_thinking", "post_thinking")
SHORT = {"no_thinking": "no", "pre_thinking": "pre", "post_thinking": "post"}
LABELS = ("A", "B", "C")
# AVAYA's form omitted Version B as BEST/WORST; recovered from notes (0-based char index).
AVAYA_RECOVERED = {
    2: {"worst": "B"},
    3: {"best": "B"},
    4: {"best": "B"},
    8: {"best": "B"},
    9: {"worst": "B"},
}


def load_key() -> list[dict]:
    raw = json.loads((HERE / "eval_form_key.json").read_text(encoding="utf-8"))
    chars = []
    for char_id, info in raw.items():
        mapping = info["mapping"]
        chars.append(
            {
                "id": char_id,
                "name": info["display_name"],
                "mapping": mapping,
                "inv": {cond: lbl for lbl, cond in mapping.items()},
            }
        )
    return chars


def _parse_score(value) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        m = re.search(r"[1-5](?:\.\d+)?", text)
        return float(m.group(0)) if m else None


def _parse_choice(value) -> tuple[str, set[str]]:
    text = str(value or "").strip()
    if not text:
        return "empty", set()
    lowered = text.lower().replace("vesion", "version")
    if "all equal" in lowered:
        return "all", set(LABELS)
    tie = re.search(
        r"tie\s*\(\s*([abc])\s*=\s*([abc])(?:\s*=\s*([abc]))?\s*\)", lowered
    )
    if tie:
        return "tie", {g.upper() for g in tie.groups() if g}
    found = {
        lab for lab in LABELS if re.search(rf"\bversion\s*{lab.lower()}\b", lowered)
    }
    if not found:
        m = re.fullmatch(r"\s*([abc])\s*", lowered)
        if m:
            found.add(m.group(1).upper())
    if len(found) == 1:
        return "single", found
    if len(found) >= 2:
        return "tie", found
    return "unknown", set()


def _find_col(header: list[str], *needles: str) -> int | None:
    lowered = [h.strip().lower() for h in header]
    for i, h in enumerate(lowered):
        if all(n.lower() in h for n in needles):
            return i
    return None


def parse_form(path: Path, cohort: str, chars: list[dict]) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if any(c.strip() for c in r)]
    eval_idx = _find_col(header, "evaluator id")
    if eval_idx is None:
        raise ValueError(f"{path}: no Evaluator ID column")
    first_a = next(i for i, h in enumerate(header) if h.strip() == "Version A")
    block = 9
    evaluators = []
    for row in rows:
        eval_id = (row[eval_idx] if eval_idx < len(row) else "").strip()
        if not eval_id:
            continue
        cells = []
        for ci, char in enumerate(chars):
            base = first_a + ci * block
            cons = {lab: _parse_score(row[base + j]) for j, lab in enumerate(LABELS)}
            nat = {lab: _parse_score(row[base + 3 + j]) for j, lab in enumerate(LABELS)}
            best_kind, best_labs = _parse_choice(
                row[base + 6] if base + 6 < len(row) else ""
            )
            worst_kind, worst_labs = _parse_choice(
                row[base + 7] if base + 7 < len(row) else ""
            )
            if eval_id == "AVAYA" and ci in AVAYA_RECOVERED:
                recov = AVAYA_RECOVERED[ci]
                if "best" in recov:
                    best_kind, best_labs = "single", {recov["best"]}
                if "worst" in recov:
                    worst_kind, worst_labs = "single", {recov["worst"]}
            by_cond = {}
            for cond in CONDS:
                lab = char["inv"][cond]
                c, n = cons[lab], nat[lab]
                by_cond[cond] = {
                    "consistency": c,
                    "naturalness": n,
                    "combined": None if c is None or n is None else (c + n) / 2.0,
                }
            cells.append(
                {
                    "char_id": char["id"],
                    "mapping": char["mapping"],
                    "by_cond": by_cond,
                    "best_kind": best_kind,
                    "best_labs": best_labs,
                    "worst_kind": worst_kind,
                    "worst_labs": worst_labs,
                }
            )
        evaluators.append({"evaluator": eval_id, "cohort": cohort, "cells": cells})
    return evaluators


def _avg(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def evaluator_means(ev: dict, metric: str = "combined") -> dict[str, float]:
    out = {}
    for cond in CONDS:
        vals = [
            cell["by_cond"][cond][metric]
            for cell in ev["cells"]
            if cell["by_cond"][cond][metric] is not None
        ]
        out[cond] = _avg(vals)
    return out


def pooled(evaluators: list[dict], metric: str) -> dict[str, list[float]]:
    buckets = {c: [] for c in CONDS}
    for ev in evaluators:
        for cell in ev["cells"]:
            for cond in CONDS:
                v = cell["by_cond"][cond][metric]
                if v is not None:
                    buckets[cond].append(v)
    return buckets


def _top(comb: dict[str, float]) -> list[str]:
    peak = max(comb.values())
    return [c for c in CONDS if abs(comb[c] - peak) < 1e-12]


def holm(pairs: list[tuple[str, float]]) -> dict[str, tuple[float, float, bool]]:
    m, stop, out = len(pairs), False, {}
    for i, (name, p) in enumerate(sorted(pairs, key=lambda x: x[1])):
        thr = 0.05 / (m - i)
        sig = (not stop) and (p <= thr)
        if not sig:
            stop = True
        out[name] = (p, thr, sig)
    return out


def pairwise_wilcoxon(evaluators: list[dict]):
    means = {ev["evaluator"]: evaluator_means(ev) for ev in evaluators}
    names = [ev["evaluator"] for ev in evaluators]
    results = []
    for a, b, label in (
        ("post_thinking", "no_thinking", "post vs no"),
        ("pre_thinking", "no_thinking", "pre vs no"),
        ("post_thinking", "pre_thinking", "post vs pre"),
    ):
        xa = np.array([means[n][a] for n in names])
        xb = np.array([means[n][b] for n in names])
        try:
            stat = wilcoxon(xa, xb, zero_method="wilcox", alternative="two-sided")
            p, w = float(stat.pvalue), float(stat.statistic)
        except ValueError:
            p, w = float("nan"), float("nan")
        results.append((label, float(np.mean(xa - xb)), w, p))
    return results, holm([(r[0], r[3]) for r in results])


def friedman(evaluators: list[dict]):
    means = [evaluator_means(ev) for ev in evaluators]
    arrays = [np.array([m[c] for m in means]) for c in CONDS]
    stat = friedmanchisquare(*arrays)
    return float(stat.statistic), float(stat.pvalue)


def mean_ranks(evaluators: list[dict]) -> dict[str, float]:
    acc = {c: [] for c in CONDS}
    for ev in evaluators:
        for cell in ev["cells"]:
            vals = [cell["by_cond"][c]["combined"] for c in CONDS]
            if any(v is None for v in vals):
                continue
            for cond, r in zip(CONDS, rankdata([-v for v in vals], method="average")):
                acc[cond].append(float(r))
    return {c: _avg(acc[c]) for c in CONDS}


def best_worst(evaluators: list[dict]) -> dict:
    best, worst = {c: 0 for c in CONDS}, {c: 0 for c in CONDS}
    n_best = n_worst = 0
    for ev in evaluators:
        for cell in ev["cells"]:
            if cell["best_kind"] == "single" and len(cell["best_labs"]) == 1:
                best[cell["mapping"][next(iter(cell["best_labs"]))]] += 1
                n_best += 1
            if cell["worst_kind"] == "single" and len(cell["worst_labs"]) == 1:
                worst[cell["mapping"][next(iter(cell["worst_labs"]))]] += 1
                n_worst += 1
    return {"best": best, "worst": worst, "n_best": n_best, "n_worst": n_worst}


def _char_metric(
    evaluators: list[dict], char_id: str, cond: str, metric: str
) -> list[float]:
    vals = []
    for ev in evaluators:
        cell = next(c for c in ev["cells"] if c["char_id"] == char_id)
        v = cell["by_cond"][cond][metric]
        if v is not None:
            vals.append(v)
    return vals


def _krippendorff_alpha(R: np.ndarray) -> float:
    values = sorted({float(v) for v in R.ravel() if not np.isnan(v)})
    if len(values) < 2:
        return 1.0
    idx, m = {v: i for i, v in enumerate(values)}, len(values)
    O = np.zeros((m, m))
    for unit in R.T:
        observed = [float(v) for v in unit if not np.isnan(v)]
        n_u = len(observed)
        if n_u < 2:
            continue
        counts: dict[float, int] = defaultdict(int)
        for a in observed:
            counts[a] += 1
        denom = n_u - 1
        for a, ma in counts.items():
            O[idx[a], idx[a]] += ma * (ma - 1) / denom
            for b, mb in counts.items():
                if a != b:
                    O[idx[a], idx[b]] += ma * mb / denom
    n = O.sum()
    if n == 0:
        return float("nan")
    n_c = O.sum(axis=1)
    delta = np.zeros((m, m))
    for c in range(m):
        for k in range(m):
            if c == k:
                continue
            lo, hi = (c, k) if c < k else (k, c)
            s = n_c[lo : hi + 1].sum() - (n_c[lo] + n_c[hi]) / 2.0
            delta[c, k] = s**2
    Do = (O * delta).sum() / n
    De = sum(
        n_c[c] * (n_c[k] - (1 if c == k else 0)) * delta[c, k]
        for c in range(m)
        for k in range(m)
    ) / (n * (n - 1))
    return 1.0 if De == 0 else 1.0 - Do / De


def krippendorff_ordinal(evaluators: list[dict], metric: str) -> float:
    units = [(i, cond) for i in range(len(evaluators[0]["cells"])) for cond in CONDS]
    R = np.full((len(evaluators), len(units)), np.nan)
    for i, ev in enumerate(evaluators):
        for j, (ci, cond) in enumerate(units):
            v = ev["cells"][ci]["by_cond"][cond][metric]
            if v is not None:
                R[i, j] = v
    return _krippendorff_alpha(R)


def _fmt3(buckets: dict[str, list[float]]) -> str:
    return " ".join(f"{np.mean(buckets[c]):7.2f}" for c in CONDS)


def report(evaluators: list[dict], chars: list[dict]) -> None:
    n = len(evaluators)
    print(f"n = {n} evaluators, {n * len(chars)} cells/cond")

    print("\nCondition means (pooled cells)")
    print(f"{'metric':<14} {'no':>7} {'pre':>7} {'post':>7}")
    for label, metric in (
        ("Consistency", "consistency"),
        ("Naturalness", "naturalness"),
        ("Combined", "combined"),
    ):
        print(f"{label:<14} {_fmt3(pooled(evaluators, metric))}")

    print("\nCohort split (combined)")
    print(f"{'cohort':<10} {'n':>3} {'no':>6} {'pre':>6} {'post':>6}")
    by: dict[str, list] = defaultdict(list)
    for ev in evaluators:
        by[ev["cohort"]].append(ev)
    for coh, evs in by.items():
        pc = pooled(evs, "combined")
        print(
            f"{coh:<10} {len(evs):3d} {np.mean(pc['no_thinking']):6.2f} {np.mean(pc['pre_thinking']):6.2f} {np.mean(pc['post_thinking']):6.2f}"
        )

    print("\nWilcoxon signed-rank (per-evaluator combined) + Holm")
    results, holm_out = pairwise_wilcoxon(evaluators)
    print(
        f"{'comparison':<14} {'mean Δ':>8} {'W':>8} {'raw p':>10} {'Holm thr':>10}  result"
    )
    for label, delta, w, p in results:
        _, thr, sig = holm_out[label]
        flag = "sig" if sig else ("ns (raw < .05)" if p < 0.05 else "ns")
        print(f"{label:<14} {delta:+8.3f} {w:8.1f} {p:10.4f} {thr:10.4f}  {flag}")
    chi, fp = friedman(evaluators)
    print(f"Friedman: χ² = {chi:.2f}, p = {fp:.3f}")

    print("\nMean rank (1 = best)")
    ranks = mean_ranks(evaluators)
    for c in sorted(CONDS, key=lambda x: ranks[x]):
        print(f"  {SHORT[c]:<5} {ranks[c]:.2f}")

    print("\nBEST / WORST single picks (ties excluded)")
    bw = best_worst(evaluators)
    print(f"{'':<8} {'no':>8} {'pre':>8} {'post':>8}")
    for kind, nkey in (("BEST", "n_best"), ("WORST", "n_worst")):
        d, k = bw[kind.lower()], bw[nkey]
        cells = "  ".join(
            f"{d[c]:3d} ({d[c] / k:4.0%})" if k else "   —" for c in CONDS
        )
        print(f"{kind} ({k:<3}) {cells}")

    print("\nPer-character winner (combined, 2–10 scale)")
    print(f"{'Character':<22} {'no':>6} {'pre':>6} {'post':>6}  winner")
    wins = {c: 0 for c in CONDS}
    for char in chars:
        means = {
            cond: _avg(
                [
                    v * 2.0
                    for v in _char_metric(evaluators, char["id"], cond, "combined")
                ]
            )
            for cond in CONDS
        }
        winners = _top(means)
        w = winners[0] if len(winners) == 1 else "/".join(SHORT[c] for c in winners)
        print(
            f"{char['name']:<22} {means['no_thinking']:6.3f} {means['pre_thinking']:6.3f} {means['post_thinking']:6.3f}  {w if '/' in w else SHORT[w]}"
        )
        if w in CONDS:
            wins[w] += 1
    print(
        f"wins: post {wins['post_thinking']}, pre {wins['pre_thinking']}, no {wins['no_thinking']}"
    )

    print("\nKrippendorff ordinal α (units = char×cond)")
    print(f"  consistency {krippendorff_ordinal(evaluators, 'consistency'):.3f}")
    print(f"  naturalness {krippendorff_ordinal(evaluators, 'naturalness'):.3f}")


def main() -> None:
    chars = load_key()
    volunteer = parse_form(HERE / "FormResponse-Volunteer.csv", "vol", chars)
    prolific = parse_form(HERE / "FormResponse-Prolific.csv", "prolific", chars)
    all_ev = volunteer + prolific
    print(f"{len(volunteer)} volunteer + {len(prolific)} prolific = {len(all_ev)}")
    report(all_ev, chars)


if __name__ == "__main__":
    main()
