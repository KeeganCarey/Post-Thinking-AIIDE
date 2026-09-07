from __future__ import annotations

import json
import time
from pathlib import Path

import eval_lib

EVAL_DIR = Path(__file__).resolve().parent
SOURCE_LATENCY_DIR = EVAL_DIR / "latency"
RUNS_DIR = EVAL_DIR / "runs"
LATENCY_DIR = RUNS_DIR / "latency"

CONDITIONS = eval_lib.CONDITIONS
KEEP_REPS = (0, 1, 2)
RECORDS_TEMPLATE = "latency_records_{condition}.jsonl"
SUMMARY_PATH = LATENCY_DIR / "latency_summary.json"
SUMMARY_METRICS = ("prefill_s", "ttft_s", "decode_tps", "perceived_s", "total_s")


def _build_config(overrides: dict) -> dict:
    cfg = {k: v for k, v in globals().items() if k.isupper() and not k.startswith("_")}
    unknown = sorted(set(overrides) - set(cfg))
    if unknown:
        raise ValueError(
            f"unknown config override(s): {unknown}; valid keys: {sorted(cfg)}"
        )
    cfg.update(overrides)
    changed = set(overrides)
    if "EVAL_DIR" in changed:
        cfg["SOURCE_LATENCY_DIR"] = Path(cfg["EVAL_DIR"]) / "latency"
        cfg["RUNS_DIR"] = Path(cfg["EVAL_DIR"]) / "runs"
        changed.add("RUNS_DIR")
    if "RUNS_DIR" in changed:
        cfg["LATENCY_DIR"] = Path(cfg["RUNS_DIR"]) / "latency"
        changed.add("LATENCY_DIR")
    if "LATENCY_DIR" in changed and "SUMMARY_PATH" not in overrides:
        cfg["SUMMARY_PATH"] = Path(cfg["LATENCY_DIR"]) / "latency_summary.json"
    return cfg


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}: malformed JSON on line {line_no}: {exc}"
                ) from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _percentile(values: list[float], q: float) -> float | None:
    vals = sorted(float(v) for v in values)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def _metric_stats(values: list[float]) -> dict:
    vals = [
        float(v)
        for v in values
        if isinstance(v, (int, float))
        and not isinstance(v, bool)
        and v == v
        and v not in (float("inf"), float("-inf"))
    ]
    if not vals:
        return {
            "n": 0,
            "median": None,
            "q25": None,
            "q75": None,
            "iqr": None,
            "p5": None,
            "p95": None,
        }
    q25 = _percentile(vals, 25)
    q75 = _percentile(vals, 75)
    return {
        "n": len(vals),
        "median": _percentile(vals, 50),
        "q25": q25,
        "q75": q75,
        "iqr": None if q25 is None or q75 is None else q75 - q25,
        "p5": _percentile(vals, 5),
        "p95": _percentile(vals, 95),
    }


def summarize_records(records: list[dict], metrics=SUMMARY_METRICS) -> dict:
    by_condition: dict[str, list[dict]] = {}
    for row in records:
        by_condition.setdefault(row.get("condition", "unknown"), []).append(row)

    out = {}
    for condition in sorted(by_condition):
        rows = by_condition[condition]
        used = [r for r in rows if r.get("throttled") is not True]
        metric_stats = {}
        for metric in metrics:
            values = []
            for row in used:
                metric_block = (
                    row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
                )
                value = metric_block.get(metric, row.get(metric))
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    values.append(value)
            metric_stats[metric] = _metric_stats(values)
        out[condition] = {
            "n_records": len(rows),
            "n_excluded_throttled": len(rows) - len(used),
            "n_used": len(used),
            "n_throttle_unclassified": sum(
                1 for r in used if r.get("throttled") is None
            ),
            "metrics": metric_stats,
        }
    return out


def prepare_latency_records(
    source_dir: Path,
    output_dir: Path,
    conditions,
    keep_reps: tuple[int, ...],
    template: str,
) -> tuple[list[dict], dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    keep = set(keep_reps)
    all_kept = []
    report = {}
    for condition in conditions:
        src = Path(source_dir) / template.format(condition=condition)
        if not src.exists():
            raise FileNotFoundError(f"missing latency records for {condition}: {src}")
        rows = _read_jsonl(src)
        kept = [r for r in rows if r.get("rep") in keep]
        dropped = [r for r in rows if r.get("rep") not in keep]
        dst = Path(output_dir) / template.format(condition=condition)
        _write_jsonl(dst, kept)
        all_kept.extend(kept)

        kept_reps = {}
        for row in kept:
            rep = row.get("rep")
            kept_reps[rep] = kept_reps.get(rep, 0) + 1
        report[condition] = {
            "source_rows": len(rows),
            "kept_rows": len(kept),
            "dropped_rows": len(dropped),
            "kept_reps": dict(sorted(kept_reps.items())),
        }
    return all_kept, report


def main(**overrides) -> dict:
    cfg = _build_config(overrides)
    all_records, report = prepare_latency_records(
        Path(cfg["SOURCE_LATENCY_DIR"]),
        Path(cfg["LATENCY_DIR"]),
        cfg["CONDITIONS"],
        tuple(cfg["KEEP_REPS"]),
        cfg["RECORDS_TEMPLATE"],
    )
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "n_reps": len(tuple(cfg["KEEP_REPS"])),
            "kept_reps": list(tuple(cfg["KEEP_REPS"])),
            "source_dir": str(cfg["SOURCE_LATENCY_DIR"]),
            "filtered_from_accidental_extra_rep": any(
                v["dropped_rows"] for v in report.values()
            ),
            "notes": (
                "Canonical latency records filtered to KEEP_REPS. "
                "Original staging records are left untouched."
            ),
            "cold_prefill": True,
            "window_n": eval_lib.WINDOW_N,
        },
        "filter_report": report,
        "conditions": summarize_records(all_records, tuple(cfg["SUMMARY_METRICS"])),
    }
    summary_path = Path(cfg["SUMMARY_PATH"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_dir": str(cfg["LATENCY_DIR"]),
                "summary_path": str(summary_path),
                "filter_report": report,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return summary


if __name__ == "__main__":
    main()
