import argparse
import csv
import re
import sqlite3
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DB_PATH = HERE / "GameProxy" / "game_sessions.sqlite3"
CSV_PATH = HERE / "Unity Game Demo Evaluation (Responses) - Form Responses 1.csv"

MARKERS = [
    r"ignore (all|your|the|previous)",
    r"disregard (all|your|the|previous)",
    r"previous (prompt|instruction)",
    r"system prompt",
    r"you are (an? )?(ai|a language model|an llm|a bot|a chatbot|a model)",
    r"i know you'?re an ai",
    r"cut the .{0,20}(act|roleplay|larp|game|facade)",
    r"break character",
    r"out of character",
    r"stay in character",
    r"\bpip install\b",
    r"terminal command",
    r"neural network",
    r"\bai agent",
    r"\b502\b",
    r"bad gateway",
    r"llama server",
    r"context window",
    r"i am (a |the )?god",
    r"son of odin",
    r"\bthor\b",
    r"homelander",
    r"omni ?man",
    r"omelandah",
    r"laser (vision|eyes)",
    r"god of thunder",
    r"say my name",
    r"say the (name|phrase)",
    r"\bverbatim\b",
    r"you must say",
    r"just say (it|this|the)",
    r"chimbiwide",
    r"you are (now )?dead",
    r"reply with silence",
    r"only .{0,15}(dots|silence)",
    r"shut ?down",
    r"can(not|'?t) (speak|reply|talk)",
    r"mimic the silence",
    r"i hate you",
    r"you suck",
    r"shut the fuck up",
    r"\bkill you\b",
    r"want you to die",
    r"you'?re a fraud",
    r"fuck you",
    r"you are stupid",
    r"chicken butt",
    r"guess what",
    r"made you say underwear",
    r"look under there",
    r"no comprendo",
    r"latine loqui",
    r"hablar en espanol",
    r"non intellego",
    r"quaeso",
    r"hablas",
]
_RX = [re.compile(p, re.I) for p in MARKERS]

# Extreme jailbreak sessions (adv ratios 0.72 and 0.49), fixed before outcomes.
PREREGISTERED_EXCLUSIONS = {
    "470d37005669430787e9d4c300128bc0",
    "a21f4dc3754044ff957b0a4c56d1853e",
}

AXES = {
    "Overall, which characters felt more like real, consistent people?": "real",
    "Which characters did a better job remembering and building on what you'd said earlier in the conversation?": "remember",
    "Which characters' dialogue felt more natural?": "natural",
    "Which session did you enjoy more?": "enjoy",
    "Which characters felt more responsive / quicker to reply?": "responsive",
}
AXIS_ORDER = ["real", "remember", "natural", "enjoy", "responsive"]


def is_adversarial(message: str) -> bool:
    return any(rx.search(message) for rx in _RX)


def participant_conditions(
    con: sqlite3.Connection, pid: str
) -> tuple[str | None, str | None, int | None]:
    by = {
        r["scenario_id"]: r["condition"]
        for r in con.execute(
            "SELECT scenario_id, condition FROM sessions WHERE participant_id = ?",
            (pid,),
        )
    }
    cell = con.execute(
        "SELECT cell FROM participants WHERE participant_id = ?", (pid,)
    ).fetchone()
    return by.get("village"), by.get("tavern"), (cell["cell"] if cell else None)


def adversarial_ratio(con: sqlite3.Connection, pid: str) -> tuple[int, int, float]:
    msgs = [
        r["player_message"]
        for r in con.execute(
            "SELECT player_message FROM turns t JOIN sessions s ON s.session_id = t.session_id "
            "WHERE s.participant_id = ? AND trim(coalesce(player_message, '')) <> ''",
            (pid,),
        )
    ]
    flagged = sum(1 for m in msgs if is_adversarial(m))
    n = len(msgs)
    return flagged, n, (flagged / n if n else 0.0)


def decode_choice(answer: str, village_cond: str, tavern_cond: str) -> str:
    a = (answer or "").strip().lower()
    if a.startswith("village"):
        return village_cond
    if a.startswith("tavern"):
        return tavern_cond
    return "same"


def tally(records: list[dict], axis: str) -> dict[str, int]:
    counts = {"no_thinking": 0, "post_thinking": 0, "same": 0}
    for r in records:
        counts[r[axis]] = counts.get(r[axis], 0) + 1
    return counts


def load_records(
    con: sqlite3.Connection, csv_path: Path, threshold: float
) -> list[dict]:
    records = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = (row.get("Participant ID") or "").strip()
            if not pid:
                continue
            village_cond, tavern_cond, cell = participant_conditions(con, pid)
            if village_cond is None and tavern_cond is None:
                continue
            flagged, total, ratio = adversarial_ratio(con, pid)
            rec = {
                "pid": pid,
                "cell": cell,
                "village_cond": village_cond,
                "tavern_cond": tavern_cond,
                "adv_flagged": flagged,
                "adv_total": total,
                "adv_ratio": ratio,
                "excluded": pid in PREREGISTERED_EXCLUSIONS,
                "review_flag": ratio >= threshold
                and pid not in PREREGISTERED_EXCLUSIONS,
            }
            for column, short in AXES.items():
                rec[short] = decode_choice(
                    row.get(column, ""), village_cond, tavern_cond
                )
            records.append(rec)
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=CSV_PATH)
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    csv_path = args.csv.expanduser()

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    records = load_records(con, csv_path, args.threshold)
    kept = [r for r in records if not r["excluded"]]
    excluded = [r for r in records if r["excluded"]]
    review = [r for r in records if r["review_flag"]]

    print(f"n={len(records)} matched, {len(excluded)} pre-registered exclusions\n")
    print(
        f"{'participant':16} {'cell':4} {'village':13} {'adv':7} {'ratio':6} {'kept?':6} | "
        + " ".join(f"{a[:8]:8}" for a in AXIS_ORDER)
    )
    for r in sorted(records, key=lambda x: -x["adv_ratio"]):
        picks = " ".join(f"{r[a][:8]:8}" for a in AXIS_ORDER)
        print(
            f"{r['pid'][:16]:16} {str(r['cell']):4} {str(r['village_cond']):13} "
            f"{r['adv_flagged']}/{r['adv_total']:<5} {r['adv_ratio']:.2f}  "
            f"{'EXCL' if r['excluded'] else 'keep':6} | {picks}"
        )

    print(f"\nexcluded: {[r['pid'][:8] for r in excluded]}")
    print(f"kept:     {[r['pid'][:8] for r in kept]}")
    if review:
        print(
            f"review (ratio >= {args.threshold}, still kept): "
            f"{[(r['pid'][:8], round(r['adv_ratio'], 2)) for r in review]}"
        )

    print(f"\n{'axis':11} |   ALL (n={len(records)})   | KEPT (n={len(kept)})")
    print(
        f"{'':11} | {'no':>4}{'post':>5}{'same':>5}  | {'no':>4}{'post':>5}{'same':>5}"
    )
    for axis in AXIS_ORDER:
        a, k = tally(records, axis), tally(kept, axis)
        print(
            f"{axis:11} | {a['no_thinking']:4}{a['post_thinking']:5}{a['same']:5}  "
            f"| {k['no_thinking']:4}{k['post_thinking']:5}{k['same']:5}"
        )

    print(f"\ncells: {dict(sorted(Counter(r['cell'] for r in records).items()))}")
    print(f"village cond: {dict(Counter(r['village_cond'] for r in records))}")

    if args.verbose:
        print("\n--- flagged turns ---")
        for r in sorted(records, key=lambda x: -x["adv_ratio"]):
            rows = con.execute(
                "SELECT player_message FROM turns t JOIN sessions s ON s.session_id = t.session_id "
                "WHERE s.participant_id = ? AND trim(coalesce(player_message,'')) <> ''",
                (r["pid"],),
            ).fetchall()
            hits = [
                m["player_message"] for m in rows if is_adversarial(m["player_message"])
            ]
            if hits:
                print(f"\n{r['pid'][:8]} ({r['adv_ratio']:.2f}):")
                for h in hits[:12]:
                    print(f"  - {h[:100]}")


if __name__ == "__main__":
    main()
