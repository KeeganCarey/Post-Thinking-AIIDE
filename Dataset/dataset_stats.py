# Dataset statistics for PostThink-RP (npc 800 + pippa 400 + rpg 800).

import csv
import json
import re
from pathlib import Path
from statistics import mean, median, pstdev

from transformers import AutoTokenizer

csv.field_size_limit(10_000_000)

ROOT = Path(__file__).resolve().parent
POST_SOURCES = {
    "npc": ROOT / "dataset/post-thinking/npc_dialogue_post_thinking.csv",
    "pippa": ROOT / "dataset/post-thinking/pippa_post_thinking_400.csv",
    "rpg": ROOT / "dataset/post-thinking/rpg_quests_post_thinking.csv",
}
PRE_JSONL = ROOT / "Pre-Thinking.jsonl"
POST_TAG = re.compile(r"<post-thinking>(.*?)</post-thinking>", re.DOTALL)
THINK_TAG = re.compile(r"<think>(.*?)</think>", re.DOTALL)
NAME_RE = re.compile(r"You are ([^.\n]+?)\.")
TOK = AutoTokenizer.from_pretrained("chimbiwide/Gemma3NPC-4B-no-thinking")


def _tok(text: str) -> int:
    return len(TOK.encode(text, add_special_tokens=False))


def _words(text: str) -> int:
    return len(text.split())


def _summary(values: list[int]) -> dict:
    if not values:
        return {"n": 0, "mean": 0.0, "median": 0.0, "min": 0, "max": 0, "std": 0.0}
    return {
        "n": len(values),
        "mean": round(mean(values), 2),
        "median": round(median(values), 2),
        "min": min(values),
        "max": max(values),
        "std": round(pstdev(values) if len(values) >= 2 else 0.0, 2),
    }


def _load_csv(path: Path) -> list[list[dict]]:
    with path.open(encoding="utf-8", newline="") as f:
        return [
            json.loads(row["post-thinking"])["messages"]
            for row in csv.DictReader(f)
            if row.get("post-thinking")
        ]


def _load_jsonl(path: Path) -> list[list[dict]]:
    convs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                convs.append(json.loads(line)["messages"])
    return convs


def _system(msgs: list[dict]) -> str | None:
    for m in msgs:
        if m.get("role") == "system":
            return m.get("content", "").strip()
    return None


def _analyze(convs: list[list[dict]], tag_re: re.Pattern) -> dict:
    characters: set[str] = set()
    turns_per_conv: list[int] = []
    trace_words: list[int] = []
    trace_toks: list[int] = []
    dialogue_words: list[int] = []
    empty = missing = user_turns = 0

    for msgs in convs:
        asst = 0
        for m in msgs:
            role, content = m.get("role"), m.get("content", "")
            if role == "system":
                if match := NAME_RE.search(content):
                    characters.add(match.group(1).strip())
            elif role == "user":
                user_turns += 1
            elif role == "assistant":
                asst += 1
                tag = tag_re.search(content)
                if tag is None:
                    missing += 1
                    dialogue_words.append(_words(content))
                    continue
                trace = tag.group(1).strip()
                if not trace:
                    empty += 1
                trace_words.append(_words(trace))
                trace_toks.append(_tok(trace))
                dialogue_words.append(_words(content[: tag.start()]))
        turns_per_conv.append(asst)

    return {
        "convs": len(convs),
        "chars": len(characters),
        "asst": sum(turns_per_conv),
        "user": user_turns,
        "traces": len(trace_words),
        "empty": empty,
        "missing": missing,
        "turns": _summary(turns_per_conv),
        "tw": _summary(trace_words),
        "tt": _summary(trace_toks),
        "dw": _summary(dialogue_words),
        "tw_sum": sum(trace_words),
        "tt_sum": sum(trace_toks),
    }


def _show(name: str, s: dict) -> None:
    t, tw, tt, dw = s["turns"], s["tw"], s["tt"], s["dw"]
    print(f"{name}")
    print(
        f"  {s['convs']} convs, {s['chars']} chars, {s['asst']} asst, "
        f"{s['traces']} traces, {s['empty']} empty, {s['missing']} missing tag"
    )
    print(f"  turns/conv  mean {t['mean']}  med {t['median']}  {t['min']}-{t['max']}")
    print(
        f"  trace words mean {tw['mean']}  med {tw['median']}  "
        f"{tw['min']}-{tw['max']}  std {tw['std']}  total {s['tw_sum']}"
    )
    print(
        f"  trace toks  mean {tt['mean']}  med {tt['median']}  "
        f"{tt['min']}-{tt['max']}  std {tt['std']}  total {s['tt_sum']}"
    )
    print(
        f"  dialogue    mean {dw['mean']}  med {dw['median']}  {dw['min']}-{dw['max']}"
    )


def main() -> None:
    per_src = {src: _load_csv(p) for src, p in POST_SOURCES.items()}
    post = {src: _analyze(convs, POST_TAG) for src, convs in per_src.items()}
    all_post = [c for convs in per_src.values() for c in convs]
    post_all = _analyze(all_post, POST_TAG)

    pre_convs = _load_jsonl(PRE_JSONL) if PRE_JSONL.exists() else []
    key_to_src: dict[str, str] = {}
    for src, convs in per_src.items():
        for msgs in convs:
            if (k := _system(msgs)) is not None:
                key_to_src.setdefault(k, src)
    pre_buckets: dict[str, list] = {src: [] for src in per_src}
    for msgs in pre_convs:
        src = key_to_src.get(_system(msgs))
        if src is not None:
            pre_buckets[src].append(msgs)
    pre = {src: _analyze(convs, THINK_TAG) for src, convs in pre_buckets.items()}
    pre_all = _analyze(pre_convs, THINK_TAG)

    print("Post-thinking  (npc 800 + pippa 400 + rpg 800)")
    print(
        f"{'src':<10} {'convs':>5} {'chars':>5} {'asst':>6} {'traces':>6} "
        f"{'empty':>5} {'t/conv':>7} {'post w':>7} {'pre w':>7} {'post t':>7} {'pre t':>7}"
    )
    for src in ("npc", "pippa", "rpg"):
        s, p = post[src], pre[src]
        print(
            f"{src:<10} {s['convs']:5} {s['chars']:5} {s['asst']:6} {s['traces']:6} "
            f"{s['empty']:5} {s['turns']['mean']:7.2f} {s['tw']['mean']:7.2f} "
            f"{p['tw']['mean']:7.2f} {s['tt']['mean']:7.2f} {p['tt']['mean']:7.2f}"
        )
    s, p = post_all, pre_all
    print(
        f"{'combined':<10} {s['convs']:5} {s['chars']:5} {s['asst']:6} {s['traces']:6} "
        f"{s['empty']:5} {s['turns']['mean']:7.2f} {s['tw']['mean']:7.2f} "
        f"{p['tw']['mean']:7.2f} {s['tt']['mean']:7.2f} {p['tt']['mean']:7.2f}"
    )

    print()
    for src in ("npc", "pippa", "rpg"):
        _show(f"post {src}", post[src])
        print(
            f"  pre traces  mean {pre[src]['tw']['mean']} words / "
            f"{pre[src]['tt']['mean']} toks"
        )
        print()
    _show("post combined", post_all)
    print()
    _show("pre combined", pre_all)

    words = sum(_words(m.get("content", "")) for msgs in all_post for m in msgs)
    toks = sum(_tok(m.get("content", "")) for msgs in all_post for m in msgs)
    print(f"\npost-thinking training text: {words:,} words, {toks:,} Gemma3 tokens")


if __name__ == "__main__":
    main()
