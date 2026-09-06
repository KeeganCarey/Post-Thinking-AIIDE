# Combine per-source post-thinking CSVs into Post-Thinking.jsonl

import csv
from pathlib import Path

csv.field_size_limit(10_000_000)

ROOT = Path(__file__).resolve().parent
SOURCES = [
    ROOT / "dataset/post-thinking/npc_dialogue_post_thinking.csv",
    ROOT / "dataset/post-thinking/pippa_post_thinking_400.csv",
    ROOT / "dataset/post-thinking/rpg_quests_post_thinking.csv",
]
OUT = ROOT / "Post-Thinking.jsonl"


def main() -> None:
    n = 0
    with OUT.open("w", encoding="utf-8") as out:
        for path in SOURCES:
            with path.open(encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    line = row.get("post-thinking")
                    if line:
                        out.write(line.rstrip("\n") + "\n")
                        n += 1
    print(f"wrote {n} conversations -> {OUT}")


if __name__ == "__main__":
    main()
