# Mix npc 800 + pippa 400 + rpg 800 into NPC-2k.jsonl

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCES = [
    ROOT / "dataset/extracted/npc_dialogue_800.jsonl",
    ROOT / "dataset/extracted/pippa_400.jsonl",
    ROOT / "dataset/extracted/rpg_quests_800.jsonl",
]
OUT = ROOT / "NPC-2k.jsonl"


def main() -> None:
    n = 0
    with OUT.open("w", encoding="utf-8") as out:
        for src in SOURCES:
            for line in src.open(encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                json.loads(line)
                out.write(line + "\n")
                n += 1
    print(f"wrote {n} conversations -> {OUT}")


if __name__ == "__main__":
    main()
