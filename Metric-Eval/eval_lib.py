import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

CONDITIONS = ("no_thinking", "pre_thinking", "post_thinking")

PRE_OPEN = "<think>"
PRE_CLOSE = "</think>"
POST_OPEN = "<post-thinking>"
POST_CLOSE = "</post-thinking>"

CONTEXT_LENGTH = 4096
WINDOW_N = 3

REACT_ANCHOR = "- React to the player's words and intentions."
GREETING_ANCHOR = "Your first response should be a greeting to the player."

INSTRUCTION_LINES = {
    "no_thinking": " ",
    "pre_thinking": " - Before each reply, in <think>...</think>, briefly plan your intent and how you'll respond, in character. ",
    "post_thinking": " - After each reply, in <post-thinking>...</post-thinking>, briefly reflect in your own voice on why you responded as you did and what matters to remember going forward. Stay in your own perspective; never narrate the player's actions or feelings. ",
}

FLAGS = (
    "missing_trace",
    "unexpected_trace",
    "malformed_tags",
    "duplicate_trace",
    "misordered_trace",
    "nested_tags",
    "empty_dialogue",
    "empty_trace",
)


@dataclass
class CharacterCard:
    character_id: str
    name: str
    source: str
    card_text: str


@dataclass
class ParsedTurn:
    dialogue: str
    trace: str | None
    flags: list[str]


_TAG_TOKENS = (
    (PRE_OPEN, "pre", "open"),
    (PRE_CLOSE, "pre", "close"),
    (POST_OPEN, "post", "open"),
    (POST_CLOSE, "post", "close"),
)

_EXPECTED_FAMILY = {"pre_thinking": "pre", "post_thinking": "post"}


def _scan_tags(raw: str) -> list[tuple[int, int, str, str]]:
    events = []
    for token, family, kind in _TAG_TOKENS:
        start = 0
        while True:
            i = raw.find(token, start)
            if i == -1:
                break
            events.append((i, len(token), family, kind))
            start = i + len(token)
    events.sort()
    return events


def _segment(raw: str, events: list) -> tuple[list, list, list, bool]:
    outside = []
    blocks = []
    orphan_closes = []
    nested = False
    depth = 0
    cursor = 0
    current = None
    for pos, tlen, family, kind in events:
        if kind == "open":
            if depth == 0:
                outside.append((cursor, pos))
                current = (family, pos, pos + tlen)
            else:
                nested = True
            depth += 1
        else:
            if depth == 0:
                outside.append((cursor, pos))
                cursor = pos + tlen
                orphan_closes.append(family)
            else:
                depth -= 1
                if depth == 0:
                    fam, start, inner_start = current
                    blocks.append(
                        {
                            "family": fam,
                            "start": start,
                            "inner": (inner_start, pos),
                            "end": pos + tlen,
                            "complete": True,
                        }
                    )
                    cursor = pos + tlen
                    current = None
    if depth > 0:
        fam, start, inner_start = current
        blocks.append(
            {
                "family": fam,
                "start": start,
                "inner": (inner_start, len(raw)),
                "end": len(raw),
                "complete": False,
            }
        )
    else:
        outside.append((cursor, len(raw)))
    return outside, blocks, orphan_closes, nested


def _family_imbalance(events: list) -> bool:
    for fam in ("pre", "post"):
        depth = 0
        for _pos, _tlen, family, kind in events:
            if family != fam:
                continue
            if kind == "open":
                depth += 1
            elif depth == 0:
                return True
            else:
                depth -= 1
        if depth > 0:
            return True
    return False


def slug(name: str) -> str:
    name = "" if name is None else str(name)
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def load_character_cards(path) -> list[CharacterCard]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    cards = []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise ValueError(f"{path}: record {i} is not a JSON object")
        if "messages" in rec:
            name = rec.get("character") or rec.get("name")
            messages = rec.get("messages")
            if (
                not name
                or not isinstance(messages, list)
                or not messages
                or not isinstance(messages[0], dict)
                or "content" not in messages[0]
            ):
                raise ValueError(
                    f"{path}: record {i}: expected Eval-45 shape "
                    "{'source', 'character', 'messages': [{'role', 'content'}, ...]}"
                )
            source = rec.get("source", "card")
            card_text = messages[0]["content"]
        elif "name" in rec and "background" in rec and "location" in rec:
            name = rec["name"]
            source = rec.get("source", "card")
            quest = str(rec.get("quest") or "").strip()
            quest_part = f"Quest: {quest} " if quest else ""
            card_text = (
                f"You are {name}. Background: {rec['background']} "
                f"Current Location: {rec['location']} {quest_part}"
                "Roleplaying Instructions: "
                "- Speak using appropriate tone and vocabulary "
                "- Reference your background and current surroundings naturally "
                "- Keep responses conversational and authentic "
                f"{REACT_ANCHOR} {GREETING_ANCHOR}"
            )
        else:
            raise ValueError(
                f"{path}: record {i} has keys {sorted(rec)}; expected either "
                "{'source', 'character', 'messages'} (Eval-45 shape) or "
                "{'name', 'background', 'location'} with optional 'quest' (flat card)"
            )
        cards.append(
            CharacterCard(
                character_id=f"{source}__{slug(name)}",
                name=str(name),
                source=str(source),
                card_text=str(card_text),
            )
        )
    return cards


def build_turn1_prompt(card, condition: str) -> str:
    text = card.card_text if isinstance(card, CharacterCard) else str(card)
    line = INSTRUCTION_LINES.get(condition)
    if line is None:
        logger.warning(f"unknown condition {condition!r}; using no-thinking prompt")
        line = INSTRUCTION_LINES["no_thinking"]
    needle = f"{REACT_ANCHOR} {GREETING_ANCHOR}"
    if needle in text:
        return text.replace(needle, f"{REACT_ANCHOR}{line}{GREETING_ANCHOR}", 1)
    if line == INSTRUCTION_LINES["no_thinking"]:
        return text
    if GREETING_ANCHOR in text:
        logger.warning(
            "anchor pair not found in card; inserting instruction line before greeting anchor"
        )
        idx = text.index(GREETING_ANCHOR)
        return text[:idx].rstrip() + line + text[idx:]
    logger.warning("both anchors missing from card; appending instruction line at end")
    return text.rstrip() + line.rstrip()


def parse_model_turn(raw: str, condition: str) -> ParsedTurn:
    raw = "" if raw is None else str(raw)
    events = _scan_tags(raw)
    outside, blocks, _orphan_closes, nested = _segment(raw, events)
    flags = set()

    expected = _EXPECTED_FAMILY.get(condition)
    if expected is None:
        if events:
            flags.add("unexpected_trace")
    else:
        other = "post" if expected == "pre" else "pre"
        if any(family == other for _p, _l, family, _k in events):
            flags.add("unexpected_trace")

    if _family_imbalance(events):
        flags.add("malformed_tags")
    if nested:
        flags.add("nested_tags")

    trace = None
    if expected is not None:
        exp_blocks = [b for b in blocks if b["family"] == expected]
        complete = [b for b in exp_blocks if b["complete"]]
        if len(complete) > 1:
            flags.add("duplicate_trace")
        chosen = complete[0] if complete else (exp_blocks[0] if exp_blocks else None)
        if chosen is None:
            flags.add("missing_trace")
        else:
            s, e = chosen["inner"]
            trace = raw[s:e].strip()
            if not trace:
                flags.add("empty_trace")
            before = "".join(raw[a:b] for a, b in outside if b <= chosen["start"])
            after = "".join(raw[a:b] for a, b in outside if a >= chosen["end"])
            if expected == "pre" and before.strip():
                flags.add("misordered_trace")
            if expected == "post" and after.strip():
                flags.add("misordered_trace")

    dialogue = "".join(raw[a:b] for a, b in outside).strip()
    if not dialogue:
        flags.add("empty_dialogue")

    return ParsedTurn(
        dialogue=dialogue, trace=trace, flags=[f for f in FLAGS if f in flags]
    )


def strip_all_tags(raw: str) -> str:
    raw = "" if raw is None else str(raw)
    events = _scan_tags(raw)
    if not events:
        return raw.strip()
    outside, _blocks, _orphans, _nested = _segment(raw, events)
    return "".join(raw[a:b] for a, b in outside).strip()


def format_model_turn(dialogue: str, trace: str | None, condition: str) -> str:
    dialogue = "" if dialogue is None else str(dialogue)
    if trace is None or condition == "no_thinking":
        return dialogue
    if condition == "pre_thinking":
        return f"{PRE_OPEN}{trace}{PRE_CLOSE}{dialogue}"
    if condition == "post_thinking":
        return f"{dialogue}{POST_OPEN}{trace}{POST_CLOSE}"
    logger.warning(f"unknown condition {condition!r}; returning bare dialogue")
    return dialogue


def apply_trace_window(
    model_turn_raws: list[str], condition: str, window_n: int = WINDOW_N
) -> list[str]:
    raws = list(model_turn_raws)
    if condition == "no_thinking":
        return raws
    n = max(0, int(window_n))
    keep_from = max(0, len(raws) - n)
    return [
        raw if i >= keep_from else strip_all_tags(raw) for i, raw in enumerate(raws)
    ]


def read_jsonl(path) -> list[dict]:
    path = Path(path)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(f"{path}: skipping malformed JSON on line {lineno}")
    return rows


def append_jsonl(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_STAGE_DIRECTION_RE = re.compile(r"\*[^*]*\*")


def sentence_split(text: str) -> list[str]:
    text = "" if text is None else str(text).strip()
    if not text:
        return []
    return [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]


def strip_stage_directions(text: str) -> str:
    text = "" if text is None else str(text)
    out = _STAGE_DIRECTION_RE.sub(" ", text)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip()
