from __future__ import annotations

import re
from dataclasses import dataclass

from .config import (
    CONDITION_NO,
    CONDITION_POST,
    CONDITION_PRE,
    POST_CLOSE,
    POST_OPEN,
    PRE_CLOSE,
    PRE_OPEN,
)

_KNOWN_TAG_RE = re.compile(
    r"<think>.*?</think>|<post-thinking>.*?</post-thinking>",
    flags=re.DOTALL | re.IGNORECASE,
)
_GENERIC_TAG_RE = re.compile(r"</?[^>\n]{1,80}>")


@dataclass(frozen=True)
class ParsedOutput:
    dialogue: str
    trace: str | None
    think_block: str | None
    flags: tuple[str, ...]


def _first_block(
    raw: str, open_tag: str, close_tag: str
) -> tuple[str | None, int, int, bool]:
    start = raw.find(open_tag)
    if start == -1:
        return None, -1, -1, False
    inner_start = start + len(open_tag)
    close = raw.find(close_tag, inner_start)
    if close == -1:
        return raw[inner_start:].strip(), start, len(raw), True
    return raw[inner_start:close].strip(), start, close + len(close_tag), False


def strip_all_known_tags(raw: str) -> str:
    text = "" if raw is None else str(raw)
    text = _KNOWN_TAG_RE.sub("", text)
    for tag in (PRE_OPEN, POST_OPEN):
        idx = text.lower().find(tag)
        if idx != -1:
            text = text[:idx]
    for tag in (PRE_CLOSE, POST_CLOSE):
        text = text.replace(tag, "")
    text = _GENERIC_TAG_RE.sub("", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def parse_output(raw: str, condition: str) -> ParsedOutput:
    raw = "" if raw is None else str(raw)
    flags: list[str] = []
    trace: str | None = None
    think_block: str | None = None

    has_pre = PRE_OPEN in raw or PRE_CLOSE in raw
    has_post = POST_OPEN in raw or POST_CLOSE in raw

    if condition == CONDITION_NO:
        if has_pre or has_post:
            flags.append("unexpected_trace")
        return ParsedOutput(strip_all_known_tags(raw), None, None, tuple(flags))

    if condition == CONDITION_PRE:
        think_block, start, end, unclosed = _first_block(raw, PRE_OPEN, PRE_CLOSE)
        if think_block is None:
            flags.append("missing_trace")
        else:
            if not think_block:
                flags.append("empty_trace")
            if raw[:start].strip():
                flags.append("misordered_trace")
            if unclosed:
                flags.append("malformed_tags")
        if raw.count(PRE_OPEN) > 1 or raw.count(PRE_CLOSE) > 1:
            flags.append("duplicate_trace")
        if has_post:
            flags.append("unexpected_trace")
        return ParsedOutput(
            strip_all_known_tags(raw), None, think_block, tuple(dict.fromkeys(flags))
        )

    if condition == CONDITION_POST:
        trace, start, end, unclosed = _first_block(raw, POST_OPEN, POST_CLOSE)
        if trace is None:
            flags.append("missing_trace")
        else:
            if not trace:
                flags.append("empty_trace")
            if end < len(raw) and raw[end:].strip():
                flags.append("misordered_trace")
            if unclosed:
                flags.append("malformed_tags")
        if raw.count(POST_OPEN) > 1 or raw.count(POST_CLOSE) > 1:
            flags.append("duplicate_trace")
        if has_pre:
            flags.append("unexpected_trace")
        return ParsedOutput(
            strip_all_known_tags(raw), trace, None, tuple(dict.fromkeys(flags))
        )

    flags.append("unknown_condition")
    return ParsedOutput(strip_all_known_tags(raw), None, None, tuple(flags))


def format_model_turn(dialogue: str, trace: str | None, condition: str) -> str:
    dialogue = "" if dialogue is None else str(dialogue).strip()
    trace = None if trace is None else str(trace).strip()
    if condition == CONDITION_PRE and trace:
        return f"{PRE_OPEN}{trace}{PRE_CLOSE}{dialogue}"
    if condition == CONDITION_POST and trace:
        return f"{dialogue}{POST_OPEN}{trace}{POST_CLOSE}"
    return dialogue
