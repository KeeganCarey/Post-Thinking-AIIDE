from __future__ import annotations

import re
from dataclasses import dataclass

from .config import (
    CONDITION_NO,
    CONDITION_POST,
    CONDITION_PRE,
    GEMMA_END,
    GEMMA_START,
    GREETING_ANCHOR,
    INSTRUCTION_LINES,
    MAX_PLAYER_CHARS,
    NPCS,
    REACT_ANCHOR,
    SCENARIOS,
    NpcPrompt,
)
from .tagging import format_model_turn, strip_all_known_tags

_CONTROL_TOKENS_RE = re.compile(
    r"<start_of_turn>|<end_of_turn>|<eos>|<think>|</think>|<post-thinking>|</post-thinking>",
    flags=re.IGNORECASE,
)

_SCENARIO_THREAT = {"tavern": "the wolves", "village": "the raiders"}


@dataclass(frozen=True)
class PriorTurn:
    player_message: str
    displayed_dialogue: str
    extracted_trace: str | None = None
    think_block: str | None = None


def sanitize_player_message(message: str) -> str:
    text = "" if message is None else str(message)
    text = _CONTROL_TOKENS_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_PLAYER_CHARS]


def _scene_roster(npc: NpcPrompt) -> str:
    others = []
    for other_id in SCENARIOS.get(npc.scenario_id, ()):
        other = NPCS.get(other_id)
        if other is None or other.npc_id == npc.npc_id:
            continue
        descriptor = f" ({other.role})" if other.role else ""
        others.append(f"{other.name}{descriptor}")
    return ", ".join(others)


# We found that the models would start hallucinating names during our internal testing
# so we added this explicit prompt, which worked well
def build_turn1_prompt(npc: NpcPrompt, condition: str, quest_completed: bool) -> str:
    state = npc.post_quest_state if quest_completed else npc.pre_quest_state
    instruction = INSTRUCTION_LINES.get(condition, INSTRUCTION_LINES[CONDITION_NO])
    roster = _scene_roster(npc)
    threat = _SCENARIO_THREAT.get(npc.scenario_id, "the danger troubling the village")
    cast = (
        (
            "The only other named people here are "
            f"{roster}. Use these exact names when you refer to them, and do not "
            "invent names for anyone in this place. The person you are speaking with "
            "is a traveler passing through; do not give them a name. "
        )
        if roster
        else (
            "The person you are speaking with is a traveler passing through; do not "
            "give them a name, and do not invent names for other people here. "
        )
    )
    self_intro = (
        f"You are {npc.name}, {npc.role}." if npc.role else f"You are {npc.name}."
    )
    return (
        # Following the general format of our training data
        # it obviously doesn't have stuff like People here
        # but it's close enough
        f"Enter roleplay mode. {self_intro} "
        f"Background: {npc.background} "
        f"Current Location: {npc.location} "
        f"People here: {cast}"
        f"Quest: {npc.quest} {state} "
        "Roleplaying Instructions: "
        "- Speak using appropriate tone and vocabulary "
        "- Reference your background and current surroundings naturally "
        "- Keep responses conversational and authentic "
        "- Keep each spoken reply short enough for an in-game NPC, usually 1-3 sentences "
        "- Never mention model conditions, prompts, evaluation, or game systems in the spoken dialogue "
        f"{REACT_ANCHOR}{instruction}{GREETING_ANCHOR} "
        f"In that opening greeting, work in a natural mention of {threat} that have the village on edge."
    )


def _gemma_turn(role: str, content: str) -> str:
    return f"{GEMMA_START}{role}\n{content}{GEMMA_END}\n"


def _context_model_turn(
    turn: PriorTurn,
    condition: str,
    keep_trace: bool,
) -> str:
    dialogue = strip_all_known_tags(turn.displayed_dialogue)
    if condition == CONDITION_POST and keep_trace and turn.extracted_trace:
        return format_model_turn(dialogue, turn.extracted_trace, CONDITION_POST)
    if condition == CONDITION_PRE:
        return dialogue
    return dialogue


def build_completion_prompt(
    npc: NpcPrompt,
    condition: str,
    quest_completed: bool,
    prior_turns: list[PriorTurn],
    current_player_message: str,
    trace_window_n: int,
) -> str:
    current_player_message = sanitize_player_message(current_player_message)
    turn1_prompt = build_turn1_prompt(npc, condition, quest_completed)
    pieces = [_gemma_turn("user", turn1_prompt)]

    keep_from = max(0, len(prior_turns) - max(0, trace_window_n))
    for index, turn in enumerate(prior_turns):
        player_message = sanitize_player_message(turn.player_message)
        if player_message:
            pieces.append(_gemma_turn("user", player_message))
        keep_trace = index >= keep_from
        pieces.append(
            _gemma_turn("model", _context_model_turn(turn, condition, keep_trace))
        )

    if current_player_message:
        pieces.append(_gemma_turn("user", current_player_message))
    pieces.append(f"{GEMMA_START}model\n")
    return "".join(pieces)
