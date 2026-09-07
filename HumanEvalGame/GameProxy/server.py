from __future__ import annotations

import asyncio
import random
import time
import uuid
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import config
from .config import (
    CONDITION_NO,
    CONDITION_POST,
    CONDITIONS,
    DEFAULT_SCENARIO,
    NPC_SCENARIO,
    NPCS,
    PAIR_CELLS,
    POST_OPEN,
    SCENARIO_STARTING_NPC,
    SCENARIOS,
)
from .llama_client import stream_complete
from .prompting import PriorTurn, build_completion_prompt, sanitize_player_message
from .store import GameStore, utc_now_iso
from .tagging import parse_output, strip_all_known_tags

app = FastAPI(title="PostThink-RP Game Proxy", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(config.CORS_ORIGINS),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "ngrok-skip-browser-warning"],
)

store = GameStore(config.DATABASE_PATH, config.TURN_LOG_JSONL_PATH)

_pending_turns: dict[tuple[str, str], asyncio.Task] = {}


class SessionRequest(BaseModel):
    starting_npc: str = "maid"
    force_condition: str | None = None


class SessionResponse(BaseModel):
    session_id: str
    starting_npc: str


class ChatRequest(BaseModel):
    session_id: str
    npc_id: str
    message: str = Field(min_length=1, max_length=config.MAX_PLAYER_CHARS * 2)


class ChatResponse(BaseModel):
    dialogue: str
    npc_id: str
    turn_index: int


class GreetingRequest(BaseModel):
    session_id: str
    npc_id: str


class HistoryRequest(BaseModel):
    session_id: str
    npc_id: str


class HistoryTurn(BaseModel):
    turn_index: int
    player_message: str
    dialogue: str
    is_greeting: bool


class HistoryResponse(BaseModel):
    session_id: str
    npc_id: str
    quest_completed: bool
    turns: list[HistoryTurn]


class QuestCompleteRequest(BaseModel):
    session_id: str
    quest_completed: bool = True


class QuestCompleteResponse(BaseModel):
    session_id: str
    quest_completed: bool


class EndSessionRequest(BaseModel):
    session_id: str


class EndSessionResponse(BaseModel):
    session_id: str
    survey_url: str


class PairPart(BaseModel):
    part_index: int
    session_id: str
    scenario_id: str
    starting_npc: str


class PairResponse(BaseModel):
    participant_id: str
    parts: list[PairPart]


class PairRequest(BaseModel):
    force_cell: int | None = None


def _enabled_conditions() -> tuple[str, ...]:
    enabled = tuple(c for c in config.ENABLED_CONDITIONS if c in CONDITIONS)
    if not enabled:
        raise RuntimeError("No valid POSTTHINK_ENABLED_CONDITIONS configured")
    return enabled


def _assign_condition(force_condition: str | None = None) -> str:
    enabled = _enabled_conditions()
    if force_condition:
        if not config.ALLOW_FORCE_CONDITION:
            raise HTTPException(status_code=403, detail="Forced condition is disabled")
        if force_condition not in enabled:
            raise HTTPException(
                status_code=400, detail="Forced condition is not enabled"
            )
        return force_condition
    counts = store.count_sessions_by_condition(enabled)
    min_count = min(counts.values())
    choices = [condition for condition, count in counts.items() if count == min_count]
    return random.choice(choices)


def _assign_cell(force_cell: int | None = None) -> int:
    n_cells = len(PAIR_CELLS)
    if force_cell is not None:
        if not config.ALLOW_FORCE_CELL:
            raise HTTPException(status_code=403, detail="Forced cell is disabled")
        if force_cell not in range(n_cells):
            raise HTTPException(status_code=400, detail="Forced cell is out of range")
        return force_cell
    preferred = config.PREFERRED_CELL
    if preferred is not None and preferred in range(n_cells):
        return preferred
    counts = store.count_participants_by_cell(n_cells)
    min_count = min(counts.values())
    return random.choice([cell for cell, count in counts.items() if count == min_count])


def _survey_url(session_id: str) -> str:
    parts = urlsplit(config.SURVEY_URL)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query[config.SURVEY_SESSION_PARAM] = session_id
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
    )


def _pair_survey_url(participant_id: str) -> str:
    parts = urlsplit(config.SURVEY_URL)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query[config.SURVEY_PARTICIPANT_PARAM] = participant_id
    for part_index, session_id in store.get_participant_session_ids(participant_id):
        query[f"part{part_index}_session"] = session_id
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
    )


def _history_for_prompt(session_id: str, npc_id: str) -> list[PriorTurn]:
    return [
        PriorTurn(
            player_message=turn.player_message,
            displayed_dialogue=turn.displayed_dialogue,
            extracted_trace=turn.extracted_trace,
            think_block=turn.think_block,
        )
        for turn in store.get_npc_turns(session_id, npc_id)
    ]


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "enabled_conditions": _enabled_conditions(),
        "trace_window_n": config.TRACE_WINDOW_N,
        "mock_llm": config.MOCK_LLM,
        "allow_force_cell": config.ALLOW_FORCE_CELL,
        "preferred_cell": config.PREFERRED_CELL,
        "npcs": list(NPCS),
    }


@app.post("/session", response_model=SessionResponse)
async def create_session(req: SessionRequest) -> SessionResponse:
    if req.starting_npc not in NPCS:
        raise HTTPException(status_code=400, detail="Unknown starting_npc")
    condition = _assign_condition(req.force_condition)
    session_id = uuid.uuid4().hex
    scenario_id = NPC_SCENARIO.get(req.starting_npc, DEFAULT_SCENARIO)
    store.create_session(
        session_id, condition, req.starting_npc, scenario_id=scenario_id
    )
    return SessionResponse(session_id=session_id, starting_npc=req.starting_npc)


@app.post("/pair", response_model=PairResponse)
async def create_pair(req: PairRequest | None = None) -> PairResponse:
    enabled = _enabled_conditions()
    if CONDITION_NO not in enabled or CONDITION_POST not in enabled:
        raise HTTPException(
            status_code=409,
            detail="Paired study requires no_thinking and post_thinking enabled",
        )
    cell = _assign_cell(req.force_cell if req else None)

    participant_id = uuid.uuid4().hex
    store.create_participant(participant_id, cell)

    parts: list[PairPart] = []
    for part_index, (scenario_id, condition) in enumerate(PAIR_CELLS[cell], start=1):
        session_id = uuid.uuid4().hex
        starting_npc = SCENARIO_STARTING_NPC[scenario_id]
        store.create_session(
            session_id,
            condition,
            starting_npc,
            scenario_id=scenario_id,
            participant_id=participant_id,
            part_index=part_index,
        )
        parts.append(
            PairPart(
                part_index=part_index,
                session_id=session_id,
                scenario_id=scenario_id,
                starting_npc=starting_npc,
            )
        )
    return PairResponse(participant_id=participant_id, parts=parts)


async def _run_streamed_turn(
    session,
    session_id: str,
    npc_id: str,
    prompt: str,
    turn_index: int,
    player_message: str,
    persist_flags: tuple[str, ...] = (),
) -> str:
    """Stream a turn; return spoken dialogue as soon as it is ready."""
    key = (session_id, npc_id)
    request_sent_ts = utc_now_iso()
    dialogue_ready = asyncio.Event()
    holder: dict[str, Any] = {}

    def _persist(
        raw: str, total_latency_ms: int, p_tokens, c_tokens, extra_flags=()
    ) -> None:
        parsed = parse_output(raw, session.condition)
        displayed = parsed.dialogue
        flags = parsed.flags + tuple(persist_flags) + tuple(extra_flags)
        if not displayed:
            displayed = "..."
            flags = flags + ("empty_dialogue_replaced",)
        store.insert_turn(
            {
                "session_id": session_id,
                "condition": session.condition,
                "npc_id": npc_id,
                "quest_completed": session.quest_completed,
                "turn_index": turn_index,
                "player_message": player_message,
                "raw_model_output": raw,
                "displayed_dialogue": displayed,
                "extracted_trace": parsed.trace,
                "think_block": parsed.think_block,
                "parse_flags": list(flags),
                "request_sent_ts": request_sent_ts,
                "response_complete_ts": utc_now_iso(),
                "model_latency_ms": total_latency_ms,
                "dialogue_latency_ms": holder.get("dialogue_latency_ms"),
                "prompt_tokens": p_tokens,
                "completion_tokens": c_tokens,
            }
        )

    async def _consume() -> None:
        start = time.perf_counter()
        buffer = ""
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        truncated = False
        try:
            async for chunk in stream_complete(session.condition, prompt):
                if chunk.text:
                    buffer += chunk.text
                    if (
                        not dialogue_ready.is_set()
                        and session.condition == CONDITION_POST
                        and POST_OPEN in buffer
                    ):
                        holder["dialogue_snapshot"] = buffer
                        holder["dialogue_latency_ms"] = int(
                            (time.perf_counter() - start) * 1000
                        )
                        dialogue_ready.set()
                if chunk.done:
                    prompt_tokens = chunk.prompt_tokens
                    completion_tokens = chunk.completion_tokens
                    truncated = chunk.truncated
        except Exception as exc:
            holder["error"] = exc
            if "dialogue_snapshot" not in holder:
                dialogue_ready.set()
                return
            _persist(
                buffer,
                int((time.perf_counter() - start) * 1000),
                None,
                None,
                extra_flags=("stream_failed_after_dialogue",),
            )
            return

        total_latency_ms = int((time.perf_counter() - start) * 1000)
        if not dialogue_ready.is_set():
            holder["dialogue_snapshot"] = buffer
            holder["dialogue_latency_ms"] = total_latency_ms
            dialogue_ready.set()
        extra = ("output_truncated",) if truncated else ()
        _persist(
            buffer,
            total_latency_ms,
            prompt_tokens,
            completion_tokens,
            extra_flags=extra,
        )

    task = asyncio.create_task(_consume())
    _pending_turns[key] = task
    task.add_done_callback(
        lambda finished: (
            _pending_turns.pop(key, None)
            if _pending_turns.get(key) is finished
            else None
        )
    )

    await dialogue_ready.wait()
    if "dialogue_snapshot" not in holder:
        raise HTTPException(
            status_code=502,
            detail=f"llama-server request failed: {holder.get('error')}",
        )
    return strip_all_known_tags(holder["dialogue_snapshot"]) or "..."


@app.post("/greeting", response_model=ChatResponse)
async def greeting(req: GreetingRequest) -> ChatResponse:
    """NPC opening line. Re-opening returns the stored greeting."""
    session = store.get_session(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    npc = NPCS.get(req.npc_id)
    if npc is None:
        raise HTTPException(status_code=400, detail="Unknown npc_id")
    if session.scenario_id and req.npc_id not in SCENARIOS.get(session.scenario_id, ()):
        raise HTTPException(
            status_code=400, detail="npc_id does not belong to this session's scenario"
        )

    pending = _pending_turns.get((req.session_id, req.npc_id))
    if pending is not None:
        try:
            await pending
        except Exception:
            pass

    existing = store.get_npc_turns(req.session_id, req.npc_id)
    if existing:
        greeting_turn = next(
            (t for t in existing if t.player_message == ""), existing[0]
        )
        return ChatResponse(
            dialogue=strip_all_known_tags(greeting_turn.displayed_dialogue) or "...",
            npc_id=req.npc_id,
            turn_index=greeting_turn.turn_index,
        )

    prompt = build_completion_prompt(
        npc=npc,
        condition=session.condition,
        quest_completed=session.quest_completed,
        prior_turns=[],
        current_player_message="",
        trace_window_n=config.TRACE_WINDOW_N,
    )
    turn_index = store.next_turn_index(req.session_id, req.npc_id)
    displayed = await _run_streamed_turn(
        session,
        req.session_id,
        req.npc_id,
        prompt,
        turn_index,
        "",
        persist_flags=("greeting",),
    )
    return ChatResponse(dialogue=displayed, npc_id=req.npc_id, turn_index=turn_index)


@app.post("/history", response_model=HistoryResponse)
async def history(req: HistoryRequest) -> HistoryResponse:
    """Saved conversation for one NPC, tags stripped."""
    session = store.get_session(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    if req.npc_id not in NPCS:
        raise HTTPException(status_code=400, detail="Unknown npc_id")

    pending = _pending_turns.get((req.session_id, req.npc_id))
    if pending is not None:
        try:
            await pending
        except Exception:
            pass

    turns = [
        HistoryTurn(
            turn_index=t.turn_index,
            player_message=t.player_message,
            dialogue=strip_all_known_tags(t.displayed_dialogue) or "...",
            is_greeting=(t.player_message == ""),
        )
        for t in store.get_npc_turns(req.session_id, req.npc_id)
    ]
    return HistoryResponse(
        session_id=req.session_id,
        npc_id=req.npc_id,
        quest_completed=session.quest_completed,
        turns=turns,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    session = store.get_session(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    npc = NPCS.get(req.npc_id)
    if npc is None:
        raise HTTPException(status_code=400, detail="Unknown npc_id")
    if session.scenario_id and req.npc_id not in SCENARIOS.get(session.scenario_id, ()):
        raise HTTPException(
            status_code=400, detail="npc_id does not belong to this session's scenario"
        )

    player_message = sanitize_player_message(req.message)
    if not player_message:
        raise HTTPException(
            status_code=400, detail="Message is empty after sanitization"
        )

    key = (req.session_id, req.npc_id)
    pending = _pending_turns.get(key)
    if pending is not None:
        try:
            await pending
        except Exception:
            pass

    prior_turns = _history_for_prompt(req.session_id, req.npc_id)
    prompt = build_completion_prompt(
        npc=npc,
        condition=session.condition,
        quest_completed=session.quest_completed,
        prior_turns=prior_turns,
        current_player_message=player_message,
        trace_window_n=config.TRACE_WINDOW_N,
    )
    turn_index = store.next_turn_index(req.session_id, req.npc_id)
    displayed = await _run_streamed_turn(
        session, req.session_id, req.npc_id, prompt, turn_index, player_message
    )
    return ChatResponse(dialogue=displayed, npc_id=req.npc_id, turn_index=turn_index)


@app.post("/quest/complete", response_model=QuestCompleteResponse)
async def quest_complete(req: QuestCompleteRequest) -> QuestCompleteResponse:
    ok = store.set_quest_completed(req.session_id, req.quest_completed)
    if not ok:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    return QuestCompleteResponse(
        session_id=req.session_id, quest_completed=req.quest_completed
    )


@app.post("/session/end", response_model=EndSessionResponse)
async def end_session(req: EndSessionRequest) -> EndSessionResponse:
    session = store.get_session(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session_id")

    is_first_part = session.part_index == 1
    store.end_session(req.session_id, survey_reached=not is_first_part)
    if is_first_part:
        return EndSessionResponse(session_id=req.session_id, survey_url="")
    if session.participant_id:
        survey_url = _pair_survey_url(session.participant_id)
    else:
        survey_url = _survey_url(req.session_id)
    return EndSessionResponse(session_id=req.session_id, survey_url=survey_url)
