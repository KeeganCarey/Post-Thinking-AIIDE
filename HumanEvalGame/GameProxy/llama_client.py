from __future__ import annotations

import asyncio
import itertools
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx

from . import config
from .config import (
    CONDITION_NO,
    CONDITION_POST,
    CONDITION_PRE,
    GEMMA_END,
    GEMMA_EOS,
    POST_OPEN,
)


@dataclass(frozen=True)
class LlamaResult:
    raw_text: str
    prompt_tokens: int | None
    completion_tokens: int | None


@dataclass(frozen=True)
class LlamaChunk:
    text: str
    done: bool
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    truncated: bool = False


_mock_counter = itertools.count(1)


def _completion_url(condition: str) -> str:
    base = config.LLAMA_ENDPOINTS[condition].rstrip("/")
    path = config.LLAMA_COMPLETION_PATH
    if not path.startswith("/"):
        path = "/" + path
    return base + path


def _extract_text(payload: dict) -> str:
    if isinstance(payload.get("content"), str):
        return payload["content"]
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            if isinstance(first.get("text"), str):
                return first["text"]
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
    return ""


def _extract_prompt_tokens(payload: dict) -> int | None:
    for key in ("tokens_evaluated", "prompt_tokens"):
        value = payload.get(key)
        if isinstance(value, int):
            return value
    usage = payload.get("usage")
    if isinstance(usage, dict) and isinstance(usage.get("prompt_tokens"), int):
        return usage["prompt_tokens"]
    return None


def _extract_completion_tokens(payload: dict) -> int | None:
    for key in ("tokens_predicted", "completion_tokens"):
        value = payload.get(key)
        if isinstance(value, int):
            return value
    usage = payload.get("usage")
    if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
        return usage["completion_tokens"]
    return None


def _mock_completion(condition: str) -> LlamaResult:
    idx = next(_mock_counter)
    dialogue = (
        "Aye, I hear you. The wolf trouble has everyone speaking in lowered "
        f"voices tonight, and I will remember what you asked. ({idx})"
    )
    if condition == CONDITION_PRE:
        raw = f"<think>I should answer plainly and keep the village trouble in mind.</think>{dialogue}"
    elif condition == CONDITION_POST:
        raw = (
            f"{dialogue}<post-thinking>I answered in a grounded village voice and "
            "should remember the traveler is following the wolf trouble through "
            "the tavern conversations.</post-thinking>"
        )
    else:
        raw = dialogue
    return LlamaResult(raw, None, None)


async def complete(condition: str, prompt: str) -> LlamaResult:
    if config.MOCK_LLM:
        return _mock_completion(condition)

    body = {
        "prompt": prompt,
        "n_predict": config.MAX_TOKENS,
        "temperature": config.TEMPERATURE,
        "top_p": config.TOP_P,
        "top_k": config.TOP_K,
        "min_p": config.MIN_P,
        "repeat_penalty": config.REPEAT_PENALTY,
        "stop": [GEMMA_END, GEMMA_EOS],
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    if config.LLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {config.LLAMA_API_KEY}"

    async with httpx.AsyncClient(timeout=config.LLAMA_TIMEOUT_S) as client:
        res = await client.post(_completion_url(condition), json=body, headers=headers)
        res.raise_for_status()
        payload = res.json()
    return LlamaResult(
        raw_text=_extract_text(payload),
        prompt_tokens=_extract_prompt_tokens(payload),
        completion_tokens=_extract_completion_tokens(payload),
    )


async def _mock_stream(condition: str) -> AsyncIterator[LlamaChunk]:
    raw = _mock_completion(condition).raw_text
    open_idx = raw.find(POST_OPEN)
    if open_idx != -1:
        yield LlamaChunk(text=raw[:open_idx], done=False)
        yield LlamaChunk(text=POST_OPEN, done=False)
        await asyncio.sleep(config.MOCK_TRACE_DELAY_S)
        yield LlamaChunk(text=raw[open_idx + len(POST_OPEN) :], done=False)
    else:
        mid = max(1, len(raw) // 2)
        yield LlamaChunk(text=raw[:mid], done=False)
        await asyncio.sleep(0)
        yield LlamaChunk(text=raw[mid:], done=False)
    yield LlamaChunk(text="", done=True)


def _parse_sse_line(line: str) -> dict | None:
    line = line.strip()
    if not line:
        return None
    if line.startswith("data:"):
        line = line[len("data:") :].strip()
    if not line or line == "[DONE]":
        return None
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


async def stream_complete(condition: str, prompt: str) -> AsyncIterator[LlamaChunk]:
    if config.MOCK_LLM:
        async for chunk in _mock_stream(condition):
            yield chunk
        return

    body = {
        "prompt": prompt,
        "n_predict": config.MAX_TOKENS,
        "temperature": config.TEMPERATURE,
        "top_p": config.TOP_P,
        "top_k": config.TOP_K,
        "min_p": config.MIN_P,
        "repeat_penalty": config.REPEAT_PENALTY,
        "stop": [GEMMA_END, GEMMA_EOS],
        "stream": True,
    }
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if config.LLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {config.LLAMA_API_KEY}"

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    truncated = False
    async with httpx.AsyncClient(timeout=config.LLAMA_TIMEOUT_S) as client:
        async with client.stream(
            "POST", _completion_url(condition), json=body, headers=headers
        ) as res:
            res.raise_for_status()
            async for line in res.aiter_lines():
                payload = _parse_sse_line(line)
                if payload is None:
                    continue
                text = payload.get("content")
                if not isinstance(text, str):
                    text = _extract_text(payload)
                if text:
                    yield LlamaChunk(text=text, done=False)
                if payload.get("stop") is True:
                    prompt_tokens = _extract_prompt_tokens(payload)
                    completion_tokens = _extract_completion_tokens(payload)
                    truncated = (
                        bool(payload.get("stopped_limit"))
                        or payload.get("stop_type") == "limit"
                    )
    yield LlamaChunk(
        text="",
        done=True,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        truncated=truncated,
    )
