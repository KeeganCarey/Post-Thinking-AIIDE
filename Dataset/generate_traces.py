# Used to generate post-thinking traces

import asyncio
import csv
import json
import logging
import re
import time
from pathlib import Path

from data_tools import *
from llm import respond
from prompts import Prompts

ROOT = Path(__file__).resolve().parent
JOBS = [
    (
        ROOT / "dataset/extracted/npc_dialogue_800.jsonl",
        ROOT / "dataset/post-thinking/npc_dialogue_post_thinking.csv",
    ),
    (
        ROOT / "dataset/extracted/pippa_400.jsonl",
        ROOT / "dataset/post-thinking/pippa_post_thinking_400.csv",
    ),
    (
        ROOT / "dataset/extracted/rpg_quests_800.jsonl",
        ROOT / "dataset/post-thinking/rpg_quests_post_thinking.csv",
    ),
]

# simple logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("./traces.log")],
)
logger = logging.getLogger(__name__)


# the RATE LIMITER
class RateLimiter:
    def __init__(self, max_per_min: int):
        self.max_rpm = max_per_min
        self._lock = asyncio.Lock()
        self._timestamps: list[float] = []

    async def wait(self):
        async with self._lock:
            now = time.monotonic()
            self._timestamps = [t for t in self._timestamps if now - t < 60]
            if len(self._timestamps) >= self.max_rpm:
                sleep_time = 60 - (now - self._timestamps[0])
                await asyncio.sleep(sleep_time)
            self._timestamps.append(time.monotonic())


def format_response(dialogue: str, post_thinking: str) -> str:
    return f"{dialogue}<post-thinking>{post_thinking.strip()}</post-thinking>"


def extract_name(message: list) -> str:
    first_msg = message[0].get("content", "")
    matched = re.search(r"You are ([^.]+)\.", first_msg)
    return matched.group(1).strip() if matched else "unknown"


async def process_row(
    row: list,
    semaphore: asyncio.Semaphore,
    prompts: Prompts,
    writer,
    limiter: RateLimiter,
    write_lock: asyncio.Lock,
):
    async with semaphore:
        # data = json.loads(row["messages"]) #a list of conversations
        data = row
        results = []
        thinking_traces = []

        name = extract_name(data)

        prompt = ""  # gradually build the prompt for DeepSeek
        for turn in data:
            if turn.get("role") == "system":
                prompt += f"system: {turn.get('content')}\n"
                results.append({"role": "system", "content": turn.get("content")})
                continue
            if turn.get("role") == "user":
                prompt += f"player: {turn.get('content')}\n"
                results.append({"role": "user", "content": turn.get("content")})
                continue
            prompt += f"assistant: {turn.get('content')}\n"

            post_thinking_traces = ""
            thinking = ""
            for attempt in range(1, 4):
                try:
                    await limiter.wait()
                    thinking, response = await respond(
                        prompts.generate_traces(), prompt
                    )
                    parsed = parse_response(response)
                    if parsed is None:
                        raise ValueError("parse_response returned None")
                    post_thinking_traces = parsed.get("post-thinking", "")
                    break
                except Exception as e:
                    if attempt < 3:
                        logger.warning(
                            f"[{name}] attempt {attempt}/3 failed: {e}, retrying"
                        )
                    else:
                        logger.error(f"[{name}] all 3 attempts failed: {e}")
            # append to the actual data list
            results.append(
                {
                    "role": "assistant",
                    "content": format_response(
                        turn.get("content"), post_thinking_traces
                    ),
                }
            )
            thinking_traces.append(thinking)

            # append the post-thinking traces back
            if post_thinking_traces:
                prompt = prompt.rsplit(f"assistant: {turn.get('content')}\n", 1)[0]
                prompt += f"assistant: {turn.get('content')}<post-thinking>{post_thinking_traces}</post-thinking>\n"
        async with write_lock:
            writer.writerow(
                {
                    "original": json.dumps(row),
                    "post-thinking": json.dumps({"messages": results}),
                    "llm_thinking": json.dumps(thinking_traces),
                }
            )
        logger.info(f"Done: {name}")


async def run_job(
    source: Path,
    output: Path,
    prompts: Prompts,
    limiter: RateLimiter,
    semaphore: asyncio.Semaphore,
):
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = read_json(source)
    fieldnames = ["original", "post-thinking", "llm_thinking"]

    already_done: set[str] = set()
    write_mode = "w"
    if output.exists():
        with open(output, newline="") as f:
            for r in csv.DictReader(f):
                already_done.add(r["original"])
            write_mode = "a"
            logger.info(f"{output.name}: resuming, skip {len(already_done)}")

    pending = [r for r in rows if json.dumps(r) not in already_done]
    logger.info(f"{output.name}: {len(pending)} rows")

    with open(output, write_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_mode == "w":
            writer.writeheader()
        write_lock = asyncio.Lock()
        await asyncio.gather(
            *[
                process_row(row, semaphore, prompts, writer, limiter, write_lock)
                for row in pending
            ]
        )


async def main():
    csv.field_size_limit(10_000_000)
    prompts = Prompts()
    limiter = RateLimiter(2000)
    semaphore = asyncio.Semaphore(50)
    for source, output in JOBS:
        await run_job(source, output, prompts, limiter, semaphore)


if __name__ == "__main__":
    asyncio.run(main())
