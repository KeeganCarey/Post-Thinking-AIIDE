# Used to generate thinking traces

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
SOURCE = ROOT / "NPC-2k.jsonl"
OUTPUT = ROOT / "Thinking.csv"

#simple logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("./traces.log")],
)
logger = logging.getLogger(__name__)

#the RATE LIMITER
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

# now its the <think> tokens first
def format_response(dialogue: str, thinking: str) -> str:
    return f"<think>{thinking.strip()}</think>{dialogue}"


def extract_name(message: list) -> str:
    first_msg = message[0].get("content", "")
    matched = re.search(r"You are ([^.]+)\.", first_msg)
    return matched.group(1).strip() if matched else "unknown"


async def process_row(
    row: list,
    semaphore:asyncio.Semaphore,
    prompts:Prompts,
    writer,
    limiter: RateLimiter,
    write_lock: asyncio.Lock
):
    async with semaphore:
        #data = json.loads(row["messages"]) #a list of conversations
        data = row
        results = []
        thinking_traces = []

        name = extract_name(data)

        sys_prompt = prompts.generate_thinking()
        prompt = "" # gradually build the prompt for DeepSeek

        for turn in data:
            if turn.get("role") == "system":
                prompt += f"system: {turn.get('content')}\n"
                results.append({"role": "system", "content": turn.get("content")})
                continue
            if turn.get("role") == "user":
                prompt += f"player: {turn.get("content")}\n"
                results.append({"role": "user", "content": turn.get("content")})
                continue
            prompt += f"NPC: {turn.get("content")}\n"

            user_prompt = f"{sys_prompt}\n Given Dialogue: {prompt}"

            # the actual generated thinking traces
            thinking = ""
            # DeepSeek's thinking traces
            thinking_trace = ""
            for attempt in range(1, 4):
                try:
                    await limiter.wait()
                    thinking_trace, response = await respond(sys_prompt, user_prompt)
                    #print(f"\n\nprompt: {user_prompt}\n\n")
                    #print(f"\n\nthinking: {thinking_trace}\n\n")
                    #print(f"\n\nresponse: {response}\n\n")
                    parsed = parse_response(response)
                    if parsed is None:
                        raise ValueError("parse_response returned None")
                    thinking = parsed.get("thinking", "")
                    break
                except Exception as e:
                    if attempt < 3:
                        logger.warning(f"[{name}] attempt {attempt}/3 failed: {e}, retrying")
                    else:
                        logger.error(f"[{name}] all 3 attempts failed: {e}")
            # append to the actual data list
            results.append(
                {
                    "role": "assistant",
                    "content": format_response(
                        turn.get("content"), thinking
                    )
                }
            )
            thinking_traces.append(thinking_trace)

            #prepend the thinking traces back
            if thinking:
                prompt = prompt.rsplit(f"NPC: {turn.get("content")}\n", 1)[0]
                prompt += f"NPC: [NPC_reasoning: {thinking}] {turn.get("content")}\n"
        async with write_lock:
            writer.writerow(
                {
                    "original": json.dumps(row),
                    "thinking": json.dumps({"messages": results}),
                    "llm_thinking": json.dumps(thinking_traces),
                }
            )
        logger.info(f"Done: {name}")

async def main():
    source = SOURCE
    output = OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)

    csv.field_size_limit(10_000_000)
    prompts = Prompts()
    limiter = RateLimiter(500)
    semaphore = asyncio.Semaphore(250)

    rows = read_json(source)
    fieldnames = ["original", "thinking", "llm_thinking"]

    already_done: set[str] = set()
    write_mode = "w"
    if output.exists():
        with open(output, newline="") as f:
            for r in csv.DictReader(f):
                already_done.add(r["original"])
            write_mode = "a"
            logger.info(f"Resuming - skipping {len(already_done)} rows")

    pending = [r for r in rows if json.dumps(r) not in already_done]
    logger.info(f"Processing {len(pending)} rows")

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

if __name__=="__main__":
    asyncio.run(main())

