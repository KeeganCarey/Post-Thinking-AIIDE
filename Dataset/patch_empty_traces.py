# Used to patch empty traces due to API error, parsing error.etc

import asyncio
import csv
import json
import logging
import re
import time
from pathlib import Path

from data_tools import parse_response
from llm import respond
from prompts import Prompts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("./patch_traces.log")],
)
logger = logging.getLogger(__name__)


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


def extract_dialogue(content: str) -> str:
    return re.sub(r"<post-thinking>.*?</post-thinking>", "", content, flags=re.DOTALL).strip()


def extract_trace(content: str) -> str:
    m = re.search(r"<post-thinking>(.*?)</post-thinking>", content, re.DOTALL)
    return m.group(1) if m else ""


def has_empty_trace(content: str) -> bool:
    return "<post-thinking></post-thinking>" in content


async def patch_row(row: dict, prompts: Prompts, limiter: RateLimiter) -> dict:
    original_messages = json.loads(row["original"])
    pt_data = json.loads(row["post-thinking"])
    pt_messages = pt_data["messages"]
    llm_thinking = json.loads(row["llm_thinking"])

    name = "unknown"
    for msg in original_messages:
        if msg.get("role") == "system":
            m = re.search(r"You are ([^.]+)\.", msg.get("content", ""))
            if m:
                name = m.group(1).strip()
            break

    prompt = ""
    new_messages = []
    new_thinking = list(llm_thinking)
    asst_idx = 0

    for pt_msg in pt_messages:
        role = pt_msg["role"]
        content = pt_msg["content"]

        if role == "system":
            prompt += f"system: {content}\n"
            new_messages.append(pt_msg)
        elif role == "user":
            prompt += f"player: {content}\n"
            new_messages.append(pt_msg)
        elif role == "assistant":
            dialogue = extract_dialogue(content)
            existing_trace = extract_trace(content)

            if has_empty_trace(content):
                prompt += f"assistant: {dialogue}\n"
                new_trace = ""
                new_llm_thinking = ""
                for attempt in range(1, 4):
                    try:
                        await limiter.wait()
                        thinking, response = await respond(prompts.generate_traces(), prompt)
                        parsed = parse_response(response)
                        if parsed is None:
                            raise ValueError("parse_response returned None")
                        new_trace = parsed.get("post-thinking", "")
                        new_llm_thinking = thinking
                        break
                    except Exception as e:
                        if attempt < 3:
                            logger.warning(f"[{name}] attempt {attempt}/3 failed: {e}, retrying")
                        else:
                            logger.error(f"[{name}] all 3 attempts failed: {e}")

                new_messages.append({"role": "assistant", "content": format_response(dialogue, new_trace)})
                if asst_idx < len(new_thinking):
                    new_thinking[asst_idx] = new_llm_thinking
                else:
                    new_thinking.append(new_llm_thinking)

                if new_trace:
                    prompt = prompt.rsplit(f"assistant: {dialogue}\n", 1)[0]
                    prompt += f"assistant: {dialogue}<post-thinking>{new_trace}</post-thinking>\n"

                logger.info(f"[{name}] patched asst turn {asst_idx + 1}")
            else:
                prompt += f"assistant: {dialogue}<post-thinking>{existing_trace}</post-thinking>\n"
                new_messages.append(pt_msg)

            asst_idx += 1

    return {
        "original": row["original"],
        "post-thinking": json.dumps({"messages": new_messages}),
        "llm_thinking": json.dumps(new_thinking),
    }


async def patch_file(path: Path, semaphore: asyncio.Semaphore, prompts: Prompts, limiter: RateLimiter):
    csv.field_size_limit(10_000_000)

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    needs_patch = [i for i, r in enumerate(rows) if "<post-thinking></post-thinking>" in r.get("post-thinking", "")]
    logger.info(f"{path.name}: {len(needs_patch)} rows to patch")

    if not needs_patch:
        return

    # Backup before overwriting
    backup = path.with_suffix(".csv.bak")
    backup.write_bytes(path.read_bytes())
    logger.info(f"Backed up to {backup.name}")

    async def patch_one(idx: int):
        async with semaphore:
            rows[idx] = await patch_row(rows[idx], prompts, limiter)
            logger.info(f"{path.name}: row {idx + 1} done")

    await asyncio.gather(*[patch_one(i) for i in needs_patch])

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["original", "post-thinking", "llm_thinking"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"{path.name}: written ({len(needs_patch)} rows patched)")


async def main():
    base = Path(__file__).resolve().parent / "dataset/post-thinking"
    targets = [
        base / "npc_dialogue_post_thinking.csv",
        base / "pippa_post_thinking_400.csv",
        base / "rpg_quests_post_thinking.csv",
    ]

    prompts = Prompts()
    limiter = RateLimiter(500)
    semaphore = asyncio.Semaphore(50)

    for target in targets:
        await patch_file(target, semaphore, prompts, limiter)


if __name__ == "__main__":
    asyncio.run(main())
