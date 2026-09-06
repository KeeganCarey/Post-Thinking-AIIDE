# Abstracted async LLM respond method used for generation

import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI, BadRequestError

logger = logging.getLogger(__name__)

# env variables and Deepseek chat
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
ds_api = os.getenv("DS_API")
nv_api = os.getenv("NV_API")

novita_client = AsyncOpenAI(
    base_url="https://api.novita.ai/openai",
    api_key=nv_api,
)

ds_client = AsyncOpenAI(
    base_url="https://api.deepseek.com",
    api_key=ds_api,
)


async def respond(
    system_prompt: str,
    user_prompt: str,
    stream_status: bool = False,
    max_retries: int = 5,
) -> tuple:
    model = "deepseek-v4-pro"
    stream = stream_status
    max_tokens = 30000
    system = system_prompt
    user = user_prompt
    temperature = 1.0
    top_p = 1.0
    min_p = 0
    presence_penalty = 0
    frequency_penalty = 0
    repetition_penalty = 1.1
    response_format = {"type": "json_object"}

    for attempt in range(1, max_retries + 1):
        try:
            chat_completion_res = await ds_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": system,
                    },
                    {
                        "role": "user",
                        "content": user,
                    },
                ],
                stream=stream,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                response_format=response_format,
                reasoning_effort="high",
                extra_body={
                    "repetition_penalty": repetition_penalty,
                    "min_p": min_p,
                    "thinking": {"type": "enabled"},
                },
            )
            thinking: str = chat_completion_res.choices[0].message.reasoning_content
            response: str = chat_completion_res.choices[0].message.content
            return (thinking, response)
        except BadRequestError as e:
            logger.warning(
                f"400 BadRequestError on attempt {attempt}/{max_retries}: {e}"
            )
            if attempt == max_retries:
                raise
            await asyncio.sleep(2**attempt)
    return ("", "")


def respond_sync(system_prompt: str, user_prompt: str) -> tuple:
    return asyncio.run(respond(system_prompt, user_prompt))
