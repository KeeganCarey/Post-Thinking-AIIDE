# Used to standardize PIPPA's system prompts

import csv
import json
import logging
from pathlib import Path


from data_tools import *
from llm import respond_sync
from openai import BadRequestError
from prompts import Prompts

logging.basicConfig(
    filename="DS.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def count_completed(output_path: str) -> int:
    p = Path(output_path)
    if not p.exists():
        return 0
    csv.field_size_limit(10_000_000)
    with open(p, newline="") as f:
        return sum(1 for _ in csv.reader(f)) - 1  # subtract header


def main():
    root = Path(__file__).resolve().parent
    data: list = read_json(root / "dataset/extracted/pippa_filtered_800.jsonl")
    data_length: int = len(data)
    prompt = Prompts()
    output_path = str(root / "dataset/cleaned/pippa_system.csv")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    completed = count_completed(output_path)
    if completed > 0:
        logger.info(f"Resuming from item {completed + 1} / {data_length} ({completed} already done)")
    data = data[completed:]

    for i, item in enumerate(data, completed + 1):
        logger.info(f"Processing item {i} / {data_length}")
        ds_system_prompt = prompt.standardize()
        user_input = prompt.user(item)
        try:
            (thinking, response) = respond_sync(ds_system_prompt, user_input)
        except BadRequestError as e:
            logger.error(
                f"Skipping item {i} / {data_length} after all retries — 400 BadRequestError: {e}"
            )
            continue
        parsed = parse_response(response)
        if parsed is None:
            logger.warning(f"Skipping item {i} / {data_length} due to parse failure.")
            continue
        name, background, location = extract_info(parsed)
        logger.info(f"Done {i} / {data_length} - character: {name}")
        write_csv(
            Path(output_path),
            [
                "name",
                "background",
                "location",
                "thinking",
                "systemPrompt",
                "sourceConversation",
            ],
            (
                name,
                background,
                location,
                thinking,
                prompt.system_prompt(name, background, location),
                json.dumps(item),
            ),
        )


if __name__ == "__main__":
    main()
