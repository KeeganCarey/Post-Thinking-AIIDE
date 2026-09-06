# Utility file for some quick functions

import csv
import json
from pathlib import Path


def read_json(path: Path, fieldname: str = "messages") -> list[dict]:
    dialogue = []
    with open(path, "r") as f:
        for row in f:
            data = json.loads(row)
            dialogue.append(data.get(fieldname))
    return dialogue


def write_csv(path: Path, header: list, row: tuple) -> None:
    file_exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


def parse_response(response: str) -> dict[str, str] | None:
    try:
        return json.loads(response)
    except:
        try:
            return json.loads(response[response.index("{") : response.rindex("}") + 1])
        except (json.JSONDecodeError, ValueError):
            print(f"Failed to parse response: {response}")
            return None


def extract_info(response: dict[str, str]) -> tuple[str, str, str]:
    return response.get("name"), response.get("background"), response.get("location")
