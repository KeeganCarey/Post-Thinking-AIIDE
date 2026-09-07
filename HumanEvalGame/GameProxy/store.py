from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class SessionRecord:
    session_id: str
    condition: str
    quest_completed: bool
    started_at: str
    ended_at: str | None
    starting_npc: str
    survey_reached: bool
    scenario_id: str | None = None
    participant_id: str | None = None
    part_index: int | None = None


@dataclass(frozen=True)
class TurnRecord:
    turn_index: int
    player_message: str
    displayed_dialogue: str
    raw_model_output: str
    extracted_trace: str | None
    think_block: str | None


class GameStore:
    def __init__(self, db_path: Path, jsonl_path: Path | None = None):
        self.db_path = Path(db_path)
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        self._lock = threading.RLock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    condition TEXT NOT NULL,
                    quest_completed INTEGER NOT NULL DEFAULT 0,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    starting_npc TEXT NOT NULL,
                    survey_reached INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    condition TEXT NOT NULL,
                    npc_id TEXT NOT NULL,
                    quest_completed INTEGER NOT NULL,
                    turn_index INTEGER NOT NULL,
                    player_message TEXT NOT NULL,
                    raw_model_output TEXT NOT NULL,
                    displayed_dialogue TEXT NOT NULL,
                    extracted_trace TEXT,
                    think_block TEXT,
                    parse_flags TEXT NOT NULL,
                    request_sent_ts TEXT NOT NULL,
                    response_complete_ts TEXT NOT NULL,
                    model_latency_ms INTEGER NOT NULL,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_turns_session_npc ON turns(session_id, npc_id, turn_index)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS participants (
                    participant_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    cell INTEGER NOT NULL
                )
                """
            )
            turn_cols = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(turns)").fetchall()
            }
            if "dialogue_latency_ms" not in turn_cols:
                conn.execute("ALTER TABLE turns ADD COLUMN dialogue_latency_ms INTEGER")
            session_cols = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
            }
            for column, decl in (
                ("scenario_id", "TEXT"),
                ("participant_id", "TEXT"),
                ("part_index", "INTEGER"),
            ):
                if column not in session_cols:
                    conn.execute(f"ALTER TABLE sessions ADD COLUMN {column} {decl}")
            conn.commit()

    def create_session(
        self,
        session_id: str,
        condition: str,
        starting_npc: str,
        scenario_id: str | None = None,
        participant_id: str | None = None,
        part_index: int | None = None,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions(
                    session_id, condition, quest_completed, started_at, starting_npc,
                    scenario_id, participant_id, part_index
                )
                VALUES (?, ?, 0, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    condition,
                    utc_now_iso(),
                    starting_npc,
                    scenario_id,
                    participant_id,
                    part_index,
                ),
            )
            conn.commit()

    def create_participant(self, participant_id: str, cell: int) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO participants(participant_id, created_at, cell) VALUES (?, ?, ?)",
                (participant_id, utc_now_iso(), cell),
            )
            conn.commit()

    def count_participants_by_cell(self, n_cells: int) -> dict[int, int]:
        counts = {cell: 0 for cell in range(n_cells)}
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT cell, COUNT(*) AS n FROM participants GROUP BY cell"
            ).fetchall()
        for row in rows:
            if row["cell"] in counts:
                counts[int(row["cell"])] = int(row["n"])
        return counts

    def get_participant_session_ids(self, participant_id: str) -> list[tuple[int, str]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT part_index, session_id FROM sessions
                WHERE participant_id = ?
                ORDER BY part_index ASC
                """,
                (participant_id,),
            ).fetchall()
        return [(int(row["part_index"]), row["session_id"]) for row in rows]

    def get_session(self, session_id: str) -> SessionRecord | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        part_index = row["part_index"]
        return SessionRecord(
            session_id=row["session_id"],
            condition=row["condition"],
            quest_completed=bool(row["quest_completed"]),
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            starting_npc=row["starting_npc"],
            survey_reached=bool(row["survey_reached"]),
            scenario_id=row["scenario_id"],
            participant_id=row["participant_id"],
            part_index=int(part_index) if part_index is not None else None,
        )

    def count_sessions_by_condition(
        self, conditions: tuple[str, ...]
    ) -> dict[str, int]:
        counts = {condition: 0 for condition in conditions}
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT condition, COUNT(*) AS n FROM sessions GROUP BY condition"
            ).fetchall()
        for row in rows:
            if row["condition"] in counts:
                counts[row["condition"]] = int(row["n"])
        return counts

    def set_quest_completed(self, session_id: str, completed: bool = True) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "UPDATE sessions SET quest_completed = ? WHERE session_id = ?",
                (1 if completed else 0, session_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def end_session(self, session_id: str, survey_reached: bool = True) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE sessions
                SET ended_at = COALESCE(ended_at, ?), survey_reached = ?
                WHERE session_id = ?
                """,
                (utc_now_iso(), 1 if survey_reached else 0, session_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def next_turn_index(self, session_id: str, npc_id: str) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT COALESCE(MAX(turn_index), 0) + 1 AS next_index
                FROM turns WHERE session_id = ? AND npc_id = ?
                """,
                (session_id, npc_id),
            ).fetchone()
        return int(row["next_index"])

    def get_npc_turns(self, session_id: str, npc_id: str) -> list[TurnRecord]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT turn_index, player_message, displayed_dialogue, raw_model_output,
                       extracted_trace, think_block
                FROM turns
                WHERE session_id = ? AND npc_id = ?
                ORDER BY turn_index ASC
                """,
                (session_id, npc_id),
            ).fetchall()
        return [
            TurnRecord(
                turn_index=int(row["turn_index"]),
                player_message=row["player_message"],
                displayed_dialogue=row["displayed_dialogue"],
                raw_model_output=row["raw_model_output"],
                extracted_trace=row["extracted_trace"],
                think_block=row["think_block"],
            )
            for row in rows
        ]

    def insert_turn(self, row: dict) -> None:
        parse_flags = json.dumps(row.get("parse_flags", []), ensure_ascii=False)
        values = (
            row["session_id"],
            row["condition"],
            row["npc_id"],
            1 if row["quest_completed"] else 0,
            row["turn_index"],
            row["player_message"],
            row["raw_model_output"],
            row["displayed_dialogue"],
            row.get("extracted_trace"),
            row.get("think_block"),
            parse_flags,
            row["request_sent_ts"],
            row["response_complete_ts"],
            row["model_latency_ms"],
            row.get("prompt_tokens"),
            row.get("completion_tokens"),
            row.get("dialogue_latency_ms"),
        )
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO turns(
                    session_id, condition, npc_id, quest_completed, turn_index,
                    player_message, raw_model_output, displayed_dialogue,
                    extracted_trace, think_block, parse_flags, request_sent_ts,
                    response_complete_ts, model_latency_ms, prompt_tokens, completion_tokens,
                    dialogue_latency_ms
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            conn.commit()
        self._append_jsonl(row | {"parse_flags": row.get("parse_flags", [])})

    def _append_jsonl(self, row: dict) -> None:
        if self.jsonl_path is None:
            return
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(row, ensure_ascii=False)
        with self._lock, open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
