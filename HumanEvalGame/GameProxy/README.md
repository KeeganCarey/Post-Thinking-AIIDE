# GameProxy

FastAPI proxy for the Unity WebGL eval game. Unity never sees conditions or traces.

Within-subjects: each participant plays **tavern** and **village**, one under `no_thinking` and one under `post_thinking`. Four counterbalance cells. 

Trace window `n=3`. Spoken line returns at the `<post-thinking>` opener; the trace finishes in the background.

## Run

From the repo root: `uv sync`. llama-servers: `../setup_llamaserver.sh` (see `model.md`). Then from `HumanEvalGame/`:

```bash
./run_proxy.sh          # or ./run_proxy.sh --mock
ngrok http 8000
```

Point Unity at the ngrok URL (`postthink_config.example.json`).

## API

| Method | Path | Notes |
|---|---|---|
| POST | `/pair` | enrolls a participant; two parts (`scenario_id`, `session_id`, `starting_npc`). No conditions in the response. |
| POST | `/session` | single session |
| POST | `/greeting` | NPC opening line |
| POST | `/history` | restore conversation |
| POST | `/chat` | `{session_id, npc_id, message}` → `{dialogue, npc_id, turn_index}` |
| POST | `/quest/complete` | hunt / raid done |
| POST | `/session/end` | end a part |
| GET | `/health` | |

Tavern NPCs: `maid` / `hunter` / `bartender`. Village: `keeper` / `guard` / `trader`. `/chat` rejects an NPC from the other scenario.

Logs (gitignored, for obvious reasons): `game_sessions.sqlite3`, `turn_logs.jsonl`.
