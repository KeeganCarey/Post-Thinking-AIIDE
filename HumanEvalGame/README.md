# HumanEvalGame

Unity WebGL client + FastAPI proxy for the Unity Game demo.

Within-subjects: each participant plays the **tavern** and **village** scenes, one under `no_thinking` and one under `post_thinking`. Conditions and traces stay on the proxy; Unity never sees them. Trace window `n=3`.

## Layout

- `GameProxy/` — FastAPI server (`run_proxy.sh`)
- `UnityClient/` — Unity project (`Scenes/PostThinkingEval.unity`)
- `UnityClient/ThirdPartyAssets/README.md` — SOI / Quaternius / people licenses
- `model.md` — llama-server
- `analyze_pair_survey.py` — survey vs session log

## Run

From the repo root: `uv sync`. Then from `HumanEvalGame/`:

```bash
./run_proxy.sh --mock          # no llama-server
./run_proxy.sh                 # real models on :8081 (no) and :8083 (post)
ngrok http 8000
```

Copy `UnityClient/Assets/StreamingAssets/postthink_config.example.json` to `postthink_config.json` and set `apiBaseUrl` to the ngrok HTTPS URL (that file is gitignored).
