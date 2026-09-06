# Serving models

```text
Unity WebGL → ngrok HTTPS → FastAPI proxy :8000 → llama-server (one GGUF per condition)
```

The proxy builds the Gemma3 prompt and calls `/completion`. Do not use `/v1/chat/completions`. The paper used **two** servers (`pre_thinking` is off).

| Condition | Port | Model |
|---|---|---|
| `no_thinking` | 8081 | `models/Gemma3-4B-no-thinking-q8.gguf` |
| `post_thinking` | 8083 | `models/Gemma3-4B-post-thinking-q8.gguf` |
| `pre_thinking` | 8082 | optional |

Download GGUFs with repo-root `setup.sh`. Sampling lives in `GameProxy/config.py` (same across conditions). Trace window `n=3`.

```bash
llama-server --model models/Gemma3-4B-no-thinking-q8.gguf \
  --host 127.0.0.1 --port 8081 --n-gpu-layers 999 --ctx-size 8192 --parallel 4 --cont-batching

llama-server --model models/Gemma3-4B-post-thinking-q8.gguf \
  --host 127.0.0.1 --port 8083 --n-gpu-layers 999 --ctx-size 8192 --parallel 4 --cont-batching
```

Then from `HumanEvalGame/`:

```bash
./run_proxy.sh                 # or ./run_proxy.sh --mock
ngrok http 8000
curl -s http://127.0.0.1:8000/health
```

Point Unity at the ngrok URL (`postthink_config.json`).
