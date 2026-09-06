#!/usr/bin/env bash
set -euo pipefail

command -v llama-server >/dev/null 2>&1 || { echo "llama-server is not on PATH" >&2; exit 1; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="$ROOT/models"

case "${1:-}" in
  --no)   port=8081; gguf="Gemma3-4B-no-thinking-q8.gguf" ;;
  --post) port=8083; gguf="Gemma3-4B-post-thinking-q8.gguf" ;;
  --pre)  port=8082; gguf="Gemma3-4B-pre-thinking-q8.gguf" ;;
  *) echo "usage: $0 --no | --post | --pre" >&2; exit 1 ;;
esac

exec llama-server \
  --model "$MODELS/$gguf" \
  --host 127.0.0.1 --port "$port" \
  --n-gpu-layers 999 --ctx-size 8192 --parallel 4 --cont-batching
