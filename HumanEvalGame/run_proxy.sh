#!/usr/bin/env bash
# Usage: ./run_proxy.sh [--mock]
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" == "--mock" ]]; then
  export POSTTHINK_MOCK_LLM=1
  shift
fi

exec uv run uvicorn GameProxy.server:app --host 0.0.0.0 --port 8000 "$@"
