#!/bin/bash

command -v hf >/dev/null 2>&1 || { echo "hf cli is required" >&2; exit 1; }

echo "starting GGUF model downloads"

hf download hf://chimbiwide/Gemma3NPC-4B-post-thinking-GGUF/Gemma3-4B-post-thinking-Q8_0.gguf --local-dir ./models
hf download hf://chimbiwide/Gemma3NPC-4B-pre-thinking-GGUF/Gemma3-4B-pre-thinking-Q8_0.gguf --local-dir ./models
hf download hf://chimbiwide/Gemma3NPC-4B-no-thinking-GGUF/Gemma3-4B-no-thinking-Q8_0.gguf --local-dir ./models

echo "starting dataset downloads"

hf download chimbiwide/NPC-RP-No-Thinking --repo-type dataset --local-dir ./Dataset/hf/NPC-RP-No-Thinking
hf download chimbiwide/NPC-RP-Pre-Thinking --repo-type dataset --local-dir ./Dataset/hf/NPC-RP-Pre-Thinking
hf download chimbiwide/NPC-RP-Post-Thinking --repo-type dataset --local-dir ./Dataset/hf/NPC-RP-Post-Thinking
