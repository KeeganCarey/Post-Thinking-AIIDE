# UnityClient

Unity project for the WebGL eval game. Open this folder in Unity 6 / 2022 LTS, WebGL target. Playable scene: `Assets/Scenes/PostThinkingEval.unity`.

Two scenarios: tavern (Mirela, Bran, Alden) and village (Odila, Tomas, Gunnar). Hunt / raid buttons appear after talking to the introducer and the advisor, then call `/quest/complete`.

## Config

Copy `Assets/StreamingAssets/postthink_config.example.json` to `postthink_config.json` and set `apiBaseUrl` to the proxy (ngrok HTTPS for a WebGL build). That file is gitignored, for obvious reasons.

The client only talks to the FastAPI proxy, never llama-server. `TagSanitizer` strips leftover tags.

## Scripts

- `Assets/Scripts/PostThinkRP/` — runtime (API, session, dialogue, quest, player)
- `Assets/Editor/` — `PostThink-RP > Generate Evaluation Scene` if you need a fresh primitive scene

Art packs and licenses: `ThirdPartyAssets/README.md`.
