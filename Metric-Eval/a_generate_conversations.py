import gc
import hashlib
import logging
import os
import re
import time
from pathlib import Path

import eval_lib
from dotenv import load_dotenv
from eval_lib import (
    append_jsonl,
    apply_trace_window,
    build_turn1_prompt,
    load_character_cards,
    parse_model_turn,
    read_jsonl,
)
from openai import OpenAI
from tqdm import tqdm

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
CARDS_PATH = EVAL_DIR / "Eval-45.jsonl"
RUNS_DIR = EVAL_DIR / "runs"
MODELS_DIR = REPO_ROOT / "models"

CONDITIONS = eval_lib.CONDITIONS
CONTEXT_LENGTH = eval_lib.CONTEXT_LENGTH
WINDOW_N = eval_lib.WINDOW_N

MODEL_PATHS = {
    "no_thinking": MODELS_DIR / "Gemma3-4B-no-thinking-Q8_0.gguf",
    "pre_thinking": MODELS_DIR / "Gemma3-4B-pre-thinking-Q8_0.gguf",
    "post_thinking": MODELS_DIR / "Gemma3-4B-post-thinking-Q8_0.gguf",
}

TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 64
MIN_P = 0.0
MAX_TOKENS_NPC = 512
SEED = 17

MODEL_TURNS = 20

TOKENS_PER_MESSAGE_OVERHEAD = 8

DS_BASE_URL = "https://api.deepseek.com"
DS_ENV_VAR = "DS_API"
DS_MODEL = "deepseek-v4-flash"
DS_TEMPERATURE = 1.0
MAX_TOKENS_USER = 200
DS_MAX_RETRIES = 5

DISPOSITIONS = ("cautious", "eager", "skeptical", "talkative")
DISPOSITION_NOTES = {
    "cautious": "cautious - wary of strangers, you weigh risks before committing and probe for hidden motives",
    "eager": "eager - enthusiastic and quick to volunteer, you push the conversation toward action",
    "skeptical": "skeptical - doubtful of grand claims, you question details and ask for proof",
    "talkative": "talkative - chatty and friendly, you share small observations and keep the exchange lively",
}

PLAYER_PERSONA_PROMPT = """You are the player character in a video game, in a spoken conversation with a non-player character (NPC).

{scene}

Your disposition for this conversation: {disposition}.

Stay fully immersed in the fiction:
- React to what {name} just said; pick up on the details they mention.
- Be curious: ask about their quest, their world, their past, and their motives when it fits the moment.
- Make in-character decisions: accept or refuse offers, bargain, choose a course of action, take sides.
- Vary your engagement from turn to turn: sometimes probe deeper, sometimes answer briefly, sometimes steer toward a different aspect of the scene.
- Reply with 1-3 sentences of plain spoken dialogue only: no stage directions, no asterisks, no quotation marks around the line, no narration of actions or feelings.
- Never break character and never refer to games, AI, models, prompts, or roleplay. No meta-commentary of any kind.

You speak only as the player. Output only your next line of dialogue."""


class UserSimError(RuntimeError):
    pass


_LOCATION_RE = re.compile(
    r"Current Location:\s*(.*?)\s*(?:Quest:|Roleplaying Instructions:)", re.DOTALL
)
_QUEST_RE = re.compile(r"Quest:\s*(.*?)\s*Roleplaying Instructions:", re.DOTALL)


def extract_scene(card_text: str) -> tuple[str, str]:
    card_text = "" if card_text is None else str(card_text)
    loc = _LOCATION_RE.search(card_text)
    quest = _QUEST_RE.search(card_text)
    return (
        loc.group(1).strip() if loc else "",
        quest.group(1).strip() if quest else "",
    )


def pick_disposition(character_id: str, dispositions=DISPOSITIONS) -> str:
    digest = hashlib.md5(character_id.encode("utf-8")).hexdigest()
    return dispositions[int(digest, 16) % len(dispositions)]


def make_conversation_id(condition: str, character_id: str) -> str:
    return f"{condition}__{character_id}"


def build_persona_prompt(
    card,
    disposition: str,
    template: str = PLAYER_PERSONA_PROMPT,
    notes=DISPOSITION_NOTES,
) -> str:
    location, quest = extract_scene(card.card_text)
    scene = f"Scene: you are speaking with {card.name}."
    if location:
        scene += f" Where this takes place: {location}"
    if quest:
        scene += f" The matter at hand: {quest}"
    return template.format(
        name=card.name, scene=scene, disposition=notes.get(disposition, disposition)
    )


def build_npc_messages(
    turn1_prompt: str,
    model_turn_raws: list[str],
    user_texts: list[str],
    condition: str,
    window_n: int = WINDOW_N,
) -> list[dict]:
    windowed = apply_trace_window(model_turn_raws, condition, window_n)
    messages = [{"role": "user", "content": turn1_prompt}]
    for i, raw in enumerate(windowed):
        messages.append({"role": "assistant", "content": raw})
        if i < len(user_texts):
            messages.append({"role": "user", "content": user_texts[i]})
    return messages


def build_user_sim_messages(
    persona_prompt: str, npc_dialogues: list[str], player_replies: list[str]
) -> list[dict]:
    messages = [{"role": "system", "content": persona_prompt}]
    for i, dialogue in enumerate(npc_dialogues):
        messages.append({"role": "user", "content": dialogue})
        if i < len(player_replies):
            messages.append({"role": "assistant", "content": player_replies[i]})
    return messages


def make_token_counter(llama, overhead: int = TOKENS_PER_MESSAGE_OVERHEAD):
    def count(messages: list[dict]) -> int:
        text = "".join(m["content"] for m in messages)
        n_tokens = len(
            llama.tokenize(text.encode("utf-8"), add_bos=True, special=False)
        )
        return n_tokens + overhead * len(messages)

    return count


def enforce_context_budget(
    messages: list[dict],
    count_tokens,
    max_tokens_npc: int,
    context_length: int = CONTEXT_LENGTH,
) -> tuple[list[dict], int]:
    messages = list(messages)
    dropped = 0
    while (
        len(messages) >= 5 and count_tokens(messages) + max_tokens_npc > context_length
    ):
        del messages[1:3]
        dropped += 1
    if count_tokens(messages) + max_tokens_npc > context_length:
        logger.warning(
            "prompt still over context budget after dropping all droppable pairs"
        )
    return messages, dropped


def deepseek_user_reply(
    ds_client,
    messages: list[dict],
    *,
    ds_model: str = DS_MODEL,
    temperature: float = DS_TEMPERATURE,
    max_tokens: int = MAX_TOKENS_USER,
    max_retries: int = DS_MAX_RETRIES,
) -> str:
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            res = ds_client.chat.completions.create(
                model=ds_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = res.choices[0].message.content
            if not content or not content.strip():
                raise ValueError("empty user-sim reply")
            return content.strip()
        except Exception as e:
            last_error = e
            logger.warning(
                f"DeepSeek user-sim attempt {attempt}/{max_retries} failed: {e}"
            )
            if attempt < max_retries:
                time.sleep(2**attempt)
    raise UserSimError(str(last_error))


def run_conversation(
    llama,
    ds_client,
    card,
    condition: str,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    max_tokens_npc: int,
    seed: int,
    context_length: int,
    window_n: int,
    model_turns: int,
    ds_model: str,
    ds_temperature: float,
    max_tokens_user: int,
    ds_max_retries: int,
    model_path,
    dispositions=DISPOSITIONS,
    persona_template: str = PLAYER_PERSONA_PROMPT,
    disposition_notes=DISPOSITION_NOTES,
    token_overhead: int = TOKENS_PER_MESSAGE_OVERHEAD,
) -> tuple[dict, dict]:
    turn1_prompt = build_turn1_prompt(card, condition)
    disposition = pick_disposition(card.character_id, dispositions)
    persona_prompt = build_persona_prompt(
        card, disposition, persona_template, disposition_notes
    )
    conversation_id = make_conversation_id(condition, card.character_id)
    count_tokens = make_token_counter(llama, token_overhead)

    model_turn_raws: list[str] = []
    npc_dialogues: list[str] = []
    user_texts: list[str] = []
    turns: list[dict] = []
    context_truncated = False
    dropped_pairs_total = 0

    for model_turn_i in range(model_turns):
        messages = build_npc_messages(
            turn1_prompt, model_turn_raws, user_texts, condition, window_n
        )
        messages, dropped = enforce_context_budget(
            messages, count_tokens, max_tokens_npc, context_length
        )
        if dropped:
            context_truncated = True
            dropped_pairs_total += dropped
            logger.info(
                f"{conversation_id}: dropped {dropped} oldest turn pair(s) "
                f"before model turn {model_turn_i + 1} (context guard)"
            )

        out = llama.create_chat_completion(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            max_tokens=max_tokens_npc,
        )
        choice = out["choices"][0]
        raw = (choice.get("message") or {}).get("content") or ""
        usage = out.get("usage") or {}
        parsed = parse_model_turn(raw, condition)

        turns.append(
            {
                "index": len(turns),
                "role": "model",
                "raw": raw,
                "dialogue": parsed.dialogue,
                "trace": parsed.trace,
                "tokens": usage.get("completion_tokens"),
                "finish_reason": choice.get("finish_reason"),
                "flags": parsed.flags,
            }
        )
        model_turn_raws.append(raw)
        npc_dialogues.append(parsed.dialogue or "...")

        if model_turn_i < model_turns - 1:
            sim_messages = build_user_sim_messages(
                persona_prompt, npc_dialogues, user_texts
            )
            reply = deepseek_user_reply(
                ds_client,
                sim_messages,
                ds_model=ds_model,
                temperature=ds_temperature,
                max_tokens=max_tokens_user,
                max_retries=ds_max_retries,
            )
            turns.append(
                {
                    "index": len(turns),
                    "role": "user",
                    "raw": reply,
                    "dialogue": None,
                    "trace": None,
                    "tokens": None,
                }
            )
            user_texts.append(reply)

    transcript = {
        "conversation_id": conversation_id,
        "character_id": card.character_id,
        "condition": condition,
        "turn1_prompt": turn1_prompt,
        "turns": turns,
        "sampling": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "max_tokens": max_tokens_npc,
            "seed": seed,
            "context_length": context_length,
            "window_n": window_n,
            "model_path": str(model_path),
            "ds_model": ds_model,
            "player_disposition": disposition,
            "context_truncated": context_truncated,
            "dropped_turn_pairs": dropped_pairs_total,
        },
    }
    replay = {
        "conversation_id": conversation_id,
        "character_id": card.character_id,
        "condition": condition,
        "turn1_prompt": turn1_prompt,
        "user_turns": user_texts,
    }
    return transcript, replay


def _ensure_newline_terminated(path: Path) -> None:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path, "rb") as f:
        f.seek(-1, os.SEEK_END)
        last = f.read(1)
    if last != b"\n":
        logger.warning(
            f"{path}: no trailing newline (torn final write); terminating partial line"
        )
        with open(path, "ab") as f:
            f.write(b"\n")


def _resume_state(transcripts_path: Path, user_turns_path: Path) -> set[str]:
    transcripts_path = Path(transcripts_path)
    user_turns_path = Path(user_turns_path)
    _ensure_newline_terminated(transcripts_path)
    _ensure_newline_terminated(user_turns_path)
    if not transcripts_path.exists():
        return set()
    done = {}
    for rec in read_jsonl(transcripts_path):
        cid = rec.get("conversation_id")
        if cid:
            done[cid] = rec
    have_replay = set()
    if user_turns_path.exists():
        have_replay = {r.get("conversation_id") for r in read_jsonl(user_turns_path)}
    for cid, rec in done.items():
        if cid not in have_replay:
            append_jsonl(
                user_turns_path,
                {
                    "conversation_id": cid,
                    "character_id": rec.get("character_id"),
                    "condition": rec.get("condition"),
                    "turn1_prompt": rec.get("turn1_prompt"),
                    "user_turns": [
                        t.get("raw")
                        for t in rec.get("turns", [])
                        if t.get("role") == "user"
                    ],
                },
            )
            logger.info(f"backfilled user-turns record for {cid}")
    return set(done)


def _setup_logging(runs_dir: Path) -> None:
    runs_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(runs_dir / "generate.log", encoding="utf-8"),
        ],
    )


def main(
    model_paths=MODEL_PATHS,
    cards_path=CARDS_PATH,
    runs_dir=RUNS_DIR,
    conditions=CONDITIONS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    min_p=MIN_P,
    max_tokens_npc=MAX_TOKENS_NPC,
    seed=SEED,
    context_length=CONTEXT_LENGTH,
    window_n=WINDOW_N,
    model_turns=MODEL_TURNS,
    ds_base_url=DS_BASE_URL,
    ds_env_var=DS_ENV_VAR,
    ds_model=DS_MODEL,
    ds_temperature=DS_TEMPERATURE,
    max_tokens_user=MAX_TOKENS_USER,
    ds_max_retries=DS_MAX_RETRIES,
    dispositions=DISPOSITIONS,
    persona_template=PLAYER_PERSONA_PROMPT,
    disposition_notes=DISPOSITION_NOTES,
    token_overhead=TOKENS_PER_MESSAGE_OVERHEAD,
):
    from llama_cpp import Llama

    runs_dir = Path(runs_dir)
    _setup_logging(runs_dir)

    api_key = os.getenv(ds_env_var)
    if not api_key:
        raise RuntimeError(f"{ds_env_var} not set; add it to the .env at the repo root")
    ds_client = OpenAI(base_url=ds_base_url, api_key=api_key)

    cards = load_character_cards(cards_path)
    logger.info(f"loaded {len(cards)} character cards from {cards_path}")

    for condition in conditions:
        transcripts_path = runs_dir / f"transcripts_{condition}.jsonl"
        user_turns_path = runs_dir / f"user_turns_{condition}.jsonl"
        done = _resume_state(transcripts_path, user_turns_path)
        pending = [
            c
            for c in cards
            if make_conversation_id(condition, c.character_id) not in done
        ]
        if done:
            logger.info(
                f"[{condition}] resume: {len(done)} conversations already "
                f"written, {len(pending)} pending"
            )
        if not pending:
            logger.info(f"[{condition}] nothing to do")
            continue

        model_path = Path(model_paths[condition])
        if not model_path.exists():
            raise FileNotFoundError(
                f"GGUF for {condition} not found: {model_path} (fill in MODEL_PATHS)"
            )

        logger.info(f"[{condition}] loading {model_path}")
        llama = Llama(
            model_path=str(model_path),
            n_gpu_layers=-1,
            n_ctx=context_length,
            seed=seed,
            verbose=False,
        )
        try:
            for card in tqdm(pending, desc=condition):
                try:
                    transcript, replay = run_conversation(
                        llama,
                        ds_client,
                        card,
                        condition,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        min_p=min_p,
                        max_tokens_npc=max_tokens_npc,
                        seed=seed,
                        context_length=context_length,
                        window_n=window_n,
                        model_turns=model_turns,
                        ds_model=ds_model,
                        ds_temperature=ds_temperature,
                        max_tokens_user=max_tokens_user,
                        ds_max_retries=ds_max_retries,
                        model_path=model_path,
                        dispositions=dispositions,
                        persona_template=persona_template,
                        disposition_notes=disposition_notes,
                        token_overhead=token_overhead,
                    )
                except UserSimError as e:
                    logger.error(
                        f"[{condition}] {card.character_id}: conversation "
                        f"aborted, user-sim failed after {ds_max_retries} "
                        f"attempts: {e}"
                    )
                    continue
                append_jsonl(transcripts_path, transcript)
                append_jsonl(user_turns_path, replay)
                logger.info(
                    f"[{condition}] wrote {transcript['conversation_id']} "
                    f"({len(transcript['turns'])} turns)"
                )
        finally:
            try:
                llama.close()
            except AttributeError:
                pass
            del llama
            gc.collect()


if __name__ == "__main__":
    main()
