import gc
import hashlib
import json
import logging
import time
from pathlib import Path

import a_generate_conversations
import eval_lib
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

CONDITIONS = eval_lib.CONDITIONS
CONTEXT_LENGTH = eval_lib.CONTEXT_LENGTH
WINDOW_N = eval_lib.WINDOW_N

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
RUNS_DIR = EVAL_DIR / "runs"
LATENCY_DIR = RUNS_DIR / "latency"
MODELS_DIR = REPO_ROOT / "models"
TRANSCRIPTS_PATHS = {c: RUNS_DIR / f"transcripts_{c}.jsonl" for c in CONDITIONS}
USER_TURNS_PATHS = {c: RUNS_DIR / f"user_turns_{c}.jsonl" for c in CONDITIONS}
RECORDS_PATHS = {c: LATENCY_DIR / f"latency_records_{c}.jsonl" for c in CONDITIONS}
SUMMARY_PATH = LATENCY_DIR / "latency_summary.json"
LOG_PATH = LATENCY_DIR / "latency_replay.log"

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

N_REPS = 10
WARMUP_RUNS = 3
COOLDOWN_S = 60.0
KV_CACHE_QUANT = True
THROTTLE_FRACTION = 0.90
GPU_INDEX = 0
N_GPU_LAYERS = -1
MSG_TOKEN_OVERHEAD = a_generate_conversations.TOKENS_PER_MESSAGE_OVERHEAD

SUMMARY_METRICS = ("prefill_s", "ttft_s", "decode_tps", "perceived_s", "total_s")

_CONFIG_KEYS = (
    "CONDITIONS",
    "CONTEXT_LENGTH",
    "WINDOW_N",
    "SUMMARY_METRICS",
    "RUNS_DIR",
    "LATENCY_DIR",
    "TRANSCRIPTS_PATHS",
    "USER_TURNS_PATHS",
    "RECORDS_PATHS",
    "SUMMARY_PATH",
    "LOG_PATH",
    "MODEL_PATHS",
    "TEMPERATURE",
    "TOP_P",
    "TOP_K",
    "MIN_P",
    "MAX_TOKENS_NPC",
    "SEED",
    "N_REPS",
    "WARMUP_RUNS",
    "COOLDOWN_S",
    "KV_CACHE_QUANT",
    "THROTTLE_FRACTION",
    "GPU_INDEX",
    "N_GPU_LAYERS",
    "MSG_TOKEN_OVERHEAD",
)


def char_to_chunk(offset: int | None, chunk_lens: list[int]) -> int | None:
    if offset is None or offset < 0:
        return None
    cum = 0
    for i, n in enumerate(chunk_lens):
        cum += n
        if offset < cum:
            return i
    return None


def locate_boundaries(raw: str, condition: str) -> dict:
    raw = "" if raw is None else str(raw)
    events = eval_lib._scan_tags(raw)
    outside, blocks, _orphans, _nested = eval_lib._segment(raw, events)

    first_dialogue = None
    for a, b in outside:
        for i in range(a, b):
            if not raw[i].isspace():
                first_dialogue = i
                break
        if first_dialogue is not None:
            break
    last_dialogue = None
    for a, b in reversed(outside):
        for i in range(b - 1, a - 1, -1):
            if not raw[i].isspace():
                last_dialogue = i
                break
        if last_dialogue is not None:
            break

    family = {"pre_thinking": "pre", "post_thinking": "post"}.get(condition)
    trace_start = trace_end = None
    if family is not None:
        fam_blocks = [bl for bl in blocks if bl["family"] == family]
        complete = [bl for bl in fam_blocks if bl["complete"]]
        chosen = complete[0] if complete else (fam_blocks[0] if fam_blocks else None)
        if chosen is not None:
            trace_start = chosen["start"]
            trace_end = chosen["end"] - 1

    return {
        "first_dialogue": first_dialogue,
        "last_dialogue": last_dialogue,
        "trace_start": trace_start,
        "trace_end": trace_end,
    }


def _verify_boundary_convention() -> None:
    cases = (
        ("Hello there.<post-thinking>note to self</post-thinking>", "post_thinking"),
        ("<think>plan the reply</think>Well met, traveler.", "pre_thinking"),
        ("Hi.<post-thinking>cut off mid-trace", "post_thinking"),
        ("Plain dialogue only.", "no_thinking"),
    )
    for raw, cond in cases:
        b = locate_boundaries(raw, cond)
        parsed = eval_lib.parse_model_turn(raw, cond)
        ok = True
        if parsed.dialogue:
            ok = ok and (
                b["first_dialogue"] is not None
                and raw[b["first_dialogue"]] == parsed.dialogue[0]
            )
            ok = ok and (
                b["last_dialogue"] is not None
                and raw[b["last_dialogue"]] == parsed.dialogue[-1]
            )
        if parsed.trace:
            span = (
                raw[b["trace_start"] : b["trace_end"] + 1]
                if b["trace_start"] is not None
                else ""
            )
            ok = ok and parsed.trace in span
        elif cond == "no_thinking":
            ok = ok and b["trace_start"] is None and b["trace_end"] is None
        if not ok:
            raise RuntimeError(
                f"locate_boundaries disagrees with eval_lib.parse_model_turn on {raw!r} "
                f"({cond}); eval_lib's private _scan_tags/_segment changed convention - "
                "update locate_boundaries to match"
            )


try:
    _verify_boundary_convention()
except AttributeError as e:
    raise RuntimeError(
        "eval_lib's private tag-segmentation API (_scan_tags/_segment) is gone; "
        "update locate_boundaries to the refactored internals"
    ) from e


def chunk_events(
    chunk_texts: list[str], chunk_times: list[float], condition: str
) -> dict:
    if not chunk_texts:
        return {
            "first_token": None,
            "first_dialogue": None,
            "last_dialogue": None,
            "trace_start": None,
            "trace_end": None,
        }
    raw = "".join(chunk_texts)
    lens = [len(t) for t in chunk_texts]
    bounds = locate_boundaries(raw, condition)

    def t_of(offset):
        i = char_to_chunk(offset, lens)
        return None if i is None else chunk_times[i]

    return {
        "first_token": chunk_times[0],
        "first_dialogue": t_of(bounds["first_dialogue"]),
        "last_dialogue": t_of(bounds["last_dialogue"]),
        "trace_start": t_of(bounds["trace_start"]),
        "trace_end": t_of(bounds["trace_end"]),
    }


def derive_metrics(events: dict, gen_end_s: float, gen_tokens: int) -> dict:
    first = events.get("first_token")
    decode_tps = None
    if first is not None and gen_end_s > first and gen_tokens > 0:
        decode_tps = gen_tokens / (gen_end_s - first)
    return {
        "prefill_s": first,
        "ttft_s": first,
        "decode_tps": decode_tps,
        "perceived_s": events.get("last_dialogue"),
        "total_s": gen_end_s,
    }


def extract_replay_units(transcript: dict, user_rec: dict | None) -> dict:
    cid = transcript.get("conversation_id")
    if user_rec is None:
        raise ValueError(f"{cid}: no matching record in the user_turns replay file")
    for key in ("conversation_id", "condition", "character_id", "turn1_prompt"):
        if transcript.get(key) != user_rec.get(key):
            raise ValueError(f"{cid}: transcript/user_turns mismatch on {key!r}")
    turns = transcript.get("turns") or []
    model_raws = [t.get("raw") for t in turns if t.get("role") == "model"]
    user_raws = [t.get("raw") for t in turns if t.get("role") == "user"]
    if any(not isinstance(r, str) for r in model_raws + user_raws):
        raise ValueError(f"{cid}: non-string raw turn in transcript")
    if user_raws != list(user_rec.get("user_turns") or []):
        raise ValueError(f"{cid}: user turns in transcript differ from the replay file")
    if len(model_raws) != len(user_raws) + 1:
        raise ValueError(
            f"{cid}: expected one more model turn than user turns, "
            f"got {len(model_raws)} model / {len(user_raws)} user"
        )
    sampling = transcript.get("sampling") or {}
    return {
        "conversation_id": cid,
        "character_id": transcript.get("character_id"),
        "condition": transcript.get("condition"),
        "turn1_prompt": transcript.get("turn1_prompt"),
        "model_raws": model_raws,
        "user_turns": user_raws,
        "recorded_dropped_pairs": sampling.get("dropped_turn_pairs"),
    }


def build_turn_messages(
    turn1_prompt: str,
    model_raws: list[str],
    user_turns: list[str],
    turn_index: int,
    condition: str,
    window_n: int = WINDOW_N,
) -> list[dict]:
    prior = eval_lib.apply_trace_window(model_raws[:turn_index], condition, window_n)
    messages = [{"role": "user", "content": turn1_prompt}]
    for j, raw in enumerate(prior):
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": user_turns[j]})
    return messages


def check_drop_totals(
    conversation_id: str, rep: int, replayed_total: int, recorded_total
) -> bool:
    if recorded_total is None:
        return True
    if replayed_total != recorded_total:
        logger.warning(
            f"{conversation_id} rep {rep}: replay context guard dropped {replayed_total} "
            f"turn pair(s) vs {recorded_total} recorded at generation time - token "
            "estimator drift; investigate before trusting prefill numbers"
        )
        return False
    return True


def _percentile_stats(values: list[float]) -> dict:
    if not values:
        return {
            "n": 0,
            "median": None,
            "q25": None,
            "q75": None,
            "iqr": None,
            "p5": None,
            "p95": None,
        }
    arr = np.asarray(values, dtype=float)
    p5, q25, med, q75, p95 = (float(v) for v in np.percentile(arr, [5, 25, 50, 75, 95]))
    return {
        "n": int(arr.size),
        "median": med,
        "q25": q25,
        "q75": q75,
        "iqr": q75 - q25,
        "p5": p5,
        "p95": p95,
    }


def summarize_records(records: list[dict], metrics: tuple = SUMMARY_METRICS) -> dict:
    by_cond: dict[str, list[dict]] = {}
    for r in records:
        by_cond.setdefault(r.get("condition", "unknown"), []).append(r)
    out = {}
    for cond in sorted(by_cond):
        rows = by_cond[cond]
        used = [r for r in rows if r.get("throttled") is not True]
        unclassified = sum(1 for r in used if r.get("throttled") is None)
        metric_stats = {}
        for m in metrics:
            vals = [r.get("metrics", {}).get(m) for r in used]
            metric_stats[m] = _percentile_stats([v for v in vals if v is not None])
        out[cond] = {
            "n_records": len(rows),
            "n_excluded_throttled": len(rows) - len(used),
            "n_used": len(used),
            "n_throttle_unclassified": unclassified,
            "metrics": metric_stats,
        }
    return out


def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )


def _nvml_handle(gpu_index: int):
    try:
        import pynvml

        pynvml.nvmlInit()
        return pynvml, pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    except Exception as e:
        logger.warning(
            f"pynvml unavailable ({e}); GPU logging, throttle detection "
            "and offload verification are DISABLED - treat timings with caution"
        )
        return None, None


def _gpu_state(nv, handle) -> dict:
    if nv is None:
        return {"temp_c": None, "sm_clock_mhz": None}
    try:
        return {
            "temp_c": int(nv.nvmlDeviceGetTemperature(handle, nv.NVML_TEMPERATURE_GPU)),
            "sm_clock_mhz": int(nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_SM)),
        }
    except Exception:
        return {"temp_c": None, "sm_clock_mhz": None}


def _nvml_mem_used(nv, handle) -> int | None:
    if nv is None:
        return None
    try:
        return int(nv.nvmlDeviceGetMemoryInfo(handle).used)
    except Exception:
        return None


def _load_model(condition: str, cfg: dict, nv, handle):
    import llama_cpp

    model_path = Path(cfg["MODEL_PATHS"][condition])
    if not model_path.exists():
        raise FileNotFoundError(
            f"GGUF for {condition} not found: {model_path} "
            "(fill MODEL_PATHS in the config block)"
        )
    used_before = _nvml_mem_used(nv, handle)
    kwargs = dict(
        model_path=str(model_path),
        n_gpu_layers=cfg["N_GPU_LAYERS"],
        n_ctx=cfg["CONTEXT_LENGTH"],
        seed=cfg["SEED"],
        verbose=False,
    )
    if cfg["KV_CACHE_QUANT"]:
        kwargs.update(
            type_k=llama_cpp.GGML_TYPE_Q8_0,
            type_v=llama_cpp.GGML_TYPE_Q8_0,
            flash_attn=True,
        )
    llm = llama_cpp.Llama(**kwargs)
    used_after = _nvml_mem_used(nv, handle)
    if used_before is not None and used_after is not None:
        delta = used_after - used_before
        model_bytes = model_path.stat().st_size
        if delta < 0.85 * model_bytes:
            logger.warning(
                f"FULL OFFLOAD NOT VERIFIED for {condition}: VRAM grew {delta / 1e9:.2f}GB "
                f"but the GGUF is {model_bytes / 1e9:.2f}GB - layers likely spilled to CPU. "
                "CPU spill invalidates the timing; do not use this pass for the paper."
            )
        else:
            logger.info(
                f"{condition}: offload check OK "
                f"(VRAM +{delta / 1e9:.2f}GB for a {model_bytes / 1e9:.2f}GB GGUF)"
            )
    else:
        logger.warning(
            f"{condition}: pynvml memory check unavailable; full offload NOT verified"
        )
    return llm


def _unload_model(llm) -> None:
    try:
        llm.close()
    except AttributeError:
        pass


def _stream_generation(llm, messages: list[dict], cfg: dict):
    llm.reset()
    t0 = time.perf_counter()
    stream = llm.create_chat_completion(
        messages=messages,
        stream=True,
        temperature=cfg["TEMPERATURE"],
        top_p=cfg["TOP_P"],
        top_k=cfg["TOP_K"],
        min_p=cfg["MIN_P"],
        max_tokens=cfg["MAX_TOKENS_NPC"],
        seed=cfg["SEED"],
    )
    chunk_texts: list[str] = []
    chunk_times: list[float] = []
    finish_reason = None
    for chunk in stream:
        t = time.perf_counter()
        choice = chunk["choices"][0]
        if choice.get("finish_reason") is not None:
            finish_reason = choice["finish_reason"]
        content = (choice.get("delta") or {}).get("content")
        if content:
            chunk_texts.append(content)
            chunk_times.append(t - t0)
    gen_end_s = time.perf_counter() - t0
    return chunk_texts, chunk_times, gen_end_s, finish_reason


def _warmup(llm, messages: list[dict], cfg: dict, nv, handle) -> int | None:
    max_clock = None
    for _ in range(cfg["WARMUP_RUNS"]):
        _stream_generation(llm, messages, cfg)
        clock = _gpu_state(nv, handle)["sm_clock_mhz"]
        if clock is not None:
            max_clock = clock if max_clock is None else max(max_clock, clock)
    return max_clock


def _time_turn(llm, messages: list[dict], condition: str, cfg: dict) -> dict:
    chunk_texts, chunk_times, gen_end_s, finish_reason = _stream_generation(
        llm, messages, cfg
    )
    raw = "".join(chunk_texts)
    events = chunk_events(chunk_texts, chunk_times, condition)
    try:
        gen_tokens = (
            len(llm.tokenize(raw.encode("utf-8"), add_bos=False, special=False))
            if raw
            else 0
        )
    except Exception:
        gen_tokens = len(chunk_texts)
    parsed = eval_lib.parse_model_turn(raw, condition)
    return {
        "events_s": {"request_start": 0.0, **events, "gen_end": gen_end_s},
        "metrics": derive_metrics(events, gen_end_s, gen_tokens),
        "gen_tokens": gen_tokens,
        "gen_chunks": len(chunk_texts),
        "gen_chars": len(raw),
        "finish_reason": finish_reason,
        "flags": parsed.flags,
        "raw_sha1": hashlib.sha1(raw.encode("utf-8")).hexdigest(),
    }


def main(**overrides):
    cfg = {k: globals()[k] for k in _CONFIG_KEYS}
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise TypeError(f"unknown config overrides: {sorted(unknown)}")
    cfg.update(overrides)
    conditions = tuple(cfg["CONDITIONS"])
    runs_dir = Path(cfg["RUNS_DIR"])
    if "LATENCY_DIR" not in overrides:
        cfg["LATENCY_DIR"] = runs_dir / "latency"
    latency_dir = Path(cfg["LATENCY_DIR"])
    if "TRANSCRIPTS_PATHS" not in overrides:
        cfg["TRANSCRIPTS_PATHS"] = {
            c: runs_dir / f"transcripts_{c}.jsonl" for c in conditions
        }
    if "USER_TURNS_PATHS" not in overrides:
        cfg["USER_TURNS_PATHS"] = {
            c: runs_dir / f"user_turns_{c}.jsonl" for c in conditions
        }
    if "RECORDS_PATHS" not in overrides:
        cfg["RECORDS_PATHS"] = {
            c: latency_dir / f"latency_records_{c}.jsonl" for c in conditions
        }
    if "SUMMARY_PATH" not in overrides:
        cfg["SUMMARY_PATH"] = latency_dir / "latency_summary.json"
    if "LOG_PATH" not in overrides:
        cfg["LOG_PATH"] = latency_dir / "latency_replay.log"

    _setup_logging(Path(cfg["LOG_PATH"]))
    logger.info(
        f"latency replay: N_REPS={cfg['N_REPS']} warmup={cfg['WARMUP_RUNS']} "
        f"cooldown={cfg['COOLDOWN_S']}s kv_quant={cfg['KV_CACHE_QUANT']} seed={cfg['SEED']}"
    )

    units_by_cond: dict[str, list[dict]] = {}
    for cond in conditions:
        tpath = Path(cfg["TRANSCRIPTS_PATHS"][cond])
        upath = Path(cfg["USER_TURNS_PATHS"][cond])
        if not tpath.exists() or not upath.exists():
            raise FileNotFoundError(
                f"missing recordings for {cond}: {tpath} / {upath} "
                "(run a_generate_conversations.py first)"
            )
        user_recs = {r.get("conversation_id"): r for r in eval_lib.read_jsonl(upath)}
        units = [
            extract_replay_units(tr, user_recs.get(tr.get("conversation_id")))
            for tr in eval_lib.read_jsonl(tpath)
        ]
        units_by_cond[cond] = units
        logger.info(
            f"{cond}: {len(units)} conversations, "
            f"{sum(len(u['model_raws']) for u in units)} model turns"
        )

    done: dict[str, set] = {}
    drop_acc: dict[str, dict] = {}
    for cond in conditions:
        done[cond] = set()
        drop_acc[cond] = {}
        rpath = Path(cfg["RECORDS_PATHS"][cond])
        if rpath.exists():
            for r in eval_lib.read_jsonl(rpath):
                done[cond].add(
                    (r.get("rep"), r.get("conversation_id"), r.get("turn_index"))
                )
                acc = drop_acc[cond].setdefault(
                    (r.get("rep"), r.get("conversation_id")), [0, 0]
                )
                acc[0] += 1
                acc[1] += int(r.get("context_dropped_pairs") or 0)
        if done[cond]:
            logger.info(f"{cond}: resuming, {len(done[cond])} records already present")
        n_turns = {
            u["conversation_id"]: len(u["model_raws"]) for u in units_by_cond[cond]
        }
        recorded = {
            u["conversation_id"]: u["recorded_dropped_pairs"]
            for u in units_by_cond[cond]
        }
        for (rep, cid), acc in drop_acc[cond].items():
            if cid in n_turns and acc[0] >= n_turns[cid]:
                check_drop_totals(cid, rep, acc[1], recorded.get(cid))

    nv, handle = _nvml_handle(cfg["GPU_INDEX"])

    for rep in range(cfg["N_REPS"]):
        for cond in conditions:
            pending = [
                (u, ti)
                for u in units_by_cond[cond]
                for ti in range(len(u["model_raws"]))
                if (rep, u["conversation_id"], ti) not in done[cond]
            ]
            is_last_pass = rep == cfg["N_REPS"] - 1 and cond == conditions[-1]
            if not pending:
                logger.info(f"rep {rep} {cond}: nothing pending, skipping pass")
                continue
            logger.info(f"rep {rep} {cond}: timing {len(pending)} turns")
            llm = _load_model(cond, cfg, nv, handle)
            count_fn = a_generate_conversations.make_token_counter(
                llm, cfg["MSG_TOKEN_OVERHEAD"]
            )

            wu = units_by_cond[cond][0]
            wu_msgs = build_turn_messages(
                wu["turn1_prompt"],
                wu["model_raws"],
                wu["user_turns"],
                len(wu["model_raws"]) - 1,
                cond,
                cfg["WINDOW_N"],
            )
            wu_msgs, _ = a_generate_conversations.enforce_context_budget(
                wu_msgs, count_fn, cfg["MAX_TOKENS_NPC"], cfg["CONTEXT_LENGTH"]
            )
            warmup_max_clock = _warmup(llm, wu_msgs, cfg, nv, handle)
            logger.info(
                f"rep {rep} {cond}: warmup max SM clock = {warmup_max_clock} MHz"
            )

            for u, ti in tqdm(pending, desc=f"rep{rep}-{cond}", unit="turn"):
                msgs = build_turn_messages(
                    u["turn1_prompt"],
                    u["model_raws"],
                    u["user_turns"],
                    ti,
                    cond,
                    cfg["WINDOW_N"],
                )
                msgs, dropped = a_generate_conversations.enforce_context_budget(
                    msgs, count_fn, cfg["MAX_TOKENS_NPC"], cfg["CONTEXT_LENGTH"]
                )
                prompt_tokens = count_fn(msgs)
                timing = _time_turn(llm, msgs, cond, cfg)
                gpu = _gpu_state(nv, handle)
                throttled = None
                if warmup_max_clock and gpu["sm_clock_mhz"] is not None:
                    throttled = (
                        gpu["sm_clock_mhz"]
                        < cfg["THROTTLE_FRACTION"] * warmup_max_clock
                    )
                record = {
                    "condition": cond,
                    "conversation_id": u["conversation_id"],
                    "character_id": u["character_id"],
                    "turn_index": ti,
                    "rep": rep,
                    "prompt_tokens_est": prompt_tokens,
                    "context_dropped_pairs": dropped,
                    **timing,
                    "gpu": {**gpu, "warmup_max_sm_clock_mhz": warmup_max_clock},
                    "throttled": throttled,
                    "ts_unix": time.time(),
                }
                eval_lib.append_jsonl(Path(cfg["RECORDS_PATHS"][cond]), record)
                done[cond].add((rep, u["conversation_id"], ti))
                acc = drop_acc[cond].setdefault((rep, u["conversation_id"]), [0, 0])
                acc[0] += 1
                acc[1] += dropped
                if acc[0] >= len(u["model_raws"]):
                    check_drop_totals(
                        u["conversation_id"], rep, acc[1], u["recorded_dropped_pairs"]
                    )

            _unload_model(llm)
            llm = None
            count_fn = None
            gc.collect()
            if not is_last_pass:
                logger.info(
                    f"rep {rep} {cond}: pass done, cooling down {cfg['COOLDOWN_S']}s"
                )
                time.sleep(cfg["COOLDOWN_S"])

    if nv is not None:
        try:
            nv.nvmlShutdown()
        except Exception:
            pass

    all_records = []
    for cond in conditions:
        rpath = Path(cfg["RECORDS_PATHS"][cond])
        if rpath.exists():
            all_records.extend(eval_lib.read_jsonl(rpath))
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "n_reps": cfg["N_REPS"],
            "warmup_runs": cfg["WARMUP_RUNS"],
            "cooldown_s": cfg["COOLDOWN_S"],
            "kv_cache_quant": cfg["KV_CACHE_QUANT"],
            "throttle_fraction": cfg["THROTTLE_FRACTION"],
            "seed": cfg["SEED"],
            "temperature": cfg["TEMPERATURE"],
            "top_p": cfg["TOP_P"],
            "top_k": cfg["TOP_K"],
            "min_p": cfg["MIN_P"],
            "max_tokens": cfg["MAX_TOKENS_NPC"],
            "context_length": cfg["CONTEXT_LENGTH"],
            "window_n": cfg["WINDOW_N"],
            "n_gpu_layers": cfg["N_GPU_LAYERS"],
            "cold_prefill": True,
            "model_paths": {c: str(p) for c, p in cfg["MODEL_PATHS"].items()},
        },
        "conditions": summarize_records(all_records, tuple(cfg["SUMMARY_METRICS"])),
    }
    summary_path = Path(cfg["SUMMARY_PATH"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    for cond, s in summary["conditions"].items():
        perceived = s["metrics"].get("perceived_s", {}).get("median")
        logger.info(
            f"{cond}: n={s['n_used']} (excluded throttled={s['n_excluded_throttled']}, "
            f"throttle-unclassified={s['n_throttle_unclassified']}) "
            f"perceived_s median={perceived}"
        )
    logger.info(f"summary written to {summary_path}")


if __name__ == "__main__":
    main()
