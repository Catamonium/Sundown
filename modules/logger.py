# ============================================================
# logger.py  —  MODULE
# File-only logging service for Sundown.
# Captures LLM thinking tokens, pipeline events, and errors.
#
# Zero dependencies on other Sundown modules — stdlib only.
# Never prints to console. All functions are safe no-ops if
# called before init().
# ============================================================

import os
import re
import json
import glob
import datetime
import traceback

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOG_DIR      = os.path.join(_PROJECT_ROOT, "configuration", "logs")
_MAX_LOGS     = 10

_log_file   = None   # open file handle for current run
_debug_mode = False  # when True, raw prompts + full responses are also written


# ─────────────────────────────────────────────────────────────
# Lifecycle
# ─────────────────────────────────────────────────────────────

def init(debug: bool = False) -> None:
    """Open a new log file for this run. Call once at startup.

    Creates log directory if missing. Rotates old logs.
    """
    global _log_file, _debug_mode
    _debug_mode = debug
    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
        _rotate_logs()
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(_LOG_DIR, f"sundown_{ts}.log")
        _log_file = open(path, "w", encoding="utf-8", buffering=1)
        _write("INFO", f"Log opened — debug={'on' if debug else 'off'}")
    except Exception:
        pass


def set_debug(enabled: bool) -> None:
    """Toggle debug mode at runtime."""
    global _debug_mode
    _debug_mode = enabled
    _write("INFO", f"Debug mode changed: {'on' if enabled else 'off'}")


def close() -> None:
    """Flush and close the log file. Call at program exit."""
    global _log_file
    if _log_file:
        try:
            _write("INFO", "Log closed.")
            _log_file.flush()
            _log_file.close()
        except Exception:
            pass
        _log_file = None


def _rotate_logs() -> None:
    try:
        logs = sorted(glob.glob(os.path.join(_LOG_DIR, "sundown_*.log")))
        for old in logs[:-(  _MAX_LOGS - 1)]:
            try:
                os.remove(old)
            except OSError:
                pass
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# Core write primitives
# ─────────────────────────────────────────────────────────────

def _write(level: str, msg: str) -> None:
    """Write a single timestamped line to the log file.

    Format: 2026-03-21 14:32:05 [LEVEL]  message
    Safe no-op if called before init().
    """
    if _log_file is None:
        return
    try:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _log_file.write(f"{ts} [{level:<5}]  {msg}\n")
    except Exception:
        pass


def info(msg: str) -> None:
    _write("INFO", msg)


def debug(msg: str) -> None:
    if _debug_mode:
        _write("DEBUG", msg)


def warn(msg: str) -> None:
    _write("WARN", msg)


def error(msg: str) -> None:
    _write("ERROR", msg)


# ─────────────────────────────────────────────────────────────
# LLM thinking capture
# ─────────────────────────────────────────────────────────────

def _extract_thinking(raw_response: str) -> tuple[str, str]:
    """Split a raw LLM response into (thinking_text, response_text).

    Handles inline <think>...</think> tags (qwen3.5, deepseek-r1, etc.).
    The Ollama /api/chat "thinking" field is handled separately by the caller.

    Returns (thinking, clean_response). Either may be empty string.
    """
    match = re.search(r"<think>(.*?)</think>", raw_response, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        clean    = (raw_response[:match.start()] + raw_response[match.end():]).strip()
        return thinking, clean
    return "", raw_response


def log_agent_thinking(thinking: str, iteration: int) -> None:
    """Write the LLM's chain-of-thought to the log file.

    Always written regardless of debug_mode — thinking tokens are the
    primary debugging signal for why the agent made its decisions.
    """
    if not thinking or _log_file is None:
        return
    try:
        header = f"─── LLM THINKING (iteration {iteration}) "
        header = header + "─" * max(0, 58 - len(header))
        ts     = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _log_file.write(f"{ts}        ┌{header}\n")
        for line in thinking.splitlines():
            _log_file.write(f"{ts}        │ {line}\n")
        _log_file.write(f"{ts}        └{'─' * 58}\n")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# Pipeline events
# ─────────────────────────────────────────────────────────────

def log_pipeline_start(video_path: str, settings: dict) -> None:
    info(f"Pipeline start — {os.path.basename(video_path)}")
    key_settings = {k: settings.get(k) for k in (
        "whisper_model", "use_llm_scoring", "llm_model",
        "max_clips", "max_clip_duration", "min_clip_score",
    )}
    info(f"Settings: {json.dumps(key_settings)}")


def log_audio_signals(signal_counts: dict) -> None:
    parts = "  ".join(f"{k}={v}" for k, v in signal_counts.items())
    info(f"Audio signals: {parts}")


# ─────────────────────────────────────────────────────────────
# Agent events
# ─────────────────────────────────────────────────────────────

def log_agent_start(model: str, memory_count: int, profile_loaded: bool) -> None:
    info(
        f"Agent starting — model={model}  memory={memory_count} entries  "
        f"profile={'yes' if profile_loaded else 'no'}"
    )


def log_agent_iteration(iteration: int, max_iterations: int) -> None:
    info(f"Agent iteration {iteration}/{max_iterations}")


def log_agent_flag(start: str, end: str, score: float,
                   humor_type: str, reason: str) -> None:
    info(f"[FLAG] {start}–{end}  score={score:.2f}  type={humor_type}")
    info(f"       reason: {reason}")


def log_agent_veto(start: str, end: str, reason: str) -> None:
    info(f"[VETO] {start}–{end}")
    info(f"       reason: {reason}")


def log_agent_context_request(timestamp: str, window_sec: int) -> None:
    info(f"[CONTEXT REQUEST] {timestamp}  window={window_sec}s")


def log_agent_done(flagged: int, vetoed: int, kept: int, iterations: int) -> None:
    info(f"Agent done — flagged={flagged}  vetoed={vetoed}  kept={kept}  iterations={iterations}")


def log_agent_suggestion(name: str, current: float, suggested: float, reason: str) -> None:
    direction = "↑" if suggested > current else "↓"
    info(f"[SUGGESTION] {name}: {current} → {suggested} {direction}")
    info(f"             reason: {reason}")


def log_suggestion_accepted(name: str, current: float, suggested: float, reason: str) -> None:
    info(f"[SUGGESTION ACCEPTED] {name}: {current} → {suggested}")


def log_suggestion_rejected(name: str, current: float, suggested: float, reason: str) -> None:
    info(f"[SUGGESTION REJECTED] {name}: {current} → {suggested}")


def log_agent_prompt(system: str, user: str) -> None:
    if not _debug_mode:
        return
    _write("DEBUG", f"--- SYSTEM PROMPT ---\n{system}\n--- USER PROMPT ---\n{user}\n--- END ---")


def log_agent_raw_response(raw: str) -> None:
    if not _debug_mode:
        return
    _write("DEBUG", f"--- OLLAMA RAW RESPONSE ---\n{raw}\n--- END ---")


# ─────────────────────────────────────────────────────────────
# Clip + whisper events
# ─────────────────────────────────────────────────────────────

def log_clip_selected(rank: int, start_sec: float, end_sec: float,
                      composite: float, sw: float, heat: float,
                      labels: list) -> None:
    def _fmt(sec: float) -> str:
        s = int(sec)
        return f"{s // 60:02d}:{s % 60:02d}"
    info(
        f"Clip {rank}: {_fmt(start_sec)}→{_fmt(end_sec)}  composite={composite:.2f}"
        f"  sw={sw:.2f}  heat={heat:.2f}"
    )
    if labels:
        info(f"  labels: {', '.join(str(lb) for lb in labels)}")


def log_whisper_cache_hit(video_path: str) -> None:
    info(f"Whisper cache hit — {os.path.basename(video_path)}")


def log_whisper_start(model: str, duration_sec: float) -> None:
    h = int(duration_sec) // 3600
    m = (int(duration_sec) % 3600) // 60
    info(f"Whisper start — model={model}  duration={h}h{m}m")


# ─────────────────────────────────────────────────────────────
# Training events
# ─────────────────────────────────────────────────────────────

def log_training_clip(fname: str, humor_type: str, why_funny: str,
                      user_reason: str, confidence: int) -> None:
    info(f"[{fname}] type={humor_type}  confidence={confidence}")
    info(f"  why: {why_funny}")
    if user_reason:
        info(f"  user: {user_reason}")


# ─────────────────────────────────────────────────────────────
# Error capture
# ─────────────────────────────────────────────────────────────

def log_error_detail(context: str, exc: Exception) -> None:
    """Always written regardless of debug_mode."""
    _write("ERROR", f"[ERROR in {context}] {type(exc).__name__}: {exc}")
    tb = traceback.format_exc()
    if tb and tb.strip() not in ("NoneType: None", ""):
        _write("ERROR", tb)
