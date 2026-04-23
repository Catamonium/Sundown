# ============================================================
# llm.py  —  MODULE
# Ollama-based transcript analysis for Smart Clip scoring.
# Inspired by SupoClip's LLM segment selection approach.
#
# No external pip dependencies — uses urllib.request (stdlib).
# Falls back to empty results gracefully if Ollama is not running.
# ============================================================

import os
import json
import urllib.request
import urllib.error

try:
    import logger as _logger
except ImportError:
    _logger = None  # type: ignore[assignment]

_PROJECT_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MEMORY_PATH     = os.path.join(_PROJECT_ROOT, "configuration", "comedy_memory.json")
_MEMORY_MAX      = 50
_REJECTION_PATH  = os.path.join(_PROJECT_ROOT, "configuration", "rejection_memory.json")
_REJECTION_MAX   = 30


# ─────────────────────────────────────────────────────────────
# Transcript formatting
# ─────────────────────────────────────────────────────────────

def _sec_to_mmss(seconds: float) -> str:
    """Convert float seconds to MM:SS string."""
    s = max(0, int(seconds))
    return f"{s // 60:02d}:{s % 60:02d}"


def _parse_mmss(ts: str) -> float:
    """Convert 'MM:SS' string to total seconds. Returns 0.0 on error."""
    try:
        parts = ts.strip().split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except Exception:
        return 0.0


def format_transcript(segments: list[dict]) -> str:
    """Convert Whisper segment dicts to SupoClip-style timestamped lines.

    Output format (one line per segment):
        [MM:SS - MM:SS] Spoken text here
    """
    lines = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        start = _sec_to_mmss(seg.get("start", 0.0))
        end   = _sec_to_mmss(seg.get("end",   seg.get("start", 0.0)))
        lines.append(f"[{start} - {end}] {text}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Comedy memory I/O
# ─────────────────────────────────────────────────────────────

def load_comedy_memory(settings: dict | None = None) -> list[dict]:
    """Load comedy_memory.json. Returns [] if missing or corrupt."""
    try:
        with open(_MEMORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return []


def save_comedy_memory(memory: list[dict], settings: dict | None = None) -> None:
    """Save memory list (truncated to _MEMORY_MAX newest entries)."""
    os.makedirs(os.path.dirname(_MEMORY_PATH), exist_ok=True)
    try:
        with open(_MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump(memory[-_MEMORY_MAX:], f, indent=2)
    except OSError as exc:
        print(f"[WARNING] Could not save comedy memory: {exc}")


def load_rejection_memory() -> list[dict]:
    """Load rejection_memory.json. Returns [] if missing or corrupt."""
    try:
        with open(_REJECTION_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return []


def save_rejection_memory(memory: list[dict]) -> None:
    """Save rejection memory list (truncated to _REJECTION_MAX newest entries)."""
    os.makedirs(os.path.dirname(_REJECTION_PATH), exist_ok=True)
    try:
        with open(_REJECTION_PATH, "w", encoding="utf-8") as f:
            json.dump(memory[-_REJECTION_MAX:], f, indent=2)
    except OSError as exc:
        print(f"[WARNING] Could not save rejection memory: {exc}")


def add_positive_feedback(text_sample: str, user_reason: str) -> None:
    """Add a user-marked good clip to comedy memory.

    Called from the post-run feedback loop when the user rates a clip as good.
    """
    memory = load_comedy_memory()
    memory.append({
        "source":      "feedback",
        "text_sample": text_sample[:200],
        "why_funny":   user_reason,
        "humor_type":  "other",
        "confidence":  100,
        "user_reason": user_reason,
    })
    save_comedy_memory(memory)


def add_negative_feedback(text_sample: str, reason: str) -> None:
    """Add a user-marked bad clip to rejection memory.

    Called from the post-run feedback loop when the user rates a clip as bad.
    """
    memory = load_rejection_memory()
    memory.append({
        "text_sample": text_sample[:200],
        "reason":      reason,
    })
    save_rejection_memory(memory)


# ─────────────────────────────────────────────────────────────
# Why-funny analysis (training only)
# ─────────────────────────────────────────────────────────────

_WHY_FUNNY_SYSTEM = """\
You are analyzing a short Twitch stream clip transcript to understand what makes it funny.
The streamer speaks primarily in Spanish.
Respond ONLY with a raw JSON object — no markdown, no explanation.
Keys:
  "why_funny"  (string, 1-3 sentences describing the specific comedic structure),
  "humor_type" (one of: panic_reaction, unexpected_escalation, voice_comedy,
                absurdist, self_deprecation, timing, running_gag, other),
  "confidence" (integer 0-100, how confident you are this is genuinely funny vs just loud)
Audio signals detected: {hints}
Clip transcript:
{transcript}\
"""


def analyze_why_funny(
    clip_transcript: str,
    audio_hints: list[str],
    settings: dict,
    user_context: str = "",
) -> dict | None:
    """Ask Ollama why a clip is funny. Returns {why_funny, humor_type, confidence} or None.

    user_context: optional free-text reason provided by the user at training time.
    When non-empty it is prepended to the transcript so the LLM builds on it rather
    than guessing from scratch. Memory entries with a user_context get confidence=100.
    """
    if not clip_transcript.strip():
        return None

    model   = settings.get("llm_model",   "llama3.2")
    host    = settings.get("llm_host",    "http://localhost:11434")
    timeout = int(settings.get("llm_timeout", 120))

    context_block = (
        f'The user who created this clip says: "{user_context}"\n'
        f'Use this as the primary explanation. Expand on it technically if useful, but do not contradict it.\n\n'
        if user_context else ""
    )
    prompt = _WHY_FUNNY_SYSTEM.format(
        hints=", ".join(audio_hints) if audio_hints else "none",
        transcript=context_block + clip_transcript,
    )

    try:
        raw = _query_ollama(prompt, model, host, timeout)
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text  = "\n".join(l for l in lines if not l.startswith("```"))
        data = json.loads(text)
        if not isinstance(data, dict):
            return None
        why_funny  = str(data.get("why_funny",  "")).strip()
        humor_type = str(data.get("humor_type", "other")).strip()
        confidence = int(data.get("confidence", 0))
        if not why_funny:
            return None
        valid_types = {
            "panic_reaction", "unexpected_escalation", "voice_comedy",
            "absurdist", "self_deprecation", "timing", "running_gag", "other",
        }
        if humor_type not in valid_types:
            humor_type = "other"
        return {"why_funny": why_funny, "humor_type": humor_type, "confidence": confidence}
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# Dynamic prompt builder
# ─────────────────────────────────────────────────────────────

# Hardcoded fallback (used when no profile and no memory)
_SYSTEM_BASE = """\
You are analyzing a Twitch stream transcript to find clips worth editing into short-form content.
The streamer speaks primarily in Spanish. Their humor style includes: swearing, screaming, \
nonsense sounds, goofy energy, and unexpected reactions.\
"""

_SYSTEM_RULES = """\
Return ONLY a raw JSON array — no explanation, no markdown, no code fences.

Each object must have exactly these keys:
  "start_time":     string MM:SS (must be a timestamp present in the transcript)
  "end_time":       string MM:SS (must be a timestamp present in the transcript)
  "comedy_score":   integer 0-40  (how funny/goofy is this moment)
  "reaction_score": integer 0-30  (strong emotional reaction — surprise, excitement, panic)
  "hook_score":     integer 0-30  (would this make someone stop scrolling)

Rules:
- Only use timestamps visible in the transcript.
- Each segment must be a single contiguous range — no stitching distant moments.
- Capture the full arc: include the setup before a reaction, not just the reaction itself.
- Minimum segment: 8 seconds. Maximum: 55 seconds.
- Return between 5 and 15 segments ranked by total score descending.
- Total of comedy + reaction + hook must not exceed 100.\
"""


def build_system_prompt(profile: dict | None, memory: list[dict]) -> str:
    """Build a dynamic system prompt from the training profile and comedy memory.

    Falls back to the hardcoded base prompt when both profile and memory are empty.
    """
    parts = [_SYSTEM_BASE]

    # Profile → natural language traits
    if profile:
        comedy = profile.get("profile", {}).get("comedy", {})
        traits = []

        v = comedy.get("scream_presence", 0.0)
        if v > 0.05:
            traits.append(f"screaming or sudden loud reactions (present in {v:.0%} of known funny clips)")

        v = comedy.get("swear_density", 0.0)
        if v > 0.1:
            traits.append(f"swearing ({v:.1f} per minute)")

        v = comedy.get("nonsense_density", 0.0)
        if v > 0.1:
            traits.append(f"nonsense sounds/vocalizations ({v:.1f} per minute)")

        v = comedy.get("voice_crack_count", 0.0)
        if v > 0.1:
            traits.append(f"voice cracks ({v:.1f} per minute on average)")

        v = comedy.get("pre_silence_count", 0.0)
        if v > 0.1:
            traits.append(f"setup silences before reactions ({v:.1f} per minute)")

        v = comedy.get("chat_spike_density", 0.0)
        if v > 0.1:
            traits.append(f"chat spikes ({v:.1f} per minute)")

        v = comedy.get("pitch_variance_mean", 0.0)
        if v > 0.3:
            traits.append("high pitch variance (wild tonal swings)")

        if traits:
            parts.append(
                "Known funny clip patterns from training:\n"
                + "\n".join(f"- {t}" for t in traits)
            )

    # Memory → concrete examples
    eligible = [
        e for e in memory
        if e.get("confidence", 0) >= 50 or e.get("source") == "training"
    ]
    eligible.sort(key=lambda e: e.get("confidence", 0), reverse=True)
    examples = eligible[:6]

    if examples:
        lines = ["Examples of known funny moments (use these as reference):"]
        for e in examples:
            sample = e.get("text_sample", "")[:80].replace("\n", " ")
            why    = e.get("why_funny", "")
            htype  = e.get("humor_type", "other")
            ur    = e.get("user_reason", "")
            notes = e.get("llm_notes", "")
            if ur and notes:
                lines.append(f'[{htype}] "{sample}"\n  → "{ur}"\n  → (LLM notes: "{notes}")')
            elif ur:
                lines.append(f'[{htype}] "{sample}"\n  → User said: "{ur}"\n  → LLM analysis: "{why}"')
            else:
                lines.append(f'- [{htype}] "{sample}" → {why}')
        parts.append("\n".join(lines))

    # Rejection memory → "do not clip these" examples
    rejections = load_rejection_memory()
    if rejections:
        rej_lines = ["Moments the editor explicitly rejected as not worth clipping (avoid similar):"]
        for r in rejections[-5:]:
            sample = r.get("text_sample", "")[:80].replace("\n", " ")
            reason = r.get("reason", "not funny")
            rej_lines.append(f'- "{sample}" → rejected: {reason}')
        parts.append("\n".join(rej_lines))

    parts.append(_SYSTEM_RULES)
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────
# Ollama REST calls
# ─────────────────────────────────────────────────────────────

def _query_ollama(prompt: str, model: str, host: str, timeout: int) -> str:
    """POST to Ollama /api/generate. Returns the raw `response` string.

    Raises urllib.error.URLError on connection failure.
    Raises ValueError if the response JSON is malformed.
    """
    url  = host.rstrip("/") + "/api/generate"
    body = json.dumps({
        "model":  model,
        "prompt": prompt,
        "format": "json",
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")

    data = json.loads(raw)
    if "error" in data:
        raise ValueError(f"Ollama error: {data['error']}")
    return data.get("response", "")


def _chat_ollama(
    messages: list[dict],
    tools: list[dict],
    model: str,
    host: str,
    timeout: int,
) -> dict:
    """POST to Ollama /api/chat with tool definitions.

    Returns the full response dict including message and tool_calls.
    Raises urllib.error.URLError on connection failure.
    Raises ValueError if the response JSON is malformed.
    """
    url  = host.rstrip("/") + "/api/chat"
    body = json.dumps({
        "model":    model,
        "messages": messages,
        "tools":    tools,
        "stream":   False,
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")

    data = json.loads(raw)
    if "error" in data:
        raise ValueError(f"Ollama error: {data['error']}")
    return data


# ─────────────────────────────────────────────────────────────
# Response parsing + validation
# ─────────────────────────────────────────────────────────────

def _parse_response(raw_response: str) -> list[dict]:
    """Parse Ollama's raw text response into a validated list of segment dicts.

    Returns a list of {start_sec, end_sec, score, comedy_score} dicts.
    Silently drops invalid/malformed segments.
    """
    text = raw_response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text  = "\n".join(l for l in lines if not l.startswith("```"))

    try:
        items = json.loads(text)
    except json.JSONDecodeError:
        return []

    if not isinstance(items, list):
        if isinstance(items, dict):
            for v in items.values():
                if isinstance(v, list):
                    items = v
                    break
            else:
                return []
        else:
            return []

    results = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            start_sec = _parse_mmss(str(item.get("start_time", "0:00")))
            end_sec   = _parse_mmss(str(item.get("end_time",   "0:00")))
            duration  = end_sec - start_sec
            if duration < 5:
                continue

            comedy   = int(item.get("comedy_score",   0))
            reaction = int(item.get("reaction_score", 0))
            hook     = int(item.get("hook_score",     0))
            total    = comedy + reaction + hook
            if total <= 0:
                continue

            score = min(1.0, total / 100.0)
            results.append({
                "start_sec":    start_sec,
                "end_sec":      end_sec,
                "score":        score,
                "comedy_score": min(1.0, comedy / 40.0),
            })
        except Exception:
            continue

    return results


# ─────────────────────────────────────────────────────────────
# Agentic tool definitions
# ─────────────────────────────────────────────────────────────

_AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "flag_clip",
            "description": (
                "Nominate a moment from the transcript as a clip candidate. "
                "Use this when you find a moment that would make a good short-form clip. "
                "Capture the full comedic arc — include setup before the reaction."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time":     {"type": "string",  "description": "MM:SS start timestamp from the transcript"},
                    "end_time":       {"type": "string",  "description": "MM:SS end timestamp from the transcript"},
                    "comedy_score":   {"type": "integer", "description": "0-40 — how funny/goofy is this moment"},
                    "reaction_score": {"type": "integer", "description": "0-30 — strength of emotional reaction"},
                    "hook_score":     {"type": "integer", "description": "0-30 — would this stop a scroller"},
                    "reason":         {"type": "string",  "description": "1-2 sentences explaining why this moment works"},
                },
                "required": ["start_time", "end_time", "comedy_score", "reaction_score", "hook_score", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "veto_clip",
            "description": (
                "Explicitly reject a time range as a clip candidate. "
                "Use this when audio signals flagged a moment but the transcript shows "
                "it is not actually funny — just loud, repetitive, or contextless."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "string", "description": "MM:SS start of the moment to reject"},
                    "end_time":   {"type": "string", "description": "MM:SS end of the moment to reject"},
                    "reason":     {"type": "string", "description": "Why this moment is not worth clipping"},
                },
                "required": ["start_time", "end_time", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_more_context",
            "description": (
                "Request additional transcript context around a specific timestamp "
                "before deciding whether to flag or veto it. "
                "Use when the transcript around a moment seems cut off or ambiguous."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "timestamp":  {"type": "string",  "description": "MM:SS of the moment you need more context around"},
                    "window_sec": {"type": "integer", "description": "How many extra seconds of context to retrieve (max 60)"},
                },
                "required": ["timestamp"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "suggest_setting_change",
            "description": (
                "Suggest an adjustment to an audio detection threshold based on what "
                "you observed in this VOD. Use this when you notice a pattern — for example, "
                "many audio-flagged moments that weren't actually funny (threshold too low), "
                "or funny moments the audio arm missed entirely (threshold too high). "
                "Only suggest changes you can justify from specific observations in this VOD."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "setting_name": {
                        "type": "string",
                        "description": "The exact settings.json key to adjust",
                    },
                    "current_value": {
                        "type": "number",
                        "description": "The current value of the setting",
                    },
                    "suggested_value": {
                        "type": "number",
                        "description": "The value you recommend",
                    },
                    "reason": {
                        "type": "string",
                        "description": (
                            "Specific observation from this VOD that justifies the change. "
                            "Reference what you saw — e.g. 'Saw 11 RMS triggers but only 2 "
                            "were actual funny moments, the rest were game explosions.'"
                        ),
                    },
                },
                "required": ["setting_name", "current_value", "suggested_value", "reason"],
            },
        },
    },
]

_TUNABLE_SETTINGS = {
    "rms_threshold_factor",
    "rms_min_sustain",
    "laughter_burst_factor",
    "laughter_burst_count",
    "onset_novelty_threshold",
    "voice_excitement_threshold",
    "smart_heat_threshold",
    "smart_heat_decay",
    "smart_heat_min_duration",
    "comedy_scream_min_pitch",
    "comedy_scream_max_duration",
    "comedy_pitch_variance_thresh",
    "comedy_silence_min_sec",
    "comedy_silence_max_sec",
    "comedy_chat_spike_factor",
    "comedy_rise_factor",
}


# ─────────────────────────────────────────────────────────────
# Agentic tool executors
# ─────────────────────────────────────────────────────────────

def _execute_flag_clip(args: dict) -> tuple:
    """Validate and parse a flag_clip tool call.

    Returns (segment_dict | None, result_text).
    """
    try:
        start_sec = _parse_mmss(str(args.get("start_time", "0:00")))
        end_sec   = _parse_mmss(str(args.get("end_time",   "0:00")))
        duration  = end_sec - start_sec
        if duration < 5:
            return None, f"Rejected: duration {duration:.0f}s is under 5s minimum."
        if duration > 60:
            return None, f"Rejected: duration {duration:.0f}s exceeds 60s maximum."

        comedy   = max(0, min(40, int(args.get("comedy_score",   0))))
        reaction = max(0, min(30, int(args.get("reaction_score", 0))))
        hook     = max(0, min(30, int(args.get("hook_score",     0))))
        total    = comedy + reaction + hook
        if total <= 0:
            return None, "Rejected: all scores are zero."

        score  = min(1.0, total / 100.0)
        reason = str(args.get("reason", "")).strip()
        seg = {
            "start_sec":    start_sec,
            "end_sec":      end_sec,
            "score":        score,
            "comedy_score": min(1.0, comedy / 40.0),
            "reason":       reason,
        }
        return seg, f"Flagged {args.get('start_time')}–{args.get('end_time')} (score {score:.2f}): {reason}"
    except Exception as exc:
        return None, f"Error: {exc}"


def _execute_veto_clip(args: dict) -> tuple:
    """Parse a veto_clip tool call.

    Returns ((start_sec, end_sec) | None, result_text).
    """
    try:
        start_sec = _parse_mmss(str(args.get("start_time", "0:00")))
        end_sec   = _parse_mmss(str(args.get("end_time",   "0:00")))
        reason    = str(args.get("reason", "")).strip()
        return (start_sec, end_sec), f"Vetoed {args.get('start_time')}–{args.get('end_time')}: {reason}"
    except Exception as exc:
        return None, f"Error: {exc}"


def _execute_request_context(args: dict, all_segments: list[dict]) -> str:
    """Return extra transcript lines around a requested timestamp."""
    ts_sec = _parse_mmss(str(args.get("timestamp", "0:00")))
    window = min(60, max(10, int(args.get("window_sec", 30))))
    lo, hi = ts_sec - window, ts_sec + window
    nearby = [s for s in all_segments if lo <= s.get("start", 0) <= hi]
    if not nearby:
        return f"No transcript found around {args.get('timestamp')}."
    return "Context:\n" + format_transcript(nearby)


def _execute_suggest_setting(args: dict, settings: dict) -> tuple:
    """Validate a suggest_setting_change tool call.

    Returns (suggestion_dict | None, result_text).
    """
    name      = str(args.get("setting_name", "")).strip()
    current   = args.get("current_value",   None)
    suggested = args.get("suggested_value", None)
    reason    = str(args.get("reason", "")).strip()

    if name not in _TUNABLE_SETTINGS:
        return None, f"Rejected: '{name}' is not a tunable setting."
    if current is None or suggested is None:
        return None, "Rejected: current_value and suggested_value are required."
    try:
        current   = float(current)
        suggested = float(suggested)
    except (TypeError, ValueError):
        return None, "Rejected: values must be numeric."
    if current == suggested:
        return None, "Rejected: suggested value is the same as current."

    suggestion = {
        "setting_name":    name,
        "current_value":   current,
        "suggested_value": suggested,
        "reason":          reason,
    }
    direction = "↑" if suggested > current else "↓"
    return suggestion, (
        f"Suggestion recorded: {name} {current} → {suggested} {direction}\n"
        f"Reason: {reason}"
    )


# ─────────────────────────────────────────────────────────────
# Score→window mapping
# ─────────────────────────────────────────────────────────────

def map_llm_scores_to_windows(
    windows: list[tuple],
    llm_segments: list[dict],
) -> list[float]:
    """Return an LLM score in [0, 1] for each window.

    A window gets the max score of any LLM segment that overlaps it by
    at least 30% of the window duration.  Windows with no qualifying
    overlap receive 0.0.
    """
    scores = []
    for win in windows:
        w_start, w_end = float(win[0]), float(win[1])
        w_dur = max(w_end - w_start, 1e-6)
        best  = 0.0
        for seg in llm_segments:
            overlap = min(w_end, seg["end_sec"]) - max(w_start, seg["start_sec"])
            if overlap / w_dur >= 0.30:
                best = max(best, seg["score"])
        scores.append(best)
    return scores


def build_llm_windows(llm_segments: list[dict]) -> list[tuple]:
    """Convert LLM segment dicts to (start, end, labels) window tuples."""
    windows = []
    for seg in llm_segments:
        label = f"llm:{seg['score']:.2f}"
        windows.append((seg["start_sec"], seg["end_sec"], [label]))
    return windows


# ─────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────

def analyze_transcript(
    segments: list[dict],
    settings: dict,
    window_signals: str = "",
) -> tuple[list[dict], list[dict]]:
    """Agentic LLM worker — runs a tool-calling loop to select and veto clips.

    The LLM calls flag_clip, veto_clip, request_more_context, and
    suggest_setting_change until done.
    Returns (flagged_segments, setting_suggestions).
    Falls back to ([], []) gracefully if Ollama is unavailable or the model
    does not support tool calling.
    """
    if len(segments) < 3:
        return [], []

    transcript = format_transcript(segments)
    if not transcript.strip():
        return [], []

    model          = settings.get("llm_model",          "qwen3.5:9b")
    host           = settings.get("llm_host",           "http://localhost:11434")
    timeout        = int(settings.get("llm_timeout",        120))
    max_iterations = int(settings.get("llm_max_iterations", 8))

    try:
        import trainer as _trainer  # lazy — avoids circular import
        profile = _trainer.load_profile()
    except Exception:
        profile = None

    memory = load_comedy_memory()
    system = build_system_prompt(profile, memory)
    system += (
        "\n\nYou are operating as an agentic worker. "
        "Use the available tools to flag moments you want clipped, veto moments "
        "that the audio detector flagged but are not actually funny, and suggest "
        "setting changes when you notice detection patterns that seem miscalibrated. "
        "You may call request_more_context to get additional transcript around any timestamp "
        "before deciding. When you have finished reviewing the transcript, stop calling tools."
    )

    user_content = f"Transcript:\n{transcript}"
    if window_signals:
        user_content += f"\n\nAudio signals (moments the audio detector flagged as interesting):\n{window_signals}"
    user_content += "\n\nReview the transcript. Flag clips worth editing and veto any audio-flagged moments that are not actually funny."

    if _logger:
        _logger.log_agent_start(model, len(memory), profile is not None)
        _logger.log_agent_prompt(system, user_content)

    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_content},
    ]

    flagged:     list[dict]              = []
    vetoed:      list[tuple[float, float]] = []
    suggestions: list[dict]              = []
    iterations = 0

    try:
        while iterations < max_iterations:
            iterations += 1
            if _logger:
                _logger.log_agent_iteration(iterations, max_iterations)

            resp = _chat_ollama(messages, _AGENT_TOOLS, model, host, timeout)
            msg  = resp.get("message", {})

            # Capture thinking tokens (qwen3.5, deepseek-r1, etc.)
            if _logger:
                thinking = msg.get("thinking", "")
                if not thinking:
                    thinking, _ = _logger._extract_thinking(msg.get("content", ""))
                if thinking:
                    _logger.log_agent_thinking(thinking, iterations)
                _logger.log_agent_raw_response(json.dumps(resp))

            messages.append({
                "role":       "assistant",
                "content":    msg.get("content", ""),
                "tool_calls": msg.get("tool_calls", []),
            })

            tool_calls = msg.get("tool_calls", [])
            if not tool_calls:
                break  # LLM is done — no more tool calls

            tool_results = []
            for tc in tool_calls:
                fn_name = tc.get("function", {}).get("name", "")
                fn_args = tc.get("function", {}).get("arguments", {})
                if isinstance(fn_args, str):
                    try:
                        fn_args = json.loads(fn_args)
                    except Exception:
                        fn_args = {}

                if fn_name == "flag_clip":
                    seg, result_text = _execute_flag_clip(fn_args)
                    if seg:
                        flagged.append(seg)
                        if _logger:
                            _logger.log_agent_flag(
                                fn_args.get("start_time", ""),
                                fn_args.get("end_time", ""),
                                seg["score"],
                                "flagged",
                                seg.get("reason", ""),
                            )
                    tool_results.append(result_text)

                elif fn_name == "veto_clip":
                    veto_range, result_text = _execute_veto_clip(fn_args)
                    if veto_range:
                        vetoed.append(veto_range)
                        if _logger:
                            _logger.log_agent_veto(
                                fn_args.get("start_time", ""),
                                fn_args.get("end_time", ""),
                                fn_args.get("reason", ""),
                            )
                    tool_results.append(result_text)

                elif fn_name == "request_more_context":
                    if _logger:
                        _logger.log_agent_context_request(
                            fn_args.get("timestamp", ""),
                            fn_args.get("window_sec", 30),
                        )
                    result_text = _execute_request_context(fn_args, segments)
                    tool_results.append(result_text)

                elif fn_name == "suggest_setting_change":
                    suggestion, result_text = _execute_suggest_setting(fn_args, settings)
                    if suggestion:
                        suggestions.append(suggestion)
                        if _logger:
                            _logger.log_agent_suggestion(
                                suggestion["setting_name"],
                                suggestion["current_value"],
                                suggestion["suggested_value"],
                                suggestion["reason"],
                            )
                    tool_results.append(result_text)

                else:
                    tool_results.append(f"Unknown tool: {fn_name}")

            messages.append({
                "role":    "tool",
                "content": "\n".join(tool_results),
            })

        def _is_vetoed(seg: dict) -> bool:
            for v_start, v_end in vetoed:
                if min(seg["end_sec"], v_end) - max(seg["start_sec"], v_start) > 0:
                    return True
            return False

        result = [s for s in flagged if not _is_vetoed(s)]

        if _logger:
            _logger.log_agent_done(len(flagged), len(vetoed), len(result), iterations)
        if vetoed:
            print(f"       [Agent] {len(vetoed)} moment(s) vetoed by LLM.")
        print(f"       [Agent] {len(result)} clip(s) flagged after {iterations} iteration(s).")
        return result, suggestions

    except urllib.error.URLError as exc:
        if _logger:
            _logger.log_error_detail("analyze_transcript/URLError", exc)
        return [], []
    except Exception as exc:
        if _logger:
            _logger.log_error_detail("analyze_transcript", exc)
        return [], []
