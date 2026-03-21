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

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MEMORY_PATH  = os.path.join(_PROJECT_ROOT, "configuration", "comedy_memory.json")
_MEMORY_MAX   = 50


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
) -> dict | None:
    """Ask Ollama why a clip is funny. Returns {why_funny, humor_type, confidence} or None."""
    if not clip_transcript.strip():
        return None

    model   = settings.get("llm_model",   "llama3.2")
    host    = settings.get("llm_host",    "http://localhost:11434")
    timeout = int(settings.get("llm_timeout", 120))

    prompt = _WHY_FUNNY_SYSTEM.format(
        hints=", ".join(audio_hints) if audio_hints else "none",
        transcript=clip_transcript,
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
            sample  = e.get("text_sample", "")[:80].replace("\n", " ")
            why     = e.get("why_funny", "")
            htype   = e.get("humor_type", "other")
            lines.append(f'- [{htype}] "{sample}" → {why}')
        parts.append("\n".join(lines))

    parts.append(_SYSTEM_RULES)
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────
# Ollama REST call
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
) -> list[dict]:
    """Analyze a Whisper transcript via Ollama and return scored segments.

    Builds a dynamic system prompt from training profile + comedy memory.
    Appends window_signals (timestamped audio hints) when provided.
    Returns [] on any error so the caller can degrade gracefully.
    """
    if len(segments) < 3:
        return []

    transcript = format_transcript(segments)
    if not transcript.strip():
        return []

    model   = settings.get("llm_model",   "llama3.2")
    host    = settings.get("llm_host",    "http://localhost:11434")
    timeout = int(settings.get("llm_timeout", 120))

    try:
        import trainer as _trainer  # lazy — avoids circular import
        profile = _trainer.load_profile()
    except Exception:
        profile = None

    memory = load_comedy_memory()
    prompt = build_system_prompt(profile, memory)

    full_prompt = f"{prompt}\n\nTranscript:\n{transcript}"
    if window_signals:
        full_prompt += f"\n\nAudio signals detected:\n{window_signals}"

    try:
        raw = _query_ollama(full_prompt, model, host, timeout)
        return _parse_response(raw)
    except urllib.error.URLError:
        return []
    except Exception:
        return []
