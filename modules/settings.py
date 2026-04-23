# ============================================================
# settings.py  —  MODULE (imported by main.py and others)
# Manages persistent user settings stored in settings.json.
# ============================================================

import json
import os
import sys

_PROJECT_ROOT         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS_FILE         = os.path.join(_PROJECT_ROOT, "configuration", "settings.json")
_TRANSCRIPT_CACHE_DIR = os.path.join(_PROJECT_ROOT, "configuration", "transcript_cache")
_COMEDY_MEMORY_FILE   = os.path.join(_PROJECT_ROOT, "configuration", "comedy_memory.json")

_U = "\033[4m" if sys.stdout.isatty() else ""
_R = "\033[0m"  if sys.stdout.isatty() else ""


def _cls() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def _transcript_cache_count() -> int:
    if not os.path.isdir(_TRANSCRIPT_CACHE_DIR):
        return 0
    import glob as _glob
    return len(_glob.glob(os.path.join(_TRANSCRIPT_CACHE_DIR, "*.json")))


def _comedy_memory_count() -> int:
    try:
        with open(_COMEDY_MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return len(data) if isinstance(data, list) else 0
    except (OSError, json.JSONDecodeError):
        return 0


def _row(key: str, label: str, value: str, description: str = "") -> str:
    label_display = label
    if _U and label:
        idx = label.lower().find(key.lower())
        if idx >= 0:
            label_display = label[:idx] + _U + label[idx] + _R + label[idx + 1:]
    pad       = max(0, 20 - len(label))
    desc_part = f"   — {description}" if description else ""
    return f"  [{key}]  {label_display}{' ' * pad}: {value}{desc_part}"


DEFAULTS: dict = {
    "pre_event_seconds":            15,
    "post_event_seconds":           5,
    "delete_original_policy":       "ask",
    "use_whisper":                  True,
    "use_transcript_cache":         True,
    "whisper_model":                "small",
    "twitch_username":              "",
    "max_clips":                    5,
    "rms_threshold_factor":         3.0,
    "rms_min_sustain":              2,
    "laughter_burst_factor":        2.0,
    "laughter_burst_count":         4,
    "onset_novelty_threshold":      3.0,
    "voice_excitement_threshold":   2.0,
    "smart_score_heat_min":         0.20,
    "smart_score_window_min":       0.20,
    "downloader_backend":           "yt-dlp",
    "twitch_browser":               "chrome",
    "twitch_cookie_file":           "",
    "download_concurrent_fragments": 1,
    "whisper_language":             "auto",
    "max_clip_duration":            50,
    "clip_trim_to_peak":            True,
    "smart_heat_decay":             0.92,
    "smart_heat_threshold":         0.55,
    "smart_heat_min_duration":      2.0,
    "comedy_rise_factor":           4.0,
    "comedy_tail_sec":              6.0,
    "min_clip_score":               0.30,
    "use_llm_scoring":              True,
    "llm_window_weight":            0.60,
    "llm_model":                    "qwen3.5:9b",
    "llm_host":                     "http://localhost:11434",
    "llm_timeout":                  120,
    "llm_max_iterations":           8,
    "comedy_chat_spike_factor":     3.0,
    "comedy_scream_min_pitch":      300,
    "comedy_scream_max_duration":   1.8,
    "comedy_pitch_variance_thresh": 80.0,
    "comedy_silence_min_sec":       0.3,
    "comedy_silence_max_sec":       2.0,
    "debug_mode":                   False,
}


# ============================================================
# Core load / save
# ============================================================

def load() -> dict:
    if not os.path.exists(SETTINGS_FILE):
        save(DEFAULTS.copy())
        return DEFAULTS.copy()
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        merged = {**DEFAULTS, **data}
        if merged != data:
            save(merged)
        return merged
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[WARNING] Could not read settings.json ({exc}). Using defaults.")
        return DEFAULTS.copy()


def save(settings: dict) -> None:
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4)
    except OSError as exc:
        print(f"[WARNING] Could not save settings.json: {exc}")


# ============================================================
# Individual field helpers
# ============================================================

def set_twitch_username(username: str) -> None:
    s = load(); s["twitch_username"] = username; save(s)

def clear_twitch_username() -> None:
    set_twitch_username("")

def get_delete_policy() -> str:
    return load()["delete_original_policy"]

def set_delete_policy(policy: str) -> None:
    s = load(); s["delete_original_policy"] = policy; save(s)


# ============================================================
# Interactive settings menu
# ============================================================

_WHISPER_MODELS  = ["small", "medium"]
_DELETE_POLICIES = ["ask", "always", "never"]


def _prompt_int(prompt: str, current: int, min_val: int = 1, max_val: int = 300) -> int:
    while True:
        raw = input(f"{prompt} [{current}]: ").strip()
        if raw == "": return current
        try:
            val = int(raw)
            if min_val <= val <= max_val: return val
            print(f"  Enter a whole number between {min_val} and {max_val}.")
        except ValueError:
            print("  Enter a whole number.")


def _prompt_float(prompt: str, current: float, min_val: float, max_val: float) -> float:
    while True:
        raw = input(f"{prompt} [{current}]: ").strip()
        if raw == "": return current
        try:
            val = float(raw)
            if min_val <= val <= max_val: return val
            print(f"  Enter a number between {min_val} and {max_val}.")
        except ValueError:
            print("  Enter a number (decimals allowed).")


def _prompt_choice(prompt: str, options: list[str], current: str) -> str:
    options_str = " / ".join(f'"{o}"' for o in options)
    while True:
        raw = input(f"{prompt} ({options_str}) [{current}]: ").strip().lower()
        if raw == "": return current
        if raw in options: return raw
        print(f"  Valid options: {options_str}")


def show_menu() -> None:
    while True:
        _cls()
        s        = load()
        username = s["twitch_username"] or "(not connected)"

        print("=" * 60)
        print("  Sundown — Settings")
        print(f"  Account: @{username}")
        print("=" * 60)

        print("\n  ── Clip output ──────────────────────────────────────────")
        print(_row("p", "Pre-event pad",    f"{s['pre_event_seconds']}s",                    "seconds captured before a trigger"))
        print(_row("o", "Post-event pad",   f"{s['post_event_seconds']}s",                   "seconds captured after a trigger"))
        print(_row("x", "Max clip length",  f"{s['max_clip_duration']}s",                    "hard cap; clips longer than this are split"))
        print(_row("1", "Trim to peak",     "on" if s.get("clip_trim_to_peak", True) else "off", "auto-shrink each clip to its loudest moment + padding"))
        print(_row("d", "Delete policy",    s["delete_original_policy"],                     "what to do with source file after clipping"))

        print("\n  ── Detection sensitivity ────────────────────────────────")
        sustain_sec = float(s.get("rms_min_sustain", 2)) * 0.5
        print(_row("l", "Loud threshold",       str(s["rms_threshold_factor"]),                   "RMS multiplier (lower = more clips)"))
        print(_row("f", "Transient filter",      f"{s.get('rms_min_sustain', 2)} frames ({sustain_sec:.1f}s)", "min sustained loudness; filters gunshots/impacts"))
        print(_row("y", "Laughter sensitivity",  str(s["laughter_burst_factor"]),                  "energy per burst (lower = more sensitive)"))
        print(_row("b", "Laughter burst count",  str(s["laughter_burst_count"]),                   "bursts needed in 2s to fire"))
        print(_row("i", "Onset novelty",         str(s.get("onset_novelty_threshold", 3.0)),       "z-score above local baseline — self-calibrates to game noise"))
        print(_row("s", "Voice excitement",      str(s.get("voice_excitement_threshold", 2.0)),    "voice-band × centroid ratio above baseline — detects screaming/hype"))

        print("\n  ── Speech recognition ───────────────────────────────────")
        print(_row("w", "Use Whisper",       "on" if s.get("use_whisper", True) else "off",      "off = skip transcription, use heat + RMS only (faster)"))
        print(_row("h", "Transcript cache",  "on" if s.get("use_transcript_cache", True) else "off", "reuse previous transcription on re-runs"))
        print(_row("m", "Whisper model",  s["whisper_model"],                "small / medium"))
        print(_row("g", "Language",       s.get("whisper_language", "auto"), "auto or ISO code: en es pt fr de ja ko zh ..."))

        print("\n  ── Clipping ─────────────────────────────────────────────")
        print(_row("n", "Max clips",      str(s.get("max_clips", 5)),          "max clips produced per run"))
        print(_row("2", "Min clip score", str(s.get("min_clip_score", 0.30)),  "quality floor: clips below this score are dropped (0 = off)"))

        print("\n  ── Smart Clip — heat model ──────────────────────────────")
        print(_row("e", "Heat decay",        str(s.get("smart_heat_decay", 0.92)),       "how fast excitement fades (higher = lingers longer)"))
        print(_row("t", "Heat threshold",    str(s.get("smart_heat_threshold", 0.55)),   "excitement needed to open a zone (lower = more clips)"))
        print(_row("z", "Min zone duration", f"{s.get('smart_heat_min_duration', 2.0)}s","shortest burst that qualifies as a clip zone"))
        print(_row("j", "Min window score",  str(s.get("smart_score_window_min", 0.20)), "dual filter: min audio score a clip must reach (0 = off)"))
        print(_row("u", "Min heat score",    str(s.get("smart_score_heat_min", 0.20)),   "dual filter: min heat level a clip must reach (0 = off)"))

        print("\n  ── Downloader ───────────────────────────────────────────")
        print(_row("a", "Backend",              s["downloader_backend"],                              "yt-dlp (recommended) or streamlink"))
        print(_row("r", "Browser",              s["twitch_browser"],                                  "browser yt-dlp reads Twitch cookies from"))
        print(_row("c", "Cookie file",          s["twitch_cookie_file"] or "(not set — using browser)", "Netscape cookies.txt (overrides browser, fixes DPAPI errors)"))
        print(_row("k", "Concurrent fragments", str(s.get("download_concurrent_fragments", 1)),       "parallel HLS segments per download (1 = safe, 4–8 = faster)"))

        print("\n  ── AI Scoring (Ollama) ──────────────────────────────────")
        llm_on = s.get("use_llm_scoring", False)
        print(_row("3", "LLM scoring", f"{'on' if llm_on else 'off'}  model={s.get('llm_model', 'qwen3.5:9b')}", "Ollama virality scoring — free & local, no API key needed"))

        print("\n  ── Debug ────────────────────────────────────────────────")
        print(_row("v", "Debug logging", "on" if s.get("debug_mode", False) else "off", "write raw LLM prompts and full responses to log file"))

        print("\n  ── Account ──────────────────────────────────────────────")
        print(_row("q", "Erase Twitch credentials", "", "revoke and remove stored auth token"))

        print("\n  ── Cache ────────────────────────────────────────────────")
        tc = _transcript_cache_count()
        mc = _comedy_memory_count()
        print(_row("4", "Clear transcript cache", f"{tc} file(s)", "delete all cached whisper transcriptions"))
        print(_row("5", "Clear comedy memory",    f"{mc} entry(s)", "wipe LLM training memory (rebuild on next train)"))

        print("\n  !! DANGER ZONE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(_row("6", "Reset all settings",     "",               "restore every setting to its default value"))
        print("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        print("\n  Press 0 or Enter to go back")
        print("-" * 60)

        choice = input("  > ").strip().lower()

        if choice in ("0", ""):
            return

        elif choice == "p":
            print("\n  Seconds of video captured BEFORE the trigger moment.")
            s["pre_event_seconds"] = _prompt_int("  Pre-event pad (s)", s["pre_event_seconds"])
            save(s); print("  Saved.")

        elif choice == "o":
            print("\n  Seconds of video captured AFTER the trigger moment.")
            s["post_event_seconds"] = _prompt_int("  Post-event pad (s)", s["post_event_seconds"])
            save(s); print("  Saved.")

        elif choice == "x":
            s["max_clip_duration"] = _prompt_int(
                "  Max clip duration (s)", s["max_clip_duration"], min_val=10, max_val=300)
            save(s); print("  Saved.")

        elif choice == "1":
            s["clip_trim_to_peak"] = not s.get("clip_trim_to_peak", True)
            save(s); print(f"  Trim to peak is now {'on' if s['clip_trim_to_peak'] else 'off'}.")

        elif choice == "d":
            print('\n  "ask" → prompt each time  "always" → always delete  "never" → never delete')
            s["delete_original_policy"] = _prompt_choice(
                "  Delete policy", _DELETE_POLICIES, s["delete_original_policy"])
            save(s); print("  Saved.")

        elif choice == "l":
            print("\n  RMS threshold multiplier. Lower = more clips; higher = only loudest moments.\n"
                  "  A frame triggers when its volume exceeds (median × this value).")
            s["rms_threshold_factor"] = _prompt_float(
                "  Loud moment sensitivity", s["rms_threshold_factor"], 1.0, 10.0)
            save(s); print("  Saved.")

        elif choice == "f":
            print("\n  Consecutive 0.5s frames that must stay loud before triggering.\n"
                  "  1 = any spike  |  2 = 1.0s sustained (default)  |  3 = 1.5s sustained")
            s["rms_min_sustain"] = _prompt_int(
                "  Transient filter (frames)", s.get("rms_min_sustain", 2), 1, 10)
            save(s); print("  Saved.")

        elif choice == "y":
            print("\n  Energy threshold per laughter burst. Lower = more sensitive to quiet laughter.")
            s["laughter_burst_factor"] = _prompt_float(
                "  Laughter sensitivity", s["laughter_burst_factor"], 1.0, 5.0)
            save(s); print("  Saved.")

        elif choice == "b":
            print("\n  Rapid energy bursts needed within 2s to classify a moment as laughter.\n"
                  "  Lower = more sensitive; higher = only obvious repeated laughter.")
            s["laughter_burst_count"] = _prompt_int(
                "  Laughter burst count", s["laughter_burst_count"], 2, 10)
            save(s); print("  Saved.")

        elif choice == "i":
            print("\n  Onset novelty threshold (z-score above local 60s baseline).\n"
                  "  Self-calibrates per game. 3.0 = default; 2.0 = more sensitive; 5.0 = only big bursts.")
            s["onset_novelty_threshold"] = _prompt_float(
                "  Onset novelty threshold", s.get("onset_novelty_threshold", 3.0), 1.5, 8.0)
            save(s); print("  Saved.")

        elif choice == "s":
            print("\n  Voice excitement threshold. Detects screaming/hype via voice-band energy\n"
                  "  and spectral brightness vs rolling baseline. Filters game SFX.\n"
                  "  2.0 = default; lower = more sensitive; higher = only very excited screaming.")
            s["voice_excitement_threshold"] = _prompt_float(
                "  Voice excitement threshold", s.get("voice_excitement_threshold", 2.0), 1.2, 6.0)
            save(s); print("  Saved.")

        elif choice == "w":
            s["use_whisper"] = not s.get("use_whisper", True)
            save(s); print(f"  Whisper is now {'on' if s['use_whisper'] else 'off'}.")

        elif choice == "h":
            s["use_transcript_cache"] = not s.get("use_transcript_cache", True)
            save(s); print(f"  Transcript cache is now {'on' if s['use_transcript_cache'] else 'off'}.")

        elif choice == "m":
            print("\n  small = faster, good accuracy  |  medium = most accurate, slower")
            s["whisper_model"] = _prompt_choice("  Whisper model", _WHISPER_MODELS, s["whisper_model"])
            save(s); print("  Saved.")

        elif choice == "g":
            print("\n  Common codes: en  es  pt  fr  de  it  ja  ko  zh  ru  ar\n"
                  "  'auto' = detect automatically (slower first pass).")
            raw = input("  Stream language [auto]: ").strip().lower()
            s["whisper_language"] = raw if raw else "auto"
            save(s); print("  Saved.")

        elif choice == "n":
            print("\n  Max clips produced per run. Smart Clip keeps top N by score;\n"
                  "  Fast Clip keeps first N by time.")
            s["max_clips"] = _prompt_int("  Max clips", s.get("max_clips", 5), 1, 20)
            save(s); print("  Saved.")

        elif choice == "2":
            print("\n  Quality floor: clips scoring below this are dropped even if fewer than max remain.\n"
                  "  0.30 (default) filters weak detections. 0.0 = always produce up to max clips.")
            s["min_clip_score"] = _prompt_float(
                "  Min clip score (0–1)", s.get("min_clip_score", 0.30), 0.0, 0.95)
            save(s); print("  Saved.")

        elif choice == "e":
            print("\n  How fast heat decays. 0.92 = natural feel. Higher = lingers. Lower = drops fast.")
            s["smart_heat_decay"] = _prompt_float(
                "  Heat decay per frame", s.get("smart_heat_decay", 0.92), 0.50, 0.99)
            save(s); print("  Saved.")

        elif choice == "t":
            print("\n  Heat level to open a clip zone. Lower = more zones; higher = only intense moments.")
            s["smart_heat_threshold"] = _prompt_float(
                "  Heat threshold (0.0–1.0)", s.get("smart_heat_threshold", 0.55), 0.10, 0.95)
            save(s); print("  Saved.")

        elif choice == "z":
            print("\n  Minimum seconds a zone must stay hot. 2.0s filters brief game sound spikes.")
            s["smart_heat_min_duration"] = _prompt_float(
                "  Min zone duration (s)", s.get("smart_heat_min_duration", 2.0), 0.5, 10.0)
            save(s); print("  Saved.")

        elif choice == "j":
            print("\n  Dual filter: clip must reach BOTH window score (j) AND heat score (u).\n"
                  "  0.20 (default) removes weakly-detected moments. 0.0 = disable.")
            s["smart_score_window_min"] = _prompt_float(
                "  Min window score (0–1)", s.get("smart_score_window_min", 0.20), 0.0, 0.95)
            save(s); print("  Saved.")

        elif choice == "u":
            print("\n  Dual filter: clip must reach BOTH heat score (u) AND window score (j).\n"
                  "  0.0 = disable heat filtering.")
            s["smart_score_heat_min"] = _prompt_float(
                "  Min heat score (0–1)", s.get("smart_score_heat_min", 0.20), 0.0, 0.95)
            save(s); print("  Saved.")

        elif choice == "a":
            print("\n  yt-dlp (recommended) uses browser cookies.\n"
                  "  streamlink requires a browser-session OAuth token.")
            s["downloader_backend"] = _prompt_choice(
                "  Downloader backend", ["yt-dlp", "streamlink"], s["downloader_backend"])
            save(s); print("  Saved.")

        elif choice == "r":
            print("\n  Browser yt-dlp reads Twitch cookies from. Must be logged into Twitch.\n"
                  '  Use "none" for public VODs only.')
            s["twitch_browser"] = _prompt_choice(
                "  Browser", ["chrome", "edge", "firefox", "brave", "zen", "none"],
                s["twitch_browser"])
            save(s); print("  Saved.")

        elif choice == "c":
            print("\n  Netscape cookies.txt path (exported from browser). Fixes Chrome DPAPI errors.\n"
                  "  Use 'Get cookies.txt LOCALLY' extension. Leave blank to keep current value.")
            raw = input(f"  Cookie file path [{s['twitch_cookie_file'] or 'none'}]: ").strip()
            if raw == "":
                pass
            elif raw.lower() in ("none", "clear", "-"):
                s["twitch_cookie_file"] = ""; save(s); print("  Cleared — will use browser cookies.")
            elif os.path.isfile(raw):
                s["twitch_cookie_file"] = raw; save(s); print("  Saved.")
            else:
                print(f"  [ERROR] File not found: {raw}")

        elif choice == "k":
            print("\n  Parallel HLS segments for yt-dlp. 1 = safest. 4–8 = faster on good connections.")
            s["download_concurrent_fragments"] = _prompt_int(
                "  Concurrent fragments", s.get("download_concurrent_fragments", 1), 1, 32)
            save(s); print("  Saved.")

        elif choice == "3":
            if s.get("use_llm_scoring", False):
                print("\n  LLM scoring is currently ON.\n  [1] Turn off   [2] Change model/host   [0] Cancel")
                sub = input("  > ").strip()
                if sub == "1":
                    s["use_llm_scoring"] = False; save(s); print("  LLM scoring disabled.")
                elif sub == "2":
                    raw = input(f"  Ollama model [{s.get('llm_model', 'qwen3.5:9b')}]: ").strip()
                    if raw: s["llm_model"] = raw
                    raw = input(f"  Ollama host [{s.get('llm_host', 'http://localhost:11434')}]: ").strip()
                    if raw: s["llm_host"] = raw
                    save(s); print("  Saved.")
            else:
                print("\n  LLM scoring uses Ollama to score clips for virality (hook, engagement, value, shareability).\n"
                      "  Requires Ollama: https://ollama.com\n"
                      "  Recommended: qwen3.5:9b  phi4-mini  gemma3:4b\n")
                raw = input(f"  Ollama model [{s.get('llm_model', 'qwen3.5:9b')}]: ").strip()
                if raw: s["llm_model"] = raw
                raw = input(f"  Ollama host [{s.get('llm_host', 'http://localhost:11434')}]: ").strip()
                if raw: s["llm_host"] = raw
                raw = input(f"  Timeout seconds [{s.get('llm_timeout', 90)}]: ").strip()
                if raw:
                    try: s["llm_timeout"] = max(10, int(raw))
                    except ValueError: pass
                s["use_llm_scoring"] = True; save(s)
                print(f"  LLM scoring enabled (model: {s['llm_model']}, host: {s['llm_host']}).\n"
                      f"  Make sure Ollama is running and the model is pulled:\n"
                      f"    ollama pull {s['llm_model']}")

        elif choice == "v":
            s["debug_mode"] = not s.get("debug_mode", False)
            save(s)
            state = "on" if s["debug_mode"] else "off"
            print(f"  Debug logging is now {state}.")
            try:
                import logger as _logger
                _logger.set_debug(s["debug_mode"])
            except Exception:
                pass

        elif choice == "q":
            import downloader as _downloader
            _downloader.erase_credentials()

        elif choice == "4":
            tc = _transcript_cache_count()
            if tc == 0:
                print("\n  No transcript cache files found.")
            else:
                ans = input(f"\n  Delete {tc} cached transcript file(s)? [y/N]: ").strip().lower()
                if ans == "y":
                    import glob as _glob
                    deleted = 0
                    for p in _glob.glob(os.path.join(_TRANSCRIPT_CACHE_DIR, "*.json")):
                        try:
                            os.remove(p)
                            deleted += 1
                        except OSError:
                            pass
                    print(f"  ✓ {deleted} file(s) deleted.")

        elif choice == "5":
            mc = _comedy_memory_count()
            if mc == 0:
                print("\n  Comedy memory is already empty.")
            else:
                ans = input(f"\n  Clear {mc} comedy memory entry(s)? [y/N]: ").strip().lower()
                if ans == "y":
                    try:
                        with open(_COMEDY_MEMORY_FILE, "w", encoding="utf-8") as f:
                            json.dump([], f)
                        print("  ✓ Comedy memory cleared.")
                    except OSError as exc:
                        print(f"  [ERROR] {exc}")

        elif choice == "6":
            ans = input("\n  Reset ALL settings to defaults? This cannot be undone. [y/N]: ").strip().lower()
            if ans == "y":
                save(DEFAULTS.copy())
                print("  ✓ All settings reset to defaults.")

        else:
            print(f"\n  '{choice}' is not a valid option.")
