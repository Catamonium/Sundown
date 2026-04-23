# ============================================================
# clipper.py  —  MODULE (imported by main.py, not run directly)
# Fast-clip pipeline: audio analysis + Whisper keyword detection.
# ============================================================

import os
import sys
import glob
import subprocess
import tempfile
from dataclasses import dataclass

import numpy as np
import settings as _settings
import trainer as _trainer

import shutil as _shutil

try:
    import logger as _logger
except ImportError:
    _logger = None  # type: ignore[assignment]

# ============================================================
# Folders
# ============================================================

INPUT_DIR        = "input"
CLIPS_DIR        = "clips"
_SMART_FRAME_SEC = 0.05   # 50 ms frames used by the heat pipeline

# Transcript cache — stores merged Whisper segments so re-runs skip transcription
_CACHE_DIR     = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configuration", "transcript_cache"
)
_CACHE_VERSION = 1


# ============================================================
# Data types
# ============================================================

@dataclass
class Trigger:
    timestamp: float   # seconds into the video
    source: str        # "rms_spike" | "laughter" | "keyword"
    label: str         # human-readable description


# ============================================================
# whisper.cpp helpers
# ============================================================

def _find_whisper_cpp() -> str | None:
    """Return the path to the whisper.cpp binary, or None if not found.

    Search order:
    1. WHISPER_CPP_PATH environment variable
    2. project_root/Whisper/whisper-cli.exe  (current whisper.cpp naming)
    3. project_root/Whisper/main.exe         (legacy whisper.cpp naming)
    4. build/ subdirectory variants of both names
    """
    env = os.environ.get("WHISPER_CPP_PATH", "").strip()
    if env and os.path.isfile(env):
        return env
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(root, "Whisper", "whisper-cli.exe"),
        os.path.join(root, "Whisper", "main.exe"),
        os.path.join(root, "Whisper", "build", "bin", "whisper-cli.exe"),
        os.path.join(root, "Whisper", "build", "bin", "main.exe"),
        os.path.join(root, "Whisper", "whisper-cli"),
        os.path.join(root, "Whisper", "main"),
        os.path.join(root, "Whisper", "build", "bin", "whisper-cli"),
        os.path.join(root, "Whisper", "build", "bin", "main"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


_WHISPER_CPP_BIN: str | None = _find_whisper_cpp()


def _whisper_cpp_model_path(model_name: str) -> str | None:
    """Return path to the ggml model file for whisper.cpp.

    Search order:
    1. WHISPER_CPP_MODELS_DIR env var + /ggml-{model_name}.bin
    2. project_root/Whisper/models/ggml-{model_name}.bin
    3. project_root/Whisper/ggml-{model_name}.bin
    """
    root  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fname = f"ggml-{model_name}.bin"
    env_dir = os.environ.get("WHISPER_CPP_MODELS_DIR", "").strip()
    candidates = []
    if env_dir:
        candidates.append(os.path.join(env_dir, fname))
    candidates += [
        os.path.join(root, "Whisper", "models", fname),
        os.path.join(root, "Whisper", fname),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _parse_whispercpp_timestamp(ts: str) -> float:
    """Convert 'HH:MM:SS,mmm' (whisper.cpp JSON format) to float seconds."""
    try:
        hms, ms = ts.strip().split(",")
        h, m, s = hms.split(":")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
    except Exception:
        return 0.0


# ============================================================
# File picker
# ============================================================

def _list_video_files() -> list[str]:
    """Return all video files found in input/ and downloads/."""
    extensions = ("*.mp4", "*.mkv", "*.mov", "*.avi", "*.ts", "*.flv")
    files: list[str] = []
    for folder in (INPUT_DIR, "downloads"):
        for ext in extensions:
            files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)


def pick_input_file() -> str | None:
    """List available video files and let the user pick one.

    Returns the chosen file path, or None if cancelled.
    """
    files = _list_video_files()

    if not files:
        print(
            f"\n[INFO] No video files found.\n"
            f"       Drop your video into the '{INPUT_DIR}/' folder and try again.\n"
            f"       Or download a Twitch VOD first (option [1])."
        )
        return None

    print(f"\n{'='*60}")
    print("  Select a video file to clip")
    print(f"{'='*60}")
    for i, path in enumerate(files, start=1):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  [{i}] {path}  ({size_mb:.1f} MB)")
    print("  [0] Cancel")
    print("-" * 60)

    while True:
        raw = input("Select a file: ").strip()
        if raw == "0":
            return None
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(files):
                return files[idx]
        except ValueError:
            pass
        print(f"  Please enter a number between 0 and {len(files)}.")


# ============================================================
# Audio extraction
# ============================================================

def extract_audio(video_path: str, tmp_dir: str) -> str:
    """Extract mono 16 kHz WAV from the video using ffmpeg.

    Returns the path to the temporary WAV file.
    """
    wav_path = os.path.join(tmp_dir, "audio.wav")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ac", "1",          # mono
        "-ar", "16000",      # 16 kHz  (Whisper's native rate)
        "-vn",               # no video
        wav_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] ffmpeg audio extraction failed:\n{result.stderr[-800:]}")
        sys.exit(1)
    return wav_path


def _load_wav_as_array(wav_path: str) -> tuple[np.ndarray, int]:
    """Read a WAV file into a float32 numpy array.

    Returns (samples, sample_rate).
    """
    import wave, struct
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        n  = wf.getnframes()
        raw = wf.readframes(n)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sr


def _write_temp_wav(samples: np.ndarray, sr: int, tmp_dir: str, prefix: str = "slice_") -> str:
    """Write a float32 numpy array to a 16-bit mono WAV file in tmp_dir."""
    import wave
    path = os.path.join(tmp_dir, f"{prefix}{id(samples)}.wav")
    int16 = (samples * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int16.tobytes())
    return path


def _transcript_cache_path(video_path: str) -> str:
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(_CACHE_DIR, f"{stem}_transcript.json")


def _load_transcript_cache(video_path: str) -> list[dict] | None:
    import json
    path = _transcript_cache_path(video_path)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") == _CACHE_VERSION and isinstance(data.get("segments"), list):
            return data["segments"]
    except (OSError, json.JSONDecodeError, KeyError):
        pass
    return None


def _save_transcript_cache(video_path: str, segments: list[dict]) -> None:
    import json
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = _transcript_cache_path(video_path)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": _CACHE_VERSION, "segments": segments}, f)
    except OSError as exc:
        print(f"[WARNING] Could not write transcript cache: {exc}")


def _transcribe_full_vod(
    wav_path: str,
    model_name: str,
    language: str,
    video_path: str,
    use_cache: bool,
) -> list[dict]:
    """Transcribe the full VOD audio with whisper.cpp and return all segments.

    Checks the transcript cache first — returns cached segments immediately on hit.
    Saves results to cache after a fresh transcription.
    Returns [] gracefully on any failure.
    """
    if use_cache:
        cached = _load_transcript_cache(video_path)
        if cached is not None:
            print("[INFO] Transcript cache hit — skipping transcription.")
            return cached
    try:
        _, segments = _run_whisper(
            wav_path, model_name, language=language, verbose=True
        )
        segments = segments or []
        if use_cache and segments:
            _save_transcript_cache(video_path, segments)
        return segments
    except Exception as exc:
        print(f"[WARNING] Transcription failed: {exc}")
        return []


# ============================================================
# Clip preference profile  (Smart Clip learning)
# ============================================================


def _profile_clip(samples: np.ndarray, sr: int, s: dict) -> dict:
    """Extract the audio+trigger feature vector for one training clip.

    Runs all 4 signal detectors and the excitement curve; returns a dict
    describing this clip's audio profile.
    """
    duration = max(len(samples) / sr, 1e-3)

    rms_t   = _rms_spike_detection(samples, sr,
                  threshold_factor=float(s.get("rms_threshold_factor", 3.0)),
                  min_sustain_frames=int(s.get("rms_min_sustain", 2)))
    laugh_t = _laughter_detection(samples, sr,
                  burst_threshold_factor=float(s.get("laughter_burst_factor", 2.0)),
                  burst_count=int(s.get("laughter_burst_count", 4)))
    onset_t = _onset_novelty_detection(samples, sr,
                  novelty_threshold=float(s.get("onset_novelty_threshold", 3.0)))
    voice_t = _voice_excitement_detection(samples, sr,
                  threshold_factor=float(s.get("voice_excitement_threshold", 2.0)))
    all_t   = rms_t + laugh_t + onset_t + voice_t

    trigger_density = len(all_t) / duration
    presence = {
        "rms_spike":       1.0 if rms_t   else 0.0,
        "laughter":        1.0 if laugh_t  else 0.0,
        "onset_novelty":   1.0 if onset_t  else 0.0,
        "voice_excitement":1.0 if voice_t  else 0.0,
    }

    # RMS mean
    frame_len = max(1, int(sr * 0.5))
    n_frames  = len(samples) // frame_len
    if n_frames > 0:
        rms_vals = np.array([
            np.sqrt(np.mean(samples[i*frame_len:(i+1)*frame_len] ** 2))
            for i in range(n_frames)
        ])
        rms_mean = float(np.mean(rms_vals))
    else:
        rms_mean = 0.0

    # Spectral features (librosa optional)
    voice_ratio_mean = 0.5
    centroid_mean    = 2000.0
    flux_variance    = 0.0
    try:
        import librosa
        hop = int(sr * _SMART_FRAME_SEC)
        stft = np.abs(librosa.stft(y=samples.astype(float), hop_length=hop))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=(stft.shape[0] - 1) * 2)
        voice_mask   = (freqs >= 150) & (freqs <= 3000)
        total_energy = stft.mean(axis=0) + 1e-8
        voice_ratio  = stft[voice_mask, :].mean(axis=0) / total_energy
        voice_ratio_mean = float(np.mean(voice_ratio))

        centroid = librosa.feature.spectral_centroid(S=stft ** 2, sr=sr, hop_length=hop)[0]
        centroid_mean = float(np.mean(centroid))

        flux = np.mean(np.abs(np.diff(stft, axis=1)), axis=0)
        flux_variance = float(np.var(flux))
    except ImportError:
        pass

    # ── Comedy features (audio-only; whisper/chat features are 0.0 here) ──
    duration_min = max(duration / 60.0, 1e-3)
    scream_t  = _scream_detection(samples, sr)
    silence_t = _pre_reaction_silence_detection(samples, sr)
    crack_t   = _voice_crack_detection(samples, sr)

    pitch_variance_mean = 0.0
    try:
        import librosa as _lr_cv
        _hop = max(1, int(sr * 0.1))
        _f0  = _lr_cv.yin(
            samples.astype(float), fmin=65.0,
            fmax=float(min(sr // 2 - 1, 2000)), sr=sr, hop_length=_hop,
        )
        _win = max(1, int(3.0 / (_hop / sr)))
        _stds = []
        for _j in range(_win, len(_f0)):
            _w = _f0[_j - _win:_j]
            _v = _w[_w > 80.0]
            if len(_v) >= _win // 3:
                _stds.append(float(np.std(_v)))
        if _stds:
            pitch_variance_mean = float(np.mean(_stds))
    except Exception:
        pass

    comedy_features = {
        "swear_density":       0.0,
        "scream_presence":     1.0 if scream_t  else 0.0,
        "nonsense_density":    0.0,
        "pitch_variance_mean": pitch_variance_mean,
        "pre_silence_count":   len(silence_t) / duration_min,
        "voice_crack_count":   len(crack_t)   / duration_min,
        "chat_spike_density":  0.0,
    }

    return {
        "rms_mean":           rms_mean,
        "voice_ratio_mean":   voice_ratio_mean,
        "centroid_mean":      centroid_mean,
        "flux_variance":      flux_variance,
        "trigger_density":    trigger_density,
        "trigger_presence":   presence,
        "comedy_features":    comedy_features,
    }


def _whisper_on_windows(
    samples: np.ndarray,
    sr: int,
    windows: list[tuple[float, float, list[str]]],
    model_name: str,
    language: str,
    tmp_dir: str,
    video_path: str,
    use_cache: bool = True,
) -> tuple[list[Trigger], list[dict]]:
    """Run Whisper only on the audio slices of candidate windows.

    Instead of transcribing the full VOD, only the pre-scored candidate
    clip windows are fed to Whisper — typically a 10–30x reduction in
    audio processed.  Timestamps in the returned segments and triggers are
    already offset to their true VOD positions.

    Returns (keyword_triggers, all_segments_with_vod_timestamps).
    """
    all_kw_triggers: list[Trigger] = []
    all_segments:    list[dict]    = []

    # Cache check — load merged segments from a previous run
    if use_cache:
        cached = _load_transcript_cache(video_path)
        if cached is not None:
            print("[INFO] Whisper transcript cache hit — skipping transcription.")
            all_kw_triggers = _scan_keywords(cached)
            print(f"[INFO] Cached keyword scan: {len(all_kw_triggers)} hit(s).")
            return all_kw_triggers, cached

    total = len(windows)
    print(f"[INFO] Running whisper.cpp on {total} window(s) (model='{model_name}')...")
    bar_width = 28
    for idx, (start, end, _labels) in enumerate(windows):
        slice_samples = samples[int(start * sr): int(end * sr)]
        slice_wav = _write_temp_wav(slice_samples, sr, tmp_dir, prefix=f"w{idx}_")
        kw, segs = _run_whisper(slice_wav, model_name,
                                language=language, verbose=False)

        # Overwrite single line with \r progress bar
        filled = bar_width * (idx + 1) // total
        bar = "=" * filled + "-" * (bar_width - filled)
        print(f"\r       [{bar}] {idx+1}/{total}  {start:.0f}s–{end:.0f}s",
              end="", flush=True)

        for t in kw:
            all_kw_triggers.append(Trigger(start + t.timestamp, t.source, t.label))
        for seg in segs:
            all_segments.append({**seg,
                                  "start": seg.get("start", 0.0) + start,
                                  "end":   seg.get("end",   0.0) + start})

    print()  # newline after the \r bar

    if use_cache:
        _save_transcript_cache(video_path, all_segments)

    print(f"[INFO] Targeted Whisper complete: {len(all_kw_triggers)} keyword hit(s) total.")
    return all_kw_triggers, all_segments


# ============================================================
# Signal detectors
# ============================================================

def _rms_spike_detection(
    samples: np.ndarray,
    sr: int,
    frame_sec: float = 0.5,
    threshold_factor: float = 3.0,
    min_sustain_frames: int = 2,
) -> list[Trigger]:
    """Detect sustained loud moments (screaming, hype).

    Splits audio into frames, computes RMS per frame, and flags runs of
    min_sustain_frames consecutive frames above threshold_factor × median.

    min_sustain_frames filters out brief transients like gunshots: at 0.5 s
    frames, the default of 2 requires the loudness to last at least 1 second.
    """
    frame_len = int(sr * frame_sec)
    n_frames  = len(samples) // frame_len
    rms_vals  = np.array([
        np.sqrt(np.mean(samples[i * frame_len:(i + 1) * frame_len] ** 2))
        for i in range(n_frames)
    ])

    median_rms = np.median(rms_vals)
    if median_rms == 0:
        return []

    threshold = median_rms * threshold_factor
    triggers  = []
    run = 0  # consecutive frames currently above threshold
    for i, rms in enumerate(rms_vals):
        if rms > threshold:
            run += 1
            if run == min_sustain_frames:
                # Emit trigger at the start of this sustained run
                ts = (i - min_sustain_frames + 1) * frame_sec
                triggers.append(Trigger(ts, "rms_spike",
                                        f"Loud moment (RMS ×{rms/median_rms:.1f}, "
                                        f"{min_sustain_frames * frame_sec:.1f}s sustained)"))
        else:
            run = 0

    return triggers


def _laughter_detection(
    samples: np.ndarray,
    sr: int,
    frame_sec: float = 0.1,
    burst_threshold_factor: float = 2.0,
    burst_count: int = 4,
    burst_window_sec: float = 2.0,
) -> list[Trigger]:
    """Heuristic laughter detector.

    Looks for rapid-fire rhythmic bursts of audio energy — short peaks
    repeating several times within a few seconds.
    """
    frame_len  = int(sr * frame_sec)
    n_frames   = len(samples) // frame_len
    rms_vals   = np.array([
        np.sqrt(np.mean(samples[i * frame_len:(i + 1) * frame_len] ** 2))
        for i in range(n_frames)
    ])

    median_rms = np.median(rms_vals)
    if median_rms == 0:
        return []

    threshold = median_rms * burst_threshold_factor
    burst_frames_needed = int(burst_window_sec / frame_sec)

    triggers: list[Trigger] = []
    i = 0
    while i < n_frames - burst_frames_needed:
        window = rms_vals[i : i + burst_frames_needed]
        peaks  = np.sum(window > threshold)
        if peaks >= burst_count:
            ts = i * frame_sec
            triggers.append(Trigger(ts, "laughter", f"Possible laughter ({peaks} bursts)"))
            i += burst_frames_needed  # skip ahead past this window
        else:
            i += 1

    return triggers


def _onset_novelty_detection(
    samples: np.ndarray,
    sr: int,
    novelty_threshold: float = 3.0,
    min_sustain_sec: float = 0.5,
) -> list[Trigger]:
    """Detect audio novelty bursts using a local z-score of onset strength.

    Unlike raw onset density (which fires on every gunshot), this uses the
    rate-of-change of spectral energy normalised against a ±60s local window.
    Steady periodic sounds (game music, recurring SFX) have low novelty and
    don't trigger.  Genuine bursts of unusual activity produce sustained
    z-score spikes above the threshold.

    min_sustain_sec filters out isolated transients (<100ms): a gunshot lasts
    1–2 onset frames (~32ms each); a real hype burst lasts many frames.
    """
    try:
        import librosa
    except ImportError:
        print("[WARNING] librosa not available — onset novelty detection skipped.")
        return []

    # onset_strength(aggregate=median) suppresses steady periodic content
    hop_length = 512  # ~32ms at 16kHz
    oenv = librosa.onset.onset_strength(
        y=samples.astype(float), sr=sr,
        hop_length=hop_length, aggregate=np.median
    )
    hop_sec = hop_length / sr
    n = len(oenv)

    # Local z-score: rolling window of ±60s (1875 frames at 32ms)
    W = min(int(60.0 / hop_sec), n)
    W = W if W % 2 == 1 else W + 1   # odd window for symmetry
    kernel = np.ones(W) / W
    local_mean = np.convolve(oenv, kernel, mode='same')[:n]
    local_sq   = np.convolve(oenv ** 2, kernel, mode='same')[:n]
    local_std  = np.sqrt(np.maximum(local_sq - local_mean ** 2, 0.0))
    novelty_z  = (oenv - local_mean) / np.maximum(local_std, 1e-6)

    min_frames = max(1, int(min_sustain_sec / hop_sec))
    triggers: list[Trigger] = []
    i = 0
    while i < n:
        if novelty_z[i] >= novelty_threshold:
            run_start = i
            peak_z = novelty_z[i]
            while i < n and novelty_z[i] >= novelty_threshold:
                peak_z = max(peak_z, novelty_z[i])
                i += 1
            run_len = i - run_start
            if run_len >= min_frames:
                ts = run_start * hop_sec
                triggers.append(Trigger(ts, "onset_novelty",
                                        f"Audio novelty burst (z={peak_z:.1f}, {run_len * hop_sec:.2f}s)"))
        else:
            i += 1

    return triggers


def _voice_excitement_detection(
    samples: np.ndarray,
    sr: int,
    threshold_factor: float = 2.0,
    min_sustain_sec: float = 0.5,
) -> list[Trigger]:
    """Detect excited/screaming voice using voice-band energy + spectral centroid.

    Distinguishes human voice excitement from pure game SFX (gunshots,
    explosions) by requiring both:
      1. Elevated voice-band energy ratio (150–3000 Hz fraction of total)
      2. Elevated spectral centroid (bright excited speech vs calm speech)
    Both must simultaneously exceed their rolling local baseline.

    Gunshots/SFX spread energy across all bands simultaneously and last
    <100ms — the min_sustain_sec requirement filters them out.
    """
    try:
        import librosa
    except ImportError:
        print("[WARNING] librosa not available — voice excitement detection skipped.")
        return []

    hop = int(sr * _SMART_FRAME_SEC)   # 50ms frames, matches heat model
    stft = np.abs(librosa.stft(y=samples.astype(float), hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=(stft.shape[0] - 1) * 2)

    # Voice-band fraction: energy in 150–3000 Hz / total energy per frame
    voice_mask   = (freqs >= 150) & (freqs <= 3000)
    voice_energy = stft[voice_mask, :].mean(axis=0)
    total_energy = stft.mean(axis=0) + 1e-8
    voice_ratio  = voice_energy / total_energy

    # Spectral centroid from the already-computed STFT (no second FFT)
    centroid = librosa.feature.spectral_centroid(S=stft ** 2, sr=sr, hop_length=hop)[0]

    n = min(len(voice_ratio), len(centroid))
    voice_ratio = voice_ratio[:n]
    centroid    = centroid[:n]

    # Rolling median over ±30 frames (~1.5s) for local baseline
    W = 61  # 61 frames at 50ms = 3s window
    def _rolling_median(arr):
        pad = W // 2
        padded = np.pad(arr, pad, mode='edge')
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(padded, W)
        return np.median(windows, axis=1)

    voice_baseline    = _rolling_median(voice_ratio)
    centroid_baseline = _rolling_median(centroid)

    # Combined excitability: product of both ratios above their baselines
    exc = (voice_ratio / np.maximum(voice_baseline, 1e-8)) * \
          (centroid / np.maximum(centroid_baseline, 1.0))

    min_frames = max(1, int(min_sustain_sec / _SMART_FRAME_SEC))
    triggers: list[Trigger] = []
    i = 0
    while i < n:
        if exc[i] >= threshold_factor:
            run_start = i
            peak_exc = exc[i]
            while i < n and exc[i] >= threshold_factor:
                peak_exc = max(peak_exc, exc[i])
                i += 1
            run_len = i - run_start
            if run_len >= min_frames:
                ts = run_start * _SMART_FRAME_SEC
                triggers.append(Trigger(ts, "voice_excitement",
                                        f"Voice excitement (×{peak_exc:.1f}, {run_len * _SMART_FRAME_SEC:.2f}s)"))
        else:
            i += 1

    return triggers


def _rms_rise_detection(
    samples: np.ndarray,
    sr: int,
    frame_sec: float = 0.5,
    rise_factor: float = 4.0,
    setup_window_sec: float = 3.0,
    reaction_window_sec: float = 1.0,
    cooldown_sec: float = 5.0,
) -> list[Trigger]:
    """Detect setup→punchline arcs: a quiet window followed by a loud reaction.

    Slides in frame_sec steps and compares RMS of the preceding
    setup_window_sec to the next reaction_window_sec.  Fires when the
    reaction is rise_factor× louder than the setup.  A 5s cooldown
    prevents duplicates on extended reactions.
    """
    frame_len       = int(sr * frame_sec)
    setup_len       = int(sr * setup_window_sec)
    react_len       = int(sr * reaction_window_sec)
    step            = frame_len
    cooldown_frames = int(cooldown_sec / frame_sec)

    triggers: list[Trigger] = []
    last_trigger_frame = -cooldown_frames

    total_frames = (len(samples) - setup_len - react_len) // step
    for i in range(total_frames):
        pos = i * step + setup_len
        if i - last_trigger_frame < cooldown_frames:
            continue
        setup_chunk = samples[pos - setup_len : pos]
        react_chunk = samples[pos : pos + react_len]
        setup_rms = float(np.sqrt(np.mean(setup_chunk ** 2))) + 1e-9
        react_rms = float(np.sqrt(np.mean(react_chunk ** 2)))
        if react_rms / setup_rms >= rise_factor:
            ts = pos / sr
            triggers.append(Trigger(ts, "rms_rise",
                f"Comedic arc (reaction ×{react_rms/setup_rms:.1f} vs setup)"))
            last_trigger_frame = i

    return triggers




def _run_whisper(
    wav_path: str,
    model_name: str,
    keywords: list[str] | None = None,
    language: str = "auto",
    verbose: bool = True,
) -> tuple[list[Trigger], list[dict]]:
    """Run whisper.cpp STT and return (keyword_triggers, segments).

    Segments are plain dicts with keys: start, end, text.
    Default keywords cover common clip commands in English and Spanish.

    verbose=False suppresses all prints and is used by _whisper_on_windows
    which shows its own progress bar instead.
    """
    if keywords is None:
        keywords = _CLIP_KEYWORDS

    # ── Resolve binary and model ───────────────────────────────────────────
    binary = _find_whisper_cpp()
    if binary is None:
        if verbose:
            print(
                "[ERROR] whisper.cpp binary not found. Transcription disabled.\n"
                "        Place whisper-cli.exe (or main.exe) in: Whisper/\n"
                "        Or set WHISPER_CPP_PATH env var to the binary path."
            )
        return [], []

    model_path = _whisper_cpp_model_path(model_name)
    if model_path is None:
        if verbose:
            print(
                f"[ERROR] whisper.cpp model 'ggml-{model_name}.bin' not found.\n"
                f"        Download from: https://huggingface.co/ggerganov/whisper.cpp\n"
                f"        Place in: Whisper/models/\n"
                f"        Or set WHISPER_CPP_MODELS_DIR env var."
            )
        return [], []

    # ── Build subprocess command ──────────────────────────────────────────
    output_base = os.path.splitext(wav_path)[0]   # whisper.cpp appends .json itself
    threads = min(os.cpu_count() // 2 if os.cpu_count() else 2, 4)
    cmd = [
        binary,
        "-m", model_path,
        "-f", wav_path,
        "--output-json",
        "--output-file", output_base,
        "-t", str(threads),
        "--print-progress",
    ]
    if language != "auto":
        cmd += ["-l", language]

    if verbose:
        print(f"[INFO] Transcribing with whisper.cpp (model={model_name}, threads={threads})...")

    # ── Run subprocess ─────────────────────────────────────────────────────
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            if verbose:
                print(f"[ERROR] whisper.cpp exited with code {proc.returncode}:\n{proc.stderr[:400]}")
            return [], []
    except Exception as exc:
        if verbose:
            print(f"[ERROR] whisper.cpp failed: {exc}")
        return [], []

    # ── Parse output JSON ──────────────────────────────────────────────────
    import json as _json_wc
    json_path = output_base + ".json"
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            data = _json_wc.load(fh)
        raw_segs = data.get("transcription", [])
    except Exception as exc:
        if verbose:
            print(f"[ERROR] Could not read whisper.cpp output JSON: {exc}")
        return [], []
    finally:
        try:
            os.remove(json_path)
        except OSError:
            pass

    segments: list[dict] = []
    for item in raw_segs:
        ts   = item.get("timestamps", {})
        text = item.get("text", "").strip()
        if not text:
            continue
        segments.append({
            "start": _parse_whispercpp_timestamp(ts.get("from", "00:00:00,000")),
            "end":   _parse_whispercpp_timestamp(ts.get("to",   "00:00:00,000")),
            "text":  text,
        })

    # ── Keyword / comedy scan ──────────────────────────────────────────────
    triggers = _scan_keywords(segments, keywords)
    if verbose:
        print(f"[INFO] whisper.cpp keyword scan: {len(triggers)} hit(s) found.")
    return triggers, segments


# ============================================================
# Trigger merging
# ============================================================

def _merge_triggers(
    triggers: list[Trigger],
    pre: float,
    post: float,
    min_gap_sec: float = 5.0,
    max_duration: float = 60.0,
) -> list[tuple[float, float, list[str]]]:
    """Collapse overlapping clip windows into non-overlapping segments.

    Merging stops if extending a window would exceed max_duration; the
    triggering event then starts a new window.  All windows are hard-clamped
    to max_duration after merging.

    Returns list of (clip_start, clip_end, source_labels).
    """
    if not triggers:
        return []

    sorted_triggers = sorted(triggers, key=lambda t: t.timestamp)
    windows: list[tuple[float, float, list[str]]] = []

    for t in sorted_triggers:
        start = max(0.0, t.timestamp - pre)
        end   = t.timestamp + post
        label = f"{t.source}: {t.label}"

        if windows and start < windows[-1][1] + min_gap_sec:
            prev_start, prev_end, prev_labels = windows[-1]
            merged_end = max(prev_end, end)
            if merged_end - prev_start <= max_duration:
                # Safe to extend
                windows[-1] = (prev_start, merged_end, prev_labels + [label])
            else:
                # Extending would blow the cap — start a new window
                windows.append((start, end, [label]))
        else:
            windows.append((start, end, [label]))

    # Hard clamp: a single trigger with large pre+post could still exceed cap
    return [(s, min(e, s + max_duration), lbl) for s, e, lbl in windows]


# ============================================================
# Clip cutting
# ============================================================

def _cut_clip(
    video_path: str,
    start: float,
    end: float,
    output_path: str,
) -> bool:
    """Cut a clip from video_path using ffmpeg stream-copy (no re-encode)."""
    duration = end - start
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", video_path,
        "-t", str(duration),
        "-c", "copy",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] ffmpeg clip cut failed for {output_path}:\n{result.stderr[-400:]}")
        return False
    return True


# ============================================================
# Post-clip: delete original
# ============================================================

def _handle_original(video_path: str) -> None:
    """Ask the user whether to delete the source file, respecting settings."""
    s      = _settings.load()
    policy = s.get("delete_original_policy", "ask")

    if policy == "always":
        os.remove(video_path)
        print(f"[INFO] Deleted original: {video_path}")
        return

    if policy == "never":
        return

    # policy == "ask"
    print(f"\n  Source file: {video_path}")
    ans = input("  Delete the original file? [y/N]: ").strip().lower()

    if ans == "y":
        remember = input("  Always delete without asking in the future? [y/N]: ").strip().lower()
        if remember == "y":
            _settings.set_delete_policy("always")
            print("  ✓ Preference saved: will always delete originals.")
        os.remove(video_path)
        print(f"  [INFO] Deleted: {video_path}")
    else:
        remember = input("  Never delete originals without asking? [y/N]: ").strip().lower()
        if remember == "y":
            _settings.set_delete_policy("never")
            print("  ✓ Preference saved: will never delete originals.")


# ============================================================
# Smart Clip: multi-language hype words
# ============================================================

_HYPE_WORDS: list[str] = [
    # English
    "amazing", "insane", "crazy", "unbelievable", "lmao", "no way",
    "what", "dude", "bro", "let's go", "yoo", "gg", "wtf", "holy",
    # Spanish
    "increíble", "locura", "loco", "impresionante", "dios mío", "qué",
    "no puede ser", "tremendo", "brutal", "épico", "vergón", "crack",
    # Portuguese / Brazilian
    "incrível", "absurdo", "cara", "mano", "que isso", "gênio",
    "que loucura", "meu deus", "caramba", "sensacional",
    # French
    "incroyable", "fou", "putain", "wesh", "carrément", "ouf",
    # German
    "wahnsinn", "krass", "boah", "unglaublich", "alter",
    # Italian
    "pazzesco", "assurdo", "mamma mia", "che roba",
    # Universal / cross-language chat culture
    "omg", "pog", "pogchamp", "lol", "xd", "kekw",
]

_COMEDY_WORDS: list[str] = [
    # English
    "i'm dead", "i am dead", "bro what", "what are you doing",
    "are you serious", "i can't", "stop stop stop", "why", "help",
    "you're cooked", "cooked", "it's over", "actual",
    # Twitch / internet
    "lmao", "lmfao", "kekw", "lul", "omegalul", "dead",
    # Spanish
    "me muero", "qué haces", "por qué", "no manches",
    "estás loco", "me cago", "ya me fui", "tronco",
]

_SWEAR_WORDS: list[str] = [
    # English
    "fuck", "shit", "damn", "bitch", "ass", "crap", "hell",
    "wtf", "omfg", "holy shit", "what the fuck", "what the hell",
    # Spanish
    "mierda", "joder", "coño", "hostia", "puta", "cabron", "cabrón",
    "me cago", "ostia", "joder",
    # Portuguese
    "porra", "merda", "caralho", "filho da puta",
    # French
    "putain", "merde", "bordel",
    # Italian
    "cazzo", "vaffanculo", "merda",
]

_CLIP_KEYWORDS: list[str] = ["clip it", "clip that", "clipeenlo", "clipeen", "clipa eso"]


def _scan_keywords(
    segments: list[dict],
    keywords: list[str] | None = None,
) -> list[Trigger]:
    """Scan transcript segments for clip keywords and comedy keywords.

    Returns Trigger objects for 'keyword' (clip commands) and
    'comedy_keyword' (comedy signal words) matches.
    """
    if keywords is None:
        keywords = _CLIP_KEYWORDS
    triggers: list[Trigger] = []
    for seg in segments:
        text  = seg.get("text", "").lower()
        start = seg.get("start", 0.0)
        for kw in keywords:
            if kw in text:
                triggers.append(Trigger(start, "keyword", f'Keyword detected: "{kw}"'))
                break
    for seg in segments:
        text  = seg.get("text", "").lower()
        start = seg.get("start", 0.0)
        for cw in _COMEDY_WORDS:
            if cw in text:
                triggers.append(Trigger(start, "comedy_keyword", f'Comedy keyword: "{cw}"'))
                break
    return triggers


def _chat_velocity_detection(
    chat_path: str | None,
    video_duration: float,
    window_sec: float = 5.0,
    spike_factor: float = 3.0,
) -> list[Trigger]:
    """Detect chat message rate spikes that indicate audience reaction moments.

    Reads yt-dlp info.json (written via --write-info-json --write-comments),
    extracts comment timestamps, builds a rolling messages/sec time series,
    and fires a Trigger wherever the rate exceeds median × spike_factor.
    Returns [] gracefully if the file doesn't exist or fails to parse.
    """
    if not chat_path or not os.path.exists(chat_path):
        return []
    try:
        import json as _json
        with open(chat_path, "r", encoding="utf-8", errors="replace") as f:
            data = _json.load(f)
        if isinstance(data, list):
            comments = data
        elif isinstance(data, dict):
            comments = data.get("comments", [])
        else:
            return []
        if not comments:
            return []

        timestamps = []
        for c in comments:
            ts = c.get("timestamp")
            if ts is not None:
                try:
                    timestamps.append(float(ts))
                except (TypeError, ValueError):
                    pass
        if not timestamps:
            return []

        dur = max(int(video_duration) + 1, len(timestamps))
        rate = np.zeros(dur, dtype=float)
        for ts in timestamps:
            idx = int(ts)
            if 0 <= idx < dur:
                rate[idx] += 1.0

        win = max(1, int(window_sec))
        kernel = np.ones(win) / window_sec
        smoothed = np.convolve(rate, kernel, mode="same")

        nonzero = smoothed[smoothed > 0]
        if nonzero.size == 0:
            return []
        median = float(np.median(nonzero))
        if median < 1e-6:
            return []
        threshold = median * spike_factor

        triggers: list[Trigger] = []
        cooldown = 10  # seconds
        last_t = -cooldown
        for i, val in enumerate(smoothed):
            if val > threshold and (i - last_t) >= cooldown:
                triggers.append(Trigger(
                    float(i), "chat_spike",
                    f"Chat spike ({val:.1f} msg/s, ×{val/median:.1f} median)"
                ))
                last_t = i
        return triggers
    except Exception:
        return []


def _swear_detection(
    transcript_segments: list[dict],
    threshold_density: float = 0.0,
) -> list[Trigger]:
    """Emit a Trigger for each Whisper segment that contains a swear word.

    One trigger per segment regardless of how many swear words appear in it.
    threshold_density is reserved for future density-based filtering.
    """
    triggers: list[Trigger] = []
    for seg in transcript_segments:
        text = seg.get("text", "").lower()
        ts   = float(seg.get("start", 0.0))
        for word in _SWEAR_WORDS:
            if word in text:
                triggers.append(Trigger(ts, "swear", f"Swear word: '{word}'"))
                break
    return triggers


def _scream_detection(
    samples: np.ndarray,
    sr: int,
    min_pitch_hz: float = 300.0,
    max_duration_sec: float = 1.8,
    rise_factor: float = 3.0,
) -> list[Trigger]:
    """Detect brief, sharp, high-pitched vocal bursts (screams / surprised yelps).

    A scream is: F0 > min_pitch_hz AND RMS > rise_factor × 1s baseline AND
    duration ≤ max_duration_sec.  Longer events are likely sustained hype,
    not a comedic yelp.
    Degrades gracefully if librosa is unavailable.
    """
    try:
        import librosa
    except ImportError:
        return []

    frame_sec = 0.1
    frame_len = max(1, int(sr * frame_sec))
    if len(samples) < frame_len * 4:
        return []

    n_frames = len(samples) // frame_len
    rms = np.array([
        float(np.sqrt(np.mean(samples[i * frame_len:(i + 1) * frame_len] ** 2)))
        for i in range(n_frames)
    ])
    try:
        f0 = librosa.yin(
            samples.astype(float),
            fmin=float(max(min_pitch_hz * 0.5, 60.0)),
            fmax=float(min(sr // 2 - 1, 2000)),
            sr=sr,
            hop_length=frame_len,
        )
        f0 = f0[:n_frames]
    except Exception:
        return []

    baseline_frames = max(1, int(1.0 / frame_sec))
    cooldown_frames = int(10.0 / frame_sec)
    triggers: list[Trigger] = []
    last_frame = -cooldown_frames
    in_burst = False
    burst_start = 0

    for i in range(baseline_frames, n_frames):
        rms_base = float(np.mean(rms[max(0, i - baseline_frames):i])) + 1e-9
        high_f0  = float(f0[i]) > min_pitch_hz
        rms_spike = rms[i] > rise_factor * rms_base
        if high_f0 and rms_spike:
            if not in_burst:
                in_burst = True
                burst_start = i
        else:
            if in_burst:
                burst_dur = (i - burst_start) * frame_sec
                if burst_dur <= max_duration_sec and (i - last_frame) >= cooldown_frames:
                    ts = burst_start * frame_sec
                    triggers.append(Trigger(ts, "scream",
                        f"Scream detected (F0≈{f0[burst_start]:.0f}Hz, {burst_dur:.1f}s)"))
                    last_frame = i
                in_burst = False

    return triggers


def _nonsense_vocalization_detection(
    transcript_segments: list[dict],
) -> list[Trigger]:
    """Emit Triggers for Whisper non-speech tags like [laughter], [noise], *sighs*.

    [laughter] is weighted 2× by emitting two triggers for the same timestamp.
    """
    import re as _re
    pattern          = _re.compile(r'\[.*?\]|\*.*?\*')
    laughter_pattern = _re.compile(r'\[laughter\]', _re.IGNORECASE)

    triggers: list[Trigger] = []
    for seg in transcript_segments:
        text = seg.get("text", "")
        ts   = float(seg.get("start", 0.0))
        if laughter_pattern.search(text):
            label = f"Laughter tag: '{text.strip()}'"
            triggers.append(Trigger(ts, "nonsense", label))
            triggers.append(Trigger(ts, "nonsense", label + " (2×)"))
        elif pattern.search(text):
            triggers.append(Trigger(ts, "nonsense",
                f"Vocalization tag: '{text.strip()}'"))
    return triggers


def _pitch_variance_detection(
    samples: np.ndarray,
    sr: int,
    variance_threshold: float = 80.0,
) -> list[Trigger]:
    """Detect moments of exaggerated vocal delivery via F0 rolling std dev.

    Uses librosa.yin over 100ms frames, computes std dev in a 3s rolling window,
    and fires where std dev > variance_threshold Hz.
    Degrades gracefully if librosa is unavailable.
    """
    try:
        import librosa
    except ImportError:
        return []

    hop = max(1, int(sr * 0.1))
    if len(samples) < hop * 30:
        return []

    try:
        f0 = librosa.yin(
            samples.astype(float),
            fmin=65.0,
            fmax=float(min(sr // 2 - 1, 2000)),
            sr=sr,
            hop_length=hop,
        )
    except Exception:
        return []

    frame_duration = hop / sr
    window_frames  = max(1, int(3.0 / frame_duration))
    cooldown_frames = int(10.0 / frame_duration)
    triggers: list[Trigger] = []
    last_frame = -cooldown_frames

    for i in range(window_frames, len(f0)):
        window = f0[i - window_frames:i]
        voiced = window[window > 80.0]
        if len(voiced) < window_frames // 3:
            continue
        std_f0 = float(np.std(voiced))
        if std_f0 > variance_threshold and (i - last_frame) >= cooldown_frames:
            ts = i * frame_duration
            triggers.append(Trigger(ts, "pitch_variance",
                f"Pitch variance spike (std {std_f0:.0f}Hz)"))
            last_frame = i

    return triggers


def _pre_reaction_silence_detection(
    samples: np.ndarray,
    sr: int,
    silence_threshold_db: float = -40.0,
    min_silence_sec: float = 0.3,
    max_silence_sec: float = 2.0,
    follow_window_sec: float = 0.5,
    follow_factor: float = 2.0,
) -> list[Trigger]:
    """Detect the comedic pause: a near-silence window immediately followed by a
    loud event, suggesting a setup→reaction beat.

    A silence qualifies if its duration falls within [min_silence_sec, max_silence_sec]
    and is followed within follow_window_sec by RMS > follow_factor × median RMS.
    """
    frame_sec = 0.05
    frame_len = max(1, int(sr * frame_sec))
    if len(samples) < frame_len * 20:
        return []

    n_frames = len(samples) // frame_len
    rms = np.array([
        float(np.sqrt(np.mean(samples[i * frame_len:(i + 1) * frame_len] ** 2))) + 1e-9
        for i in range(n_frames)
    ])
    rms_db      = 20.0 * np.log10(rms)
    median_rms  = float(np.median(rms))

    min_frames    = max(1, int(min_silence_sec / frame_sec))
    max_frames    = max(min_frames, int(max_silence_sec / frame_sec))
    follow_frames = max(1, int(follow_window_sec / frame_sec))
    cooldown_frames = int(10.0 / frame_sec)

    triggers: list[Trigger] = []
    last_frame = -cooldown_frames
    i = 0

    while i < n_frames:
        if rms_db[i] < silence_threshold_db:
            silence_start = i
            while i < n_frames and rms_db[i] < silence_threshold_db:
                i += 1
            silence_end = i
            silence_len = silence_end - silence_start
            if min_frames <= silence_len <= max_frames:
                follow_end   = min(silence_end + follow_frames, n_frames)
                follow_chunk = rms[silence_end:follow_end]
                if len(follow_chunk) > 0 and float(np.max(follow_chunk)) > follow_factor * median_rms:
                    if (silence_start - last_frame) >= cooldown_frames:
                        ts = silence_start * frame_sec
                        triggers.append(Trigger(ts, "pre_silence",
                            f"Pre-reaction silence ({silence_len * frame_sec:.2f}s)"))
                        last_frame = silence_start
        else:
            i += 1

    return triggers


def _voice_crack_detection(
    samples: np.ndarray,
    sr: int,
    crack_threshold_hz: float = 150.0,
    min_jump_sec: float = 0.0,
) -> list[Trigger]:
    """Detect voice cracks: sudden F0 jumps within a voiced segment.

    Uses librosa.yin for F0 estimation and RMS-based voicing detection.
    Fires where consecutive voiced frames differ by > crack_threshold_hz.
    Degrades gracefully if librosa is unavailable.
    """
    try:
        import librosa
    except ImportError:
        return []

    hop = max(1, int(sr * 0.05))
    if len(samples) < hop * 20:
        return []

    try:
        f0 = librosa.yin(
            samples.astype(float),
            fmin=60.0,
            fmax=float(min(sr // 2 - 1, 1500)),
            sr=sr,
            hop_length=hop,
        )
    except Exception:
        return []

    n_frames = len(f0)
    rms = np.array([
        float(np.sqrt(np.mean(
            samples[i * hop:min((i + 1) * hop, len(samples))] ** 2
        )))
        for i in range(n_frames)
    ])
    voiced_thresh   = max(float(np.median(rms)) * 0.3, 1e-6)
    frame_duration  = hop / sr
    cooldown_frames = int(10.0 / frame_duration)
    triggers: list[Trigger] = []
    last_frame = -cooldown_frames

    for i in range(1, n_frames):
        if rms[i] < voiced_thresh or rms[i - 1] < voiced_thresh:
            continue
        jump = abs(float(f0[i]) - float(f0[i - 1]))
        if jump > crack_threshold_hz and (i - last_frame) >= cooldown_frames:
            ts = i * frame_duration
            triggers.append(Trigger(ts, "voice_crack",
                f"Voice crack (F0 jump {jump:.0f}Hz)"))
            last_frame = i

    return triggers


# ============================================================
# Heat pipeline  (Smart Clip only)
# ============================================================

def _compute_excitement_curve(samples: np.ndarray, sr: int) -> np.ndarray:
    """Return a per-frame excitement score in [0, 1] at _SMART_FRAME_SEC resolution.

    Blends three signals:
      - Normalized RMS      (weight 0.5) — loudness
      - Spectral flux        (weight 0.3) — sudden frequency changes
      - Zero-crossing rate   (weight 0.2) — signal noisiness / chaos

    Falls back to RMS-only if librosa is unavailable.
    """
    frame_len = int(sr * _SMART_FRAME_SEC)
    n_frames  = len(samples) // frame_len

    rms_vals = np.array([
        np.sqrt(np.mean(samples[i * frame_len:(i + 1) * frame_len] ** 2))
        for i in range(n_frames)
    ])
    rms_norm = _minmax_norm(rms_vals)

    try:
        import librosa
        # Spectral flux: mean absolute diff between adjacent STFT magnitude frames
        S    = np.abs(librosa.stft(samples, hop_length=frame_len))
        flux = np.mean(np.abs(np.diff(S, axis=1)), axis=0)
        flux = np.concatenate([[0.0], flux])[:n_frames]
        flux_norm = _minmax_norm(flux.astype(np.float32))

        zcr = librosa.feature.zero_crossing_rate(
            samples, frame_length=frame_len, hop_length=frame_len
        )[0][:n_frames]
        zcr_norm = _minmax_norm(zcr.astype(np.float32))

        return 0.5 * rms_norm + 0.3 * flux_norm + 0.2 * zcr_norm

    except ImportError:
        return rms_norm  # pure-RMS fallback


def _apply_heat_model(excitement: np.ndarray, decay_per_frame: float) -> np.ndarray:
    """IIR accumulator: heat builds with excitement and decays when quiet.

    heat[0] = excitement[0]
    heat[i] = heat[i-1] * decay_per_frame + excitement[i]

    Result is max-normalized to [0, 1].
    """
    heat = np.empty_like(excitement)
    heat[0] = excitement[0]
    for i in range(1, len(excitement)):
        heat[i] = heat[i - 1] * decay_per_frame + excitement[i]
    peak = heat.max()
    if peak > 0:
        heat /= peak
    return heat


def _find_hot_zones(
    heat: np.ndarray,
    frame_sec: float,
    threshold: float,
    release: float,
    min_duration_sec: float,
    pre: float,
    post: float,
    max_duration: float,
) -> list[tuple[float, float, float]]:
    """Find continuous zones where heat stays elevated.

    Uses hysteresis: a zone opens when heat >= threshold and closes when
    heat < release (release < threshold prevents jitter at the boundary).

    Returns list of (start_sec, end_sec, peak_heat) after expanding by
    pre/post, merging overlapping zones, and capping to max_duration.
    """
    zones: list[tuple[float, float, float]] = []
    in_zone     = False
    zone_start  = 0
    zone_peak   = 0.0

    for i, h in enumerate(heat):
        if not in_zone:
            if h >= threshold:
                in_zone    = True
                zone_start = i
                zone_peak  = h
        else:
            zone_peak = max(zone_peak, h)
            if h < release:
                zones.append((zone_start * frame_sec, i * frame_sec, zone_peak))
                in_zone   = False
                zone_peak = 0.0

    if in_zone:  # close any open zone at EOF
        zones.append((zone_start * frame_sec, len(heat) * frame_sec, zone_peak))

    # Duration filter
    zones = [(s, e, p) for s, e, p in zones if (e - s) >= min_duration_sec]

    # Expand by pre/post padding
    zones = [(max(0.0, s - pre), e + post, p) for s, e, p in zones]

    # Merge overlapping zones (take max peak_heat)
    merged: list[tuple[float, float, float]] = []
    for s, e, p in zones:
        if merged and s < merged[-1][1]:
            ps, pe, pp = merged[-1]
            merged[-1] = (ps, max(pe, e), max(pp, p))
        else:
            merged.append((s, e, p))

    # Cap to max_duration
    return [(s, min(e, s + max_duration), p) for s, e, p in merged]


def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize arr to [0, 1]. Returns zeros if all values are equal."""
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _trim_to_peak(
    start: float, end: float,
    samples: np.ndarray, sr: int,
    pre: float, post: float,
    max_dur: float,
    frame_sec: float = 0.5,
    activity_fraction: float = 0.30,
) -> tuple[float, float]:
    """Shrink a clip window to center around its RMS activity peak.

    Finds the loudest 0.5s frame, expands outward until activity drops to
    activity_fraction × peak, then re-applies pre/post padding.  Guarantees
    the returned window is at least (pre + post) seconds wide and at most
    max_dur seconds wide.
    """
    s_start = int(start * sr)
    s_end   = int(end * sr)
    chunk   = samples[s_start:s_end]
    if len(chunk) == 0:
        return start, end

    frame_len = max(1, int(sr * frame_sec))
    n_frames  = len(chunk) // frame_len
    if n_frames == 0:
        return start, end

    rms = np.array([
        np.sqrt(np.mean(chunk[i * frame_len:(i + 1) * frame_len] ** 2))
        for i in range(n_frames)
    ])
    peak_idx  = int(np.argmax(rms))
    threshold = rms[peak_idx] * activity_fraction

    lo = peak_idx
    while lo > 0 and rms[lo - 1] >= threshold:
        lo -= 1
    hi = peak_idx
    while hi < n_frames - 1 and rms[hi + 1] >= threshold:
        hi += 1

    active_start = start + lo * frame_sec
    active_end   = start + (hi + 1) * frame_sec

    new_start = max(0.0, active_start - pre)
    new_end   = active_end + post
    if new_end - new_start > max_dur:
        new_end = new_start + max_dur
    if new_end - new_start < pre + post:
        new_end = new_start + pre + post

    return new_start, new_end


def _score_windows(
    windows: list[tuple[float, float, list[str]]],
    samples: np.ndarray,
    sr: int,
    transcript_segments: list[dict],
    llm_scores: list[float] | None = None,
    llm_weight: float = 0.60,
) -> list[tuple[float, float, list[str], float, float]]:
    """Score each candidate window using two independent arms.

    Audio arm (diversity 40% + spectral 35% + hype 25%) → sw_score in [0, 1]
    LLM arm   (comedy, context, meaning)                 → llm_score in [0, 1]

    Final composite = llm_weight × llm_score + (1 - llm_weight) × sw_score
    When no LLM score available: composite = sw_score

    Returns 5-tuples: (start, end, labels, sw_score, composite_score).
    """
    import librosa  # lazy import — only needed for Smart Clip

    n = len(windows)
    diversity_raw = np.zeros(n)
    spectral_raw  = np.zeros(n)
    hype_raw      = np.zeros(n)

    for i, (start, end, labels) in enumerate(windows):
        # --- 1. Trigger diversity ---
        sources = {lbl.split(":")[0].strip() for lbl in labels}
        diversity_raw[i] = min(len(sources) / 3.0, 1.0)

        # --- 2. Spectral excitement (librosa) ---
        s_start = int(start * sr)
        s_end   = int(end   * sr)
        chunk   = samples[s_start:s_end]
        if len(chunk) > 0:
            contrast = librosa.feature.spectral_contrast(y=chunk, sr=sr)
            zcr      = librosa.feature.zero_crossing_rate(y=chunk)
            spectral_raw[i] = float(np.mean(contrast)) + float(np.var(zcr))

        duration = max(end - start, 1.0)

        # --- 3. Hype word density ---
        word_hits = 0
        for seg in transcript_segments:
            seg_start = seg.get("start", 0.0)
            seg_end   = seg.get("end",   seg_start)
            if seg_end < start or seg_start > end:
                continue
            text = seg.get("text", "").lower()
            for hw in _HYPE_WORDS:
                if hw in text:
                    word_hits += 1
        hype_raw[i] = word_hits / duration

    diversity_norm = diversity_raw           # already in [0, 1]
    spectral_norm  = _minmax_norm(spectral_raw)
    hype_norm      = _minmax_norm(hype_raw)

    scored: list[tuple[float, float, list[str], float, float]] = []
    for i, (start, end, labels) in enumerate(windows):
        sw_score = (
            0.40 * diversity_norm[i]
            + 0.35 * spectral_norm[i]
            + 0.25 * hype_norm[i]
        )
        if llm_scores and i < len(llm_scores) and llm_scores[i] > 0.0:
            composite = llm_weight * llm_scores[i] + (1.0 - llm_weight) * sw_score
        else:
            composite = sw_score
        scored.append((start, end, labels, float(sw_score), float(composite)))

    return scored


# ============================================================
# Fast-clip scoring
# ============================================================

def _score_fast_windows(
    windows: list[tuple[float, float, list[str]]],
    samples: np.ndarray,
    sr: int,
) -> list[tuple[float, float, list[str], float]]:
    """Score Fast Clip candidate windows and return them sorted best-first.

    Three signals, min-max normalised across the candidate set:
      - Trigger diversity (0.45): distinct detector sources (rms, laughter,
        onset_novelty, voice_excitement, keyword)
      - RMS peak           (0.35): loudest 0.5s frame within the window
      - Trigger density    (0.20): trigger count per second of clip
    """
    n = len(windows)
    diversity_raw = np.zeros(n)
    rms_peak_raw  = np.zeros(n)
    density_raw   = np.zeros(n)

    frame_len = int(sr * 0.5)

    for i, (start, end, labels) in enumerate(windows):
        # Diversity: distinct source names
        sources = {lbl.split(":")[0].strip() for lbl in labels}
        diversity_raw[i] = min(len(sources) / 5.0, 1.0)   # 5 possible sources

        # RMS peak in window
        s_start = int(start * sr)
        s_end   = int(end * sr)
        chunk   = samples[s_start:s_end]
        if len(chunk) >= frame_len:
            n_frames = len(chunk) // frame_len
            rms_vals = np.array([
                np.sqrt(np.mean(chunk[j*frame_len:(j+1)*frame_len] ** 2))
                for j in range(n_frames)
            ])
            rms_peak_raw[i] = float(np.max(rms_vals))
        elif len(chunk) > 0:
            rms_peak_raw[i] = float(np.sqrt(np.mean(chunk ** 2)))

        # Trigger density
        duration = max(end - start, 1.0)
        density_raw[i] = len(labels) / duration

    diversity_norm = diversity_raw             # already [0, 1]
    rms_norm       = _minmax_norm(rms_peak_raw)
    density_norm   = _minmax_norm(density_raw)

    scored: list[tuple[float, float, list[str], float]] = []
    for i, (start, end, labels) in enumerate(windows):
        score = (
            0.45 * diversity_norm[i]
            + 0.35 * rms_norm[i]
            + 0.20 * density_norm[i]
        )
        scored.append((start, end, labels, float(score)))

    scored.sort(key=lambda x: x[3], reverse=True)
    return scored


# ============================================================
# Clip deduplication
# ============================================================

def _deduplicate_clips(clips: list) -> list:
    """Remove clips that overlap with a higher-ranked (earlier in list) clip.

    Input must be sorted best-first.  Each element must have [0]=start, [1]=end.
    Returns a new list with no two clips sharing any overlapping time range.
    """
    kept: list = []
    for clip in clips:
        s, e = clip[0], clip[1]
        if not any(s < k[1] and e > k[0] for k in kept):
            kept.append(clip)
    return kept


# ============================================================
# Public entry points
# ============================================================

def _progress(step: int, total: int, label: str) -> None:
    """Print a step-based progress bar.  Each call overwrites nothing —
    subprocess output (Whisper, ffmpeg) can appear between steps safely."""
    bar_width = 24
    filled = bar_width * step // total
    bar = "=" * filled + "-" * (bar_width - filled)
    print(f"\n  [{bar}] {step}/{total}  {label}")


def _extend_comedy_clips(
    clips: list,
    tail_sec: float,
    max_dur: float,
    comedy_threshold: float = 0.4,
) -> list:
    """Extend the end of high-composite clips to capture delayed laugh reactions.

    clips: 5-tuples (start, end, labels, sw_score, composite).
    Returns a new list with end times extended for clips above the threshold.
    """
    result = []
    for clip in clips:
        start, end, labels, sw, composite = clip
        if composite > comedy_threshold:
            new_end = min(end + tail_sec, start + max_dur)
            result.append((start, new_end, labels, sw, composite))
        else:
            result.append(clip)
    return result


def _fmt_time(sec: float) -> str:
    """Convert seconds to a compact human-readable timestamp, e.g. 2m05s."""
    m = int(sec) // 60
    s = int(sec) % 60
    return f"{m}m{s:02d}s"


def _build_window_signal_summary(
    trigger_windows: list,
    chat_triggers:     list,
    swear_triggers:    list,
    scream_triggers:   list,
    crack_triggers:    list,
    silence_triggers:  list,
    nonsense_triggers: list,
    video_duration:    float,
) -> str:
    """Build a flat timestamped signal list for the LLM prompt (max 120 lines).

    Each line: [MM:SS] signal_type — sorted by timestamp.
    """
    pairs: list[tuple[float, str]] = []
    for t in scream_triggers:   pairs.append((t.timestamp, "scream"))
    for t in crack_triggers:    pairs.append((t.timestamp, "voice_crack"))
    for t in silence_triggers:  pairs.append((t.timestamp, "pre_silence"))
    for t in swear_triggers:    pairs.append((t.timestamp, "swear"))
    for t in nonsense_triggers: pairs.append((t.timestamp, "nonsense"))
    for t in chat_triggers:     pairs.append((t.timestamp, "chat_spike"))

    pairs.sort(key=lambda x: x[0])
    lines = [
        f"[{int(ts)//60:02d}:{int(ts)%60:02d}] {label}"
        for ts, label in pairs
    ]
    return "\n".join(lines[:120])


def _get_transcript_excerpt(
    segments: list[dict],
    start: float,
    end: float,
    max_chars: int = 400,
) -> str:
    """Return transcript text covering the clip window [start, end]."""
    parts = []
    for seg in segments:
        s_start = seg.get("start", 0.0)
        s_end   = seg.get("end",   s_start)
        if s_end < start or s_start > end:
            continue
        text = seg.get("text", "").strip()
        if text:
            parts.append(text)
    result = " ".join(parts)
    return result[:max_chars]


def _find_llm_reason(llm_segs: list[dict], start: float, end: float) -> str:
    """Return the LLM's reason for the flagged segment most overlapping [start, end]."""
    win_dur      = max(end - start, 1e-6)
    best_overlap = 0.0
    best_reason  = ""
    for seg in llm_segs:
        overlap = min(end, seg["end_sec"]) - max(start, seg["start_sec"])
        if overlap / win_dur >= 0.20 and overlap > best_overlap:
            best_overlap = overlap
            best_reason  = seg.get("reason", "")
    return best_reason


def _write_manifest(
    base: str,
    pipeline: str,
    video_path: str,
    clip_records: list[dict],
    settings_snapshot: dict,
) -> str | None:
    """Write a JSON manifest of the clips produced in this run.

    Returns the manifest path on success, None on failure.
    """
    import json as _json
    import datetime as _dt
    ts       = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base}_{pipeline}_manifest_{ts}.json"
    path     = os.path.join(CLIPS_DIR, filename)
    key_settings = {k: settings_snapshot.get(k) for k in (
        "whisper_model", "use_llm_scoring", "llm_model",
        "max_clips", "max_clip_duration", "min_clip_score",
    )}
    manifest = {
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "pipeline":   pipeline,
        "video":      os.path.basename(video_path),
        "clip_count": len(clip_records),
        "settings":   key_settings,
        "clips":      clip_records,
    }
    try:
        with open(path, "w", encoding="utf-8") as fh:
            _json.dump(manifest, fh, indent=2, ensure_ascii=False)
        return path
    except OSError as exc:
        print(f"[WARNING] Could not write manifest: {exc}")
        return None


def _run_feedback_loop(clip_records: list[dict], settings: dict) -> None:
    """Show each clip and prompt the user to rate it for comedy memory training.

    Good → comedy_memory.json positive example.
    Bad  → rejection_memory.json negative example.
    """
    successful = [r for r in clip_records if r.get("cut_success")]
    if not successful:
        return
    ans = input("\n  Rate clips for learning? [y/N]: ").strip().lower()
    if ans != "y":
        return

    import llm as _llm_feedback
    print()
    for rec in successful:
        print(f"  Clip {rec['rank']}: {rec['start_fmt']} → {rec['end_fmt']}  "
              f"(score {rec.get('score_composite', 0.0):.2f})")
        if rec.get("llm_reason"):
            print(f"  LLM: {rec['llm_reason']}")
        if rec.get("transcript_excerpt"):
            excerpt = rec["transcript_excerpt"][:120].replace("\n", " ")
            print(f"  \"{excerpt}\"")
        print("  [g] Good  [b] Bad  [s] Skip  [q] Quit rating")
        choice = input("  > ").strip().lower()
        print()
        if choice == "q":
            break
        elif choice == "g":
            why = input("  Why is it funny? (or Enter to skip): ").strip()
            _llm_feedback.add_positive_feedback(
                rec.get("transcript_excerpt", ""),
                why or "good clip",
            )
            print("  ✓ Added to comedy memory.\n")
        elif choice == "b":
            why = input("  Why is it bad? (or Enter to skip): ").strip()
            _llm_feedback.add_negative_feedback(
                rec.get("transcript_excerpt", ""),
                why or "not funny",
            )
            print("  ✓ Added to rejection memory.\n")


def run_fast_clip(dry_run: bool = False) -> None:
    """Fast-clip pipeline — called from main.py."""
    print("\n" + "=" * 60)
    print("  Sundown — Fast Clip" + (" (DRY RUN)" if dry_run else ""))
    print("=" * 60)

    # 1. Pick file
    video_path = pick_input_file()
    if video_path is None:
        return

    s              = _settings.load()
    pre            = float(s["pre_event_seconds"])
    post           = float(s["post_event_seconds"])
    model_name     = s["whisper_model"]
    use_whisper    = bool(s.get("use_whisper",             True))
    use_cache      = bool(s.get("use_transcript_cache",    True))
    top_n          = int(s.get("max_clips",                5))
    rms_factor     = float(s.get("rms_threshold_factor",   3.0))
    rms_sustain    = int(s.get("rms_min_sustain",          2))
    laugh_factor   = float(s.get("laughter_burst_factor",  2.0))
    laugh_count    = int(s.get("laughter_burst_count",     4))
    onset_thresh   = float(s.get("onset_novelty_threshold",  3.0))
    voice_thresh   = float(s.get("voice_excitement_threshold", 2.0))
    max_clip_dur   = float(s.get("max_clip_duration",      50.0))
    trim_to_peak   = bool(s.get("clip_trim_to_peak",       True))
    whisper_lang   = s.get("whisper_language", "auto")
    total_steps    = 9 if use_whisper else 8

    if _logger:
        _logger.log_pipeline_start(video_path, s)

    os.makedirs(CLIPS_DIR, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. Extract audio
        _progress(1, total_steps, "Extracting audio")
        wav_path = extract_audio(video_path, tmp_dir)
        samples, sr = _load_wav_as_array(wav_path)

        # 2–5. Fast signal detectors (run all four, print one summary)
        _progress(2, total_steps, "Detecting loud moments (RMS)")
        rms_triggers   = _rms_spike_detection(samples, sr,
                             threshold_factor=rms_factor,
                             min_sustain_frames=rms_sustain)

        _progress(3, total_steps, "Detecting laughter")
        laugh_triggers = _laughter_detection(samples, sr,
                             burst_threshold_factor=laugh_factor,
                             burst_count=laugh_count)

        _progress(4, total_steps, "Detecting audio novelty bursts")
        onset_triggers = _onset_novelty_detection(samples, sr,
                             novelty_threshold=onset_thresh)

        _progress(5, total_steps, "Detecting voice excitement")
        voice_triggers = _voice_excitement_detection(samples, sr,
                             threshold_factor=voice_thresh)

        all_triggers: list[Trigger] = rms_triggers + laugh_triggers + onset_triggers + voice_triggers
        print(f"       Found: {len(rms_triggers)} RMS | {len(laugh_triggers)} laughter | "
              f"{len(onset_triggers)} novelty | {len(voice_triggers)} voice excitement")
        if _logger:
            _logger.log_audio_signals({
                "rms": len(rms_triggers), "laughter": len(laugh_triggers),
                "onset": len(onset_triggers), "voice": len(voice_triggers),
            })

        if not all_triggers:
            print("\n[INFO] No trigger moments detected. Try lowering thresholds in Settings.")
            return

        # 6. Merge overlapping windows (candidate set for Whisper + scoring)
        _progress(6, total_steps, "Merging triggers")
        windows = _merge_triggers(all_triggers, pre=pre, post=post, max_duration=max_clip_dur)
        print(f"       {len(windows)} candidate window(s).")

        # 7. Targeted Whisper — only on candidate window slices
        fast_segments: list[dict] = []
        if use_whisper:
            _progress(7, total_steps, "Transcribing candidate windows (Whisper)")
            kw_triggers, fast_segments = _whisper_on_windows(
                samples, sr, windows, model_name, whisper_lang,
                tmp_dir, video_path, use_cache=use_cache
            )
            # Re-merge with keyword triggers added
            all_triggers += kw_triggers
            windows = _merge_triggers(all_triggers, pre=pre, post=post, max_duration=max_clip_dur)
        else:
            print("\n       [Whisper disabled — skipping transcription]")

        # 8. Score and rank — best N clips regardless of time order
        _progress(8 if use_whisper else 7, total_steps, "Ranking clips")
        scored = _score_fast_windows(windows, samples, sr)
        selected = _deduplicate_clips(scored[:top_n])
        min_score = float(s.get("min_clip_score", 0.30))
        if min_score > 0:
            qualifying = [c for c in selected if c[3] >= min_score]
            if qualifying:
                dropped = len(selected) - len(qualifying)
                if dropped:
                    print(f"       Quality filter: {dropped} clip(s) below {min_score:.2f} dropped.")
                selected = qualifying
            else:
                print(f"       [INFO] No clips reached quality floor {min_score:.2f} — keeping best clip.")
                selected = selected[:1]
        print(f"       Top {len(selected)} of {len(windows)} window(s) selected by score.")

        # 9. Cut clips
        _progress(total_steps, total_steps, "Cutting clips" + (" (dry run)" if dry_run else ""))
        base         = os.path.splitext(os.path.basename(video_path))[0]
        cut_count    = 0
        clip_records: list[dict] = []
        for rank, (start, end, labels, score) in enumerate(selected, start=1):
            if trim_to_peak:
                start, end = _trim_to_peak(start, end, samples, sr, pre, post, max_clip_dur)
            out_name = os.path.join(CLIPS_DIR,
                                    f"{base}_clip_{rank:02d}_{_fmt_time(start)}.mp4")
            print(f"\n  Clip {rank}/{len(selected)}: {_fmt_time(start)} → {_fmt_time(end)}  (score {score:.2f})")
            for lbl in labels:
                print(f"    ↳ {lbl}")
            if _logger:
                _logger.log_clip_selected(rank, start, end, score, score, 0.0, labels)
            if dry_run:
                print(f"    [DRY RUN] Would save → {out_name}")
                cut_success = True
                cut_count  += 1
            else:
                cut_success = _cut_clip(video_path, start, end, out_name)
                if cut_success:
                    print(f"    ✓ Saved → {out_name}")
                    cut_count += 1
            clip_records.append({
                "rank":               rank,
                "start_sec":          round(start, 2),
                "end_sec":            round(end,   2),
                "start_fmt":          _fmt_time(start),
                "end_fmt":            _fmt_time(end),
                "filename":           os.path.basename(out_name),
                "score_composite":    round(score, 4),
                "score_sw":           round(score, 4),
                "score_heat":         0.0,
                "labels":             labels,
                "transcript_excerpt": _get_transcript_excerpt(fast_segments, start, end),
                "llm_reason":         "",
                "cut_success":        cut_success,
            })

    dry_tag = " (dry run — no files cut)" if dry_run else ""
    print(f"\n[SUCCESS] {cut_count}/{len(selected)} clip(s){dry_tag}.")

    if not dry_run:
        if clip_records:
            manifest_path = _write_manifest(base, "fast", video_path, clip_records, s)
            if manifest_path:
                print(f"  Manifest → {manifest_path}")
        _run_feedback_loop(clip_records, s)
        if cut_count > 0:
            _handle_original(video_path)


def run_smart_clip(dry_run: bool = False) -> None:
    """Smart-clip pipeline — LLM + audio dual-arm detection with heat model fusion.

    Pipeline order:
      1. Extract audio
      2. Whisper transcribes the full VOD (if use_whisper)
      3. Ollama selects clip moments from the full transcript (if use_llm)
      4-11. Audio detectors (RMS, laughter, onset, voice, comedy arc, screams,
             silences, cracks/pitch/chat/swear/nonsense)
      12. Heat model (excitement curve → hot zones)
      13. Three-way fusion: LLM windows + trigger windows + heat windows
      14. Score, deduplicate, cut
    """
    print("\n" + "=" * 60)
    print("  Sundown — Smart Clip" + (" (DRY RUN)" if dry_run else ""))
    print("=" * 60)

    # 1. Pick file
    video_path = pick_input_file()
    if video_path is None:
        return

    s              = _settings.load()
    pre            = float(s["pre_event_seconds"])
    post           = float(s["post_event_seconds"])
    model_name     = s["whisper_model"]
    use_whisper    = bool(s.get("use_whisper",             True))
    use_cache      = bool(s.get("use_transcript_cache",    True))
    top_n          = int(s.get("max_clips",                5))
    rms_factor     = float(s.get("rms_threshold_factor",   3.0))
    rms_sustain    = int(s.get("rms_min_sustain",          2))
    laugh_factor   = float(s.get("laughter_burst_factor",  2.0))
    laugh_count    = int(s.get("laughter_burst_count",     4))
    onset_thresh   = float(s.get("onset_novelty_threshold",   3.0))
    voice_thresh   = float(s.get("voice_excitement_threshold", 2.0))
    max_clip_dur   = float(s.get("max_clip_duration",      50.0))
    trim_to_peak   = bool(s.get("clip_trim_to_peak",       True))
    whisper_lang   = s.get("whisper_language", "auto")
    heat_decay     = float(s.get("smart_heat_decay",        0.92))
    heat_thresh    = float(s.get("smart_heat_threshold",    0.55))
    heat_min_dur   = float(s.get("smart_heat_min_duration", 2.0))
    heat_release   = max(heat_thresh * 0.64, heat_thresh - 0.20)
    heat_min_score = float(s.get("smart_score_heat_min",    0.20))
    win_min_score  = float(s.get("smart_score_window_min",  0.20))
    use_llm        = bool(s.get("use_llm_scoring",    True))
    llm_weight     = float(s.get("llm_window_weight", 0.60))

    if _logger:
        _logger.log_pipeline_start(video_path, s)
    # Step counts:  whisper+llm=15  whisper-only=14  no-whisper=13
    # Audio detectors start after transcription (if any); LLM runs after audio.
    total_steps  = 15 if (use_llm and use_whisper) else (14 if use_whisper else 13)
    audio_offset = 2  if use_whisper else 1

    os.makedirs(CLIPS_DIR, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Step 1: Extract audio
        _progress(1, total_steps, "Extracting audio")
        wav_path = extract_audio(video_path, tmp_dir)
        samples, sr = _load_wav_as_array(wav_path)
        video_duration = len(samples) / sr

        # Step 2: Transcribe full VOD (if use_whisper)
        transcript_segments: list[dict] = []
        if use_whisper:
            _progress(2, total_steps, "Transcribing full VOD (whisper.cpp)")
            if use_cache and _load_transcript_cache(video_path) is not None:
                if _logger:
                    _logger.log_whisper_cache_hit(video_path)
            else:
                if _logger:
                    _logger.log_whisper_start(model_name, video_duration)
            transcript_segments = _transcribe_full_vod(
                wav_path, model_name, whisper_lang, video_path, use_cache
            )
            print(f"       {len(transcript_segments)} segment(s) transcribed.")

        # Steps audio_offset+1 … audio_offset+8: Audio detectors
        # (LLM runs after audio so window_signals can be passed in)
        _progress(audio_offset + 1, total_steps, "Detecting loud moments (RMS)")
        rms_triggers = _rms_spike_detection(samples, sr,
                           threshold_factor=rms_factor,
                           min_sustain_frames=rms_sustain)

        _progress(audio_offset + 2, total_steps, "Detecting laughter")
        laugh_triggers = _laughter_detection(samples, sr,
                             burst_threshold_factor=laugh_factor,
                             burst_count=laugh_count)

        _progress(audio_offset + 3, total_steps, "Detecting audio novelty bursts")
        onset_triggers = _onset_novelty_detection(samples, sr,
                             novelty_threshold=onset_thresh)

        _progress(audio_offset + 4, total_steps, "Detecting voice excitement")
        voice_triggers = _voice_excitement_detection(samples, sr,
                             threshold_factor=voice_thresh)

        all_triggers: list[Trigger] = rms_triggers + laugh_triggers + onset_triggers + voice_triggers
        print(f"       Found: {len(rms_triggers)} RMS | {len(laugh_triggers)} laughter | "
              f"{len(onset_triggers)} novelty | {len(voice_triggers)} voice excitement")

        _progress(audio_offset + 5, total_steps, "Detecting comedic arcs (setup→reaction)")
        rise_triggers = _rms_rise_detection(samples, sr,
            rise_factor=float(s.get("comedy_rise_factor", 4.0)))
        all_triggers += rise_triggers
        print(f"       {len(rise_triggers)} comedic arc(s) detected.")

        _progress(audio_offset + 6, total_steps, "Detecting screams")
        scream_triggers = _scream_detection(
            samples, sr,
            min_pitch_hz=float(s.get("comedy_scream_min_pitch", 300)),
            max_duration_sec=float(s.get("comedy_scream_max_duration", 1.8)),
        )
        all_triggers += scream_triggers
        print(f"       {len(scream_triggers)} scream(s) detected.")

        _progress(audio_offset + 7, total_steps, "Detecting pre-reaction silences")
        silence_triggers = _pre_reaction_silence_detection(
            samples, sr,
            min_silence_sec=float(s.get("comedy_silence_min_sec", 0.3)),
            max_silence_sec=float(s.get("comedy_silence_max_sec", 2.0)),
        )
        all_triggers += silence_triggers
        print(f"       {len(silence_triggers)} pre-reaction silence(s) detected.")

        _progress(audio_offset + 8, total_steps, "Detecting voice cracks + chat + swear/nonsense")
        crack_triggers = _voice_crack_detection(samples, sr)
        pitch_triggers = _pitch_variance_detection(
            samples, sr,
            variance_threshold=float(s.get("comedy_pitch_variance_thresh", 80.0)),
        )
        import downloader as _dl
        chat_path     = _dl.get_chat_path(video_path)
        chat_triggers = _chat_velocity_detection(
            chat_path, video_duration,
            spike_factor=float(s.get("comedy_chat_spike_factor", 3.0)),
        )
        all_triggers += crack_triggers + pitch_triggers + chat_triggers
        print(f"       {len(crack_triggers)} crack(s), {len(pitch_triggers)} pitch spike(s), "
              f"{len(chat_triggers)} chat spike(s)."
              + (" (no chat log)" if not chat_path else ""))

        swear_triggers:    list = []
        nonsense_triggers: list = []
        if use_whisper and transcript_segments:
            swear_triggers    = _swear_detection(transcript_segments)
            nonsense_triggers = _nonsense_vocalization_detection(transcript_segments)
            all_triggers += swear_triggers + nonsense_triggers
            print(f"       {len(swear_triggers)} swear(s), {len(nonsense_triggers)} nonsense tag(s).")

        if _logger:
            _logger.log_audio_signals({
                "rms": len(rms_triggers), "laughter": len(laugh_triggers),
                "onset": len(onset_triggers), "voice": len(voice_triggers),
                "rise": len(rise_triggers), "scream": len(scream_triggers),
                "silence": len(silence_triggers), "crack": len(crack_triggers),
                "pitch": len(pitch_triggers), "chat": len(chat_triggers),
                "swear": len(swear_triggers), "nonsense": len(nonsense_triggers),
            })

        trigger_windows = _merge_triggers(
            all_triggers, pre=pre, post=post, max_duration=max_clip_dur
        ) if all_triggers else []
        print(f"       Trigger arm: {len(trigger_windows)} window(s).")

        # Ollama selects clip moments — runs after audio so window_signals is available
        llm_segs:         list[dict]  = []
        llm_suggestions:  list[dict]  = []
        llm_windows:      list[tuple] = []
        if use_llm and use_whisper:
            import llm as _llm
            _progress(audio_offset + 9, total_steps, "Asking Ollama to select clip moments")
            if transcript_segments:
                window_signals = _build_window_signal_summary(
                    trigger_windows, chat_triggers, swear_triggers,
                    scream_triggers, crack_triggers, silence_triggers, nonsense_triggers,
                    video_duration,
                )
                llm_segs, llm_suggestions = _llm.analyze_transcript(
                    transcript_segments, s, window_signals=window_signals
                )
                if llm_segs:
                    print(f"       {len(llm_segs)} LLM window(s) selected.")
                    llm_windows = _llm.build_llm_windows(llm_segs)
                else:
                    print("       [INFO] Ollama not running or returned no results — audio arm only.")
            else:
                print("       [INFO] No transcript — skipping LLM selection.")

        # Heat arm — excitement curve + hot zones
        _progress(total_steps - 3, total_steps, "Computing excitement curve + heat model")
        excitement = _compute_excitement_curve(samples, sr)
        heat       = _apply_heat_model(excitement, decay_per_frame=heat_decay)

        _progress(total_steps - 2, total_steps, "Finding heat zones")
        zones = _find_hot_zones(
            heat, _SMART_FRAME_SEC,
            threshold=heat_thresh, release=heat_release,
            min_duration_sec=heat_min_dur,
            pre=pre, post=post, max_duration=max_clip_dur,
        )
        heat_windows = [(s_z, e_z, [f"heat:{p:.2f}"]) for s_z, e_z, p in zones]
        print(f"       Heat arm: {len(heat_windows)} zone(s).")

        # Three-way fusion: LLM windows + trigger windows + heat windows
        _progress(total_steps - 1, total_steps, "Fusing + scoring")
        all_windows = sorted(llm_windows + trigger_windows + heat_windows, key=lambda w: w[0])
        unified: list[list] = []
        for win_start, win_end, win_labels in all_windows:
            if unified and win_start < unified[-1][1]:
                unified[-1][1] = max(unified[-1][1], win_end)
                unified[-1][2] = unified[-1][2] + win_labels
            else:
                unified.append([win_start, win_end, list(win_labels)])
        unified_windows = [
            (s_w, min(e_w, s_w + max_clip_dur), lbl)
            for s_w, e_w, lbl in unified
        ]

        if not unified_windows:
            print("\n[INFO] No moments detected. Try lowering the heat threshold or thresholds in Settings.")
            return

        print(f"       {len(unified_windows)} fused candidate(s).")

        llm_scores_mapped: list[float] | None = None
        if llm_segs:
            import llm as _llm
            llm_scores_mapped = _llm.map_llm_scores_to_windows(unified_windows, llm_segs)

        scored = _score_windows(
            unified_windows, samples, sr, transcript_segments,
            llm_scores_mapped, llm_weight=llm_weight,
        )

        def _extract_peak_heat(labels: list[str]) -> float:
            for lbl in labels:
                if lbl.startswith("heat:"):
                    try:
                        return float(lbl[5:])
                    except ValueError:
                        pass
            return 0.0

        combined = [
            (s_c, e_c, lbl, sw, 0.5 * composite + 0.5 * _extract_peak_heat(lbl))
            for s_c, e_c, lbl, sw, composite in scored
        ]
        combined.sort(key=lambda w: w[4], reverse=True)

        # Dual filter: clip must independently pass both minimum thresholds
        passing = [c for c in combined
                   if _extract_peak_heat(c[2]) >= heat_min_score and c[3] >= win_min_score]
        if not passing:
            print(f"       [INFO] No clips passed dual filter "
                  f"(heat≥{heat_min_score}, score≥{win_min_score}) — using top-N unfiltered.")
            passing = combined
        else:
            n_filtered = len(combined) - len(passing)
            if n_filtered:
                print(f"       Dual filter: {n_filtered} clip(s) below thresholds removed.")

        selected = _deduplicate_clips(passing[:top_n])
        min_score = float(s.get("min_clip_score", 0.30))
        if min_score > 0:
            qualifying = [c for c in selected if c[4] >= min_score]
            if qualifying:
                dropped = len(selected) - len(qualifying)
                if dropped:
                    print(f"       Quality filter: {dropped} clip(s) below {min_score:.2f} dropped.")
                selected = qualifying
            else:
                print(f"       [INFO] No clips reached quality floor {min_score:.2f} — keeping best clip.")
                selected = selected[:1]
        print(f"       Keeping top {len(selected)} of {len(combined)} candidate(s).")

        comedy_tail = float(s.get("comedy_tail_sec", 6.0))
        selected = _extend_comedy_clips(selected, comedy_tail, max_clip_dur)

        _progress(total_steps, total_steps, "Cutting clips" + (" (dry run)" if dry_run else ""))
        selected_by_time = sorted(selected, key=lambda w: w[0])
        base         = os.path.splitext(os.path.basename(video_path))[0]
        cut_count    = 0
        clip_records: list[dict] = []

        for i, (start, end, labels, sw_score, composite) in enumerate(selected_by_time, start=1):
            if trim_to_peak:
                start, end = _trim_to_peak(start, end, samples, sr, pre, post, max_clip_dur)
            out_name = os.path.join(
                CLIPS_DIR,
                f"{base}_smart_{i:02d}_{_fmt_time(start)}.mp4",
            )
            peak_heat = _extract_peak_heat(labels)
            print(f"\n  Clip {i}/{len(selected_by_time)}: "
                  f"{_fmt_time(start)} → {_fmt_time(end)}  "
                  f"(composite: {composite:.2f}  sw: {sw_score:.2f}  "
                  f"heat: {peak_heat:.2f})")
            for lbl in labels:
                print(f"    ↳ {lbl}")
            if _logger:
                _logger.log_clip_selected(
                    i, start, end, composite, sw_score, peak_heat, labels,
                )
            if dry_run:
                print(f"    [DRY RUN] Would save → {out_name}")
                cut_success = True
                cut_count  += 1
            else:
                cut_success = _cut_clip(video_path, start, end, out_name)
                if cut_success:
                    print(f"    ✓ Saved → {out_name}")
                    cut_count += 1
            clip_records.append({
                "rank":               i,
                "start_sec":          round(start, 2),
                "end_sec":            round(end,   2),
                "start_fmt":          _fmt_time(start),
                "end_fmt":            _fmt_time(end),
                "filename":           os.path.basename(out_name),
                "score_composite":    round(composite, 4),
                "score_sw":           round(sw_score,  4),
                "score_heat":         round(peak_heat, 4),
                "labels":             labels,
                "transcript_excerpt": _get_transcript_excerpt(transcript_segments, start, end),
                "llm_reason":         _find_llm_reason(llm_segs, start, end),
                "cut_success":        cut_success,
            })

    dry_tag = " (dry run — no files cut)" if dry_run else ""
    print(f"\n[SUCCESS] {cut_count}/{len(selected)} clip(s){dry_tag}.")

    if not dry_run:
        if clip_records:
            manifest_path = _write_manifest(base, "smart", video_path, clip_records, s)
            if manifest_path:
                print(f"  Manifest → {manifest_path}")
        _run_feedback_loop(clip_records, s)

    if llm_suggestions:
        print(f"\n  [Agent] {len(llm_suggestions)} setting suggestion(s) from this run:\n")
        import settings as _settings_mod
        s_current = _settings_mod.load()

        for suggestion in llm_suggestions:
            name      = suggestion["setting_name"]
            current   = suggestion["current_value"]
            proposed  = suggestion["suggested_value"]
            reason    = suggestion["reason"]
            direction = "↑" if proposed > current else "↓"

            print(f"  {name}: {current} → {proposed} {direction}")
            print(f"  \"{reason}\"")
            ans = input("  Apply? [y/N]: ").strip().lower()

            if ans == "y":
                s_current[name] = proposed
                _settings_mod.save(s_current)
                print("  ✓ Applied.")
                if _logger:
                    _logger.log_suggestion_accepted(name, current, proposed, reason)
            else:
                print("  Skipped.")
                if _logger:
                    _logger.log_suggestion_rejected(name, current, proposed, reason)
            print()

    if not dry_run and cut_count > 0:
        _handle_original(video_path)
