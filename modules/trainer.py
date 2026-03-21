# ============================================================
# trainer.py  —  MODULE (imported by main.py and clipper.py)
# Training pipeline: profile I/O, preference similarity,
# comedy profile similarity, and run_train().
# ============================================================

import os
import json
import glob
import tempfile

import numpy as np
import settings as _settings

_PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILE_PATH    = os.path.join(_PROJECT_ROOT, "configuration", "clip_profile.json")
PROFILE_VERSION = 2        # v2 adds comedy sub-profile
TRAINING_DIR    = "training"

_COMEDY_KEYS = [
    "swear_density", "scream_presence", "nonsense_density",
    "pitch_variance_mean", "pre_silence_count", "voice_crack_count",
    "chat_spike_density",
]


# ============================================================
# Profile I/O
# ============================================================

def load_profile() -> dict | None:
    """Load the saved clip preference profile. Returns None if not trained yet.

    Accepts both v1 (no comedy) and v2 profiles.
    """
    if not os.path.exists(PROFILE_PATH):
        return None
    try:
        with open(PROFILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") in (1, 2) and "profile" in data:
            return data
    except (OSError, json.JSONDecodeError, KeyError):
        pass
    return None


def save_profile(data: dict) -> None:
    """Write a profile dict to disk."""
    os.makedirs(os.path.dirname(PROFILE_PATH), exist_ok=True)
    try:
        with open(PROFILE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError as exc:
        print(f"[WARNING] Could not save clip profile: {exc}")


def clear_profile() -> None:
    """Delete the saved training profile so all clips are reprocessed on next train."""
    if not os.path.exists(PROFILE_PATH):
        print("\n[INFO] No training profile found — nothing to clear.")
        return
    try:
        os.remove(PROFILE_PATH)
        print(f"\n[SUCCESS] Training profile cleared. All clips in '{TRAINING_DIR}/' will be "
              f"reprocessed on the next training run.")
    except OSError as exc:
        print(f"[ERROR] Could not delete profile: {exc}")


# ============================================================
# Similarity functions
# ============================================================

def preference_similarity(window_features: dict, profile: dict) -> float:
    """Cosine similarity between a window's feature vector and the preference profile.

    Features used: rms_mean, voice_ratio_mean, centroid_mean, flux_variance,
    trigger_density, and the 4 trigger_presence values (all normalized).
    Returns a value in [0, 1].
    """
    p = profile.get("profile", {})
    keys = ["rms_mean", "voice_ratio_mean", "centroid_mean",
            "flux_variance", "trigger_density"]
    pres_keys = ["rms_spike", "laughter", "onset_novelty", "voice_excitement"]

    pv, wv = [], []
    for k in keys:
        pv.append(float(p.get(k, 0.0)))
        wv.append(float(window_features.get(k, 0.0)))
    for k in pres_keys:
        pv.append(float(p.get("trigger_presence", {}).get(k, 0.0)))
        wv.append(float(window_features.get("trigger_presence", {}).get(k, 0.0)))

    pv_arr = np.array(pv, dtype=float)
    wv_arr = np.array(wv, dtype=float)
    denom  = np.linalg.norm(pv_arr) * np.linalg.norm(wv_arr)
    if denom < 1e-9:
        return 0.0
    return float(np.clip(np.dot(pv_arr, wv_arr) / denom, 0.0, 1.0))


def comedy_profile_similarity(window_comedy_features: dict, profile: dict) -> float:
    """Cosine similarity between a window's comedy features and the learned comedy profile.

    window_comedy_features keys (rates per minute or binary presence):
      swear_density, scream_presence, nonsense_density, pitch_variance_mean,
      pre_silence_count, voice_crack_count, chat_spike_density

    Returns 0.0 gracefully when no comedy sub-profile exists.
    """
    comedy = profile.get("profile", {}).get("comedy")
    if not comedy:
        return 0.0

    wv = np.array([float(window_comedy_features.get(k, 0.0)) for k in _COMEDY_KEYS],
                  dtype=float)
    pv = np.array([float(comedy.get(k, 0.0)) for k in _COMEDY_KEYS], dtype=float)

    denom = np.linalg.norm(pv) * np.linalg.norm(wv)
    if denom < 1e-9:
        return 0.0
    return float(np.clip(np.dot(pv, wv) / denom, 0.0, 1.0))


# ============================================================
# Training pipeline
# ============================================================

def run_train() -> None:
    """Profile clips in training/ to build a preference model.

    The resulting clip_profile.json is used by Smart Clip's _score_windows
    to boost candidate windows that match the user's preferred clip style.
    Learns comedy patterns (swear/scream/silence/crack/pitch) from training clips.
    """
    import clipper as _clipper  # lazy import — avoids circular dependency

    print("\n" + "=" * 60)
    print("  Sundown — Train Smart Clip")
    print("=" * 60)

    os.makedirs(TRAINING_DIR, exist_ok=True)
    extensions = ("*.mp4", "*.mkv", "*.mov", "*.avi", "*.ts", "*.flv")
    training_files: list[str] = []
    for ext in extensions:
        training_files.extend(glob.glob(os.path.join(TRAINING_DIR, ext)))
    training_files = sorted(training_files)

    if not training_files:
        print(f"\n[INFO] No video files found in '{TRAINING_DIR}/'.")
        print(f"       Drop highlight clips into '{TRAINING_DIR}/' and run again.")
        return

    s = _settings.load()

    # Load existing profile or start fresh
    existing = load_profile() or {
        "version": PROFILE_VERSION,
        "clip_count": 0,
        "trained_clips": [],
        "profile": {},
    }
    trained_clips: list[str] = existing.get("trained_clips", [])

    new_files = [f for f in training_files
                 if os.path.basename(f) not in trained_clips]

    if not new_files:
        print(f"\n[INFO] All {len(training_files)} clip(s) in '{TRAINING_DIR}/' were already trained.")
        print("       Add new clips to the folder to update the profile.")
        return

    print(f"\n  Found {len(new_files)} new clip(s) to process "
          f"({len(trained_clips)} already in profile).")

    # Accumulate feature sums for weighted averaging
    sums: dict = {}
    pres_sums: dict = {"rms_spike": 0.0, "laughter": 0.0,
                       "onset_novelty": 0.0, "voice_excitement": 0.0}
    comedy_sums: dict = {k: 0.0 for k in _COMEDY_KEYS}
    scalar_keys = ["rms_mean", "voice_ratio_mean", "centroid_mean",
                   "flux_variance", "trigger_density"]

    n_existing = existing["clip_count"]
    if n_existing > 0:
        ep = existing.get("profile", {})
        for k in scalar_keys:
            sums[k] = ep.get(k, 0.0) * n_existing
        for k in ("rms_spike", "laughter", "onset_novelty", "voice_excitement"):
            pres_sums[k] = ep.get("trigger_presence", {}).get(k, 0.0) * n_existing
        ec = ep.get("comedy", {})
        for k in _COMEDY_KEYS:
            comedy_sums[k] = ec.get(k, 0.0) * n_existing
    else:
        for k in scalar_keys:
            sums[k] = 0.0

    processed = 0
    new_memory_entries: list[dict] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for idx, fpath in enumerate(new_files, start=1):
            fname = os.path.basename(fpath)
            print(f"\n  [{idx}/{len(new_files)}] {fname}")
            try:
                wav_path = _clipper.extract_audio(fpath, tmp_dir)
                samples, sr = _clipper._load_wav_as_array(wav_path)
                feats = _clipper._profile_clip(samples, sr, s)

                for k in scalar_keys:
                    sums[k] += feats.get(k, 0.0)
                for k in ("rms_spike", "laughter", "onset_novelty", "voice_excitement"):
                    pres_sums[k] += feats["trigger_presence"].get(k, 0.0)

                cf = feats.get("comedy_features", {})
                for k in _COMEDY_KEYS:
                    comedy_sums[k] += cf.get(k, 0.0)

                trained_clips.append(fname)
                processed += 1
                print(f"       triggers: {feats['trigger_presence']}  "
                      f"rms: {feats['rms_mean']:.4f}  "
                      f"scream: {'yes' if cf.get('scream_presence') else 'no'}  "
                      f"cracks: {cf.get('voice_crack_count', 0.0):.2f}/min")

                # Transcribe clip for comedy memory analysis
                try:
                    _, clip_segs = _clipper._run_whisper(
                        wav_path,
                        model_name=s.get("whisper_model", "small"),
                        language=s.get("whisper_language", "auto"),
                        verbose=False,
                    )
                except Exception:
                    clip_segs = []

                # Collect audio hints from extracted features
                audio_hints = []
                if cf.get("scream_presence", 0)                          > 0:   audio_hints.append("scream")
                if cf.get("voice_crack_count", 0)                        > 1.0: audio_hints.append("voice_crack")
                if cf.get("pre_silence_count", 0)                        > 1.0: audio_hints.append("pre_silence")
                if cf.get("swear_density", 0)                            > 0:   audio_hints.append("swear")
                if cf.get("nonsense_density", 0)                         > 0:   audio_hints.append("nonsense")
                if feats.get("trigger_presence", {}).get("laughter", 0)  > 0:   audio_hints.append("laughter")

                # Ask LLM why this clip is funny
                if clip_segs:
                    import llm as _llm
                    analysis = _llm.analyze_why_funny(
                        _llm.format_transcript(clip_segs), audio_hints, s
                    )
                    if analysis and analysis.get("confidence", 0) >= 40:
                        new_memory_entries.append({
                            "clip_name":   fname,
                            "text_sample": " ".join(
                                seg.get("text", "").strip() for seg in clip_segs
                            )[:300],
                            "why_funny":   analysis["why_funny"],
                            "humor_type":  analysis["humor_type"],
                            "confidence":  analysis["confidence"],
                            "audio_hints": audio_hints,
                            "score":       1.0,
                            "source":      "training",
                        })
                        print(f"       LLM: [{analysis['humor_type']}] {analysis['why_funny'][:80]}...")
                    else:
                        print("       LLM: skipped (low confidence or Ollama not running)")
                else:
                    print("       LLM: skipped (transcription failed)")

                try:
                    os.remove(wav_path)
                except OSError:
                    pass
            except Exception as exc:
                print(f"       [WARNING] Skipped ({exc})")

    if processed == 0:
        print("\n[INFO] No clips were successfully processed.")
        return

    total = n_existing + processed
    new_profile: dict = {k: sums[k] / total for k in scalar_keys}
    new_profile["trigger_presence"] = {
        k: pres_sums[k] / total
        for k in ("rms_spike", "laughter", "onset_novelty", "voice_excitement")
    }
    new_profile["comedy"] = {k: comedy_sums[k] / total for k in _COMEDY_KEYS}

    profile_data = {
        "version":       PROFILE_VERSION,
        "clip_count":    total,
        "trained_clips": trained_clips,
        "profile":       new_profile,
    }
    save_profile(profile_data)

    dominant = sorted(new_profile["trigger_presence"].items(),
                      key=lambda x: x[1], reverse=True)
    dominant_str = ", ".join(f"{k} ({v:.0%})" for k, v in dominant if v > 0)
    cp = new_profile["comedy"]
    print(f"\n[SUCCESS] Profile updated from {total} clip(s).")
    print(f"          Comedy: scream in {cp['scream_presence']:.0%}, "
          f"cracks {cp['voice_crack_count']:.2f}/min, "
          f"silences {cp['pre_silence_count']:.2f}/min")
    print(f"          Dominant signals: {dominant_str or 'none'}")
    print(f"          Saved to: {PROFILE_PATH}")

    # Save comedy memory entries from this training run
    if new_memory_entries:
        import llm as _llm
        memory   = _llm.load_comedy_memory(s)
        existing_names = {e.get("clip_name", "") for e in memory if e.get("source") == "training"}
        truly_new = [e for e in new_memory_entries if e["clip_name"] not in existing_names]
        if truly_new:
            memory.extend(truly_new)
            _llm.save_comedy_memory(memory, s)
            print(f"\n  ✓ Comedy memory: {len(truly_new)} new entry(s) ({len(memory)} total)")
            for e in truly_new:
                print(f"    [{e['humor_type']}] {e['clip_name']}: {e['why_funny'][:80]}...")

    # Offer to reset memory if it exists
    import llm as _llm
    existing_memory = _llm.load_comedy_memory(s)
    if existing_memory:
        ans = input(
            f"\n  Reset LLM memory to match new profile? ({len(existing_memory)} entries) [y/N]: "
        ).strip().lower()
        if ans == "y":
            _llm.save_comedy_memory([], s)
            print("  ✓ Memory cleared — will rebuild on next training run.")
