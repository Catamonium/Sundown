# Sundown

An automated VOD clipping tool for Twitch streamers. Sundown processes long-form stream recordings and surfaces the moments most worth editing into short-form content — it does not produce finished clips, it finds the raw material so editors can work faster.

---

## What it does

Sundown runs two separate pipelines on a downloaded VOD:

**Fast Clip** — Audio-only detection. Scans for loud moments, laughter patterns, audio novelty bursts, and voice excitement using signal processing. Fast, no LLM required, good for a quick pass.

**Smart Clip** — Full pipeline. Transcribes the entire VOD with whisper.cpp, feeds the transcript to a locally hosted LLM (via Ollama) to select clip candidates, runs all audio detectors in parallel, fuses both sets of results, scores and ranks them. The LLM layer is informed by a comedy memory built up over time from your training clips — it learns what your specific audience finds funny and gets better the more you train it.

The system is intentionally designed as an **editor's assistant, not a replacement**. Sundown outputs raw MP4 clips with human-readable filenames and a score breakdown. Creative decisions stay with the editor.

---

## Requirements

**Python packages**
```
pip install streamlink yt-dlp requests ffmpeg-python numpy librosa
```

**External tools — not pip packages**
- [ffmpeg](https://ffmpeg.org/download.html) — must be on PATH
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) — see Whisper setup section below
- [Ollama](https://ollama.com) — runs the local LLM for Smart Clip. Pull your preferred model: `ollama pull llama3.2`
- [Twitch CLI](https://dev.twitch.tv/docs/cli/) — used for authentication setup

---

## Whisper setup

Sundown uses [whisper.cpp](https://github.com/ggerganov/whisper.cpp) for transcription.
The binary and model files are not included in this repo and must be set up manually.

**Step 1 — Get the binary**

Download a prebuilt release from [whisper.cpp releases](https://github.com/ggerganov/whisper.cpp/releases)
or build from source. For AMD GPU acceleration on Windows, build with Vulkan:
```
cmake -DGGML_VULKAN=1 -B build
cmake --build build --config Release
```
Place `main.exe` (or `whisper-cli.exe`) and all `.dll` files into the `Whisper/` folder.

**Step 2 — Download a model**

Models are hosted on Hugging Face at [ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp/tree/main).

| Model | Size | Notes |
|---|---|---|
| `ggml-small.bin` | ~466 MB | Good starting point |
| `ggml-medium.bin` | ~1.5 GB | Better accuracy, recommended for non-English |

Download your chosen model and place it in `Whisper/models/`.

**Step 3 — Verify**

Run `python main.py` — if whisper.cpp is found, startup will confirm it.
If not found, a warning will show with the expected path.

---

## Setup

1. Clone the repo
2. Install Python packages above
3. Place whisper.cpp binary and models in `Whisper/`
4. Run `python main.py`
5. Select **[5] Set up Twitch CLI auth** and follow the prompts
6. Drop clips you consider funny into `training/` and run **[6] Train Smart Clip** — this builds the audio profile and bootstraps the comedy memory via Ollama

---

## Project structure

```
Sundown/
├── main.py                  — entry point and menu
├── modules/
│   ├── clipper.py           — Fast Clip and Smart Clip pipelines
│   ├── downloader.py        — VOD + chat log downloading via yt-dlp / streamlink
│   ├── trainer.py           — training pipeline, audio profile, comedy memory bootstrap
│   ├── llm.py               — Ollama integration, comedy memory I/O, why-funny analysis
│   └── settings.py          — persistent settings with interactive editor
├── configuration/
│   ├── settings.json        — user settings
│   ├── clip_profile.json    — learned audio preference profile
│   ├── comedy_memory.json   — LLM comedy memory (built during training)
│   └── transcript_cache/    — cached Whisper transcripts (skips re-transcription on re-runs)
├── training/                — drop funny clips here before running Train
├── downloads/               — VODs downloaded by Sundown
├── clips/                   — output clips
├── input/                   — alternative input folder for local files
└── Whisper/                 — whisper.cpp binary and models (not tracked by git)
```

---

## How the comedy learning works

Training clips are ground truth. When you run **Train Smart Clip**:
1. Each clip is transcribed with whisper.cpp
2. Audio features are extracted (scream presence, voice cracks, pre-reaction silences, etc.)
3. The LLM is asked to explain *why* each clip is funny — the specific comedic structure, not just "loud moment"
4. Those explanations are saved to `comedy_memory.json`

On future Smart Clip runs, the LLM reads the comedy memory and uses those explanations as reference examples when scoring the new VOD's transcript. The more you train, the more specifically it understands your humor style.

---

## Settings

Run `python main.py` and select **[4] Settings** for the interactive editor. Key settings:

| Setting | Default | Description |
|---|---|---|
| `whisper_model` | `medium` | `small` or `medium` recommended for Spanish |
| `whisper_language` | `es` | ISO language code or `auto` |
| `use_llm_scoring` | `true` | Enable Ollama-based segment selection |
| `llm_model` | `llama3.2` | Ollama model name |
| `max_clips` | `10` | Max clips produced per run |
| `max_clip_duration` | `50` | Hard cap per clip in seconds |

---

## AMD GPU note

whisper.cpp supports Vulkan on Windows, which works with AMD GPUs (RDNA and newer). Build whisper.cpp with `-DGGML_VULKAN=1` to enable it. Ollama also handles AMD GPU acceleration automatically on Windows.

PyTorch-based Whisper does not support AMD GPUs on Windows — this is why Sundown uses whisper.cpp instead of openai-whisper.

---

## AI Disclosure

The architecture, design decisions, feature selection, and direction of this project were conceived and driven entirely by the developer. AI coding assistants (Claude) were used extensively during implementation to accelerate writing boilerplate, translating design intent into working code, and debugging. Every technical decision — what signals to detect, how the pipeline is structured, what makes a clip worth keeping, how the learning system works — was made by the developer. AI was the hands, not the brain.
