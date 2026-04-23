# Sundown — Claude Instructions

## Hard Constraints

- **Never modify, create, or delete any file inside `Whisper/`.**
  This folder contains the whisper.cpp binary and models and is managed externally.
  Read paths from it (for `_find_whisper_cpp` / `_whisper_cpp_model_path`) but never touch its contents.
