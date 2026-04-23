# ============================================================
# main.py  —  Entry Point
# Run this file to start Sundown.
#   python main.py
# ============================================================

import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "modules"))

import downloader   # noqa: E402
import clipper      # noqa: E402
import settings     # noqa: E402
import trainer      # noqa: E402
import logger       # noqa: E402


# ------------------------------------------------------------
# Startup utilities
# ------------------------------------------------------------

_REQUIRED_DIRS = ("input", "downloads", "clips", "configuration", "training")


def _cls() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def _ensure_folders() -> None:
    for folder in _REQUIRED_DIRS:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            print(f"[INFO] Created folder: {folder}/")


def _check_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        print("\n[WARNING] ffmpeg not found on PATH — clipping will not work.\n"
              "          Download: https://ffmpeg.org/download.html\n")


def _check_whisper_cpp() -> None:
    if clipper._WHISPER_CPP_BIN is None:
        print("\n[WARNING] whisper.cpp binary not found — transcription disabled.\n"
              "          Place whisper-cli.exe (or main.exe) in: Whisper/\n")


# ------------------------------------------------------------
# Menu actions
# ------------------------------------------------------------

def action_download_vod() -> None:
    print()
    while True:
        url = input("Paste the Twitch VOD URL and press Enter:\n> ").strip()
        if not url:
            print("[ERROR] URL cannot be empty. Please try again.\n")
        elif "twitch.tv" not in url:
            print("[ERROR] That doesn't look like a Twitch URL. Please try again.\n")
        else:
            break
    downloader.set_vod_url(url)
    downloader.run_download()


def action_fast_clip()        -> None: clipper.run_fast_clip()
def action_smart_clip()       -> None: clipper.run_smart_clip()
def action_fast_clip_dry()    -> None: clipper.run_fast_clip(dry_run=True)
def action_smart_clip_dry()   -> None: clipper.run_smart_clip(dry_run=True)
def action_settings()         -> None: settings.show_menu()
def action_setup_twitch_cli() -> None: downloader.setup_twitch_cli()


def action_train_smart_clip() -> None:
    while True:
        _cls()
        profile = trainer.load_profile()
        clip_count = profile["clip_count"] if profile else 0
        status = f"{clip_count} clip(s) in profile" if profile else "no profile yet"
        print("=" * 60)
        print("  Sundown — Training")
        print(f"  Status: {status}")
        print("=" * 60)
        print("  [1] Train  (drop clips into training/ then run this)")
        print("  [2] Clear training profile  (reprocess all clips)")
        print("  [0] Back")
        print("-" * 60)
        choice = input("  > ").strip()
        if choice in ("0", ""):   return
        elif choice == "1":       trainer.run_train()
        elif choice == "2":       trainer.clear_profile()
        else:                     print(f"\n  '{choice}' is not a valid option.")


# ------------------------------------------------------------
# Menu
# ------------------------------------------------------------

_MENU: dict[str, tuple[str, callable]] = {
    "1": ("Download a Twitch VOD",                          action_download_vod),
    "2": ("Fast Clip",                                      action_fast_clip),
    "3": ("Smart Clip",                                     action_smart_clip),
    "4": ("Settings",                                       action_settings),
    "5": ("Set up Twitch CLI auth",                         action_setup_twitch_cli),
    "6": ("Train Smart Clip  (drop clips into training/)",  action_train_smart_clip),
    "7": ("Fast Clip  — dry run  (score only, no cutting)", action_fast_clip_dry),
    "8": ("Smart Clip — dry run  (score only, no cutting)", action_smart_clip_dry),
}


def show_menu() -> None:
    _cls()
    s        = settings.load()
    username = s.get("twitch_username") or "(not connected)"
    print("=" * 60)
    print("  Sundown — Main Menu")
    print(f"  Account: @{username}")
    print("=" * 60)
    for key, (label, _) in _MENU.items():
        print(f"  [{key}] {label}")
    print("  [0] Exit")
    print("-" * 60)


def main() -> None:
    _ensure_folders()
    _check_ffmpeg()
    _check_whisper_cpp()

    s = settings.load()
    logger.init(debug=s.get("debug_mode", False))

    while True:
        show_menu()
        choice = input("Select an option: ").strip()
        if choice == "0":
            print("\nGoodbye!\n")
            logger.close()
            sys.exit(0)
        if choice in _MENU:
            _MENU[choice][1]()
        else:
            print(f"\n[ERROR] '{choice}' is not a valid option. Please try again.")
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
