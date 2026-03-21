# ============================================================
# downloader.py  —  MODULE (imported by main.py, not run directly)
# Handles authentication, VOD verification, and downloading.
# ============================================================

# --- Imports ------------------------------------------------
import subprocess
import sys
import os
import requests
from datetime import datetime

# --- Twitch Credentials -------------------------------------
# Credentials are fetched automatically from the Twitch CLI.
# Run the following once to set up:
#   twitch configure          ← enter your Client ID & Secret
#   twitch token -u           ← opens browser to authorize
#
# If you prefer to set them manually, replace the constants below
# and the auto-fetch functions will not be called.

TWITCH_CLIENT_ID   = ""   # Leave empty to read from Twitch CLI
TWITCH_OAUTH_TOKEN = ""   # Leave empty to read from Twitch CLI


# ============================================================
# Twitch CLI Auth Helpers
# ============================================================

import json as _json

# Absolute path to the project root (parent of the modules/ folder).
# Using __file__ makes every path resolve correctly regardless of the
# working directory when main.py is launched.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CREDS_FILE   = os.path.join(_PROJECT_ROOT, "configuration", "twitch_credentials.json")


# ============================================================
# Twitch CLI config-file helpers
# ============================================================

def _twitch_cli_config_paths() -> list[str]:
    """Return candidate paths for the Twitch CLI config file (Windows-first)."""
    candidates = []
    appdata = os.environ.get("APPDATA", "")
    home    = os.path.expanduser("~")
    if appdata:
        candidates.append(os.path.join(appdata, "twitch-cli", "config"))
        candidates.append(os.path.join(appdata, "twitch-cli", ".twitch-cli.env"))
    candidates += [
        os.path.join(home, ".config", "twitch-cli", "config"),
        os.path.join(home, ".twitch-cli", "config"),
        os.path.join(home, ".twitch-cli", ".twitch-cli.env"),
    ]
    return candidates


def _read_twitch_cli_config() -> dict[str, str]:
    """Parse the Twitch CLI config file into a normalised {KEY: value} dict.

    The CLI (backed by viper) writes plain key=value pairs, e.g.::

        clientid = xxxxxxxxxxxxxxxx
        clientsecret = xxxxxxxxxxxxxxxx
        accesstoken = xxxxxxxxxxxxxxxx

    Keys are normalised to uppercase so callers can use e.g. ``cfg["CLIENTID"]``.
    """
    for path in _twitch_cli_config_paths():
        if not os.path.exists(path):
            continue
        cfg: dict[str, str] = {}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for raw in fh:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    # Accept both   key = value   and   key=value
                    if "=" in line:
                        k, _, v = line.partition("=")
                        cfg[k.strip().upper()] = v.strip().strip('"').strip("'")
            if cfg:
                return cfg
        except OSError:
            continue
    return {}


def _saved_credentials() -> tuple[str, str]:
    """Return (client_id, token) previously saved by setup_twitch_cli()."""
    if not os.path.exists(_CREDS_FILE):
        return "", ""
    try:
        with open(_CREDS_FILE, "r", encoding="utf-8") as fh:
            data = _json.load(fh)
        return data.get("client_id", ""), data.get("token", "")
    except Exception:
        return "", ""


def _save_credentials(client_id: str, token: str) -> None:
    """Persist credentials to configuration/twitch_credentials.json."""
    os.makedirs(os.path.dirname(_CREDS_FILE), exist_ok=True)
    try:
        with open(_CREDS_FILE, "w", encoding="utf-8") as fh:
            _json.dump({"client_id": client_id, "token": token}, fh, indent=4)
    except OSError as exc:
        print(f"[WARNING] Could not save credentials: {exc}")


def get_client_id_from_cli() -> str:
    """Return the Twitch Client ID, checking (in order):

    1. Our own saved credentials (configuration/twitch_credentials.json)
    2. The Twitch CLI config file
    3. The hardcoded TWITCH_CLIENT_ID constant
    """
    saved_id, _ = _saved_credentials()
    if saved_id:
        return saved_id

    cfg = _read_twitch_cli_config()
    for key in ("CLIENTID", "CLIENT_ID"):
        if cfg.get(key):
            return cfg[key]

    return TWITCH_CLIENT_ID


def get_token_from_cli() -> str:
    """Return the OAuth token, checking (in order):

    1. Our own saved credentials (configuration/twitch_credentials.json)
    2. The Twitch CLI config file  (key: ACCESSTOKEN)
    3. The ``twitch token`` CLI command output
    4. The hardcoded TWITCH_OAUTH_TOKEN constant
    """
    _, saved_token = _saved_credentials()
    if saved_token:
        return saved_token

    cfg = _read_twitch_cli_config()
    for key in ("ACCESSTOKEN", "TOKEN", "USERTOKEN"):
        if cfg.get(key):
            return cfg[key]

    # Last resort: generate a fresh app access token via `twitch token`
    # (docs: output is "App Access Token: <token>")
    # Note: this is a client-credentials token, valid for public content only.
    try:
        result = subprocess.run(
            ["twitch", "token"],
            capture_output=True, text=True, timeout=15,
        )
        for line in (result.stdout + result.stderr).splitlines():
            if "access token:" in line.lower():
                t = line.split(":", 1)[-1].strip()
                if t:
                    return t
    except FileNotFoundError:
        print("[WARNING] Twitch CLI not found. Using TWITCH_OAUTH_TOKEN constant.")
    except Exception as exc:
        print(f"[WARNING] Could not fetch token from Twitch CLI: {exc}")

    return TWITCH_OAUTH_TOKEN


def setup_twitch_cli() -> bool:
    """Guide the user through first-time Twitch CLI setup.

    After the CLI commands complete, reads and saves the credentials to
    configuration/twitch_credentials.json so Sundown can find them
    reliably without depending on the CLI config format in future runs.

    Returns True if usable credentials were saved.
    """
    print("\n[Twitch CLI Setup]")
    print("  Step 1 — Configure your Client ID & Secret:")
    print("    Run: twitch configure")
    print()
    print("  Step 2 — Authorize your Twitch account (opens browser):")
    print("    Run: twitch token -u")
    print()
    print("  OAuth Redirect URL must be set to:  http://localhost:3000")
    print("  (set this in your app at dev.twitch.tv/console/apps)")
    print()

    choice = input("  Open Twitch CLI configuration now? [y/N]: ").strip().lower()
    if choice == "y":
        try:
            subprocess.run(["twitch", "configure"], check=True)
            subprocess.run(["twitch", "token", "-u"], check=True)
        except FileNotFoundError:
            print("[ERROR] 'twitch' command not found.")
            print("        Download the Twitch CLI from: https://dev.twitch.tv/docs/cli/")
            return False
        except subprocess.CalledProcessError:
            print("[ERROR] Twitch CLI setup failed. Please try running manually.")
            return False

    # Read fresh credentials directly from the Twitch CLI config file
    # (don't use _saved_credentials() here — we want the newly issued token)
    cfg       = _read_twitch_cli_config()
    client_id = cfg.get("CLIENTID", "") or cfg.get("CLIENT_ID", "") or TWITCH_CLIENT_ID
    token     = (cfg.get("ACCESSTOKEN", "")
                 or cfg.get("TOKEN", "")
                 or cfg.get("USERTOKEN", "")
                 or TWITCH_OAUTH_TOKEN)

    # If auto-detection failed, ask the user to paste them manually
    if not client_id:
        print("\n  Could not read Client ID from Twitch CLI config automatically.")
        client_id = input("  Paste your Twitch Client ID here: ").strip()

    if not token:
        print("\n  Could not read OAuth token from Twitch CLI config automatically.")
        print("  Run  twitch token  in a terminal and paste the token shown.")
        token = input("  Paste your OAuth token here: ").strip()

    if client_id and token:
        _save_credentials(client_id, token)

        # Fetch and store the Twitch display name so the menu shows it
        username = _fetch_twitch_username(token, client_id)
        if username:
            import settings as _settings
            _settings.set_twitch_username(username)
            print(f"\n[SUCCESS] Logged in as @{username}")
        else:
            print("\n[SUCCESS] Credentials saved to configuration/twitch_credentials.json")
        return True

    print("[ERROR] Could not obtain credentials. Run setup again after completing Twitch CLI auth.")
    return False


def has_credentials() -> bool:
    """Return True if a usable token + client ID are available."""
    token     = TWITCH_OAUTH_TOKEN or get_token_from_cli()
    client_id = TWITCH_CLIENT_ID   or get_client_id_from_cli()
    return bool(token and client_id)


def _fetch_twitch_username(token: str, client_id: str) -> str:
    """Call /helix/users to get the display name for the authenticated account.

    Returns the display name string, or "" on failure.
    """
    try:
        resp = requests.get(
            "https://api.twitch.tv/helix/users",
            headers={
                "Authorization": f"Bearer {token.replace('oauth:', '').strip()}",
                "Client-Id": client_id,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if data:
            return data[0].get("display_name", "")
    except Exception:
        pass
    return ""


def erase_credentials() -> bool:
    """Revoke and erase the Twitch CLI user token.

    Runs `twitch token --revoke <token>` to revoke the stored user token,
    which also removes it from the CLI's local store.
    Returns True on success.
    """
    token = TWITCH_OAUTH_TOKEN or get_token_from_cli()

    if not token:
        print("[INFO] No stored credentials found — nothing to erase.")
        return True

    confirm = input("  Are you sure you want to erase your Twitch credentials? [y/N]: ").strip().lower()
    if confirm != "y":
        print("  Cancelled.")
        return False

    client_id = get_client_id_from_cli()

    try:
        # Docs: twitch token -r <token> --client-id <id>
        cmd = ["twitch", "token", "--revoke", token]
        if client_id:
            cmd += ["--client-id", client_id]
        subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        # Whether or not the CLI revoke succeeded, always wipe our local files
    except Exception:
        pass

    # Clear the stored token but keep the file/client_id so re-setup is easier
    _save_credentials(client_id or "", "")

    import settings as _settings
    _settings.clear_twitch_username()

    print("[SUCCESS] Token revoked. Use option [5] to re-authorise.")
    return True


# --- VOD / Stream URL ---------------------------------------
# Paste the full URL of the Twitch VOD you want to download.
# Examples:
#   VOD  → "https://www.twitch.tv/videos/123456789"
#   Live → "https://www.twitch.tv/your_channel_name"

VOD_URL = "https://www.twitch.tv/videos/YOUR_VOD_ID_HERE"


def set_vod_url(url: str) -> None:
    """Allow main.py (or any caller) to set the VOD URL at runtime.

    Example usage in main.py::

        import downloader
        downloader.set_vod_url("https://www.twitch.tv/videos/123456789")
        downloader.run_download()
    """
    global VOD_URL
    VOD_URL = url

# --- Output Settings ----------------------------------------
# Folder where the downloaded file will be saved.
OUTPUT_DIR = "downloads"

# Preferred stream quality.
# Options (best → worst): "best", "1080p60", "720p60", "720p", "480p", "360p", "worst"
QUALITY = "best"

# ============================================================
# Downloader Logic  (no edits needed below this line)
# ============================================================

def build_output_path(output_dir: str, url: str, quality: str, title: str = "") -> str:
    """Build an output file path in the format: StreamTitle_Quality_Timestamp.mp4"""
    os.makedirs(output_dir, exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")

    if title:
        # Sanitize: keep alphanumeric, spaces, hyphens, underscores; collapse the rest
        safe = "".join(c if c.isalnum() or c in " _-" else " " for c in title)
        safe = "_".join(safe.split())   # collapse whitespace → underscores
        safe = safe[:80].rstrip("_") or "vod"
    else:
        safe = url.rstrip("/").split("/")[-1]   # fall back to VOD ID

    filename = f"{safe}_{quality}_{date_str}.mp4"
    return os.path.join(output_dir, filename)


def validate_credentials() -> tuple[bool, str, str]:
    """Resolve and validate credentials, auto-fetching from the Twitch CLI.

    Returns (ok, client_id, token).  ok is False if credentials could
    not be resolved, in which case an error has already been printed.
    """
    client_id = TWITCH_CLIENT_ID or get_client_id_from_cli()
    token     = TWITCH_OAUTH_TOKEN or get_token_from_cli()

    problems = []
    if not client_id:
        problems.append("  • Could not find a Twitch Client ID (run: twitch configure).")
    if not token:
        problems.append("  • Could not find an OAuth token (run: twitch token -u).")
    if VOD_URL.endswith("YOUR_VOD_ID_HERE"):
        problems.append("  • VOD_URL has not been set.")

    if problems:
        print("[ERROR] Cannot proceed — missing credentials:\n")
        for p in problems:
            print(p)
        print()
        print("  Tip: run option [2] from the main menu to configure Twitch CLI.")
        return False, "", ""

    return True, client_id, token


def verify_vod_accessible(url: str, oauth_token: str, client_id: str) -> tuple[bool, str]:
    """Verify the VOD is accessible via the Twitch API.

    Returns (accessible, title).  title is an empty string for live streams
    or when the API call fails.  Only checks /videos endpoint for VOD URLs.
    """
    if "/videos/" not in url:
        print("[INFO] URL appears to be a live stream — skipping VOD accessibility check.")
        return True, ""

    video_id = url.rstrip("/").split("/videos/")[-1]
    api_url = f"https://api.twitch.tv/helix/videos?id={video_id}"

    token = oauth_token.replace("oauth:", "").strip()
    headers = {
        "Client-ID": client_id,
        "Authorization": f"Bearer {token}",
    }

    try:
        response = requests.get(api_url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        if not data.get("data"):
            print(f"[ERROR] VOD not found or not accessible (ID: {video_id}).")
            print("        Make sure the VOD exists and your credentials have the correct scopes.")
            return False, ""

        vod_info = data["data"][0]
        print(f"[INFO] VOD found: \"{vod_info['title']}\"")
        print(f"       Channel : {vod_info['user_name']}")
        print(f"       Duration: {vod_info['duration']}")
        return True, vod_info["title"]

    except requests.RequestException as exc:
        print(f"[WARNING] Could not verify VOD via Twitch API: {exc}")
        print("          Proceeding with download attempt anyway...")
        return True, ""  # Allow download attempt even if API check fails


def _ytdlp_cmd() -> list[str]:
    """Return the command prefix for invoking yt-dlp (PATH or module fallback)."""
    import shutil
    if shutil.which("yt-dlp"):
        return ["yt-dlp"]
    return [sys.executable, "-m", "yt_dlp"]


# Maps the QUALITY constant to yt-dlp format selectors
_YTDLP_FORMAT: dict[str, str] = {
    "best":    "bestvideo+bestaudio/best",
    "1080p60": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
    "720p60":  "bestvideo[height<=720]+bestaudio/best[height<=720]",
    "720p":    "bestvideo[height<=720]+bestaudio/best[height<=720]",
    "480p":    "bestvideo[height<=480]+bestaudio/best[height<=480]",
    "360p":    "bestvideo[height<=360]+bestaudio/best[height<=360]",
    "worst":   "worstvideo+worstaudio/worst",
}


def download_vod_ytdlp(url: str, quality: str, output_path: str, oauth_token: str) -> None:
    """Download a Twitch VOD using yt-dlp."""
    import settings as _settings
    s           = _settings.load()
    fmt         = _YTDLP_FORMAT.get(quality, "bestvideo+bestaudio/best")
    base_cmd    = _ytdlp_cmd()
    browser     = s.get("twitch_browser", "chrome")
    cookie_file = s.get("twitch_cookie_file", "").strip()

    fragments = int(s.get("download_concurrent_fragments", 1))
    command = base_cmd + [
        "--output", output_path,
        "--format", fmt,
        "--merge-output-format", "mp4",
        "--concurrent-fragments", str(fragments),
    ]

    # Cookie file takes priority — user exported from browser to bypass DPAPI/ABE.
    # Otherwise fall back to --cookies-from-browser.
    # Zen Browser uses Firefox's profile format, so map it to "firefox".
    if cookie_file and os.path.isfile(cookie_file):
        print(f"[INFO] Using cookie file: {cookie_file}")
        command += ["--cookies", cookie_file]
    else:
        effective_browser = "firefox" if browser == "zen" else browser
        if effective_browser and effective_browser != "none":
            command += ["--cookies-from-browser", effective_browser]
        else:
            # Last resort: Helix OAuth token header (often rejected by GQL)
            clean_token = oauth_token.replace("oauth:", "").strip()
            if clean_token:
                command += ["--add-header", f"Authorization:OAuth {clean_token}"]

    command += ["--write-comments", "--write-info-json"]
    command.append(url)

    print(f"\n[INFO] Starting download (yt-dlp)...")
    print(f"       URL     : {url}")
    print(f"       Quality : {quality}")
    print(f"       Output  : {output_path}")

    try:
        subprocess.run(command, check=True)
        print(f"\n[SUCCESS] Download complete → {output_path}")
    except subprocess.CalledProcessError as exc:
        print(f"\n[ERROR] yt-dlp exited with code {exc.returncode}.")
        print("        Check the yt-dlp output above for the exact reason.")
        print("        Common causes:")
        print("          • DPAPI/Chrome ABE error  →  export cookies to a file and set")
        print("            the path in Settings [C], or switch browser to Edge/Firefox")
        print("          • Invalid or expired token  →  re-run option [5] to refresh")
        print("          • VOD deleted/subscriber-only →  verify the URL is accessible")
        sys.exit(exc.returncode)
    except FileNotFoundError:
        print("\n[ERROR] Could not launch yt-dlp.")
        print("        Run:  pip install yt-dlp")
        sys.exit(1)


def _streamlink_cmd() -> list[str]:
    """Return the command prefix for invoking streamlink.

    Tries the ``streamlink`` executable on PATH first; if not found (common
    on Windows when pip's Scripts/ folder isn't in PATH), falls back to
    ``python -m streamlink`` using the same interpreter that's running now.
    """
    import shutil
    if shutil.which("streamlink"):
        return ["streamlink"]
    return [sys.executable, "-m", "streamlink"]


def download_vod(url: str, quality: str, output_path: str, oauth_token: str) -> None:
    """Invoke streamlink to download the VOD to *output_path*."""
    base_cmd = _streamlink_cmd()
    clean_token = oauth_token.replace("oauth:", "").strip()
    command = base_cmd + [
        "--output", output_path,
        "--twitch-api-header", f"Authorization=OAuth {clean_token}",
        "--twitch-supported-codecs", "h264,h265,av1",
        url,
        quality,
    ]

    print(f"\n[INFO] Starting download...")
    print(f"       URL     : {url}")
    print(f"       Quality : {quality}")
    print(f"       Output  : {output_path}")

    try:
        subprocess.run(command, check=True)
        print(f"\n[SUCCESS] Download complete → {output_path}")
    except subprocess.CalledProcessError as exc:
        print(f"\n[ERROR] streamlink exited with code {exc.returncode}.")
        print("        Check the streamlink output above for the exact reason.")
        print("        Common causes:")
        print("          • Invalid or expired token  →  re-run option [5] to refresh")
        print("          • Quality unavailable        →  try 'best' in Settings")
        print("          • VOD deleted/subscriber-only →  verify the URL is accessible")
        sys.exit(exc.returncode)
    except FileNotFoundError:
        print("\n[ERROR] Could not launch streamlink.")
        print(f"        Tried: {' '.join(command[:2])}")
        print("        Make sure you have installed the requirements:")
        print("          pip install -r requirements.txt")
        sys.exit(1)


def run_download() -> None:
    """Public entry point — call this from main.py after set_vod_url()."""
    print("=" * 60)
    print("  Sundown — Twitch VOD Downloader")
    print("=" * 60)

    # 1. Resolve & validate credentials (auto-fetches from Twitch CLI)
    ok, client_id, token = validate_credentials()
    if not ok:
        sys.exit(1)

    # 2. Verify the VOD is accessible via the Twitch API
    accessible, vod_title = verify_vod_accessible(VOD_URL, token, client_id)
    if not accessible:
        sys.exit(1)

    # 3. Build the output file path (uses VOD title when available)
    output_path = build_output_path(OUTPUT_DIR, VOD_URL, QUALITY, title=vod_title)

    # 4. Download using the configured backend
    import settings as _settings
    backend = _settings.load().get("downloader_backend", "yt-dlp")

    if backend == "streamlink":
        download_vod(VOD_URL, QUALITY, output_path, token)
    else:
        download_vod_ytdlp(VOD_URL, QUALITY, output_path, token)


def get_chat_path(video_path: str) -> str | None:
    """Return the chat JSON path for a given video path if it exists, else None.

    yt-dlp --write-info-json --write-comments writes a <base>.info.json file
    containing a 'comments' array alongside the downloaded video.
    """
    base = os.path.splitext(video_path)[0]
    info_json = base + ".info.json"
    return info_json if os.path.exists(info_json) else None
