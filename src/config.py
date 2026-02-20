"""
Clone Hero Content Manager - Configuration
All settings loaded from environment variables with sensible defaults.

The application is designed to run as a stateless container.  Persistent
data (songs, database, assets) lives on Nextcloud via WebDAV.  Local disk
is used only for transient staging (temp files) during processing.
"""

import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
_ = load_dotenv()

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
APP_ENV = os.getenv("APP_ENV", "development")
APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
DEBUG = os.getenv("DEBUG", "true").lower() == "true"
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")

if APP_ENV == "production" and SECRET_KEY == "change-me-in-production":
    raise RuntimeError(
        "SECRET_KEY must be changed from the default value in production. "
        "Set the SECRET_KEY environment variable to a random secret."
    )

# ---------------------------------------------------------------------------
# Authentication (simple single-user login)
# ---------------------------------------------------------------------------
AUTH_USERNAME = os.getenv("AUTH_USERNAME", "nuniesmith")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "")  # MUST be set in .env
CHARTER_NAME = os.getenv("CHARTER_NAME", "nuniesmith")
# Session cookie name and max age (seconds) — default 30 days
SESSION_COOKIE_NAME = "ch_session"
SESSION_MAX_AGE = int(os.getenv("SESSION_MAX_AGE", str(60 * 60 * 24 * 30)))

# ---------------------------------------------------------------------------
# Paths — local disk is used only for transient data
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Temporary staging area for archive extraction, audio processing, etc.
# Uses the system temp directory so nothing persists across container restarts.
TEMP_DIR = Path(os.getenv("TEMP_DIR", os.path.join(tempfile.gettempdir(), "clonehero")))

# SQLite database lives in a temp location; it is downloaded from Nextcloud
# on startup and uploaded back periodically + on shutdown.
DB_PATH = Path(os.getenv("DB_PATH", os.path.join(TEMP_DIR, "clonehero.db")))

TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# ---------------------------------------------------------------------------
# Logging — stdout only (no file logging for stateless containers)
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")

# ---------------------------------------------------------------------------
# Nextcloud WebDAV
# ---------------------------------------------------------------------------
NEXTCLOUD_URL = os.getenv("NEXTCLOUD_URL", "")  # e.g. https://cloud.example.com
NEXTCLOUD_USERNAME = os.getenv("NEXTCLOUD_USERNAME", "")
NEXTCLOUD_PASSWORD = os.getenv("NEXTCLOUD_PASSWORD", "")
NEXTCLOUD_REMOTE_PATH = os.getenv(
    "NEXTCLOUD_REMOTE_PATH", "/remote.php/dav/files/{username}/CloneHero"
)

# Root folder on Nextcloud where songs are stored (relative to WEBDAV_BASE_URL)
NEXTCLOUD_SONGS_PATH = os.getenv("NEXTCLOUD_SONGS_PATH", "/Songs")

# Path on Nextcloud where the SQLite database is persisted
NEXTCLOUD_DB_PATH = os.getenv("NEXTCLOUD_DB_PATH", "/Database/clonehero.db")

# How often (in seconds) to upload the DB to Nextcloud as a backup
DB_SYNC_INTERVAL = int(os.getenv("DB_SYNC_INTERVAL", "300"))  # default 5 minutes

# How often (in seconds) to auto-sync the song library from Nextcloud.
# Set to 0 to disable auto-sync (manual only).
LIBRARY_SYNC_INTERVAL = int(os.getenv("LIBRARY_SYNC_INTERVAL", "120"))  # default 2 min
# Whether to run a full library sync on application startup
LIBRARY_SYNC_ON_STARTUP = os.getenv("LIBRARY_SYNC_ON_STARTUP", "true").lower() == "true"

# Build the full WebDAV base URL
WEBDAV_BASE_URL: str = (
    NEXTCLOUD_URL.rstrip("/")
    + NEXTCLOUD_REMOTE_PATH.format(username=NEXTCLOUD_USERNAME)
    if NEXTCLOUD_URL and NEXTCLOUD_USERNAME
    else ""
)

# ---------------------------------------------------------------------------
# Nextcloud folder mappings
#
# Maps content types to their Nextcloud folder paths (relative to the
# WebDAV base URL / CloneHero root).  Songs have their own dedicated
# config var above; these cover everything else.
# ---------------------------------------------------------------------------
NEXTCLOUD_FOLDERS = {
    "songs": NEXTCLOUD_SONGS_PATH,
    "backgrounds": os.getenv("NEXTCLOUD_BACKGROUNDS_PATH", "/Backgrounds"),
    "colors": os.getenv("NEXTCLOUD_COLORS_PATH", "/Colors"),
    "highways": os.getenv("NEXTCLOUD_HIGHWAYS_PATH", "/Highways"),
    "generator": os.getenv("NEXTCLOUD_GENERATOR_PATH", "/Generator"),
    "temp": os.getenv("NEXTCLOUD_TEMP_PATH", "/Temp"),
    "database": os.getenv("NEXTCLOUD_DATABASE_PATH", "/Database"),
}

# Local temp sub-directory names used during processing.
CONTENT_FOLDERS = {
    "backgrounds": "backgrounds",
    "colors": "colors",
    "highways": "highways",
    "temp": "temp",
}

# ---------------------------------------------------------------------------
# Upload limits
# ---------------------------------------------------------------------------
MAX_FILE_SIZE_GB = int(os.getenv("MAX_FILE_SIZE_GB", "10"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_GB * 1024 * 1024 * 1024

ALLOWED_ARCHIVE_EXTENSIONS = {".zip", ".rar"}
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".ogg", ".wav", ".flac", ".opus"}
ALLOWED_UPLOAD_EXTENSIONS = (
    ALLOWED_ARCHIVE_EXTENSIONS | ALLOWED_IMAGE_EXTENSIONS | ALLOWED_AUDIO_EXTENSIONS
)

# ---------------------------------------------------------------------------
# Song.ini optional metadata fields
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# MusicBrainz / metadata lookup
# ---------------------------------------------------------------------------
MUSICBRAINZ_ENABLED = os.getenv("MUSICBRAINZ_ENABLED", "true").lower() == "true"
# User-Agent string required by MusicBrainz API policy
MUSICBRAINZ_USER_AGENT = os.getenv(
    "MUSICBRAINZ_USER_AGENT",
    "CloneHeroManager/1.0.0 (https://github.com/nuniesmith/games_clonehero)",
)
# Cover Art Archive base URL
COVER_ART_ARCHIVE_URL = os.getenv(
    "COVER_ART_ARCHIVE_URL", "https://coverartarchive.org"
)

OPTIONAL_SONG_FIELDS = [
    "genre",
    "year",
    "album_track",
    "playlist_track",
    "charter",
    "icon",
    "diff_guitar",
    "diff_rhythm",
    "diff_bass",
    "diff_guitar_coop",
    "diff_drums",
    "diff_drums_real",
    "diff_guitarghl",
    "diff_bassghl",
    "diff_rhythm_ghl",
    "diff_guitar_coop_ghl",
    "diff_keys",
    "song_length",
    "preview_start_time",
    "video_start_time",
    "modchart",
    "loading_phrase",
    "delay",
]


def ensure_directories() -> None:
    """Create the local temp directories needed for transient processing.

    Only the temp staging area and the DB parent directory are needed on
    local disk.  All persistent storage is on Nextcloud.
    """
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
