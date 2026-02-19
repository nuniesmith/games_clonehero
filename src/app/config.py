"""
Clone Hero Content Manager - Configuration
All settings loaded from environment variables with sensible defaults.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
APP_ENV = os.getenv("APP_ENV", "development")
APP_VERSION = os.getenv("APP_VERSION", "2.0.0")
DEBUG = os.getenv("DEBUG", "true").lower() == "true"
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data")))
CONTENT_DIR = DATA_DIR / "clonehero_content"
DB_PATH = DATA_DIR / "clonehero.db"

TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# Content subdirectories
CONTENT_FOLDERS = {
    "songs": "songs",
    "backgrounds": "backgrounds",
    "colors": "colors",
    "highways": "highways",
    "generator": "generator",
    "temp": "temp",
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = Path(os.getenv("LOG_DIR", str(DATA_DIR / "logs")))
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
LOG_FILE_SIZE = os.getenv("LOG_FILE_SIZE", "10MB")
LOG_RETENTION = os.getenv("LOG_RETENTION", "7 days")

# ---------------------------------------------------------------------------
# Nextcloud WebDAV
# ---------------------------------------------------------------------------
NEXTCLOUD_URL = os.getenv("NEXTCLOUD_URL", "")  # e.g. https://cloud.example.com
NEXTCLOUD_USERNAME = os.getenv("NEXTCLOUD_USERNAME", "")
NEXTCLOUD_PASSWORD = os.getenv("NEXTCLOUD_PASSWORD", "")
NEXTCLOUD_REMOTE_PATH = os.getenv(
    "NEXTCLOUD_REMOTE_PATH", "/remote.php/dav/files/{username}/CloneHero"
)

# Build the full WebDAV base URL
if NEXTCLOUD_URL and NEXTCLOUD_USERNAME:
    WEBDAV_BASE_URL = NEXTCLOUD_URL.rstrip("/") + NEXTCLOUD_REMOTE_PATH.format(
        username=NEXTCLOUD_USERNAME
    )
else:
    WEBDAV_BASE_URL = ""

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
    """Create all required directories if they don't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for folder in CONTENT_FOLDERS.values():
        (CONTENT_DIR / folder).mkdir(parents=True, exist_ok=True)
