# 🎸 Clone Hero Content Manager

A single-container content management system for Clone Hero with **Nextcloud-powered storage**. Browse, upload, generate, and edit songs — all from one lightweight, fully stateless service. Your entire library (songs, assets, and database) lives on Nextcloud via WebDAV.

## Architecture

This project was simplified from a 14-container microservice architecture down to a **single stateless Docker container** running:

- **FastAPI** web application serving both the UI and REST API
- **SQLite** embedded database as a local metadata cache (persisted to Nextcloud)
- **Jinja2** templates with a responsive sidebar UI
- **Nextcloud WebDAV** as the primary storage backend for everything
- **librosa** audio analysis for automatic chart generation

The container is fully stateless — no volumes or bind mounts are required. The SQLite database is downloaded from Nextcloud on startup, uploaded back periodically (every 5 minutes by default), and saved one final time on graceful shutdown. Songs, assets, and all persistent data live exclusively on Nextcloud.

```
┌─────────────────────────────────────────────┐
│       Stateless Docker Container            │
│                                             │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐  │
│  │ FastAPI  │  │  SQLite  │  │  librosa  │  │
│  │ + Jinja2 │  │ (cache)  │  │  (audio)  │  │
│  └────┬─────┘  └────┬─────┘  └───────────┘  │
│       │              │                      │
│       ├── HTML Pages (/, /songs, /upload...) │
│       ├── REST API (/api/*)                 │
│       └── Static Files (/static/*)          │
│                      │                      │
│              ┌───────┴────────┐             │
│              │ DB downloaded  │             │
│              │ on startup,    │             │
│              │ uploaded every  │             │
│              │ 5 min + on     │             │
│              │ shutdown        │             │
│              └───────┬────────┘             │
└──────────────────────┼──────────────────────┘
                       │  WebDAV (GET/PUT/MKCOL/
                       │  PROPFIND/DELETE/MOVE)
                 ┌─────┴─────┐
                 │ Nextcloud  │  ← everything
                 │  WebDAV    │    lives here
                 └───────────┘
```

### Data Flow

```
Upload .zip/.rar ──► Extract to /tmp ──► Parse song.ini ──► Upload to Nextcloud ──► Cache metadata in SQLite
                                                                                          │
Generate chart ────► Analyse audio ───► Create chart ─────► Upload to Nextcloud ──► Cache metadata in SQLite
                                                                                          │
Refresh Library ──────────────────────► Scan Nextcloud ──► Parse each song.ini ──► Upsert into SQLite
                                                                                   + purge stale entries
```

### Database Lifecycle

The SQLite database is treated as a **local cache** that is periodically synced to Nextcloud:

1. **Startup:** Download `clonehero.db` from Nextcloud (`/Database/clonehero.db` by default). If not found, a fresh database is created.
2. **Runtime:** A background task uploads the database to Nextcloud every `DB_SYNC_INTERVAL` seconds (default: 300 / 5 minutes).
3. **Shutdown:** The database is uploaded to Nextcloud one final time before the process exits.
4. **Atomic writes:** Uploads go to a `.tmp` file first, then are renamed via WebDAV MOVE for safety.

> **Note:** SQLite does not support concurrent writers across multiple processes. Run a single instance of this container. If you need multi-instance scaling, migrate to PostgreSQL.

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/your-username/games_clonehero.git
cd games_clonehero
cp .env.example .env
# Edit .env with your Nextcloud credentials (required for all storage)
```

### 2. Start with Docker

```bash
# Using the utility script (interactive menu):
./utils.sh

# Or directly with Docker Compose:
docker compose up -d --build

# The app will be available at http://localhost:8000
```

No volumes are mounted — the container is fully stateless.

### 3. Start for development (no Docker)

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run with auto-reload
./utils.sh  # Select option [5] "Run locally (dev mode)"

# Or manually:
PYTHONPATH=$(pwd) python -m uvicorn src.main:app --reload --port 8000
```

### 4. Set up Nextcloud

1. Generate an **App Password** in Nextcloud: Settings → Security → Devices & sessions
2. Create a `CloneHero` folder in your Nextcloud files (or customise `NEXTCLOUD_REMOTE_PATH`)
3. Create subfolders: `Songs`, `Backgrounds`, `Colors`, `Highways`, `Database`, `Generator`, `Temp`
4. Set the `NEXTCLOUD_URL`, `NEXTCLOUD_USERNAME`, and `NEXTCLOUD_PASSWORD` variables in your `.env`
5. Restart the service
6. Click **🔄 Refresh Library** on the home page or songs page to scan Nextcloud

## Features

### 📁 Song Library
Browse, search, and manage your Clone Hero song collection. Full-text search across title, artist, and album with pagination. Song metadata is cached locally in SQLite for fast queries; the actual files live on Nextcloud.

### 📤 Content Upload
Upload `.zip` or `.rar` archives containing Clone Hero songs. The system automatically extracts archives, parses `song.ini` files, uploads each song folder to Nextcloud, and caches metadata locally. Non-song assets (backgrounds, colors, highways) are also uploaded to their respective Nextcloud folders.

### ✏️ Song Editor
Edit song metadata (title, artist, album, genre, difficulty ratings, etc.) through a rich form interface. Changes are saved to both the local database cache and the `song.ini` file on Nextcloud, keeping them in sync.

### 🎵 Song Generator
Upload audio files (MP3, WAV, OGG, FLAC, OPUS) and automatically generate Clone Hero charts:
- Tempo detection via beat tracking
- Note placement from onset detection
- Section markers (Intro, Verse, Chorus, etc.)
- Multiple difficulty levels (Easy, Medium, Hard, Expert)
- Generates `notes.chart` + `song.ini` + copies audio
- Output is uploaded directly to Nextcloud

### 🔄 Library Sync
One-click refresh that scans the Nextcloud songs directory, discovers all `song.ini` files, parses their metadata, and upserts everything into the local cache. Songs that have been removed from Nextcloud are automatically purged from the local database.

### ☁️ Nextcloud Browser
Browse your entire Nextcloud file tree via WebDAV directly from the app:
- Navigate folders, view file metadata
- **Import** song folders into your library (metadata-only — song stays on Nextcloud)
- **Upload** files from your computer to Nextcloud
- Create folders, delete files/folders

## Project Structure

```
games_clonehero/
├── src/
│   ├── main.py                 # FastAPI app, lifespan (DB sync), logging
│   ├── config.py               # Configuration from environment
│   ├── database.py             # SQLite database (async + sync)
│   ├── webdav.py               # Nextcloud WebDAV client + song helpers
│   ├── routes/
│   │   ├── pages.py            # HTML page routes (Jinja2)
│   │   └── api.py              # REST API endpoints
│   ├── services/
│   │   ├── content_manager.py  # Song parsing, extraction, Nextcloud upload
│   │   └── song_generator.py   # Audio analysis, chart generation, upload
│   ├── templates/              # Jinja2 HTML templates
│   │   ├── base.html           # Base layout with sidebar
│   │   ├── home.html           # Dashboard
│   │   ├── songs.html          # Song library browser
│   │   ├── editor.html         # Song metadata editor
│   │   ├── upload.html         # Content upload (drag & drop)
│   │   ├── generator.html      # Song chart generator
│   │   └── browser.html        # Nextcloud file browser
│   └── static/
│       ├── css/style.css       # Application styles
│       ├── js/main.js          # Client-side utilities
│       └── assets/             # Icons, images
├── docker/
│   ├── Dockerfile              # Single multi-stage Dockerfile
│   └── entrypoint.sh           # Privilege drop (root → appuser)
├── docker-compose.yml          # Single-service compose file (no volumes)
├── requirements.txt            # Python dependencies
├── .env.example                # Environment configuration template
├── utils.sh                    # Management utility script
└── README.md
```

## Configuration

All settings are controlled via environment variables in the `.env` file. See [`.env.example`](.env.example) for the full list with descriptions.

### Key Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_PORT` | `8000` | Port the web server listens on |
| `APP_ENV` | `development` | `development` or `production` |
| `DEBUG` | `true` | Enable debug logging and API docs |
| `LOG_LEVEL` | `DEBUG` | Log level for stdout output |
| `NEXTCLOUD_URL` | _(empty)_ | Nextcloud instance URL (required) |
| `NEXTCLOUD_USERNAME` | _(empty)_ | Nextcloud username (required) |
| `NEXTCLOUD_PASSWORD` | _(empty)_ | Nextcloud password or app password (required) |
| `NEXTCLOUD_REMOTE_PATH` | `/remote.php/dav/files/{username}/CloneHero` | WebDAV path template |
| `NEXTCLOUD_SONGS_PATH` | `/Songs` | Song directory on Nextcloud (relative to WebDAV root) |
| `NEXTCLOUD_DB_PATH` | `/Database/clonehero.db` | Where the SQLite DB is stored on Nextcloud |
| `DB_SYNC_INTERVAL` | `300` | Seconds between periodic DB uploads to Nextcloud |
| `NEXTCLOUD_BACKGROUNDS_PATH` | `/Backgrounds` | Nextcloud folder for background assets |
| `NEXTCLOUD_COLORS_PATH` | `/Colors` | Nextcloud folder for colour assets |
| `NEXTCLOUD_HIGHWAYS_PATH` | `/Highways` | Nextcloud folder for highway assets |
| `MAX_FILE_SIZE_GB` | `10` | Maximum upload file size |

### Nextcloud Folder Layout

The app expects the following folder structure under your `CloneHero` WebDAV root:

```
CloneHero/
├── Songs/          # Song folders (each with song.ini, notes.chart, audio)
├── Backgrounds/    # Custom background images
├── Colors/         # Custom colour themes
├── Highways/       # Custom highway textures
├── Database/       # SQLite database (managed automatically)
├── Generator/      # Generated chart output
└── Temp/           # Temporary processing files
```

These folders are created automatically when you upload content, but it's recommended to create them in Nextcloud ahead of time.

### Nextcloud Setup

1. Generate an **App Password** in Nextcloud: Settings → Security → Devices & sessions
2. Create a `CloneHero` folder in your Nextcloud (or customise `NEXTCLOUD_REMOTE_PATH`)
3. Create the subfolders listed above (or let the app create them on first upload)
4. Set the three `NEXTCLOUD_*` credential variables in your `.env`
5. Restart the service
6. Click **🔄 Refresh Library** to populate the local metadata cache

## API Reference

When running in development mode, interactive API documentation is available at:

- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/songs` | List songs (with search & pagination) |
| `GET` | `/api/songs/{id}` | Get song by ID |
| `PUT` | `/api/songs/{id}` | Update song metadata (DB + song.ini on Nextcloud) |
| `DELETE` | `/api/songs/{id}` | Delete song (Nextcloud + DB) |
| `POST` | `/api/upload` | Upload content archive (songs → Nextcloud) |
| `POST` | `/api/generate` | Generate chart from audio (output → Nextcloud) |
| `POST` | `/api/library/sync` | Refresh library: scan Nextcloud → update DB cache |
| `GET` | `/api/library/status` | Library stats + Nextcloud connection info |
| `GET` | `/api/webdav/status` | Check Nextcloud connection |
| `GET` | `/api/webdav/browse` | Browse Nextcloud directory |
| `POST` | `/api/webdav/import` | Import a song folder's metadata into the library |
| `POST` | `/api/webdav/upload` | Upload a file to Nextcloud |
| `POST` | `/api/webdav/mkdir` | Create a directory on Nextcloud |
| `DELETE` | `/api/webdav/delete` | Delete a file/directory on Nextcloud |

## Utility Script

The `utils.sh` script provides an interactive menu for common operations:

```bash
./utils.sh        # Interactive menu
./utils.sh -y     # Non-interactive (auto-start service)
```

| Option | Action |
|--------|--------|
| 0 | Start service (Docker) |
| 1 | Stop service |
| 2 | Restart service |
| 3 | Show status & health |
| 4 | View logs (live) |
| 5 | Run locally (dev mode, no Docker) |
| 6 | Build & push Docker image |
| 7 | _(reserved)_ |
| 8 | _(reserved)_ |
| 9 | Docker cleanup (prune all) |

## Song.ini Format

Songs must include a `song.ini` file with at minimum:

```ini
[song]
name = Song Title
artist = Artist Name
album = Album Name
```

Optional fields: `genre`, `year`, `charter`, `icon`, `loading_phrase`, `song_length`, `preview_start_time`, `delay`, `modchart`, and difficulty ratings (`diff_guitar`, `diff_drums`, `diff_bass`, `diff_keys`, etc.)

## What Changed (v1 → v2)

| Before (v1) | After (v2) |
|-------------|------------|
| 14 Docker containers | 1 stateless Docker container |
| PostgreSQL database | SQLite (embedded, synced to Nextcloud) |
| Syncthing file sync | Nextcloud WebDAV (primary storage) |
| Django + FastAPI + Streamlit | Single FastAPI app |
| Nginx reverse proxy | Direct Uvicorn/Gunicorn |
| Redis cache | Not needed |
| Prometheus + Grafana + Datadog | Loguru stdout logging |
| Multiple Dockerfiles | Single multi-stage Dockerfile |
| 7 Docker images to build | 1 Docker image |
| Songs stored locally | Everything stored on Nextcloud |
| Host volume mounts required | No volumes — fully stateless |
| File-based logging | stdout only (Docker logs) |

## License

See [LICENSE](LICENSE) for details.