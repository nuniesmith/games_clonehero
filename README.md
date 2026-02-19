# 🎸 Clone Hero Content Manager

A single-container content management system for Clone Hero with Nextcloud WebDAV integration. Browse, upload, generate, and edit songs — all from one lightweight service.

## Architecture

This project was simplified from a 14-container microservice architecture down to a **single Docker container** running:

- **FastAPI** web application serving both the UI and REST API
- **SQLite** embedded database (no separate database container needed)
- **Jinja2** templates with a responsive sidebar UI
- **Nextcloud WebDAV** integration for centralized cloud storage
- **librosa** audio analysis for automatic chart generation

```
┌─────────────────────────────────────────────┐
│           Single Docker Container           │
│                                             │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐  │
│  │ FastAPI  │  │  SQLite  │  │  librosa  │  │
│  │ + Jinja2 │  │    DB    │  │  (audio)  │  │
│  └────┬─────┘  └──────────┘  └───────────┘  │
│       │                                     │
│       ├── HTML Pages (/, /songs, /upload...) │
│       ├── REST API (/api/*)                 │
│       └── Static Files (/static/*)          │
│                                             │
└───────────────┬─────────────────────────────┘
                │
          ┌─────┴─────┐
          │ Nextcloud  │  (optional)
          │  WebDAV    │
          └───────────┘
```

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/your-username/games_clonehero.git
cd games_clonehero
cp .env.example .env
# Edit .env with your settings (Nextcloud credentials are optional)
```

### 2. Start with Docker

```bash
# Using the utility script (interactive menu):
./utils.sh

# Or directly with Docker Compose:
docker compose up -d --build

# The app will be available at http://localhost:8000
```

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
PYTHONPATH=$(pwd) python -m uvicorn src.app.main:app --reload --port 8000
```

## Features

### 📁 Song Library
Browse, search, and manage your Clone Hero song collection. Full-text search across title, artist, and album with pagination.

### 📤 Content Upload
Upload `.zip` or `.rar` archives containing Clone Hero songs. The system automatically extracts archives, parses `song.ini` files, organizes content into `Artist/Title` folders, and registers metadata in the database.

### ✏️ Song Editor
Edit song metadata (title, artist, album, genre, difficulty ratings, etc.) through a rich form interface. Changes are saved to both the database and the `song.ini` file on disk. Supports all standard Clone Hero metadata fields.

### 🎵 Song Generator
Upload audio files (MP3, WAV, OGG, FLAC, OPUS) and automatically generate Clone Hero charts:
- Tempo detection via beat tracking
- Note placement from onset detection
- Section markers (Intro, Verse, Chorus, etc.)
- Multiple difficulty levels (Easy, Medium, Hard, Expert)
- Generates `notes.chart` + `song.ini` + copies audio

### ☁️ Nextcloud Browser
Browse your Nextcloud file tree via WebDAV directly from the app:
- Navigate folders, view file metadata
- **Import** songs or archives from Nextcloud into your local library
- **Upload** files from your computer to Nextcloud
- **Sync** local songs to Nextcloud with one click
- Create folders, delete files/folders

## Project Structure

```
games_clonehero/
├── src/
│   └── app/
│       ├── main.py              # FastAPI application entry point
│       ├── config.py            # Configuration from environment
│       ├── database.py          # SQLite database (async + sync)
│       ├── webdav.py            # Nextcloud WebDAV client
│       ├── routes/
│       │   ├── pages.py         # HTML page routes (Jinja2)
│       │   └── api.py           # REST API endpoints
│       ├── services/
│       │   ├── content_manager.py  # Song parsing, archive extraction
│       │   └── song_generator.py   # Audio analysis, chart generation
│       ├── templates/           # Jinja2 HTML templates
│       │   ├── base.html        # Base layout with sidebar
│       │   ├── home.html        # Dashboard
│       │   ├── songs.html       # Song library browser
│       │   ├── editor.html      # Song metadata editor
│       │   ├── upload.html      # Content upload (drag & drop)
│       │   ├── generator.html   # Song chart generator
│       │   └── browser.html     # Nextcloud file browser
│       └── static/
│           ├── css/style.css    # Application styles
│           ├── js/main.js       # Client-side utilities
│           └── assets/          # Icons, images
├── docker/
│   └── Dockerfile              # Single multi-stage Dockerfile
├── data/                       # Persistent data (gitignored)
│   ├── clonehero.db            # SQLite database
│   ├── clonehero_content/      # Song files, assets
│   │   ├── songs/
│   │   ├── backgrounds/
│   │   ├── colors/
│   │   ├── highways/
│   │   └── generator/
│   └── logs/
├── docker-compose.yml          # Single-service compose file
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
| `NEXTCLOUD_URL` | _(empty)_ | Nextcloud instance URL |
| `NEXTCLOUD_USERNAME` | _(empty)_ | Nextcloud username |
| `NEXTCLOUD_PASSWORD` | _(empty)_ | Nextcloud password or app password |
| `NEXTCLOUD_REMOTE_PATH` | `/remote.php/dav/files/{username}/CloneHero` | WebDAV path template |
| `MAX_FILE_SIZE_GB` | `10` | Maximum upload file size |

### Nextcloud Setup

1. Generate an **App Password** in Nextcloud: Settings → Security → Devices & sessions
2. Create a `CloneHero` folder in your Nextcloud (or customize `NEXTCLOUD_REMOTE_PATH`)
3. Set the three `NEXTCLOUD_*` variables in your `.env`
4. Restart the service

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
| `PUT` | `/api/songs/{id}` | Update song metadata |
| `DELETE` | `/api/songs/{id}` | Delete song (DB + files) |
| `POST` | `/api/upload` | Upload content archive |
| `POST` | `/api/generate` | Generate chart from audio |
| `GET` | `/api/webdav/status` | Check Nextcloud connection |
| `GET` | `/api/webdav/browse` | Browse Nextcloud directory |
| `POST` | `/api/webdav/download` | Download from Nextcloud |
| `POST` | `/api/webdav/upload` | Upload to Nextcloud |
| `POST` | `/api/webdav/sync-to-nextcloud` | Sync local song to cloud |
| `POST` | `/api/webdav/sync-from-nextcloud` | Import song from cloud |

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
| 7 | Fix data directory permissions |
| 8 | Backup data directory |
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
| 14 Docker containers | 1 Docker container |
| PostgreSQL database | SQLite (embedded) |
| Syncthing file sync | Nextcloud WebDAV |
| Django + FastAPI + Streamlit | Single FastAPI app |
| Nginx reverse proxy | Direct Uvicorn/Gunicorn |
| Redis cache | Not needed |
| Prometheus + Grafana + Datadog | Loguru file logging |
| Multiple Dockerfiles | Single multi-stage Dockerfile |
| 7 Docker images to build | 1 Docker image |

## License

See [LICENSE](LICENSE) for details.