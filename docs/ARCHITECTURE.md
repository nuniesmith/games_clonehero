# Clone Hero Content Manager — Architecture (v2)

> **v2** simplified the project from a 14-container microservice stack down to a
> **single Docker container**. This document describes the current architecture.

---

## High-Level Overview

```
┌──────────────────────────────────────────────────────────────┐
│                   Single Docker Container                    │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │               Gunicorn + Uvicorn Workers              │    │
│  │                    (ASGI server)                       │    │
│  └──────────────────────┬───────────────────────────────┘    │
│                         │                                    │
│  ┌──────────────────────▼───────────────────────────────┐    │
│  │                 FastAPI Application                   │    │
│  │                                                       │    │
│  │  ┌─────────────┐ ┌─────────────┐ ┌───────────────┐   │    │
│  │  │ Page Routes  │ │  API Routes │ │ Static Files  │   │    │
│  │  │ (Jinja2 HTML)│ │ (/api/*)    │ │ (/static/*)   │   │    │
│  │  └──────┬──────┘ └──────┬──────┘ └───────────────┘   │    │
│  │         │               │                             │    │
│  │  ┌──────▼───────────────▼──────┐                      │    │
│  │  │         Services            │                      │    │
│  │  │  ┌───────────────────────┐  │                      │    │
│  │  │  │  content_manager.py   │  │  song.ini parsing,   │    │
│  │  │  │                       │  │  archive extraction,  │    │
│  │  │  │                       │  │  file organisation    │    │
│  │  │  └───────────────────────┘  │                      │    │
│  │  │  ┌───────────────────────┐  │                      │    │
│  │  │  │  song_generator.py    │  │  librosa analysis,   │    │
│  │  │  │                       │  │  notes.chart gen      │    │
│  │  │  └───────────────────────┘  │                      │    │
│  │  └─────────────────────────────┘                      │    │
│  │                                                       │    │
│  │  ┌──────────────┐  ┌──────────────────────────────┐   │    │
│  │  │   SQLite DB   │  │   Nextcloud WebDAV Client    │   │    │
│  │  │  (aiosqlite)  │  │   (httpx — optional)         │   │    │
│  │  └──────────────┘  └──────────────┬───────────────┘   │    │
│  └───────────────────────────────────┼───────────────────┘    │
│                                      │                        │
└──────────────────────────────────────┼────────────────────────┘
                                       │
                              ┌────────▼────────┐
                              │   Nextcloud      │  (optional,
                              │   WebDAV Server  │   remote)
                              └─────────────────┘
```

---

## Component Breakdown

### FastAPI Application (`src/app/main.py`)

The single entry point. Responsibilities:

| Concern | Detail |
|---------|--------|
| **Lifespan** | Creates data directories, initialises SQLite, configures loguru |
| **Templates** | Jinja2 via `app.state.templates` |
| **Static** | Mounted at `/static` (CSS, JS, images) |
| **Middleware** | Request logging with timing |
| **Routers** | `api_router` (`/api/*`) and `pages_router` (`/*`) |

### Page Routes (`src/app/routes/pages.py`)

HTML views served with Jinja2. Every route receives `request` and builds a
context dict (always including `page_title` so the sidebar highlights correctly).

| Route | Template | Purpose |
|-------|----------|---------|
| `GET /` | `home.html` | Dashboard with stats and feature cards |
| `GET /songs` | `songs.html` | Paginated song library with search |
| `GET /songs/{id}/edit` | `editor.html` | Edit all song.ini metadata fields |
| `GET /upload` | `upload.html` | Drag-and-drop content upload |
| `GET /generator` | `generator.html` | Audio → chart generation UI |
| `GET /browser` | `browser.html` | Nextcloud WebDAV file browser |

### API Routes (`src/app/routes/api.py`)

JSON endpoints for programmatic access and AJAX calls from the UI.

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/api/health` | Health check (DB status, uptime, version) |
| `GET` | `/api/songs` | List songs (search, limit, offset) |
| `GET` | `/api/songs/{id}` | Get single song |
| `PUT` | `/api/songs/{id}` | Update song metadata (DB + song.ini) |
| `DELETE` | `/api/songs/{id}` | Delete song (DB + disk files) |
| `POST` | `/api/upload` | Upload archive or asset file |
| `POST` | `/api/generate` | Upload audio → generate chart |
| `GET` | `/api/scan` | Scan local content dir for song.ini files |
| `GET` | `/api/webdav/status` | Nextcloud connection status |
| `GET` | `/api/webdav/browse` | List Nextcloud directory |
| `POST` | `/api/webdav/download` | Download file from Nextcloud and import |
| `POST` | `/api/webdav/upload` | Upload file to Nextcloud |
| `POST` | `/api/webdav/mkdir` | Create directory on Nextcloud |
| `DELETE` | `/api/webdav/delete` | Delete file/directory on Nextcloud |
| `POST` | `/api/webdav/sync-to-nextcloud` | Push local song → Nextcloud |
| `POST` | `/api/webdav/sync-from-nextcloud` | Pull song ← Nextcloud |

### Database (`src/app/database.py`)

Embedded **SQLite** via `aiosqlite` (async, used in routes) and `sqlite3`
(sync, used in service-layer code).

```
songs
├── id              INTEGER PRIMARY KEY AUTOINCREMENT
├── title           TEXT NOT NULL
├── artist          TEXT NOT NULL
├── album           TEXT
├── file_path       TEXT UNIQUE NOT NULL
├── metadata        TEXT DEFAULT '{}'        -- JSON string
├── created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
└── updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP  (trigger)
```

- Indexes on `title`, `artist`, `album` (case-insensitive).
- Trigger `update_songs_timestamp` keeps `updated_at` current.
- Metadata stored as a serialised JSON string (use `json.dumps` / `json.loads`).

### WebDAV Client (`src/app/webdav.py`)

Async Nextcloud WebDAV client built on **httpx** with Basic Auth.

| Operation | HTTP Method | WebDAV Verb |
|-----------|-------------|-------------|
| List directory | `PROPFIND` | Depth: 1 |
| Download file | `GET` | — |
| Upload file | `PUT` | — |
| Create directory | `MKCOL` | — |
| Delete | `DELETE` | — |
| Move / rename | `MOVE` | — |
| File info | `PROPFIND` | Depth: 0 |

### Content Manager (`src/app/services/content_manager.py`)

- `parse_song_ini` / `write_song_ini` — read and write Clone Hero `song.ini`
- `extract_archive` — handles `.zip` (zipfile) and `.rar` (rarfile)
- `process_extracted_songs` — walks extracted tree, finds `song.ini`, moves to
  `songs/<Artist>/<Title>_<uuid>/`, registers in DB
- `process_upload` — full pipeline: extract → process → cleanup
- `scan_local_songs` — discovery scan (no DB insert)
- `delete_song_files` — safe deletion within `CONTENT_DIR`

### Song Generator (`src/app/services/song_generator.py`)

- `analyze_audio` — librosa beat tracking + onset detection
- `generate_notes_chart` — creates a Clone Hero `notes.chart` with `[Song]`,
  `[SyncTrack]`, `[Events]`, and note sections
- `process_song_file` — full pipeline: analyse → chart → song.ini → copy audio
  → register in DB
- Supports Easy / Medium / Hard / Expert difficulties

### Configuration (`src/app/config.py`)

All settings read from environment variables (`.env` file via `python-dotenv`).
Key groups: application, paths, logging, Nextcloud WebDAV, upload limits,
song.ini optional fields.

---

## Data Flow Diagrams

### Upload Flow

```
Browser                    FastAPI                 Service Layer
───────                    ───────                 ─────────────
  │  POST /api/upload        │                         │
  │  (multipart file) ──────►│                         │
  │                          │  save to temp file      │
  │                          │─────────────────────────►│
  │                          │                         │ extract_archive()
  │                          │                         │ process_extracted_songs()
  │                          │                         │   ├─ parse song.ini
  │                          │                         │   ├─ move to songs/<Artist>/<Title>/
  │                          │                         │   └─ insert_song_sync() → SQLite
  │                          │◄─────────────────────────│
  │  JSON response ◄─────────│                         │
  │  { songs: [...] }        │                         │
```

### Generate Flow

```
Browser                    FastAPI                 Service Layer
───────                    ───────                 ─────────────
  │  POST /api/generate      │                         │
  │  (audio file) ──────────►│                         │
  │                          │  save to temp file      │
  │                          │  asyncio.to_thread ─────►│
  │                          │                         │ analyze_audio() — librosa
  │                          │                         │ generate_notes_chart()
  │                          │                         │ write_song_ini()
  │                          │                         │ copy audio → song folder
  │                          │                         │ insert_song_sync() → SQLite
  │                          │◄─────────────────────────│
  │  JSON response ◄─────────│                         │
  │  { tempo, notes, id }    │                         │
```

### Nextcloud Sync-to-Cloud Flow

```
Browser                    FastAPI                 WebDAV Client
───────                    ───────                 ─────────────
  │  POST /api/webdav/       │                         │
  │    sync-to-nextcloud     │                         │
  │  (song_id) ─────────────►│                         │
  │                          │  get_song_by_id()       │
  │                          │  read local files       │
  │                          │  mkdir(remote) ─────────►│ MKCOL
  │                          │  upload_file() ─────────►│ PUT (for each file)
  │                          │◄─────────────────────────│
  │  JSON response ◄─────────│                         │
  │  { uploaded: [...] }     │                         │
```

---

## File System Layout

```
data/                                 # Persistent volume (mounted at /app/data)
├── clonehero.db                      # SQLite database
├── clonehero_content/
│   ├── songs/                        # Imported / uploaded songs
│   │   └── <Artist>/
│   │       └── <Title>_<uuid>/
│   │           ├── song.ini
│   │           ├── notes.chart
│   │           ├── song.ogg
│   │           └── album.png
│   ├── backgrounds/                  # Background images
│   ├── colors/                       # Color themes
│   ├── highways/                     # Highway textures
│   ├── generator/                    # Generated chart outputs
│   └── temp/                         # Temporary processing files
└── logs/
    └── app.log                       # Rotated application log
```

---

## Project Source Layout

```
src/app/
├── main.py                           # App factory, lifespan, middleware
├── config.py                         # Environment-based configuration
├── database.py                       # SQLite schema + async/sync CRUD
├── webdav.py                         # Nextcloud WebDAV client (httpx)
├── routes/
│   ├── pages.py                      # HTML page routes (Jinja2)
│   └── api.py                        # JSON API routes (/api/*)
├── services/
│   ├── content_manager.py            # Song parsing, extraction, storage
│   └── song_generator.py             # Audio analysis, chart generation
├── templates/
│   ├── base.html                     # Base layout with sidebar
│   ├── home.html                     # Dashboard
│   ├── songs.html                    # Song library browser
│   ├── editor.html                   # Song metadata editor
│   ├── upload.html                   # Content upload (drag & drop)
│   ├── generator.html                # Chart generator
│   └── browser.html                  # Nextcloud file browser
└── static/
    ├── css/style.css                 # Application stylesheet
    ├── js/main.js                    # Client-side utilities
    └── assets/ch_icon.png            # Application icon
```

---

## Docker Architecture

### Multi-Stage Dockerfile (`docker/Dockerfile`)

| Stage | Base | Purpose |
|-------|------|---------|
| **builder** | `python:3.13-slim` | Install system build deps + pip packages into `/opt/venv` |
| **runtime** | `python:3.13-slim` | Copy venv, app source; run as `appuser` (non-root) |

Runtime system packages: `libsndfile1`, `unrar-free`, `p7zip-full`, `curl`.

### docker-compose.yml

Single service (`app`):

- Builds from `docker/Dockerfile`
- Reads `.env` for configuration
- Mounts `./data` → `/app/data` for persistence
- Health check on `/api/health` every 30 s
- Exposes `$APP_PORT` (default 8000)

### Startup

```
Gunicorn (2 workers)
  └── UvicornWorker
        └── FastAPI app
              ├── ensure_directories()
              ├── init_db()           → SQLite schema
              ├── mount /static
              ├── register api_router  (/api/*)
              └── register pages_router (/* — last)
```

---

## v1 → v2 Migration Summary

| Concern | v1 (14 containers) | v2 (1 container) |
|---------|---------------------|-------------------|
| Web server | Nginx → Django + FastAPI + Streamlit | FastAPI + Jinja2 |
| Database | PostgreSQL (separate container) | SQLite (embedded) |
| File sync | Syncthing (separate container) | Nextcloud WebDAV (client only) |
| Caching | Redis | Not needed |
| Monitoring | Prometheus + Grafana + Datadog + exporters | Loguru file logging |
| Reverse proxy | Nginx | Direct Gunicorn/Uvicorn |
| Build artifacts | 7+ Docker images | 1 Docker image |
| Compose services | 14 | 1 |

---

## Key Design Decisions

1. **SQLite over PostgreSQL** — Removes an entire container and external
   dependency. Perfectly adequate for single-user / small-team song management.
2. **Nextcloud WebDAV over Syncthing** — Centralised cloud storage with explicit
   sync rather than automatic bidirectional replication.
3. **Unified FastAPI app** — One process serves HTML pages, REST API, and static
   files. Eliminates inter-service HTTP calls.
4. **Async + sync helpers** — Routes use `aiosqlite`; CPU-bound service code
   (librosa, archive extraction) uses sync helpers wrapped in
   `asyncio.to_thread()`.
5. **No build step for frontend** — Plain HTML/CSS/JS with Jinja2 templating.
   No bundler, no transpiler, no node_modules.