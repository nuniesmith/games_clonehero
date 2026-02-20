# Clone Hero Content Manager — Architecture (v2)

> **v2** simplified the project from a 14-container microservice stack down to a
> **single stateless Docker container**. This document describes the current
> architecture where **all persistent data lives on Nextcloud** via WebDAV.

---

## High-Level Overview

```
┌──────────────────────────────────────────────────────────────┐
│              Stateless Docker Container                       │
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
│  │  │  │                       │  │  Nextcloud upload     │    │
│  │  │  └───────────────────────┘  │                      │    │
│  │  │  ┌───────────────────────┐  │                      │    │
│  │  │  │  song_generator.py    │  │  librosa analysis,   │    │
│  │  │  │                       │  │  notes.chart gen      │    │
│  │  │  └───────────────────────┘  │                      │    │
│  │  └─────────────────────────────┘                      │    │
│  │                                                       │    │
│  │  ┌──────────────┐  ┌──────────────────────────────┐   │    │
│  │  │   SQLite DB   │  │   Nextcloud WebDAV Client    │   │    │
│  │  │ (temp cache)  │  │   (httpx — required)         │   │    │
│  │  └──────┬───────┘  └──────────────┬───────────────┘   │    │
│  │         │  downloaded on startup  │                    │    │
│  │         │  uploaded periodically  │                    │    │
│  │         │  + on shutdown          │                    │    │
│  └─────────┼─────────────────────────┼───────────────────┘    │
│            │                         │                        │
└────────────┼─────────────────────────┼────────────────────────┘
             │                         │  WebDAV (GET/PUT/MKCOL/
             └─────────────────────────┤  PROPFIND/DELETE/MOVE)
                              ┌────────▼────────┐
                              │   Nextcloud      │  ← everything
                              │   WebDAV Server  │    lives here
                              └─────────────────┘
```

### Key Principle: Stateless Container

The Docker container has **no volume mounts**. All persistent state is stored on
Nextcloud:

- **Songs** — uploaded and browsed via WebDAV
- **Assets** — backgrounds, colours, highways uploaded to Nextcloud folders
- **Database** — SQLite file synced to/from Nextcloud at startup/shutdown
- **Logs** — stdout only (captured by Docker logging driver)

Local disk (`/tmp/clonehero/`) is used only for transient staging during archive
extraction, audio analysis, and chart generation. Everything is cleaned up after
processing.

---

## Component Breakdown

### FastAPI Application (`src/main.py`)

The single entry point. Responsibilities:

| Concern | Detail |
|---------|--------|
| **Lifespan** | Downloads DB from Nextcloud, initialises SQLite, starts periodic DB sync, uploads DB on shutdown |
| **Templates** | Jinja2 via `app.state.templates` |
| **Static** | Mounted at `/static` (CSS, JS, images) |
| **Middleware** | Request logging with timing (stdout only) |
| **Routers** | `api_router` (`/api/*`) and `pages_router` (`/*`) |
| **Background tasks** | Periodic DB upload to Nextcloud (configurable interval) |

### Page Routes (`src/routes/pages.py`)

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

### API Routes (`src/routes/api.py`)

JSON endpoints for programmatic access and AJAX calls from the UI.

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/api/health` | Health check (DB status, uptime, version) |
| `GET` | `/api/songs` | List songs (search, limit, offset) |
| `GET` | `/api/songs/{id}` | Get single song |
| `PUT` | `/api/songs/{id}` | Update song metadata (DB + song.ini on Nextcloud) |
| `DELETE` | `/api/songs/{id}` | Delete song (Nextcloud + DB) |
| `POST` | `/api/upload` | Upload archive or asset file (everything → Nextcloud) |
| `POST` | `/api/generate` | Upload audio → generate chart → upload to Nextcloud |
| `POST` | `/api/library/sync` | Scan Nextcloud songs dir → refresh local DB cache |
| `GET` | `/api/library/status` | Library stats + Nextcloud connection info |
| `GET` | `/api/webdav/status` | Nextcloud connection status |
| `GET` | `/api/webdav/browse` | List Nextcloud directory |
| `POST` | `/api/webdav/import` | Import a song folder's metadata into the library |
| `POST` | `/api/webdav/upload` | Upload file to Nextcloud |
| `POST` | `/api/webdav/mkdir` | Create directory on Nextcloud |
| `DELETE` | `/api/webdav/delete` | Delete file/directory on Nextcloud |

### Database (`src/database.py`)

Embedded **SQLite** via `aiosqlite` (async, used in routes) and `sqlite3`
(sync, used in service-layer code). Acts as a **metadata cache** — the
canonical song data lives on Nextcloud.

The database file is stored in a temp directory (`/tmp/clonehero/clonehero.db`)
and synced to Nextcloud. See [Database Lifecycle](#database-lifecycle) below.

```
songs
├── id              INTEGER PRIMARY KEY AUTOINCREMENT
├── title           TEXT NOT NULL
├── artist          TEXT NOT NULL
├── album           TEXT
├── file_path       TEXT DEFAULT ''                      -- legacy, kept for compat
├── remote_path     TEXT UNIQUE NOT NULL                 -- Nextcloud WebDAV path
├── metadata        TEXT DEFAULT '{}'                    -- JSON string
├── synced_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- last Nextcloud sync
├── created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
└── updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP  (trigger)
```

- `remote_path` is the canonical identifier (e.g. `/Songs/Artist/Title_abc123`).
- Indexes on `title`, `artist`, `album` (case-insensitive) and `remote_path`.
- Trigger `update_songs_timestamp` keeps `updated_at` current.
- Metadata stored as a serialised JSON string (use `json.dumps` / `json.loads`).
- Schema migrations run automatically on startup to add `remote_path` and
  `synced_at` columns when upgrading from the local-only schema.

### WebDAV Client (`src/webdav.py`)

Async Nextcloud WebDAV client built on **httpx** with Basic Auth. Also provides
higher-level helpers for recursive directory walking, remote `song.ini` parsing,
and song folder upload/write operations.

| Operation | HTTP Method | WebDAV Verb |
|-----------|-------------|-------------|
| List directory | `PROPFIND` | Depth: 1 |
| Download file | `GET` | — |
| Stream download | `GET` | (chunked) |
| Upload file | `PUT` | — |
| Stream upload | `PUT` | (chunked) |
| Create directory | `MKCOL` | — |
| Delete | `DELETE` | — |
| Move / rename | `MOVE` | — |
| File info | `PROPFIND` | Depth: 0 |

Higher-level helpers:

- `walk_directory` — recursive PROPFIND to list all files/dirs under a path
- `find_song_folders` — walk the songs dir, return folders containing `song.ini`
- `parse_remote_song_ini` — download + parse a remote `song.ini`
- `upload_song_folder` — upload an entire local staging directory to Nextcloud
- `write_remote_song_ini` — write/overwrite a `song.ini` on Nextcloud
- `list_song_folder_files` — list files in a single song folder

### Content Manager (`src/services/content_manager.py`)

All persistent storage lives on Nextcloud. Local disk is only used as a
transient staging area for extraction and processing before upload.

- `parse_song_ini` / `write_song_ini` — read and write Clone Hero `song.ini`
- `extract_archive` — handles `.zip` (zipfile) and `.rar` (rarfile)
- `process_extracted_songs` — walks extracted tree, finds `song.ini`, uploads
  each song folder to Nextcloud, registers in DB with `remote_path`
- `process_upload` — full pipeline: extract → upload to Nextcloud → cleanup temp
- `_upload_asset_to_nextcloud` — uploads non-song files (backgrounds, colours,
  highways) to the appropriate Nextcloud folder
- `sync_library_from_nextcloud` — scan Nextcloud, parse every `song.ini`,
  upsert into DB, purge stale entries
- `delete_song_from_nextcloud` — safe deletion on Nextcloud (under songs root)

### Song Generator (`src/services/song_generator.py`)

- `analyze_audio` — librosa beat tracking + onset detection
- `generate_notes_chart` — creates a Clone Hero `notes.chart` with `[Song]`,
  `[SyncTrack]`, `[Events]`, and note sections
- `process_song_file` — generate chart + song.ini + copy audio to temp staging
- `process_and_upload_song` — async wrapper: generate → upload staging dir to
  Nextcloud → register in DB with `remote_path` → cleanup staging
- Supports Easy / Medium / Hard / Expert difficulties

### Configuration (`src/config.py`)

All settings read from environment variables (`.env` file via `python-dotenv`).

Key groups:
- **Application** — host, port, env, debug, secret key
- **Paths** — `TEMP_DIR`, `DB_PATH` (both in `/tmp`, transient)
- **Logging** — `LOG_LEVEL` (stdout only, no file logging)
- **Nextcloud WebDAV** — URL, credentials, folder paths
- **Nextcloud folder mappings** — `NEXTCLOUD_FOLDERS` dict mapping content types
  to Nextcloud paths (Songs, Backgrounds, Colors, Highways, Database, etc.)
- **Upload limits** — max file size, allowed extensions
- **Song.ini fields** — list of optional metadata fields

---

## Database Lifecycle

The SQLite database is treated as a local cache that is periodically synced
to Nextcloud for persistence across container restarts.

```
Container Start
      │
      ▼
┌─────────────────────────┐
│ Download DB from        │  GET /Database/clonehero.db
│ Nextcloud (if exists)   │  → save to /tmp/clonehero/clonehero.db
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ init_db()               │  Create tables if fresh,
│                         │  run schema migrations
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Start periodic sync     │  asyncio background task
│ task (every N seconds)  │  default: 300s (5 minutes)
└────────────┬────────────┘
             │
             ▼
      [Application running]
             │
         periodic ──► Upload DB to Nextcloud
         timer         PUT .tmp → MOVE to final path
             │
             ▼
┌─────────────────────────┐
│ Graceful shutdown       │
│  1. Cancel periodic     │
│  2. Final DB upload     │  PUT /Database/clonehero.db
│  3. Exit                │
└─────────────────────────┘
```

**Upload strategy:**
1. Upload to `/Database/clonehero.db.tmp` via PUT
2. Atomic rename via WebDAV MOVE to `/Database/clonehero.db`
3. If MOVE fails, fall back to direct PUT overwrite

**Concurrency caveat:** SQLite does not support concurrent writers across
multiple processes. Only run a single instance of this container. For
multi-instance scaling, migrate to PostgreSQL.

---

## Data Flow Diagrams

### Library Sync Flow

```
Browser                    FastAPI                 WebDAV Client          Nextcloud
───────                    ───────                 ─────────────          ─────────
  │  POST /api/              │                         │                     │
  │    library/sync          │                         │                     │
  │ ─────────────────────────►│                         │                     │
  │                          │  find_song_folders()     │                     │
  │                          │  (recursive PROPFIND) ──►│ PROPFIND ──────────►│
  │                          │◄─────────────────────────│◄────────────────────│
  │                          │                          │                     │
  │                          │  for each song.ini:      │                     │
  │                          │    download_file() ──────►│ GET ───────────────►│
  │                          │◄─────────────────────────│◄────────────────────│
  │                          │    parse + upsert DB     │                     │
  │                          │                          │                     │
  │                          │  purge_stale_songs()     │                     │
  │  JSON { synced, purged } │                          │                     │
  │◄──────────────────────────│                         │                     │
```

### Upload Flow (Songs)

```
Browser                    FastAPI                 Service Layer          Nextcloud
───────                    ───────                 ─────────────          ─────────
  │  POST /api/upload        │                         │                     │
  │  (multipart file) ──────►│                         │                     │
  │                          │  save to /tmp           │                     │
  │                          │─────────────────────────►│                     │
  │                          │                         │ extract_archive()   │
  │                          │                         │ parse song.ini      │
  │                          │                         │ upload_song_folder()│
  │                          │                         │ ───────────────────►│
  │                          │                         │ upsert DB          │◄── PUT
  │                          │                         │ cleanup /tmp        │
  │                          │◄─────────────────────────│                     │
  │  JSON response ◄─────────│                         │                     │
  │  { songs: [...] }        │                         │                     │
```

### Upload Flow (Non-Song Assets)

```
Browser                    FastAPI                 Service Layer          Nextcloud
───────                    ───────                 ─────────────          ─────────
  │  POST /api/upload        │                         │                     │
  │  (file, type=backgrounds)│                         │                     │
  │ ─────────────────────────►│                         │                     │
  │                          │  save to /tmp           │                     │
  │                          │─────────────────────────►│                     │
  │                          │                         │ upload_file() ─────►│
  │                          │                         │   PUT /Backgrounds/ │
  │                          │                         │ cleanup /tmp        │
  │                          │◄─────────────────────────│◄────────────────────│
  │  JSON response ◄─────────│                         │                     │
```

### Generate Flow

```
Browser                    FastAPI                 Service Layer          Nextcloud
───────                    ───────                 ─────────────          ─────────
  │  POST /api/generate      │                         │                     │
  │  (audio file) ──────────►│                         │                     │
  │                          │  save to /tmp           │                     │
  │                          │  asyncio.to_thread ─────►│                     │
  │                          │                         │ analyze_audio()     │
  │                          │                         │ generate_chart()    │
  │                          │                         │ write song.ini     │
  │                          │                         │ copy audio → /tmp   │
  │                          │                         │ upload_song_folder()│
  │                          │                         │ ───────────────────►│
  │                          │                         │ upsert DB          │
  │                          │                         │ cleanup /tmp        │
  │                          │◄─────────────────────────│                     │
  │  JSON response ◄─────────│                         │                     │
  │  { tempo, notes, id }    │                         │                     │
```

### Edit Song Metadata Flow

```
Browser                    FastAPI                 WebDAV Client          Nextcloud
───────                    ───────                 ─────────────          ─────────
  │  PUT /api/songs/{id}     │                         │                     │
  │  (JSON body) ────────────►│                         │                     │
  │                          │  update_song() (DB)     │                     │
  │                          │  write_remote_          │                     │
  │                          │    song_ini() ──────────►│ PUT song.ini ──────►│
  │                          │◄─────────────────────────│◄────────────────────│
  │  JSON response ◄──────────│                         │                     │
```

---

## Nextcloud Folder Layout

All persistent data lives under the WebDAV root (default: `CloneHero/`):

```
CloneHero/                            (NEXTCLOUD_REMOTE_PATH)
├── Songs/                            Song folders
│   └── <Artist>/
│       └── <Title>_<uuid>/
│           ├── song.ini
│           ├── notes.chart
│           ├── song.ogg
│           └── album.png
├── Backgrounds/                      Custom background images
├── Colors/                           Custom colour themes
├── Highways/                         Custom highway textures
├── Database/                         Managed automatically
│   └── clonehero.db                  SQLite database
├── Generator/                        Generated chart staging
└── Temp/                             Temporary processing files
```

---

## Project Source Layout

```
src/
├── main.py                           # App factory, lifespan, DB sync, logging
├── config.py                         # Environment-based configuration
├── database.py                       # SQLite schema + async/sync CRUD
├── webdav.py                         # Nextcloud WebDAV client (httpx)
├── routes/
│   ├── pages.py                      # HTML page routes (Jinja2)
│   └── api.py                        # JSON API routes (/api/*)
├── services/
│   ├── content_manager.py            # Song parsing, extraction, Nextcloud upload
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

No `DATA_DIR` or `LOG_DIR` environment variables — the container is stateless.

### docker-compose.yml

Single service (`app`):

- Builds from `docker/Dockerfile`
- Reads `.env` for configuration
- **No volume mounts** — fully stateless
- Health check on `/api/health` every 30 s
- Exposes `$APP_PORT` (default 8000)
- `restart: unless-stopped`

### Entrypoint (`docker/entrypoint.sh`)

Minimal script that drops privileges from root to `appuser` before executing
the main process. No directory creation or permission fixing needed since there
are no volume mounts.

### Startup Sequence

```
Gunicorn (2 workers)
  └── UvicornWorker
        └── FastAPI app lifespan
              ├── ensure_directories()    → create /tmp/clonehero/
              ├── download DB from Nextcloud (if available)
              ├── init_db()               → SQLite schema + migrations
              ├── start periodic DB sync  → asyncio background task
              ├── mount /static
              ├── register api_router     (/api/*)
              └── register pages_router   (/* — last)
```

### Shutdown Sequence

```
SIGTERM received
  └── FastAPI lifespan teardown
        ├── cancel periodic DB sync task
        ├── upload DB to Nextcloud (final save)
        └── log shutdown
```

---

## Logging

All logging goes to **stdout only** via loguru. No file-based logging.

- Console format includes timestamp, level, module, function, and line number
- Colourised output in development, structured in production
- Request middleware logs method, path, status, and duration
- Static file requests (`/static/*`) are suppressed at INFO level
- Log level controlled by `LOG_LEVEL` env var (default: `DEBUG`)
- Docker captures stdout via its logging driver (`docker logs`, `docker compose logs`)

---

## v1 → v2 Migration Summary

| Concern | v1 (14 containers) | v2 (1 stateless container) |
|---------|---------------------|----------------------------|
| Web server | Nginx → Django + FastAPI + Streamlit | FastAPI + Jinja2 |
| Database | PostgreSQL (separate container) | SQLite (embedded, synced to Nextcloud) |
| File sync | Syncthing (separate container) | Nextcloud WebDAV (client only) |
| Caching | Redis | Not needed |
| Monitoring | Prometheus + Grafana + Datadog + exporters | Loguru stdout logging |
| Reverse proxy | Nginx | Direct Gunicorn/Uvicorn |
| Build artifacts | 7+ Docker images | 1 Docker image |
| Compose services | 14 | 1 |
| Volume mounts | Multiple data volumes | None — fully stateless |
| Song storage | Local filesystem | Nextcloud WebDAV |
| Asset storage | Local filesystem | Nextcloud WebDAV |
| DB persistence | PostgreSQL container volume | SQLite synced to Nextcloud |
| Logging | File-based (rotated) | stdout only |

---

## Key Design Decisions

1. **Stateless container** — No volume mounts. The entire container can be
   destroyed and recreated without data loss because everything is on Nextcloud.
   This simplifies deployment, scaling, and container orchestration.

2. **SQLite over PostgreSQL** — Removes an entire container and external
   dependency. Perfectly adequate for single-user / small-team song management.
   The DB is synced to Nextcloud for persistence.

3. **Nextcloud WebDAV for everything** — Songs, assets, and the database all
   live on Nextcloud. Single source of truth with explicit sync rather than
   automatic bidirectional replication.

4. **Periodic DB sync** — The database is uploaded to Nextcloud every 5 minutes
   (configurable) and on shutdown. This balances durability with performance.
   An atomic upload strategy (PUT to `.tmp` then MOVE) prevents corruption.

5. **stdout-only logging** — No file logging means no need for log directories
   or rotation. Docker's logging driver handles capture, rotation, and shipping.

6. **Unified FastAPI app** — One process serves HTML pages, REST API, and static
   files. Eliminates inter-service HTTP calls.

7. **Async + sync helpers** — Routes use `aiosqlite`; CPU-bound service code
   (librosa, archive extraction) uses sync helpers wrapped in
   `asyncio.to_thread()`.

8. **No build step for frontend** — Plain HTML/CSS/JS with Jinja2 templating.
   No bundler, no transpiler, no node_modules.

9. **Single instance only** — SQLite does not support concurrent writers. If
   multi-instance scaling is needed, migrate to PostgreSQL with a network
   database rather than a file synced via WebDAV.