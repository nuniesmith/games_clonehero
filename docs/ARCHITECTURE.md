# Clone Hero Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DOCKER COMPOSE SERVICES                          │
└─────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────┐
                              │    NGINX     │
                              │   Port 80    │
                              └──────┬───────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
         ┌──────────▼──────┐  ┌─────▼─────┐  ┌──────▼──────┐
         │   FRONTEND      │  │    API    │  │   BACKEND   │
         │   (Django)      │  │ (FastAPI) │  │   (Worker)  │
         │   Port 8501     │  │ Port 8000 │  │  Port 8001  │
         └─────────┬───────┘  └─────┬─────┘  └──────┬──────┘
                   │                │                │
                   │    ┌───────────┼────────────────┘
                   │    │           │
         ┌─────────▼────▼───────────▼─────────┐
         │       PostgreSQL Database          │
         │          Port 5432                 │
         │  Table: songs (id, title, artist,  │
         │         album, file_path, metadata)│
         └────────────────────────────────────┘

         ┌─────────────────────────────────────┐
         │      File Storage (Volumes)         │
         │  /app/data/clonehero_content/       │
         │    ├── songs/                       │
         │    ├── backgrounds/                 │
         │    ├── colors/                      │
         │    ├── highways/                    │
         │    └── generator/                   │
         └────────────┬────────────────────────┘
                      │
         ┌────────────▼────────────┐
         │      SYNCTHING          │
         │      Port 8384          │
         │  (Cross-device sync)    │
         └─────────────────────────┘

         ┌─────────────────────────┐
         │   Clone Hero Server     │
         │      Port 14242         │
         │   (Multiplayer)         │
         └─────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         MONITORING STACK                                 │
└─────────────────────────────────────────────────────────────────────────┘

    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
    │Prometheus│    │ Grafana  │    │ Datadog  │    │  Redis   │
    │Port 9090 │    │Port 3000 │    │Port 8126 │    │Port 6379 │
    └──────────┘    └──────────┘    └──────────┘    └──────────┘
         │                │                │
         └────────────────┼────────────────┘
                          │
    ┌─────────────────────┴─────────────────────┐
    │             Exporters                     │
    │  Redis (9121), Postgres (9187),          │
    │  Nginx (9113)                             │
    └───────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│                      DJANGO FRONTEND ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────┐
    │                Browser (http://localhost:8501)          │
    └─────────────────────────┬───────────────────────────────┘
                              │
    ┌─────────────────────────▼───────────────────────────────┐
    │                  Gunicorn WSGI Server                   │
    │                   (2 workers)                           │
    └─────────────────────────┬───────────────────────────────┘
                              │
    ┌─────────────────────────▼───────────────────────────────┐
    │                   Django Application                    │
    │  ┌──────────────────────────────────────────────────┐   │
    │  │                    URLs                          │   │
    │  │  / → home                                        │   │
    │  │  /database/ → database_explorer                  │   │
    │  │  /songs/upload/ → songs_upload                   │   │
    │  │  /generator/ → song_generator                    │   │
    │  │  /colors/ → colors                               │   │
    │  │  /backgrounds/ → backgrounds                     │   │
    │  │  /highways/ → highways                           │   │
    │  │  /admin/ → Django Admin                          │   │
    │  └──────────────────────┬───────────────────────────┘   │
    │                         │                               │
    │  ┌──────────────────────▼───────────────────────────┐   │
    │  │                    Views                         │   │
    │  │  - API communication (requests)                  │   │
    │  │  - File upload handling                          │   │
    │  │  - Search and pagination                         │   │
    │  │  - Context preparation                           │   │
    │  └──────────────────────┬───────────────────────────┘   │
    │                         │                               │
    │  ┌──────────────────────▼───────────────────────────┐   │
    │  │                  Models                          │   │
    │  │  Song (id, title, artist, album, file_path,     │   │
    │  │        metadata, created_at, updated_at)        │   │
    │  │  - Maps to existing 'songs' table               │   │
    │  │  - managed=False (no migrations)                │   │
    │  └──────────────────────┬───────────────────────────┘   │
    │                         │                               │
    │  ┌──────────────────────▼───────────────────────────┐   │
    │  │                 Templates                        │   │
    │  │  base.html (sidebar navigation)                  │   │
    │  │  home.html, database_explorer.html,              │   │
    │  │  songs/upload.html, backgrounds.html,            │   │
    │  │  colors.html, highways.html,                     │   │
    │  │  song_generator.html                             │   │
    │  └──────────────────────┬───────────────────────────┘   │
    │                         │                               │
    │  ┌──────────────────────▼───────────────────────────┐   │
    │  │              Static Files                        │   │
    │  │  CSS: style.css (500+ lines)                     │   │
    │  │  JS: main.js, tabs.js, upload.js                 │   │
    │  │  Assets: ch_icon.png                             │   │
    │  │  Served via WhiteNoise (compressed)              │   │
    │  └──────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW                                      │
└─────────────────────────────────────────────────────────────────────────┘

User Upload Flow:
─────────────────
Browser → Django View → FastAPI → Extract Archive → Parse song.ini
                                                           │
                                                           ▼
                                        Insert to PostgreSQL (songs table)
                                                           │
                                                           ▼
                                        Move files to /app/data/clonehero_content/songs/
                                                           │
                                                           ▼
                                               Syncthing syncs to other devices

User Search Flow:
─────────────────
Browser → Django View → PostgreSQL Query → Django Template → Browser
                             │
                             ▼
                    API call to /content/ or direct ORM query


File Structure Flow:
────────────────────
/app/data/clonehero_content/
    songs/
        <artist>/
            <title>_<uuid>/
                song.ini
                notes.chart
                song.ogg
                album.jpg
                ...

Database Schema:
────────────────
songs
├── id (SERIAL PRIMARY KEY)
├── title (TEXT NOT NULL)
├── artist (TEXT NOT NULL)
├── album (TEXT)
├── file_path (TEXT UNIQUE NOT NULL)
├── metadata (JSONB DEFAULT '{}')
├── created_at (TIMESTAMP)
└── updated_at (TIMESTAMP)
```

## Key Architectural Decisions

1. **Django Frontend**: Replaced Streamlit for production performance
2. **Read-Only Models**: Django doesn't manage DB schema (managed=False)
3. **API Proxy**: Frontend forwards file operations to FastAPI backend
4. **Static Files**: WhiteNoise serves compressed CSS/JS without nginx
5. **AJAX Operations**: No page reloads for uploads/deletes
6. **Responsive Design**: Mobile-first CSS with sidebar navigation
7. **Docker Native**: All services containerized with health checks
8. **File Storage**: Shared volume between services
9. **Database**: Single source of truth for metadata
10. **Monitoring**: Full observability with Prometheus/Grafana

## Service Communication

- **Frontend ↔ API**: HTTP requests via `requests` library
- **API ↔ Database**: `psycopg2` connection pool (synchronous)
- **API ↔ File Storage**: Direct filesystem access via mounted volume
- **Frontend ↔ Browser**: HTML/CSS/JS with AJAX
- **Backend ↔ API**: Health check polling every 30s
- **Syncthing ↔ File Storage**: Watches for file changes

## Port Reference

| Service | Internal Port | External Port | Protocol |
|---------|--------------|---------------|----------|
| NGINX | 80 | ${NGINX_PORT} | HTTP |
| Frontend | 8501 | 8501 | HTTP |
| API | 8000 | 8000 | HTTP |
| Backend | 8001 | 8001 | HTTP |
| Database | 5432 | ${DB_PORT} | PostgreSQL |
| CH Server | 14242 | 14242 | TCP |
| Syncthing | 8384 | ${SYNC_PORT} | HTTP |
| Prometheus | 9090 | 9090 | HTTP |
| Grafana | 3000 | 3000 | HTTP |
| Redis | 6379 | 6379 | Redis |
