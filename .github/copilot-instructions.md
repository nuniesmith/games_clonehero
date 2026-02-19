# Clone Hero Dockerized Content Manager - AI Coding Agent Instructions

## Architecture Overview

This is a **multi-service dockerized Clone Hero content management system** with:
- **Frontend**: Django web application (`src/frontend_django/`) at port 8501 with Gunicorn/WhiteNoise
- **API**: FastAPI backend (`src/api/`) at port 8000, using Gunicorn with Uvicorn workers
- **Backend Worker**: Health monitoring service (`src/backend/worker.py`) at port 8001
- **Database**: PostgreSQL with connection pooling via `psycopg2` (not async)
- **Content Storage**: File-based in `/app/data/clonehero_content/` with metadata in PostgreSQL
- **Multiplayer Server**: Clone Hero standalone server at port 14242
- **Sync**: Syncthing for cross-device file synchronization
- **Monitoring Stack**: Prometheus, Grafana, Datadog with exporters for Redis, Postgres, Nginx

**Key Insight**: Services communicate internally via Docker network. Frontend calls API endpoints directly. All services depend on health checks before starting.

## Critical Development Workflows

### Local Development
```bash
# Use the utility script for all Docker operations
./utils.sh  # Interactive menu-driven tool

# Or for non-interactive CI/CD:
./utils.sh -y  # Auto-confirm all prompts

# Manual Docker Compose operations:
docker compose up -d --build           # Start all services
docker compose down --remove-orphans   # Stop and clean up
docker compose logs -f <service>       # View logs for specific service
```

### Building & Deploying
```bash
# Build and push all images to Docker Hub (nuniesmith/clonehero):
./utils.sh  # Select option [2] "Build & Push Docker Images"

# Images are tagged as: nuniesmith/clonehero:<service>
# Where service: api, backend, frontend, nginx, postgres, server, sync
```

### Database Management
```bash
# Schema is initialized automatically on first API startup
# See src/sql/schema.sql for the schema definition

# To reinitialize DB manually:
docker compose exec api python -c "from src.database import init_db; init_db()"

# Direct DB access:
docker compose exec db psql -U clonehero -d clonehero
```

## Code Patterns & Conventions

### Database Access Pattern
**Always use context manager** - never manual connection handling:
```python
from src.database import get_connection
from psycopg2.extras import DictCursor

with get_connection() as conn:
    with conn.cursor(cursor_factory=DictCursor) as cursor:
        cursor.execute("SELECT * FROM songs WHERE id = %s", (song_id,))
        result = cursor.fetchone()
    conn.commit()  # Only for write operations
```

**Important**: This codebase uses `psycopg2` (synchronous), NOT `asyncpg`. Wrap DB calls with `asyncio.to_thread()` in async contexts.

### Content Storage Pattern
Content is stored in structured directories under `CONTENT_BASE_DIR` (`/app/data/clonehero_content/`):
```python
from src.services.content_utils import get_final_directory

# Always use get_final_directory() - it ensures directory exists
songs_dir = get_final_directory("songs")  # Returns Path object
# Valid types: songs, backgrounds, colors, highways, generator, temp
```

Songs require `song.ini` files with `[song]` section containing `name`, `artist`, `album` (required) plus optional metadata fields defined in `OPTIONAL_FIELDS` list in `content_manager.py`.

### Logging Pattern
**Use Loguru everywhere**:
```python
from loguru import logger

logger.info("✅ Success message with emoji")
logger.warning("⚠️ Warning with context")
logger.error("❌ Error occurred")
logger.exception("❌ Exception with full traceback")
```

Logs are configured in each service's main file with rotation and retention settings from environment variables.

### Async File Operations
Use `aiofiles` for file I/O in FastAPI routes:
```python
import aiofiles

async with aiofiles.open(file_path, "wb") as f:
    await f.write(content)
```

### Archive Extraction
Supports both ZIP and RAR formats via `content_utils.extract_archive()`. RAR requires `unrar` binary installed in Docker image.

## Service-Specific Patterns

### FastAPI (src/api/)
- **Startup**: Uses `lifespan` context manager with exponential backoff DB connection retry
- **Middleware**: Logs all requests with duration and status code
- **Routes**: Organized in `src/routes/` by feature, imported in `main.py`
- **Response Format**: Always return `Dict[str, Any]` from endpoints
- **File Uploads**: Validate extension and size (10GB default), use temp files, cleanup in `finally` blocks

### Django Frontend (src/frontend_django/)
- **Architecture**: MVC with Models, Views, Templates
- **Models**: `Song` model maps to existing `songs` table (managed=False, no migrations)
- **Views**: Located in `content/views.py`, handle all page logic and API communication
- **Templates**: HTML/Jinja2 in `templates/` with base template containing sidebar navigation
- **Static Files**: CSS/JS in `static/`, served via WhiteNoise with compression
- **AJAX**: Upload, delete, search operations use fetch API with CSRF tokens
- **URLs**: RESTful patterns in `content/urls.py`
- **Server**: Runs with Gunicorn (2 workers) on port 8501
- **Admin**: Django admin available at `/admin/` for database browsing

### Backend Worker (src/backend/worker.py)
- **Purpose**: Health check monitoring (polls API `/health` endpoint)
- **Shutdown**: Gracefully handles SIGTERM/SIGINT signals
- **Retry Logic**: Exponential backoff with cap at 60s

## Docker & Deployment

### Multi-Stage Dockerfile Pattern
All services use standardized multi-stage builds (see `docker/docker_schema.txt`):
1. Build stage: Install all dependencies
2. Runtime stage: Copy only essentials, run as non-root user
3. Health checks defined per service
4. PYTHONPATH set to `/app`

### Environment Variables
Expected in `.env` file (not committed to repo):
- Database: `DB_HOST`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_PORT`, `DB_URL`
- Service ports: `API_PORT`, `FRONTEND_PORT`, `BACKEND_PORT`, `NGINX_PORT`, `CH_SERVER_PORT`
- Content: `CONTENT_BASE_DIR`, `MAX_FILE_SIZE_GB`
- Logging: `LOG_DIR`, `LOG_LEVEL`, `LOG_FILE_SIZE`, `LOG_RETENTION`
- Redis: `REDIS_PASSWORD`

### Health Check Strategy
All services have health checks. Dependent services use `depends_on: condition: service_healthy` in docker-compose.yml.

## Testing & Debugging

### Viewing Logs
```bash
docker compose logs -f              # All services
docker compose logs -f api          # Specific service
docker exec clonehero_api cat /var/log/api/app.log  # Log files
```

### Testing API Endpoints
```bash
# API docs available at: http://localhost:8000/docs
curl http://localhost:8000/health
```

### Common Issues
1. **DB connection failures**: Check if `clonehero_db` container is healthy before API starts
2. **File permission errors**: Use `utils.sh` option [4] to fix directory permissions
3. **Import errors**: Ensure `PYTHONPATH=/app` is set and using absolute imports from `src`

## Key Files Reference

- **Database schema**: `src/sql/schema.sql` - single `songs` table with JSONB metadata
- **DB connection pool**: `src/database.py` - creates pool on import, retry logic on failure
- **Content processing**: `src/services/content_manager.py` - song.ini parsing, DB insertion
- **File extraction**: `src/services/content_utils.py` - archive handling, directory management
- **API routes**: `src/routes/*.py` - feature-organized endpoints
- **Docker orchestration**: `docker-compose.yml` - 13 services with dependencies
- **Utility script**: `utils.sh` - manages Docker operations, builds, system updates

## When Adding Features

1. **New API endpoint**: Create route in `src/routes/`, register in `src/api/main.py`
2. **New Django page**: 
   - Add view function in `content/views.py`
   - Create URL pattern in `content/urls.py`
   - Create HTML template in `templates/`
   - Add navigation link in `templates/base.html` sidebar
3. **New Streamlit page** (legacy): Add to `src/pages/`, import and register in `src/frontend/sidebar.py`
4. **New content type**: Update `CONTENT_FOLDERS` in `content_utils.py` and add to `ALLOWED_CONTENT_TYPES`
5. **DB schema changes**: Modify `src/sql/schema.sql`, rebuild API container or manually run schema
6. **New service**: Add to `docker-compose.yml` with health check, add Dockerfile to `docker/`, update `services` dict in `utils.sh`
7. **Static file changes**: Update CSS/JS in `src/frontend_django/static/`, run `collectstatic` or rebuild container
