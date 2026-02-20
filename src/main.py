"""
Clone Hero Content Manager - Main Application

Single-container FastAPI application that serves:
- HTML pages via Jinja2 templates
- Static files (CSS, JS, images)
- REST API endpoints for songs CRUD, upload, generation, and WebDAV
- Health check endpoint

The application is fully stateless: the SQLite database is downloaded from
Nextcloud on startup and uploaded back periodically and on shutdown.  All
persistent data (songs, assets, DB) lives on Nextcloud via WebDAV.
"""

import asyncio
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger

from src.config import (
    APP_ENV,
    APP_HOST,
    APP_PORT,
    APP_VERSION,
    DB_PATH,
    DB_SYNC_INTERVAL,
    DEBUG,
    LOG_LEVEL,
    NEXTCLOUD_DB_PATH,
    STATIC_DIR,
    TEMPLATES_DIR,
    ensure_directories,
)
from src.database import init_db
from src.routes.api import router as api_router
from src.routes.pages import router as pages_router

# ---------------------------------------------------------------------------
# Logging setup — stdout only (no file logging for stateless containers)
# ---------------------------------------------------------------------------
logger.remove()

logger.add(
    sys.stdout,
    level="DEBUG" if DEBUG else LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
)

# ---------------------------------------------------------------------------
# Database sync helpers (Nextcloud ↔ local temp file)
# ---------------------------------------------------------------------------
_db_sync_task: asyncio.Task | None = None


async def _download_db_from_nextcloud() -> bool:
    """Download the SQLite database from Nextcloud into the local temp path.

    Returns True if a database was downloaded, False if not found or on error
    (in which case a fresh DB will be created by init_db).
    """
    from src.webdav import download_file_stream, is_configured

    if not is_configured():
        logger.info("☁️  Nextcloud not configured — using fresh local database")
        return False

    logger.info(
        "⬇️  Attempting to download database from Nextcloud ({}) …", NEXTCLOUD_DB_PATH
    )

    try:
        ok = await download_file_stream(NEXTCLOUD_DB_PATH, str(DB_PATH))
        if ok:
            logger.success(
                "✅ Database downloaded from Nextcloud ({} bytes)",
                DB_PATH.stat().st_size,
            )
            return True
        else:
            logger.info(
                "ℹ️  No existing database on Nextcloud — will create a fresh one"
            )
            return False
    except Exception as e:
        logger.warning(
            "⚠️  Failed to download database from Nextcloud: {} — starting fresh", e
        )
        return False


async def _upload_db_to_nextcloud() -> bool:
    """Upload the local SQLite database to Nextcloud.

    Uses a two-step approach: upload to a .tmp file, then MOVE to the final
    path for atomicity (if the server supports it).  Falls back to direct PUT.

    On success, updates the ``_last_db_upload`` timestamp in the API module
    so the system status endpoint can report it.
    """
    import time as _time

    from src.webdav import is_configured, move_remote, upload_file

    if not is_configured():
        return False

    if not DB_PATH.exists():
        logger.warning("⚠️  No local database to upload")
        return False

    try:
        content = DB_PATH.read_bytes()
        tmp_remote = NEXTCLOUD_DB_PATH + ".tmp"

        ok = await upload_file(
            tmp_remote, content, content_type="application/x-sqlite3"
        )
        if not ok:
            logger.error("❌ Failed to upload database to Nextcloud (tmp)")
            return False

        # Try atomic rename; if MOVE fails fall back — the .tmp is already there
        moved = await move_remote(tmp_remote, NEXTCLOUD_DB_PATH)
        if not moved:
            # Fallback: overwrite directly (the tmp upload already succeeded,
            # so the worst case is we have a .tmp file left behind)
            logger.warning("⚠️  MOVE failed, falling back to direct PUT")
            ok = await upload_file(
                NEXTCLOUD_DB_PATH, content, content_type="application/x-sqlite3"
            )
            if not ok:
                logger.error("❌ Direct PUT fallback also failed")
                return False

        logger.info("⬆️  Database uploaded to Nextcloud ({} bytes)", len(content))

        # Record the upload timestamp so /api/system/status can report it
        try:
            import src.routes.api as _api_mod

            _api_mod._last_db_upload = _time.time()
        except Exception:
            pass

        return True
    except Exception as e:
        logger.error("❌ Error uploading database to Nextcloud: {}", e)
        return False


async def _periodic_db_sync():
    """Background task that uploads the database to Nextcloud every N seconds."""
    while True:
        try:
            await asyncio.sleep(DB_SYNC_INTERVAL)
            logger.debug(
                "🔄 Periodic DB sync triggered (interval={}s)", DB_SYNC_INTERVAL
            )
            await _upload_db_to_nextcloud()
        except asyncio.CancelledError:
            logger.debug("🔄 Periodic DB sync task cancelled")
            break
        except Exception as e:
            logger.error("❌ Periodic DB sync error: {}", e)
            # Keep running — we'll try again next interval


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    On startup:
        1. Create required temp directories
        2. Download database from Nextcloud (if available)
        3. Initialize / migrate the SQLite database
        4. Start periodic DB sync background task

    On shutdown:
        5. Cancel periodic sync
        6. Upload database to Nextcloud one final time
        7. Log shutdown
    """
    global _db_sync_task

    # --- Startup ---
    logger.info("🚀 Starting Clone Hero Content Manager v{}", APP_VERSION)
    logger.info("📋 Environment: {} | Debug: {}", APP_ENV, DEBUG)

    # Step 1: Ensure temp directories exist
    ensure_directories()
    logger.info("📁 Temp directories initialized")

    # Step 2: Try to download database from Nextcloud
    await _download_db_from_nextcloud()

    # Step 3: Initialize database (creates tables / runs migrations)
    try:
        init_db()
    except Exception as e:
        logger.critical("❌ Database initialization failed: {}", e)
        raise

    # Step 4: Start periodic DB sync
    _db_sync_task = asyncio.create_task(_periodic_db_sync())
    logger.info("🔄 Periodic DB sync started (every {}s)", DB_SYNC_INTERVAL)

    logger.success("✅ Application ready — listening on {}:{}", APP_HOST, APP_PORT)

    yield

    # --- Shutdown ---
    logger.info("🛑 Shutting down Clone Hero Content Manager …")

    # Step 5: Cancel periodic sync
    if _db_sync_task and not _db_sync_task.done():
        _db_sync_task.cancel()
        try:
            await _db_sync_task
        except asyncio.CancelledError:
            pass

    # Step 6: Final DB upload
    logger.info("⬆️  Uploading database to Nextcloud before shutdown …")
    await _upload_db_to_nextcloud()

    logger.info("👋 Shutdown complete")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Clone Hero Content Manager",
        description=(
            "A single-container content management system for Clone Hero. "
            "Browse, upload, generate, and edit songs with Nextcloud WebDAV integration."
        ),
        version=APP_VERSION,
        lifespan=lifespan,
        docs_url="/docs" if DEBUG else None,
        redoc_url="/redoc" if DEBUG else None,
    )

    # ------------------------------------------------------------------
    # Jinja2 templates
    # ------------------------------------------------------------------
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.state.templates = templates

    # ------------------------------------------------------------------
    # Static files
    # ------------------------------------------------------------------
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # ------------------------------------------------------------------
    # Request logging middleware
    # ------------------------------------------------------------------
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log every incoming HTTP request with timing information."""
        start = time.time()
        response = None
        try:
            response = await call_next(request)
        except Exception as exc:
            duration = round(time.time() - start, 3)
            logger.error(
                "❌ {method} {path} — unhandled error after {duration}s: {exc}",
                method=request.method,
                path=request.url.path,
                duration=duration,
                exc=exc,
            )
            raise
        else:
            duration = round(time.time() - start, 3)
            status = response.status_code

            if status >= 500:
                logger.error(
                    "📤 {method} {path} — {status} [{duration}s]",
                    method=request.method,
                    path=request.url.path,
                    status=status,
                    duration=duration,
                )
            elif status >= 400:
                logger.warning(
                    "📤 {method} {path} — {status} [{duration}s]",
                    method=request.method,
                    path=request.url.path,
                    status=status,
                    duration=duration,
                )
            else:
                if not request.url.path.startswith("/static"):
                    logger.info(
                        "📤 {method} {path} — {status} [{duration}s]",
                        method=request.method,
                        path=request.url.path,
                        status=status,
                        duration=duration,
                    )

            return response

    # ------------------------------------------------------------------
    # Register routers
    # ------------------------------------------------------------------
    app.include_router(api_router)  # /api/*  — JSON endpoints
    app.include_router(pages_router)  # /*      — HTML pages (must be last)

    return app


# ---------------------------------------------------------------------------
# Create the app instance (used by Gunicorn / Uvicorn)
# ---------------------------------------------------------------------------
app = create_app()


# ---------------------------------------------------------------------------
# Direct execution (development)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=DEBUG,
        log_level="debug" if DEBUG else "info",
    )
