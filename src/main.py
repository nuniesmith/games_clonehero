"""
Clone Hero Content Manager - Main Application

Single-container FastAPI application that serves:
- HTML pages via Jinja2 templates
- Static files (CSS, JS, images)
- REST API endpoints for songs CRUD, upload, generation, and WebDAV
- Health check endpoint
- Simple session-based authentication
- Auto-sync with Nextcloud library

The application is fully stateless: the SQLite database is downloaded from
Nextcloud on startup and uploaded back periodically and on shutdown.  All
persistent data (songs, assets, DB) lives on Nextcloud via WebDAV.
"""

import asyncio
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger

from src.auth import (
    auth_required,
    clear_session_cookie,
    get_current_user,
    render_login_page,
    set_session_cookie,
    verify_credentials,
)
from src.config import (
    APP_ENV,
    APP_HOST,
    APP_PORT,
    APP_VERSION,
    AUTH_PASSWORD,
    DB_PATH,
    DB_SYNC_INTERVAL,
    DEBUG,
    LIBRARY_SYNC_INTERVAL,
    LIBRARY_SYNC_ON_STARTUP,
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
_db_sync_task: asyncio.Task[None] | None = None
_library_sync_task: asyncio.Task[None] | None = None


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
# Library auto-sync (Nextcloud → local DB)
# ---------------------------------------------------------------------------


async def _run_library_sync() -> None:
    """Run a single library sync (non-streaming version for background use)."""
    from src.services.content_manager import sync_library_from_nextcloud
    from src.webdav import is_configured

    if not is_configured():
        return

    try:
        logger.info("🔄 Auto-sync: refreshing library from Nextcloud …")
        result = await sync_library_from_nextcloud()
        if "error" in result:
            logger.warning("⚠️  Auto-sync error: {}", result["error"])
        else:
            synced = result.get("synced", 0)
            purged = result.get("purged", 0)
            logger.info(
                "🔄 Auto-sync complete: {} synced, {} purged",
                synced,
                purged,
            )
    except Exception as e:
        logger.error("❌ Auto-sync error: {}", e)


async def _periodic_library_sync():
    """Background task that syncs the song library from Nextcloud periodically."""
    if LIBRARY_SYNC_INTERVAL <= 0:
        logger.info("ℹ️  Library auto-sync is disabled (interval=0)")
        return

    while True:
        try:
            await asyncio.sleep(LIBRARY_SYNC_INTERVAL)
            await _run_library_sync()
        except asyncio.CancelledError:
            logger.debug("🔄 Periodic library sync task cancelled")
            break
        except Exception as e:
            logger.error("❌ Periodic library sync error: {}", e)


# ---------------------------------------------------------------------------
# Background task error handler
# ---------------------------------------------------------------------------
def _on_background_task_done(task: asyncio.Task) -> None:
    """Log errors from fire-and-forget background tasks."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("❌ Background task failed: {}", exc)


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
        5. Run initial library sync (if enabled)
        6. Start periodic library sync background task

    On shutdown:
        7. Cancel periodic tasks
        8. Upload database to Nextcloud one final time
        9. Log shutdown
    """
    global _db_sync_task, _library_sync_task

    # --- Startup ---
    logger.info("🚀 Starting Clone Hero Content Manager v{}", APP_VERSION)
    logger.info("📋 Environment: {} | Debug: {}", APP_ENV, DEBUG)

    if AUTH_PASSWORD:
        logger.info("🔒 Authentication enabled")
    else:
        logger.warning("🔓 Authentication DISABLED (no AUTH_PASSWORD set)")

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

    # Step 5: Initial library sync on startup
    if LIBRARY_SYNC_ON_STARTUP:
        # Run in background so we don't block startup
        task = asyncio.create_task(_run_library_sync())
        task.add_done_callback(_on_background_task_done)
        logger.info("🔄 Startup library sync triggered (runs in background)")

    # Step 6: Start periodic library sync
    _library_sync_task = asyncio.create_task(_periodic_library_sync())
    if LIBRARY_SYNC_INTERVAL > 0:
        logger.info(
            "🔄 Periodic library sync started (every {}s)", LIBRARY_SYNC_INTERVAL
        )

    logger.success("✅ Application ready — listening on {}:{}", APP_HOST, APP_PORT)

    yield

    # --- Shutdown ---
    logger.info("🛑 Shutting down Clone Hero Content Manager …")

    # Step 7: Cancel periodic tasks
    for task, name in [
        (_db_sync_task, "DB sync"),
        (_library_sync_task, "Library sync"),
    ]:
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            logger.debug("🔄 {} task cancelled", name)

    # Step 8: Final DB upload
    logger.info("⬆️  Uploading database to Nextcloud before shutdown …")
    await _upload_db_to_nextcloud()

    # Step 9: Close shared httpx client
    from src.webdav import close_client

    await close_client()

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
    # Authentication middleware
    # ------------------------------------------------------------------
    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        """Redirect unauthenticated requests to the login page."""
        if auth_required(request):
            # For API requests, return 401 instead of redirect
            if request.url.path.startswith("/api/"):
                from fastapi.responses import JSONResponse

                return JSONResponse(
                    status_code=401,
                    content={"detail": "Authentication required"},
                )
            return RedirectResponse(url="/login", status_code=302)

        response = await call_next(request)
        return response

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
    # Login / Logout routes (mounted directly on app, before routers)
    # ------------------------------------------------------------------
    @app.get("/login")
    async def login_page(request: Request):
        """Show the login form."""
        # If already logged in, redirect to home
        if get_current_user(request):
            return RedirectResponse(url="/", status_code=302)
        return render_login_page()

    @app.post("/login")
    async def login_post(
        request: Request,
        username: str = Form(...),
        password: str = Form(...),
    ):
        """Handle login form submission."""
        if verify_credentials(username, password):
            logger.info("🔓 User '{}' logged in", username)
            response = RedirectResponse(url="/", status_code=302)
            set_session_cookie(response, username)
            return response
        else:
            logger.warning("🔒 Failed login attempt for '{}'", username)
            return render_login_page(
                error="Invalid username or password",
                prefill_user=username,
            )

    @app.get("/logout")
    async def logout(request: Request):
        """Log out and redirect to login page."""
        user = get_current_user(request)
        if user:
            logger.info("🔒 User '{}' logged out", user)
        response = RedirectResponse(url="/login", status_code=302)
        clear_session_cookie(response)
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
