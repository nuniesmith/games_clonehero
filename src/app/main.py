"""
Clone Hero Content Manager - Main Application

Single-container FastAPI application that serves:
- HTML pages via Jinja2 templates
- Static files (CSS, JS, images)
- REST API endpoints for songs CRUD, upload, generation, and WebDAV
- Health check endpoint

Replaces the previous multi-container architecture (14 containers) with a single
unified service using SQLite and Nextcloud WebDAV for centralized storage.
"""

import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from loguru import logger

from src.app.config import (
    APP_ENV,
    APP_HOST,
    APP_PORT,
    APP_VERSION,
    DEBUG,
    LOG_DIR,
    LOG_FILE_SIZE,
    LOG_LEVEL,
    LOG_RETENTION,
    STATIC_DIR,
    TEMPLATES_DIR,
    ensure_directories,
)
from src.app.database import init_db
from src.app.routes.api import router as api_router
from src.app.routes.pages import router as pages_router

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
# Remove default loguru handler to avoid duplicate output
logger.remove()

# Console logging
logger.add(
    sys.stdout,
    level="DEBUG" if DEBUG else "INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
)


def _setup_file_logging() -> None:
    """Configure file-based logging (called after directories are ensured)."""
    log_file = LOG_DIR / "app.log"
    logger.add(
        str(log_file),
        rotation=LOG_FILE_SIZE,
        retention=LOG_RETENTION,
        level=LOG_LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    On startup:
        1. Create required directories
        2. Set up file logging
        3. Initialize the SQLite database

    On shutdown:
        4. Log shutdown and clean up
    """
    # --- Startup ---
    logger.info("🚀 Starting Clone Hero Content Manager v{}", APP_VERSION)
    logger.info("📋 Environment: {} | Debug: {}", APP_ENV, DEBUG)

    # Step 1: Ensure all directories exist
    ensure_directories()
    logger.info("📁 Data directories initialized")

    # Step 2: Set up file logging now that directories exist
    _setup_file_logging()

    # Step 3: Initialize database
    try:
        init_db()
    except Exception as e:
        logger.critical("❌ Database initialization failed: {}", e)
        raise

    logger.success("✅ Application ready — listening on {}:{}", APP_HOST, APP_PORT)

    yield

    # --- Shutdown ---
    logger.info("🛑 Shutting down Clone Hero Content Manager...")


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

    # Store templates on app.state so routes can access them via request.app
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

            # Use different log levels based on status code
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
                # Don't log every static file request at INFO level
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
        "src.app.main:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=DEBUG,
        log_level="debug" if DEBUG else "info",
    )
