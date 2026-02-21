"""
Clone Hero Content Manager - Page Routes

Serves HTML pages using Jinja2 templates. These routes handle all the
browser-facing views: home, songs browser, song editor, upload page,
song generator, chart viewer/editor, chart validation, library organizer,
and the Nextcloud WebDAV file browser.

Songs are stored on Nextcloud via WebDAV.  The local SQLite database is
a metadata cache that is refreshed by the library sync process.
"""

import json
from typing import Optional

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse
from loguru import logger

from src.auth import get_current_user
from src.config import APP_VERSION, CONTENT_FOLDERS, NEXTCLOUD_SONGS_PATH
from src.database import count_songs, get_song_by_id, get_songs
from src.webdav import check_connection, is_configured, list_directory

router = APIRouter(tags=["Pages"])

# Number of songs per page in listings
PAGE_SIZE = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_metadata(raw) -> dict:
    """Safely parse metadata from the database (stored as JSON string or dict)."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _paginate(page: int, total: int, page_size: int = PAGE_SIZE) -> dict:
    """Build a pagination context dict."""
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = max(1, min(page, total_pages))
    offset = (page - 1) * page_size
    return {
        "page": page,
        "total_pages": total_pages,
        "offset": offset,
        "page_size": page_size,
        "total": total,
        "has_previous": page > 1,
        "has_next": page < total_pages,
        "previous_page": page - 1 if page > 1 else None,
        "next_page": page + 1 if page < total_pages else None,
    }


# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------
@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Landing page / dashboard."""
    total = await count_songs()
    webdav_configured = is_configured()
    webdav_connected = False

    if webdav_configured:
        status = await check_connection()
        webdav_connected = status.get("connected", False)

    context = {
        "request": request,
        "page_title": "Home",
        "current_user": get_current_user(request),
        "total_songs": total,
        "webdav_configured": webdav_configured,
        "webdav_connected": webdav_connected,
        "nextcloud_songs_path": NEXTCLOUD_SONGS_PATH,
        "version": APP_VERSION,
        "content_types": list(CONTENT_FOLDERS.keys()),
    }
    return request.app.state.templates.TemplateResponse("home.html", context)


# ---------------------------------------------------------------------------
# Songs Browser / Database Explorer
# ---------------------------------------------------------------------------
@router.get("/songs", response_class=HTMLResponse)
async def songs_page(
    request: Request,
    search: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
):
    """Browse and search songs stored in the local metadata cache."""
    search_query = search.strip() if search else None

    total = await count_songs(search=search_query)
    pag = _paginate(page, total)

    songs = await get_songs(
        search=search_query,
        limit=pag["page_size"],
        offset=pag["offset"],
    )

    # Parse metadata for each song
    for song in songs:
        song["metadata"] = _parse_metadata(song.get("metadata"))

    context = {
        "request": request,
        "page_title": "Songs",
        "current_user": get_current_user(request),
        "songs": songs,
        "search_query": search_query or "",
        "pagination": pag,
        "webdav_configured": is_configured(),
        "nextcloud_songs_path": NEXTCLOUD_SONGS_PATH,
    }
    return request.app.state.templates.TemplateResponse("songs.html", context)


# ---------------------------------------------------------------------------
# Song Editor
# ---------------------------------------------------------------------------
@router.get("/songs/{song_id}/edit", response_class=HTMLResponse)
async def song_editor(request: Request, song_id: int):
    """Edit a song's metadata (song.ini fields)."""
    song = await get_song_by_id(song_id)

    if not song:
        context = {
            "request": request,
            "page_title": "Song Not Found",
            "current_user": get_current_user(request),
            "error": f"Song with ID {song_id} was not found.",
        }
        return request.app.state.templates.TemplateResponse("editor.html", context)

    song["metadata"] = _parse_metadata(song.get("metadata"))

    context = {
        "request": request,
        "page_title": f"Edit: {song.get('title', 'Song')}",
        "current_user": get_current_user(request),
        "song": song,
        "webdav_configured": is_configured(),
    }
    return request.app.state.templates.TemplateResponse("editor.html", context)


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
@router.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Upload songs and content archives."""
    context = {
        "request": request,
        "page_title": "Upload Content",
        "current_user": get_current_user(request),
        "content_types": ["songs", "backgrounds", "colors", "highways"],
        "webdav_configured": is_configured(),
    }
    return request.app.state.templates.TemplateResponse("upload.html", context)


# ---------------------------------------------------------------------------
# Song Generator
# ---------------------------------------------------------------------------
@router.get("/generator", response_class=HTMLResponse)
async def generator_page(request: Request):
    """Process audio files into Clone Hero charts."""
    context = {
        "request": request,
        "page_title": "Song Generator",
        "current_user": get_current_user(request),
        "webdav_configured": is_configured(),
    }
    return request.app.state.templates.TemplateResponse("generator.html", context)


# ---------------------------------------------------------------------------
# Chart Viewer / Editor
# ---------------------------------------------------------------------------
@router.get("/chart-viewer", response_class=HTMLResponse)
async def chart_viewer_page(
    request: Request,
    song_id: Optional[int] = Query(None),
    difficulty: Optional[str] = Query(None),
):
    """
    Interactive chart viewer / editor.

    If *song_id* is provided the viewer will auto-load that song's chart
    from Nextcloud.  Otherwise the user can upload a ``.chart`` file or
    pick a song from the library.
    """
    song = None
    if song_id is not None:
        song = await get_song_by_id(song_id)
        if song:
            song["metadata"] = _parse_metadata(song.get("metadata"))

    context = {
        "request": request,
        "page_title": "Chart Viewer",
        "current_user": get_current_user(request),
        "webdav_configured": is_configured(),
        "song": song,
        "song_id": song_id,
        "initial_difficulty": difficulty or "expert",
    }
    return request.app.state.templates.TemplateResponse("chart_viewer.html", context)


# ---------------------------------------------------------------------------
# Chart Validation
# ---------------------------------------------------------------------------
@router.get("/validation", response_class=HTMLResponse)
async def validation_page(request: Request):
    """Validate Clone Hero chart files for issues."""
    webdav_configured = is_configured()

    # Fetch a few recent songs for the "quick pick" list
    songs = []
    if webdav_configured:
        try:
            songs = await get_songs(limit=20)
            for song in songs:
                song["metadata"] = _parse_metadata(song.get("metadata"))
        except Exception:
            pass

    context = {
        "request": request,
        "page_title": "Chart Validation",
        "current_user": get_current_user(request),
        "webdav_configured": webdav_configured,
        "songs": songs,
    }
    return request.app.state.templates.TemplateResponse("validation.html", context)


# ---------------------------------------------------------------------------
# Library Organizer
# ---------------------------------------------------------------------------
@router.get("/organize", response_class=HTMLResponse)
async def organize_page(request: Request):
    """Library organization tools — fix paths, fetch art, clean duplicates."""
    webdav_configured = is_configured()

    context = {
        "request": request,
        "page_title": "Library Organizer",
        "current_user": get_current_user(request),
        "webdav_configured": webdav_configured,
        "nextcloud_songs_path": NEXTCLOUD_SONGS_PATH,
    }
    return request.app.state.templates.TemplateResponse("organize.html", context)


# ---------------------------------------------------------------------------
# Nextcloud WebDAV Browser
# ---------------------------------------------------------------------------
@router.get("/browser", response_class=HTMLResponse)
async def browser_page(
    request: Request,
    path: str = Query("/", alias="path"),
):
    """Browse files on the connected Nextcloud instance via WebDAV."""
    webdav_configured = is_configured()
    items = []
    connection_status = {}
    error = None

    if webdav_configured:
        connection_status = await check_connection()
        if connection_status.get("connected"):
            try:
                items_raw = await list_directory(path)
                items = [item.to_dict() for item in items_raw]
            except Exception as e:
                logger.error(f"❌ Error browsing Nextcloud path '{path}': {e}")
                error = str(e)
        else:
            error = connection_status.get("error", "Connection failed")

    # Build breadcrumb navigation
    breadcrumbs = [{"name": "Root", "path": "/"}]
    if path and path != "/":
        parts = path.strip("/").split("/")
        accumulated = ""
        for part in parts:
            accumulated += f"/{part}"
            breadcrumbs.append({"name": part, "path": accumulated})

    # Determine parent path for the "up" navigation
    parent_path = "/".join(path.strip("/").split("/")[:-1])
    parent_path = f"/{parent_path}" if parent_path else "/"

    context = {
        "request": request,
        "page_title": "Nextcloud Browser",
        "current_user": get_current_user(request),
        "webdav_configured": webdav_configured,
        "connected": connection_status.get("connected", False),
        "connection_error": error,
        "items": items,
        "current_path": path,
        "parent_path": parent_path,
        "breadcrumbs": breadcrumbs,
        "nextcloud_songs_path": NEXTCLOUD_SONGS_PATH,
    }
    return request.app.state.templates.TemplateResponse("browser.html", context)
