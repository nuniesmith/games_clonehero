"""
Clone Hero Content Manager - SQLite Database

Replaces the multi-container PostgreSQL setup with an embedded SQLite database.
Uses aiosqlite for async operations within FastAPI and plain sqlite3 for sync helpers.
"""

import sqlite3
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Dict, List, Optional

import aiosqlite
from loguru import logger

from src.config import DB_PATH

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS songs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    artist TEXT NOT NULL,
    album TEXT,
    file_path TEXT UNIQUE NOT NULL,
    metadata TEXT DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_songs_title ON songs(title COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_songs_artist ON songs(artist COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_songs_album ON songs(album COLLATE NOCASE);

CREATE TRIGGER IF NOT EXISTS update_songs_timestamp
    AFTER UPDATE ON songs
    FOR EACH ROW
BEGIN
    UPDATE songs SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
END;
"""


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------
def init_db() -> None:
    """Initialize the SQLite database and create tables if needed."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()
        logger.success(f"✅ Database initialized at {DB_PATH}")
    except Exception as e:
        logger.critical(f"❌ Failed to initialize database: {e}")
        raise


# ---------------------------------------------------------------------------
# Async context manager (for use in FastAPI routes)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def get_async_connection():
    """Async context manager for an aiosqlite connection with row factory."""
    db = await aiosqlite.connect(str(DB_PATH))
    db.row_factory = aiosqlite.Row
    try:
        yield db
    finally:
        await db.close()


# ---------------------------------------------------------------------------
# Sync context manager (for use in services / background tasks)
# ---------------------------------------------------------------------------
@contextmanager
def get_connection():
    """Synchronous context manager for a sqlite3 connection with row factory."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Helper: convert sqlite3.Row / aiosqlite.Row to plain dict
# ---------------------------------------------------------------------------
def row_to_dict(row) -> Dict[str, Any]:
    """Convert a database row to a plain dictionary."""
    if row is None:
        return {}
    return dict(row)


# ---------------------------------------------------------------------------
# CRUD operations (async)
# ---------------------------------------------------------------------------
async def insert_song(
    title: str,
    artist: str,
    album: Optional[str],
    file_path: str,
    metadata: str = "{}",
) -> int:
    """Insert a song and return its id. Returns existing id on duplicate file_path."""
    async with get_async_connection() as db:
        # Check for duplicate by file_path
        cursor = await db.execute(
            "SELECT id FROM songs WHERE file_path = ?", (file_path,)
        )
        existing = await cursor.fetchone()
        if existing:
            logger.warning(f"⚠️ Duplicate file_path, skipping insert: {file_path}")
            return existing["id"]

        # Check for duplicate by title + artist + album
        cursor = await db.execute(
            "SELECT id FROM songs WHERE title = ? AND artist = ? AND album = ?",
            (title, artist, album),
        )
        existing = await cursor.fetchone()
        if existing:
            logger.warning(
                f"⚠️ Duplicate song detected (title/artist/album), skipping: {title} - {artist}"
            )
            return existing["id"]

        cursor = await db.execute(
            """
            INSERT INTO songs (title, artist, album, file_path, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (title, artist, album, file_path, metadata),
        )
        await db.commit()
        song_id = cursor.lastrowid or 0
        logger.success(f"✅ Song added (id={song_id}): {title} - {artist}")
        return song_id


async def get_songs(
    search: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Fetch songs with optional search and pagination."""
    async with get_async_connection() as db:
        if search and search.strip():
            pattern = f"%{search.strip()}%"
            cursor = await db.execute(
                """
                SELECT * FROM songs
                WHERE title LIKE ? OR artist LIKE ? OR album LIKE ?
                ORDER BY id DESC
                LIMIT ? OFFSET ?
                """,
                (pattern, pattern, pattern, limit, offset),
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM songs ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
        rows = await cursor.fetchall()
        return [row_to_dict(r) for r in rows]


async def get_song_by_id(song_id: int) -> Optional[Dict[str, Any]]:
    """Fetch a single song by its id."""
    async with get_async_connection() as db:
        cursor = await db.execute("SELECT * FROM songs WHERE id = ?", (song_id,))
        row = await cursor.fetchone()
        return row_to_dict(row) if row else None


async def update_song(song_id: int, **fields) -> bool:
    """Update specific fields of a song. Returns True if a row was modified."""
    if not fields:
        return False

    allowed = {"title", "artist", "album", "file_path", "metadata"}
    filtered = {k: v for k, v in fields.items() if k in allowed}
    if not filtered:
        return False

    set_clause = ", ".join(f"{k} = ?" for k in filtered)
    values = list(filtered.values()) + [song_id]

    async with get_async_connection() as db:
        cursor = await db.execute(
            f"UPDATE songs SET {set_clause} WHERE id = ?",
            values,
        )
        await db.commit()
        updated = cursor.rowcount > 0
        if updated:
            logger.info(f"✏️ Song id={song_id} updated: {list(filtered.keys())}")
        return updated


async def delete_song(song_id: int) -> bool:
    """Delete a song by id. Returns True if a row was deleted."""
    async with get_async_connection() as db:
        cursor = await db.execute("DELETE FROM songs WHERE id = ?", (song_id,))
        await db.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info(f"🗑️ Song id={song_id} deleted")
        else:
            logger.warning(f"⚠️ Song id={song_id} not found for deletion")
        return deleted


async def count_songs(search: Optional[str] = None) -> int:
    """Return total number of songs, optionally filtered by search."""
    async with get_async_connection() as db:
        if search and search.strip():
            pattern = f"%{search.strip()}%"
            cursor = await db.execute(
                """
                SELECT COUNT(*) as cnt FROM songs
                WHERE title LIKE ? OR artist LIKE ? OR album LIKE ?
                """,
                (pattern, pattern, pattern),
            )
        else:
            cursor = await db.execute("SELECT COUNT(*) as cnt FROM songs")
        row = await cursor.fetchone()
        return row["cnt"] if row else 0


# ---------------------------------------------------------------------------
# Sync CRUD helpers (for use in non-async service code)
# ---------------------------------------------------------------------------
def insert_song_sync(
    title: str,
    artist: str,
    album: Optional[str],
    file_path: str,
    metadata: str = "{}",
) -> int:
    """Synchronous version of insert_song for use in service-layer code."""
    with get_connection() as conn:
        cursor = conn.execute("SELECT id FROM songs WHERE file_path = ?", (file_path,))
        existing = cursor.fetchone()
        if existing:
            logger.warning(f"⚠️ Duplicate file_path, skipping insert: {file_path}")
            return existing["id"]

        cursor = conn.execute(
            "SELECT id FROM songs WHERE title = ? AND artist = ? AND album = ?",
            (title, artist, album),
        )
        existing = cursor.fetchone()
        if existing:
            logger.warning(
                f"⚠️ Duplicate song (title/artist/album), skipping: {title} - {artist}"
            )
            return existing["id"]

        cursor = conn.execute(
            """
            INSERT INTO songs (title, artist, album, file_path, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (title, artist, album, file_path, metadata),
        )
        conn.commit()
        song_id = cursor.lastrowid or 0
        logger.success(f"✅ Song added (id={song_id}): {title} - {artist}")
        return song_id
