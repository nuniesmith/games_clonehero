"""
Clone Hero Content Manager - SQLite Database

Replaces the multi-container PostgreSQL setup with an embedded SQLite database.
Uses aiosqlite for async operations within FastAPI and plain sqlite3 for sync helpers.

The `remote_path` column stores the Nextcloud WebDAV path for each song
(e.g. ``/songs/Artist/Title_abc123``).  The legacy `file_path` column is
kept for backwards compatibility but is no longer the primary identifier —
`remote_path` is now the canonical location of a song.
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
    file_path TEXT DEFAULT '',
    remote_path TEXT UNIQUE NOT NULL,
    metadata TEXT DEFAULT '{}',
    synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_songs_title ON songs(title COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_songs_artist ON songs(artist COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_songs_album ON songs(album COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_songs_remote_path ON songs(remote_path);

CREATE TRIGGER IF NOT EXISTS update_songs_timestamp
    AFTER UPDATE ON songs
    FOR EACH ROW
BEGIN
    UPDATE songs SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
END;
"""

# ---------------------------------------------------------------------------
# Migration helpers
# ---------------------------------------------------------------------------
_MIGRATIONS = [
    # Migration 1: Add remote_path column if it doesn't exist (upgrade from
    #              the local-only schema).
    {
        "check": "SELECT COUNT(*) FROM pragma_table_info('songs') WHERE name='remote_path'",
        "apply": [
            "ALTER TABLE songs ADD COLUMN remote_path TEXT DEFAULT ''",
            "UPDATE songs SET remote_path = file_path WHERE remote_path = ''",
            "CREATE INDEX IF NOT EXISTS idx_songs_remote_path ON songs(remote_path)",
        ],
        "description": "Add remote_path column",
    },
    # Migration 2: Add synced_at column
    {
        "check": "SELECT COUNT(*) FROM pragma_table_info('songs') WHERE name='synced_at'",
        "apply": [
            "ALTER TABLE songs ADD COLUMN synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
        ],
        "description": "Add synced_at column",
    },
    # Migration 3: Add tags column (JSON array for indexing / categorisation)
    {
        "check": "SELECT COUNT(*) FROM pragma_table_info('songs') WHERE name='tags'",
        "apply": [
            "ALTER TABLE songs ADD COLUMN tags TEXT DEFAULT '[]'",
        ],
        "description": "Add tags column",
    },
]


def _run_migrations(conn: sqlite3.Connection) -> None:
    """Run any pending schema migrations."""
    for migration in _MIGRATIONS:
        cursor = conn.execute(str(migration["check"]))
        (count,) = cursor.fetchone()
        if count == 0:
            logger.info("🔄 Running migration: {}", migration["description"])
            for stmt in migration["apply"]:
                conn.execute(stmt)
            conn.commit()
            logger.success("✅ Migration applied: {}", migration["description"])


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------
def init_db() -> None:
    """Initialize the SQLite database, create tables, and run migrations."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()
            _run_migrations(conn)
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
    remote_path: str,
    metadata: str = "{}",
    file_path: str = "",
) -> int:
    """Insert a song and return its id. Returns existing id on duplicate remote_path."""
    async with get_async_connection() as db:
        # Check for duplicate by remote_path
        cursor = await db.execute(
            "SELECT id FROM songs WHERE remote_path = ?", (remote_path,)
        )
        existing = await cursor.fetchone()
        if existing:
            logger.warning(f"⚠️ Duplicate remote_path, skipping insert: {remote_path}")
            return existing["id"]

        cursor = await db.execute(
            """
            INSERT INTO songs (title, artist, album, file_path, remote_path, metadata, synced_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (title, artist, album, file_path, remote_path, metadata),
        )
        await db.commit()
        song_id = cursor.lastrowid or 0
        logger.success(f"✅ Song added (id={song_id}): {title} - {artist}")
        return song_id


async def upsert_song(
    title: str,
    artist: str,
    album: Optional[str],
    remote_path: str,
    metadata: str = "{}",
    file_path: str = "",
) -> int:
    """
    Insert or update a song keyed by *remote_path*.

    If a row with the same ``remote_path`` already exists its metadata
    fields are updated; otherwise a new row is created.

    Returns the song id.
    """
    async with get_async_connection() as db:
        cursor = await db.execute(
            "SELECT id FROM songs WHERE remote_path = ?", (remote_path,)
        )
        existing = await cursor.fetchone()

        if existing:
            song_id = existing["id"]
            await db.execute(
                """
                UPDATE songs
                SET title = ?, artist = ?, album = ?, file_path = ?,
                    metadata = ?, synced_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (title, artist, album, file_path, metadata, song_id),
            )
            await db.commit()
            logger.info(f"🔄 Song updated (id={song_id}): {title} - {artist}")
            return song_id
        else:
            cursor = await db.execute(
                """
                INSERT INTO songs (title, artist, album, file_path, remote_path, metadata, synced_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (title, artist, album, file_path, remote_path, metadata),
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


async def get_song_by_remote_path(remote_path: str) -> Optional[Dict[str, Any]]:
    """Fetch a single song by its Nextcloud remote path."""
    async with get_async_connection() as db:
        cursor = await db.execute(
            "SELECT * FROM songs WHERE remote_path = ?", (remote_path,)
        )
        row = await cursor.fetchone()
        return row_to_dict(row) if row else None


async def get_all_remote_paths() -> List[str]:
    """Return a list of all remote_path values currently in the database."""
    async with get_async_connection() as db:
        cursor = await db.execute("SELECT remote_path FROM songs ORDER BY id")
        rows = await cursor.fetchall()
        return [r["remote_path"] for r in rows]


async def update_song(song_id: int, **fields) -> bool:
    """Update specific fields of a song. Returns True if a row was modified."""
    if not fields:
        return False

    allowed = {
        "title",
        "artist",
        "album",
        "file_path",
        "remote_path",
        "metadata",
        "tags",
    }
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
            logger.info(f"🗑️ Song id={song_id} deleted from database")
        else:
            logger.warning(f"⚠️ Song id={song_id} not found for deletion")
        return deleted


async def delete_song_by_remote_path(remote_path: str) -> bool:
    """Delete a song by its remote_path. Returns True if a row was deleted."""
    async with get_async_connection() as db:
        cursor = await db.execute(
            "DELETE FROM songs WHERE remote_path = ?", (remote_path,)
        )
        await db.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info(f"🗑️ Song deleted (remote_path={remote_path})")
        return deleted


async def get_artists(search: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return distinct artists with song counts, optionally filtered by search."""
    async with get_async_connection() as db:
        if search and search.strip():
            pattern = f"%{search.strip()}%"
            cursor = await db.execute(
                """
                SELECT artist, COUNT(*) as song_count
                FROM songs
                WHERE title LIKE ? OR artist LIKE ? OR album LIKE ?
                GROUP BY artist
                ORDER BY artist COLLATE NOCASE ASC
                """,
                (pattern, pattern, pattern),
            )
        else:
            cursor = await db.execute(
                """
                SELECT artist, COUNT(*) as song_count
                FROM songs
                GROUP BY artist
                ORDER BY artist COLLATE NOCASE ASC
                """
            )
        rows = await cursor.fetchall()
        return [row_to_dict(r) for r in rows]


async def get_songs_by_artist(
    artist: str,
    search: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetch all songs for a specific artist, optionally filtered by search."""
    async with get_async_connection() as db:
        if search and search.strip():
            pattern = f"%{search.strip()}%"
            cursor = await db.execute(
                """
                SELECT * FROM songs
                WHERE artist = ?
                  AND (title LIKE ? OR artist LIKE ? OR album LIKE ?)
                ORDER BY album COLLATE NOCASE ASC, title COLLATE NOCASE ASC
                """,
                (artist, pattern, pattern, pattern),
            )
        else:
            cursor = await db.execute(
                """
                SELECT * FROM songs
                WHERE artist = ?
                ORDER BY album COLLATE NOCASE ASC, title COLLATE NOCASE ASC
                """,
                (artist,),
            )
        rows = await cursor.fetchall()
        return [row_to_dict(r) for r in rows]


async def get_artist_variants() -> List[Dict[str, Any]]:
    """
    Find artist names that differ only by case / minor punctuation.

    Returns a list of variant groups.  Each group is a dict with:
      - ``key``   – the lower-cased artist name used for grouping
      - ``variants`` – list of dicts ``{"artist": str, "song_count": int}``

    Only groups with 2+ distinct spellings are returned.
    """
    async with get_async_connection() as db:
        cursor = await db.execute(
            """
            SELECT artist, COUNT(*) as song_count
            FROM songs
            GROUP BY artist
            ORDER BY artist COLLATE NOCASE ASC
            """
        )
        rows = await cursor.fetchall()

    # Group by lower-cased + stripped key
    from collections import OrderedDict

    groups: OrderedDict[str, list] = OrderedDict()
    for row in rows:
        r = row_to_dict(row)
        key = r["artist"].strip().lower()
        groups.setdefault(key, []).append(r)

    # Keep only groups with more than one distinct spelling
    return [
        {"key": key, "variants": variants}
        for key, variants in groups.items()
        if len(variants) > 1
    ]


async def rename_artist(old_name: str, new_name: str) -> int:
    """
    Rename every song whose artist matches *old_name* (exact, case-sensitive)
    to *new_name*.  Returns the number of rows updated.
    """
    async with get_async_connection() as db:
        cursor = await db.execute(
            "UPDATE songs SET artist = ? WHERE artist = ?",
            (new_name, old_name),
        )
        await db.commit()
        updated = cursor.rowcount
        if updated:
            logger.info(
                "✏️ Renamed artist '{}' → '{}' ({} song{})",
                old_name,
                new_name,
                updated,
                "s" if updated != 1 else "",
            )
        return updated


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


async def purge_stale_songs(valid_remote_paths: List[str]) -> int:
    """
    Remove songs whose ``remote_path`` is **not** in *valid_remote_paths*.

    This is used during a library sync to clean up songs that have been
    deleted from Nextcloud.  Returns the number of rows removed.
    """
    if not valid_remote_paths:
        return 0

    async with get_async_connection() as db:
        # Fetch all current remote_paths
        cursor = await db.execute("SELECT id, remote_path FROM songs")
        rows = await cursor.fetchall()

        valid_set = set(valid_remote_paths)
        ids_to_delete = [r["id"] for r in rows if r["remote_path"] not in valid_set]

        if not ids_to_delete:
            return 0

        placeholders = ", ".join("?" for _ in ids_to_delete)
        await db.execute(
            f"DELETE FROM songs WHERE id IN ({placeholders})",
            ids_to_delete,
        )
        await db.commit()
        logger.info(
            "🗑️ Purged {} stale song(s) no longer on Nextcloud", len(ids_to_delete)
        )
        return len(ids_to_delete)


# ---------------------------------------------------------------------------
# Tag helpers
# ---------------------------------------------------------------------------


def parse_tags(raw) -> List[str]:
    """Parse a tags value from the database into a Python list of strings.

    Tags are stored as a JSON array (e.g. ``'["rock","metal"]'``).
    Accepts a list, a JSON string, or ``None`` and always returns a
    deduplicated, lowercase-sorted list.
    """
    import json as _json

    if isinstance(raw, list):
        tags = raw
    elif isinstance(raw, str):
        try:
            parsed = _json.loads(raw)
            tags = parsed if isinstance(parsed, list) else []
        except (_json.JSONDecodeError, TypeError):
            tags = []
    else:
        tags = []
    # Normalise: strip whitespace, lowercase, deduplicate, sort
    seen: set[str] = set()
    out: List[str] = []
    for t in tags:
        t_clean = str(t).strip().lower()
        if t_clean and t_clean not in seen:
            seen.add(t_clean)
            out.append(t_clean)
    out.sort()
    return out


async def get_all_tags() -> List[Dict[str, Any]]:
    """Return every distinct tag across all songs with usage counts.

    Returns a list of ``{"tag": str, "count": int}`` dicts sorted
    alphabetically by tag name.
    """
    import json as _json

    async with get_async_connection() as db:
        cursor = await db.execute(
            "SELECT tags FROM songs WHERE tags IS NOT NULL AND tags != '[]'"
        )
        rows = await cursor.fetchall()

    tag_counts: Dict[str, int] = {}
    for row in rows:
        for tag in parse_tags(row["tags"]):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    return [{"tag": tag, "count": count} for tag, count in sorted(tag_counts.items())]


async def get_song_tags(song_id: int) -> List[str]:
    """Return the tag list for a single song."""
    async with get_async_connection() as db:
        cursor = await db.execute("SELECT tags FROM songs WHERE id = ?", (song_id,))
        row = await cursor.fetchone()
    if not row:
        return []
    return parse_tags(row["tags"])


async def set_song_tags(song_id: int, tags: List[str]) -> bool:
    """Replace all tags on a song. Returns True if the row was updated."""
    import json as _json

    clean = parse_tags(tags)  # normalise
    tags_json = _json.dumps(clean)
    async with get_async_connection() as db:
        cursor = await db.execute(
            "UPDATE songs SET tags = ? WHERE id = ?",
            (tags_json, song_id),
        )
        await db.commit()
        updated = cursor.rowcount > 0
        if updated:
            logger.info("🏷️ Song id={} tags set to {}", song_id, clean)
        return updated


async def add_song_tag(song_id: int, tag: str) -> List[str]:
    """Add a single tag to a song (idempotent). Returns the new tag list."""
    import json as _json

    current = await get_song_tags(song_id)
    tag_clean = tag.strip().lower()
    if not tag_clean:
        return current
    if tag_clean in current:
        return current
    current.append(tag_clean)
    current.sort()
    tags_json = _json.dumps(current)
    async with get_async_connection() as db:
        await db.execute(
            "UPDATE songs SET tags = ? WHERE id = ?",
            (tags_json, song_id),
        )
        await db.commit()
    logger.info("🏷️ Song id={} +tag '{}'", song_id, tag_clean)
    return current


async def remove_song_tag(song_id: int, tag: str) -> List[str]:
    """Remove a single tag from a song. Returns the new tag list."""
    import json as _json

    current = await get_song_tags(song_id)
    tag_clean = tag.strip().lower()
    if tag_clean not in current:
        return current
    current.remove(tag_clean)
    tags_json = _json.dumps(current)
    async with get_async_connection() as db:
        await db.execute(
            "UPDATE songs SET tags = ? WHERE id = ?",
            (tags_json, song_id),
        )
        await db.commit()
    logger.info("🏷️ Song id={} -tag '{}'", song_id, tag_clean)
    return current


async def get_songs_by_tag(
    tag: str,
    search: Optional[str] = None,
    limit: int = 200,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Fetch songs that contain a specific tag, with optional search filter.

    Tags are stored as JSON arrays so we use a LIKE match on the
    serialised string (reliable because tags are always lowercase,
    stripped, and sorted).
    """
    tag_clean = tag.strip().lower()
    tag_pattern = f'%"{tag_clean}"%'

    async with get_async_connection() as db:
        if search and search.strip():
            search_pattern = f"%{search.strip()}%"
            cursor = await db.execute(
                """
                SELECT * FROM songs
                WHERE tags LIKE ?
                  AND (title LIKE ? OR artist LIKE ? OR album LIKE ?)
                ORDER BY artist COLLATE NOCASE ASC, title COLLATE NOCASE ASC
                LIMIT ? OFFSET ?
                """,
                (
                    tag_pattern,
                    search_pattern,
                    search_pattern,
                    search_pattern,
                    limit,
                    offset,
                ),
            )
        else:
            cursor = await db.execute(
                """
                SELECT * FROM songs
                WHERE tags LIKE ?
                ORDER BY artist COLLATE NOCASE ASC, title COLLATE NOCASE ASC
                LIMIT ? OFFSET ?
                """,
                (tag_pattern, limit, offset),
            )
        rows = await cursor.fetchall()
        return [row_to_dict(r) for r in rows]


async def count_songs_by_tag(tag: str, search: Optional[str] = None) -> int:
    """Return the number of songs that have a given tag."""
    tag_clean = tag.strip().lower()
    tag_pattern = f'%"{tag_clean}"%'

    async with get_async_connection() as db:
        if search and search.strip():
            search_pattern = f"%{search.strip()}%"
            cursor = await db.execute(
                """
                SELECT COUNT(*) as cnt FROM songs
                WHERE tags LIKE ?
                  AND (title LIKE ? OR artist LIKE ? OR album LIKE ?)
                """,
                (tag_pattern, search_pattern, search_pattern, search_pattern),
            )
        else:
            cursor = await db.execute(
                "SELECT COUNT(*) as cnt FROM songs WHERE tags LIKE ?",
                (tag_pattern,),
            )
        row = await cursor.fetchone()
        return row["cnt"] if row else 0


# ---------------------------------------------------------------------------
# Sync CRUD helpers (for use in non-async service code)
# ---------------------------------------------------------------------------
def insert_song_sync(
    title: str,
    artist: str,
    album: Optional[str],
    remote_path: str,
    metadata: str = "{}",
    file_path: str = "",
) -> int:
    """Synchronous version of insert_song for use in service-layer code."""
    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT id FROM songs WHERE remote_path = ?", (remote_path,)
        )
        existing = cursor.fetchone()
        if existing:
            logger.warning(f"⚠️ Duplicate remote_path, skipping insert: {remote_path}")
            return existing["id"]

        cursor = conn.execute(
            """
            INSERT INTO songs (title, artist, album, file_path, remote_path, metadata, synced_at)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (title, artist, album, file_path, remote_path, metadata),
        )
        conn.commit()
        song_id = cursor.lastrowid or 0
        logger.success(f"✅ Song added (id={song_id}): {title} - {artist}")
        return song_id


def upsert_song_sync(
    title: str,
    artist: str,
    album: Optional[str],
    remote_path: str,
    metadata: str = "{}",
    file_path: str = "",
) -> int:
    """Synchronous upsert keyed by *remote_path*."""
    with get_connection() as conn:
        cursor = conn.execute(
            "SELECT id FROM songs WHERE remote_path = ?", (remote_path,)
        )
        existing = cursor.fetchone()

        if existing:
            song_id = existing["id"]
            conn.execute(
                """
                UPDATE songs
                SET title = ?, artist = ?, album = ?, file_path = ?,
                    metadata = ?, synced_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (title, artist, album, file_path, metadata, song_id),
            )
            conn.commit()
            logger.info(f"🔄 Song updated (id={song_id}): {title} - {artist}")
            return song_id
        else:
            cursor = conn.execute(
                """
                INSERT INTO songs (title, artist, album, file_path, remote_path, metadata, synced_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (title, artist, album, file_path, remote_path, metadata),
            )
            conn.commit()
            song_id = cursor.lastrowid or 0
            logger.success(f"✅ Song added (id={song_id}): {title} - {artist}")
            return song_id
