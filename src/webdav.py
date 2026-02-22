"""
Clone Hero Content Manager - Nextcloud WebDAV Client

Provides async methods to browse, upload, download, and manage files
on a Nextcloud instance via the WebDAV protocol.

Uses httpx for async HTTP operations with Basic Auth against the
Nextcloud WebDAV endpoint.

The song library lives entirely on Nextcloud.  This module provides the
low-level WebDAV primitives **plus** higher-level helpers for recursive
directory walking and remote song.ini parsing that the rest of the
application relies on.
"""

from __future__ import annotations

import asyncio
import configparser
import io
import xml.etree.ElementTree as ET
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, AsyncIterator
from urllib.parse import quote, unquote, urlparse

import httpx
from loguru import logger

from src.config import (
    NEXTCLOUD_PASSWORD,
    NEXTCLOUD_SONGS_PATH,
    NEXTCLOUD_URL,
    NEXTCLOUD_USERNAME,
    OPTIONAL_SONG_FIELDS,
    WEBDAV_BASE_URL,
)

# WebDAV XML namespaces
DAV_NS = "DAV:"
OC_NS = "http://owncloud.org/ns"
NC_NS = "http://nextcloud.org/ns"

NSMAP = {
    "d": DAV_NS,
    "oc": OC_NS,
    "nc": NC_NS,
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class WebDAVItem:
    """Represents a file or directory returned by a PROPFIND request."""

    name: str
    path: str  # relative path from the WebDAV root
    is_directory: bool
    size: int = 0
    content_type: str = ""
    last_modified: str = ""
    etag: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def extension(self) -> str:
        return PurePosixPath(self.name).suffix.lower()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "is_directory": self.is_directory,
            "size": self.size,
            "content_type": self.content_type,
            "last_modified": self.last_modified,
            "etag": self.etag,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_configured() -> bool:
    """Return True if Nextcloud WebDAV credentials are configured."""
    return bool(NEXTCLOUD_URL and NEXTCLOUD_USERNAME and NEXTCLOUD_PASSWORD)


def _build_url(remote_path: str = "/") -> str:
    """Build the full WebDAV URL for a given remote path."""
    base = WEBDAV_BASE_URL.rstrip("/")
    clean = remote_path.strip("/")
    if clean:
        # Encode path segments individually to preserve slashes
        encoded = "/".join(quote(seg, safe="") for seg in clean.split("/"))
        return f"{base}/{encoded}"
    return base


def _auth() -> httpx.BasicAuth:
    return httpx.BasicAuth(NEXTCLOUD_USERNAME, NEXTCLOUD_PASSWORD)


# ---------------------------------------------------------------------------
# Shared client for batch operations (connection pooling)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def shared_client(
    timeout: float = 120.0,
    max_connections: int = 20,
    max_keepalive_connections: int = 10,
) -> AsyncIterator[httpx.AsyncClient]:
    """
    Context manager providing a **shared** :class:`httpx.AsyncClient` with
    connection pooling.  Use this for batch operations (e.g. library sync)
    instead of creating a fresh client per request.

    The connection pool dramatically reduces TLS handshake overhead and
    avoids exhausting Nextcloud's connection limits when downloading
    thousands of files.
    """
    limits = httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive_connections,
    )
    async with httpx.AsyncClient(
        auth=_auth(),
        timeout=timeout,
        limits=limits,
    ) as client:
        yield client


def _parse_propfind_response(xml_text: str, _base_path: str = "/") -> list[WebDAVItem]:
    """Parse the XML response from a PROPFIND request into WebDAVItem list."""
    items: list[WebDAVItem] = []

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.error(f"❌ Failed to parse WebDAV XML response: {e}")
        return items

    for response_el in root.findall(f"{{{DAV_NS}}}response"):
        href_el = response_el.find(f"{{{DAV_NS}}}href")
        if href_el is None or href_el.text is None:
            continue

        href = unquote(href_el.text)

        # Extract properties from the successful propstat
        props = None
        for propstat in response_el.findall(f"{{{DAV_NS}}}propstat"):
            status_el = propstat.find(f"{{{DAV_NS}}}status")
            if status_el is not None and "200" in (status_el.text or ""):
                props = propstat.find(f"{{{DAV_NS}}}prop")
                break

        if props is None:
            continue

        # Determine if it's a collection (directory)
        resourcetype = props.find(f"{{{DAV_NS}}}resourcetype")
        is_dir = False
        if resourcetype is not None:
            is_dir = resourcetype.find(f"{{{DAV_NS}}}collection") is not None

        # Extract size
        size = 0
        content_length = props.find(f"{{{DAV_NS}}}getcontentlength")
        if content_length is not None and content_length.text:
            try:
                size = int(content_length.text)
            except ValueError:
                pass

        # Extract content type
        content_type = ""
        ct_el = props.find(f"{{{DAV_NS}}}getcontenttype")
        if ct_el is not None and ct_el.text:
            content_type = ct_el.text

        # Extract last modified
        last_modified = ""
        lm_el = props.find(f"{{{DAV_NS}}}getlastmodified")
        if lm_el is not None and lm_el.text:
            last_modified = lm_el.text

        # Extract etag
        etag = ""
        etag_el = props.find(f"{{{DAV_NS}}}getetag")
        if etag_el is not None and etag_el.text:
            etag = etag_el.text.strip('"')

        # Compute the relative path from the WebDAV base
        parsed_base = urlparse(WEBDAV_BASE_URL).path.rstrip("/")
        if href.startswith(parsed_base):
            relative = href[len(parsed_base) :]
        else:
            relative = href

        relative = relative.strip("/")
        name = PurePosixPath(relative).name if relative else ""

        items.append(
            WebDAVItem(
                name=name,
                path="/" + relative if relative else "/",
                is_directory=is_dir,
                size=size,
                content_type=content_type,
                last_modified=last_modified,
                etag=etag,
            )
        )

    return items


# ---------------------------------------------------------------------------
# Public API (async)
# ---------------------------------------------------------------------------
async def check_connection() -> dict[str, Any]:
    """
    Test the WebDAV connection to Nextcloud.
    Returns a status dict with 'connected' bool and optional error info.
    """
    if not _is_configured():
        return {
            "connected": False,
            "error": "Nextcloud WebDAV is not configured. Set NEXTCLOUD_URL, NEXTCLOUD_USERNAME, and NEXTCLOUD_PASSWORD.",
        }

    try:
        async with httpx.AsyncClient(auth=_auth(), timeout=15.0) as client:
            response = await client.request(
                "PROPFIND",
                _build_url("/"),
                headers={"Depth": "0"},
            )
            if response.status_code in (200, 207):
                logger.info("✅ Nextcloud WebDAV connection successful")
                return {"connected": True, "url": NEXTCLOUD_URL}
            else:
                msg = f"HTTP {response.status_code}: {response.text[:200]}"
                logger.warning(f"⚠️ Nextcloud connection issue: {msg}")
                return {"connected": False, "error": msg}
    except httpx.HTTPError as e:
        logger.error(f"❌ Nextcloud connection failed: {e}")
        return {"connected": False, "error": str(e)}
    except Exception as e:
        logger.error(f"❌ Unexpected error connecting to Nextcloud: {e}")
        return {"connected": False, "error": str(e)}


async def list_directory(remote_path: str = "/") -> list[WebDAVItem]:
    """
    List files and directories at the given remote path.
    Returns a list of WebDAVItem objects (excluding the directory itself).
    """
    if not _is_configured():
        logger.warning("⚠️ Nextcloud WebDAV not configured, returning empty listing")
        return []

    url = _build_url(remote_path)
    propfind_body = """<?xml version="1.0" encoding="UTF-8"?>
    <d:propfind xmlns:d="DAV:" xmlns:oc="http://owncloud.org/ns" xmlns:nc="http://nextcloud.org/ns">
        <d:prop>
            <d:resourcetype/>
            <d:getcontentlength/>
            <d:getcontenttype/>
            <d:getlastmodified/>
            <d:getetag/>
            <oc:size/>
        </d:prop>
    </d:propfind>"""

    try:
        async with httpx.AsyncClient(auth=_auth(), timeout=30.0) as client:
            response = await client.request(
                "PROPFIND",
                url,
                headers={
                    "Depth": "1",
                    "Content-Type": "application/xml; charset=utf-8",
                },
                content=propfind_body,
            )

            if response.status_code not in (200, 207):
                logger.error(
                    f"❌ WebDAV PROPFIND failed ({response.status_code}): {response.text[:300]}"
                )
                return []

            items = _parse_propfind_response(response.text, remote_path)

            # Remove the first item if it represents the directory itself
            clean_path = remote_path.strip("/")
            filtered = [item for item in items if item.path.strip("/") != clean_path]

            # Sort: directories first, then alphabetically
            filtered.sort(key=lambda i: (not i.is_directory, i.name.lower()))

            logger.info(f"📂 Listed {len(filtered)} items in Nextcloud:{remote_path}")
            return filtered

    except httpx.HTTPError as e:
        logger.error(f"❌ WebDAV list failed for {remote_path}: {e}")
        return []
    except Exception as e:
        logger.error(f"❌ Unexpected error listing {remote_path}: {e}")
        return []


async def download_file(
    remote_path: str,
    *,
    client: httpx.AsyncClient | None = None,
    retries: int = 0,
) -> bytes | None:
    """
    Download a file from Nextcloud and return its content as bytes.
    Returns None on failure.

    If *client* is provided (e.g. from :func:`shared_client`), it will be
    reused instead of creating a throwaway connection.  *retries* controls
    how many additional attempts are made on transient errors.
    """
    if not _is_configured():
        logger.warning("⚠️ Nextcloud WebDAV not configured")
        return None

    url = _build_url(remote_path)
    attempts = 1 + max(0, retries)

    async def _do_download(c: httpx.AsyncClient) -> bytes | None:
        for attempt in range(1, attempts + 1):
            try:
                response = await c.get(url)
                if response.status_code == 200:
                    logger.debug(
                        "⬇️ Downloaded {} ({} bytes)",
                        remote_path,
                        len(response.content),
                    )
                    return response.content
                else:
                    logger.warning(
                        "⚠️ Download attempt {}/{} failed ({}) for {}",
                        attempt,
                        attempts,
                        response.status_code,
                        remote_path,
                    )
            except httpx.HTTPError as e:
                logger.warning(
                    "⚠️ Download attempt {}/{} error for {}: {}",
                    attempt,
                    attempts,
                    remote_path,
                    e,
                )
            except Exception as e:
                logger.warning(
                    "⚠️ Download attempt {}/{} unexpected error for {}: {}",
                    attempt,
                    attempts,
                    remote_path,
                    e,
                )

            # Back off before retrying
            if attempt < attempts:
                await asyncio.sleep(0.5 * attempt)

        logger.error("❌ Download failed after {} attempts: {}", attempts, remote_path)
        return None

    if client is not None:
        return await _do_download(client)

    # Fallback: create a one-off client (backwards compatible)
    try:
        async with httpx.AsyncClient(auth=_auth(), timeout=120.0) as c:
            return await _do_download(c)
    except Exception as e:
        logger.error(f"❌ Unexpected download error for {remote_path}: {e}")
        return None


async def download_file_stream(remote_path: str, local_path: str) -> bool:
    """
    Download a file from Nextcloud and save it to a local path (streaming).
    Returns True on success, False on failure.
    """
    if not _is_configured():
        return False

    url = _build_url(remote_path)
    try:
        async with httpx.AsyncClient(auth=_auth(), timeout=300.0) as client:
            async with client.stream("GET", url) as response:
                if response.status_code != 200:
                    logger.error(
                        f"❌ Stream download failed ({response.status_code}): {remote_path}"
                    )
                    return False

                with open(local_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=65536):
                        _ = f.write(chunk)

        logger.info(f"⬇️ Streamed {remote_path} -> {local_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Stream download error for {remote_path}: {e}")
        return False


async def upload_file(
    remote_path: str,
    content: bytes,
    content_type: str = "application/octet-stream",
) -> bool:
    """
    Upload bytes to a remote path on Nextcloud.
    Creates intermediate directories if needed.
    Returns True on success.
    """
    if not _is_configured():
        logger.warning("⚠️ Nextcloud WebDAV not configured")
        return False

    # Ensure parent directories exist
    parent = str(PurePosixPath(remote_path).parent)
    if parent and parent != "/":
        _ = await mkdir(parent)

    url = _build_url(remote_path)
    try:
        async with httpx.AsyncClient(auth=_auth(), timeout=300.0) as client:
            response = await client.put(
                url,
                content=content,
                headers={"Content-Type": content_type},
            )
            if response.status_code in (200, 201, 204):
                logger.info(f"⬆️ Uploaded {remote_path} ({len(content)} bytes)")
                return True
            else:
                logger.error(
                    f"❌ Upload failed ({response.status_code}): {response.text[:200]}"
                )
                return False
    except httpx.HTTPError as e:
        logger.error(f"❌ Upload error for {remote_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected upload error for {remote_path}: {e}")
        return False


async def upload_file_stream(
    remote_path: str,
    file_obj: io.BufferedIOBase,
    content_type: str = "application/octet-stream",
) -> bool:
    """
    Upload a file-like object to Nextcloud (streaming upload).
    Returns True on success.
    """
    if not _is_configured():
        return False

    parent = str(PurePosixPath(remote_path).parent)
    if parent and parent != "/":
        _ = await mkdir(parent)

    url = _build_url(remote_path)
    try:
        data = file_obj.read()
        async with httpx.AsyncClient(auth=_auth(), timeout=300.0) as client:
            response = await client.put(
                url,
                content=data,
                headers={"Content-Type": content_type},
            )
            if response.status_code in (200, 201, 204):
                logger.info(f"⬆️ Streamed upload to {remote_path}")
                return True
            else:
                logger.error(
                    f"❌ Stream upload failed ({response.status_code}): {response.text[:200]}"
                )
                return False
    except Exception as e:
        logger.error(f"❌ Stream upload error for {remote_path}: {e}")
        return False


async def mkdir(remote_path: str) -> bool:
    """
    Create a directory (and parent directories) on Nextcloud via MKCOL.
    Returns True on success or if the directory already exists.
    """
    if not _is_configured():
        return False

    # Build list of directories to create from root to target
    parts = PurePosixPath(remote_path.strip("/")).parts
    current = ""
    for part in parts:
        current = f"{current}/{part}"
        url = _build_url(current)
        try:
            async with httpx.AsyncClient(auth=_auth(), timeout=15.0) as client:
                response = await client.request("MKCOL", url)
                if response.status_code in (201, 405):
                    # 201 = created, 405 = already exists
                    continue
                elif response.status_code == 409:
                    # Parent doesn't exist (shouldn't happen since we go in order)
                    logger.warning(f"⚠️ MKCOL conflict for {current}")
                else:
                    logger.warning(
                        f"⚠️ MKCOL unexpected status {response.status_code} for {current}"
                    )
        except Exception as e:
            logger.error(f"❌ MKCOL error for {current}: {e}")
            return False

    logger.info(f"📁 Ensured directory exists: {remote_path}")
    return True


async def delete_remote(remote_path: str) -> bool:
    """
    Delete a file or directory on Nextcloud.
    Returns True on success.
    """
    if not _is_configured():
        return False

    url = _build_url(remote_path)
    try:
        async with httpx.AsyncClient(auth=_auth(), timeout=30.0) as client:
            response = await client.request("DELETE", url)
            if response.status_code in (200, 204):
                logger.info(f"🗑️ Deleted remote: {remote_path}")
                return True
            elif response.status_code == 404:
                logger.warning(f"⚠️ Remote path not found: {remote_path}")
                return False
            else:
                logger.error(
                    f"❌ Delete failed ({response.status_code}): {remote_path}"
                )
                return False
    except Exception as e:
        logger.error(f"❌ Delete error for {remote_path}: {e}")
        return False


async def move_remote(source_path: str, dest_path: str) -> bool:
    """
    Move/rename a file or directory on Nextcloud.
    Returns True on success.
    """
    if not _is_configured():
        return False

    source_url = _build_url(source_path)
    dest_url = _build_url(dest_path)

    try:
        async with httpx.AsyncClient(auth=_auth(), timeout=30.0) as client:
            response = await client.request(
                "MOVE",
                source_url,
                headers={
                    "Destination": dest_url,
                    "Overwrite": "T",
                },
            )
            if response.status_code in (200, 201, 204):
                logger.info(f"📦 Moved: {source_path} -> {dest_path}")
                return True
            else:
                logger.error(
                    f"❌ Move failed ({response.status_code}): {source_path} -> {dest_path}"
                )
                return False
    except Exception as e:
        logger.error(f"❌ Move error: {e}")
        return False


async def get_file_info(remote_path: str) -> WebDAVItem | None:
    """
    Get metadata for a single file or directory on Nextcloud.
    Returns a WebDAVItem or None if not found.
    """
    if not _is_configured():
        return None

    url = _build_url(remote_path)
    propfind_body = """<?xml version="1.0" encoding="UTF-8"?>
    <d:propfind xmlns:d="DAV:">
        <d:prop>
            <d:resourcetype/>
            <d:getcontentlength/>
            <d:getcontenttype/>
            <d:getlastmodified/>
            <d:getetag/>
        </d:prop>
    </d:propfind>"""

    try:
        async with httpx.AsyncClient(auth=_auth(), timeout=15.0) as client:
            response = await client.request(
                "PROPFIND",
                url,
                headers={
                    "Depth": "0",
                    "Content-Type": "application/xml; charset=utf-8",
                },
                content=propfind_body,
            )

            if response.status_code not in (200, 207):
                return None

            items = _parse_propfind_response(response.text, remote_path)
            return items[0] if items else None

    except Exception as e:
        logger.error(f"❌ Error getting info for {remote_path}: {e}")
        return None


def is_configured() -> bool:
    """Public check for whether Nextcloud WebDAV is configured."""
    return _is_configured()


# ---------------------------------------------------------------------------
# Higher-level helpers – recursive walk & song discovery
# ---------------------------------------------------------------------------


async def walk_directory(
    remote_path: str = "/",
    *,
    concurrency: int = 5,
) -> list[WebDAVItem]:
    """
    Recursively list *all* files and directories under *remote_path*.

    Returns a flat list of :class:`WebDAVItem` objects.  Directories
    themselves are included in the results so callers can distinguish
    them if needed.

    Uses bounded concurrency (*concurrency* parallel PROPFIND requests)
    to speed up large directory trees while avoiding overwhelming the
    server.
    """
    if not _is_configured():
        logger.warning("⚠️ Nextcloud WebDAV not configured, returning empty walk")
        return []

    all_items: list[WebDAVItem] = []
    dirs_to_visit: list[str] = [remote_path]
    sem = asyncio.Semaphore(concurrency)

    async def _list_one(path: str) -> list[WebDAVItem]:
        async with sem:
            return await list_directory(path)

    while dirs_to_visit:
        # Fan out listing for all pending directories
        tasks = [_list_one(d) for d in dirs_to_visit]
        dirs_to_visit = []
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, BaseException):
                logger.warning("⚠️ walk_directory listing error: {}", res)
                continue
            for item in res:
                all_items.append(item)
                if item.is_directory:
                    dirs_to_visit.append(item.path)

    return all_items


async def find_song_folders(root: str | None = None) -> list[str]:
    """
    Walk the Nextcloud songs directory and return a list of remote
    directory paths that contain a ``song.ini`` file.

    Each returned path is the *folder* containing the ``song.ini``,
    e.g. ``/songs/Artist/Title_abc123``.
    """
    root = root or NEXTCLOUD_SONGS_PATH
    all_items = await walk_directory(root)

    song_ini_files = [
        item
        for item in all_items
        if not item.is_directory and item.name.lower() == "song.ini"
    ]

    # Return the parent directory of each song.ini
    folders: list[str] = []
    for item in song_ini_files:
        parent = str(PurePosixPath(item.path).parent)
        if parent not in folders:
            folders.append(parent)

    logger.info(
        "🔍 Found {} song folder(s) on Nextcloud under {}",
        len(folders),
        root,
    )
    return folders


def _decode_ini_bytes(raw: bytes) -> str:
    """
    Decode raw bytes from a song.ini using multiple encoding fallbacks.

    Tries (in order): UTF-8-sig, UTF-8, Latin-1 (which never fails).
    """
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, ValueError):
            continue
    # latin-1 accepts every byte value — guaranteed to succeed
    return raw.decode("latin-1")


def _find_song_section(config: configparser.ConfigParser) -> str | None:
    """
    Find the ``[song]`` section in a case-insensitive manner.

    Clone Hero song.ini files almost always use ``[song]``, but community
    charts occasionally have ``[Song]``, ``[SONG]``, or other variants.
    ConfigParser section names are case-sensitive by default.

    Returns the actual section name found, or ``None``.
    """
    for section in config.sections():
        if section.lower() == "song":
            return section
    return None


async def parse_remote_song_ini(
    remote_path: str,
    *,
    client: httpx.AsyncClient | None = None,
    retries: int = 2,
) -> dict[str, Any] | None:
    """
    Download and parse a ``song.ini`` file from Nextcloud.

    *remote_path* should point to the **folder** containing the
    ``song.ini`` (the ``/song.ini`` suffix is appended automatically if
    the path doesn't already end with it).

    If *client* is supplied the download reuses that connection instead
    of creating a fresh one (important for batch sync).  *retries*
    controls how many extra download attempts are made on failure.

    Returns a dict with ``title``, ``artist``, ``album``, ``metadata``
    keys on success, or ``None`` if the file cannot be downloaded /
    parsed or is missing required fields.
    """
    ini_remote = remote_path.rstrip("/")
    if not ini_remote.lower().endswith("/song.ini"):
        ini_remote = f"{ini_remote}/song.ini"

    content = await download_file(ini_remote, client=client, retries=retries)
    if content is None:
        logger.warning("⚠️ Could not download song.ini from {}", ini_remote)
        return None

    # --- Decode with encoding fallbacks ---
    try:
        text = _decode_ini_bytes(content)
    except Exception as e:
        logger.error("❌ Failed to decode song.ini from {}: {}", ini_remote, e)
        return None

    # --- Parse ini ---
    config = configparser.ConfigParser(strict=False)
    try:
        config.read_string(text)
    except Exception as e:
        logger.error("❌ Failed to parse song.ini from {}: {}", ini_remote, e)
        return None

    # --- Case-insensitive section lookup ---
    section = _find_song_section(config)
    if section is None:
        logger.warning("⚠️ Missing [song] section in {}", ini_remote)
        return None

    # --- Field extraction with fallbacks ---
    # Some charts use "title" instead of "name"
    name = config.get(section, "name", fallback=None) or config.get(
        section, "title", fallback=None
    )
    artist = config.get(section, "artist", fallback=None)
    album = config.get(section, "album", fallback=None)

    # Try to salvage a title from the folder name if missing
    if not name:
        folder_name = remote_path.rstrip("/").rsplit("/", 1)[-1]
        if folder_name:
            # Strip trailing hash suffixes like "_abc123"
            name = folder_name.rsplit("_", 1)[0].replace("_", " ").strip()

    if not name:
        logger.warning("⚠️ Missing song name in {}", ini_remote)
        return None

    # If artist is missing, use "Unknown Artist" so the song isn't lost
    if not artist:
        artist = "Unknown Artist"
        logger.debug("ℹ️ No artist field in {}, defaulting to '{}'", ini_remote, artist)

    metadata: dict[str, Any] = {}
    for field_name in OPTIONAL_SONG_FIELDS:
        if config.has_option(section, field_name):
            value = config.get(section, field_name, fallback=None)
            if value is not None:
                metadata[field_name] = value.strip()

    # Also grab "frets" / "charter" which are used interchangeably
    for charter_key in ("charter", "frets"):
        if config.has_option(section, charter_key):
            val = config.get(section, charter_key, fallback=None)
            if val and "charter" not in metadata:
                metadata["charter"] = val.strip()

    return {
        "title": name.strip(),
        "artist": artist.strip(),
        "album": album.strip() if album else "Unknown",
        "metadata": metadata,
    }


async def upload_song_folder(
    local_dir: str,
    remote_base: str | None = None,
    artist: str = "Unknown",
    title: str = "Unknown",
    suffix: str = "",
) -> str | None:
    """
    Upload an entire local song folder to Nextcloud.

    Creates the remote directory structure under *remote_base* (defaults
    to :data:`NEXTCLOUD_SONGS_PATH`) and uploads every file found in
    *local_dir*.

    Returns the remote folder path on success, or ``None`` on failure.
    """
    if not _is_configured():
        logger.warning("⚠️ Nextcloud not configured – cannot upload song folder")
        return None

    from pathlib import Path

    local_path = Path(local_dir)
    if not local_path.is_dir():
        logger.error("❌ Local directory does not exist: {}", local_dir)
        return None

    base = remote_base or NEXTCLOUD_SONGS_PATH
    safe_artist = _safe_name(artist)
    safe_title = _safe_name(title)
    folder_name = f"{safe_title}_{suffix}" if suffix else safe_title
    remote_dir = f"{base.rstrip('/')}/{safe_artist}/{folder_name}"

    _ = await mkdir(remote_dir)

    uploaded = 0
    for local_file in local_path.rglob("*"):
        if not local_file.is_file():
            continue
        relative = local_file.relative_to(local_path).as_posix()
        remote_file = f"{remote_dir}/{relative}"
        try:
            data = local_file.read_bytes()
            ok = await upload_file(remote_file, data)
            if ok:
                uploaded += 1
            else:
                logger.warning("⚠️ Failed to upload {}", remote_file)
        except Exception as e:
            logger.error("❌ Error uploading {} → {}: {}", local_file, remote_file, e)

    if uploaded == 0:
        logger.error("❌ No files were uploaded for {}", remote_dir)
        return None

    logger.info("⬆️ Uploaded {} file(s) to {}", uploaded, remote_dir)
    return remote_dir


async def write_remote_song_ini(
    remote_folder: str, song_data: dict[str, str | dict[str, str]]
) -> bool:
    """
    Write (overwrite) a ``song.ini`` on Nextcloud from *song_data*.

    *song_data* should contain ``title``, ``artist``, ``album`` and
    optionally a ``metadata`` dict.
    """
    config = configparser.ConfigParser()
    config.add_section("song")

    title: str = str(song_data.get("title", ""))
    artist: str = str(song_data.get("artist", ""))
    album: str = str(song_data.get("album", ""))
    config.set("song", "name", title)
    config.set("song", "artist", artist)
    config.set("song", "album", album)

    raw_metadata = song_data.get("metadata", {})
    metadata: dict[str, str] = {}
    if isinstance(raw_metadata, str):
        import json

        try:
            parsed = json.loads(raw_metadata)
            if isinstance(parsed, dict):
                metadata = {str(k): str(v) for k, v in parsed.items()}
        except (json.JSONDecodeError, TypeError):
            metadata = {}
    elif isinstance(raw_metadata, dict):
        metadata = {str(k): str(v) for k, v in raw_metadata.items()}

    for key, value in metadata.items():
        if value is not None:
            config.set("song", key, value)

    buf = io.StringIO()
    config.write(buf)
    ini_bytes = buf.getvalue().encode("utf-8")

    remote_ini = f"{remote_folder.rstrip('/')}/song.ini"
    return await upload_file(remote_ini, ini_bytes, "text/plain; charset=utf-8")


async def list_song_folder_files(remote_folder: str) -> list[WebDAVItem]:
    """
    List the files inside a single song folder on Nextcloud.

    Returns only files (not sub-directories).
    """
    items = await list_directory(remote_folder)
    return [i for i in items if not i.is_directory]


def _safe_name(value: str) -> str:
    """Sanitise a string for use as a remote directory name."""
    replacements = {
        "/": "_",
        "\\": "_",
        ":": "-",
        "*": "",
        "?": "",
        '"': "",
        "<": "",
        ">": "",
        "|": "",
    }
    result = value
    for old, new in replacements.items():
        result = result.replace(old, new)
    result = result.strip(" .")
    return result or "unknown"
