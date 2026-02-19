"""
Clone Hero Content Manager - Nextcloud WebDAV Client

Provides async methods to browse, upload, download, and manage files
on a Nextcloud instance via the WebDAV protocol.

Uses httpx for async HTTP operations with Basic Auth against the
Nextcloud WebDAV endpoint.
"""

import io
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional
from urllib.parse import quote, unquote, urlparse

import httpx
from loguru import logger

from src.config import (
    NEXTCLOUD_PASSWORD,
    NEXTCLOUD_URL,
    NEXTCLOUD_USERNAME,
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
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def extension(self) -> str:
        return PurePosixPath(self.name).suffix.lower()

    def to_dict(self) -> Dict[str, Any]:
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


def _parse_propfind_response(xml_text: str, base_path: str = "/") -> List[WebDAVItem]:
    """Parse the XML response from a PROPFIND request into WebDAVItem list."""
    items: List[WebDAVItem] = []

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
async def check_connection() -> Dict[str, Any]:
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


async def list_directory(remote_path: str = "/") -> List[WebDAVItem]:
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


async def download_file(remote_path: str) -> Optional[bytes]:
    """
    Download a file from Nextcloud and return its content as bytes.
    Returns None on failure.
    """
    if not _is_configured():
        logger.warning("⚠️ Nextcloud WebDAV not configured")
        return None

    url = _build_url(remote_path)
    try:
        async with httpx.AsyncClient(auth=_auth(), timeout=120.0) as client:
            response = await client.get(url)
            if response.status_code == 200:
                logger.info(
                    f"⬇️ Downloaded {remote_path} ({len(response.content)} bytes)"
                )
                return response.content
            else:
                logger.error(
                    f"❌ Download failed ({response.status_code}): {remote_path}"
                )
                return None
    except httpx.HTTPError as e:
        logger.error(f"❌ Download error for {remote_path}: {e}")
        return None
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
                        f.write(chunk)

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
        await mkdir(parent)

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
    file_obj: io.IOBase,
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
        await mkdir(parent)

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
                    "Overwrite": "F",
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


async def get_file_info(remote_path: str) -> Optional[WebDAVItem]:
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
