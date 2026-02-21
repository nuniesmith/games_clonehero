"""
Clone Hero Content Manager - Metadata Lookup Service

Looks up song metadata from free public APIs:
- MusicBrainz (https://musicbrainz.org/doc/MusicBrainz_API) for song info
- Cover Art Archive (https://coverartarchive.org) for album artwork
- Provides filename parsing helpers to extract artist/title from filenames

All lookups are best-effort: failures are logged and empty results returned
so the generator pipeline is never blocked by external API issues.
"""

import asyncio
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import httpx
from loguru import logger

from src.config import (
    COVER_ART_ARCHIVE_URL,
    MUSICBRAINZ_ENABLED,
    MUSICBRAINZ_USER_AGENT,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MUSICBRAINZ_API_URL = "https://musicbrainz.org/ws/2"

# MusicBrainz asks for ≤1 request/second; we add a small buffer
_MB_RATE_LIMIT = 1.1

# Common filename separators between artist and title
_ARTIST_TITLE_SEPARATORS = [
    " - ",  # Most common: "Artist - Title"
    " – ",  # En-dash variant
    " — ",  # Em-dash variant
    " _ ",  # Underscore with spaces
]

# Characters to strip/replace when cleaning filenames
_STRIP_PATTERNS = [
    (r"\(Official\s*(Music\s*)?Video\)", ""),
    (r"\(Official\s*Audio\)", ""),
    (r"\(Lyric\s*Video\)", ""),
    (r"\(Audio\)", ""),
    (r"\(Visualizer\)", ""),
    (r"\[Official\s*(Music\s*)?Video\]", ""),
    (r"\[Official\s*Audio\]", ""),
    (r"\[Lyric\s*Video\]", ""),
    (r"\[Audio\]", ""),
    (r"\[Visualizer\]", ""),
    (r"\(feat\.?\s*[^)]+\)", ""),  # (feat. Someone)
    (r"\[feat\.?\s*[^\]]+\]", ""),  # [feat. Someone]
    (r"\(ft\.?\s*[^)]+\)", ""),  # (ft. Someone)
    (r"\[ft\.?\s*[^\]]+\]", ""),  # [ft. Someone]
    (r"\d{3,4}\s*kbps", ""),  # bitrate tags
    (r"\(\d{4}\)", ""),  # (2023) year tags
    (r"\[\d{4}\]", ""),  # [2023] year tags
]

# Words that should stay lowercase in title case (unless first/last)
_LOWERCASE_WORDS = {
    "a",
    "an",
    "the",
    "and",
    "but",
    "or",
    "nor",
    "for",
    "yet",
    "so",
    "in",
    "on",
    "at",
    "to",
    "by",
    "of",
    "up",
    "as",
    "is",
    "if",
    "it",
    "vs",
    "vs.",
}


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------


def clean_name(raw: str) -> str:
    """
    Clean a raw string (filename stem, metadata field, etc.) into a
    human-readable title.

    - Replaces underscores and hyphens used as word separators with spaces
    - Collapses multiple spaces
    - Strips common video/audio tags
    - Applies smart title case
    """
    text = raw.strip()

    # Apply strip patterns (remove tags like "Official Video", etc.)
    for pattern, replacement in _STRIP_PATTERNS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Replace underscores and standalone hyphens with spaces
    # But preserve hyphens inside words (e.g. "re-enter")
    text = text.replace("_", " ")
    # Replace " - " style separators that remain after other processing
    # (we don't do this blindly — only isolated dashes)
    text = re.sub(r"\s*-\s+|\s+-\s*", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Apply smart title case
    text = _smart_title_case(text)

    return text


def _smart_title_case(text: str) -> str:
    """
    Apply title case that respects common English articles/prepositions
    and preserves intentional ALL-CAPS words (e.g. "DNA", "OK").
    """
    words = text.split()
    result = []
    for i, word in enumerate(words):
        # Preserve all-caps words of 2+ chars (likely acronyms)
        if len(word) >= 2 and word.isupper():
            result.append(word)
        # First and last word always capitalized
        elif i == 0 or i == len(words) - 1:
            result.append(word.capitalize())
        # Lowercase articles/prepositions (unless after colon)
        elif word.lower() in _LOWERCASE_WORDS and (
            i == 0 or not result[-1].endswith(":")
        ):
            result.append(word.lower())
        else:
            result.append(word.capitalize())
    return " ".join(result)


def parse_filename(filename: str) -> Dict[str, str]:
    """
    Parse an audio filename to extract artist and song title.

    Handles common patterns:
    - "Artist - Song Title.mp3"
    - "Artist_-_Song_Title.flac"
    - "Artist — Song Title.wav"
    - "Song Title.ogg"  (no artist detected)
    - "01 - Song Title.mp3" (track number prefix)
    - "01. Song Title.mp3" (track number prefix)
    - "05-Artist-Song Title.mp3" (track number with dash)

    Returns a dict with 'song_name' and 'artist' keys.
    Both values are cleaned and title-cased.
    """
    stem = Path(filename).stem

    # Step 1: Strip leading track-number prefixes.
    # Handles "05-...", "05. ...", "05 ...", "5 - ..." etc.
    track_match = re.match(r"^\d{1,3}\s*[-\.]\s*", stem)
    if track_match:
        stem = stem[track_match.end() :]

    # Step 2: Try each separator pattern (highest-confidence first)
    artist = ""
    title = stem

    for sep in _ARTIST_TITLE_SEPARATORS:
        if sep in stem:
            parts = stem.split(sep, 1)
            if len(parts) == 2:
                candidate_artist = parts[0].strip()
                candidate_title = parts[1].strip()

                # Skip if either side is empty after stripping
                if not candidate_artist or not candidate_title:
                    continue

                # Check if the "artist" part is just a track number
                # (handles edge case where track number wasn't caught above)
                if re.match(r"^\d{1,3}$", candidate_artist):
                    # "01 - Song Title" → no artist, just title
                    title = candidate_title
                else:
                    artist = candidate_artist
                    title = candidate_title
                break

    # Step 3: If no spaced separator matched, try bare "-" but only when at
    # least one side contains a space (multi-word) so we don't split
    # hyphenated single words like "Re-Enter".
    if not artist and "-" in title:
        parts = title.split("-", 1)
        if len(parts) == 2:
            left = parts[0].strip()
            right = parts[1].strip()
            if left and right and (" " in left or " " in right):
                if not re.match(r"^\d{1,3}$", left):
                    artist = left
                    title = right
                else:
                    title = right

    # Step 4: If still no artist and no separator worked, check for
    # remaining "01. Title" or "01 Title" patterns (fallback)
    if not artist:
        match = re.match(r"^\d{1,3}\.?\s+(.+)$", title)
        if match:
            title = match.group(1)

    # Clean both parts
    title = clean_name(title)
    artist = clean_name(artist) if artist else ""

    return {
        "song_name": title,
        "artist": artist,
    }


# ---------------------------------------------------------------------------
# MusicBrainz lookups
# ---------------------------------------------------------------------------


async def lookup_song_metadata(
    title: str,
    artist: str = "",
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """
    Look up song metadata from MusicBrainz by title and optional artist.

    Returns a dict with any found fields:
    - title: str (canonical title)
    - artist: str (canonical artist)
    - album: str
    - year: str
    - genre: str (first tag if any)
    - genres: list[str] (all tags)
    - musicbrainz_id: str (recording MBID)
    - release_mbid: str (release MBID, for cover art lookup)
    - release_group_mbid: str

    Returns empty dict on failure or if MusicBrainz is disabled.
    """
    if not MUSICBRAINZ_ENABLED:
        logger.debug("MusicBrainz lookup disabled")
        return {}

    if not title:
        return {}

    try:
        # Build the search query
        query_parts = [f'recording:"{_mb_escape(title)}"']
        if artist:
            query_parts.append(f'artist:"{_mb_escape(artist)}"')

        query = " AND ".join(query_parts)

        headers = {
            "User-Agent": MUSICBRAINZ_USER_AGENT,
            "Accept": "application/json",
        }

        async with httpx.AsyncClient(timeout=timeout) as client:
            url = f"{MUSICBRAINZ_API_URL}/recording"
            params = {
                "query": query,
                "limit": "5",
                "fmt": "json",
            }

            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            recordings = data.get("recordings", [])
            if not recordings:
                logger.debug("No MusicBrainz results for '{}' by '{}'", title, artist)
                return {}

            # Pick the best match (first result with highest score)
            recording = recordings[0]
            result: Dict[str, Any] = {}

            result["title"] = recording.get("title", title)
            result["musicbrainz_id"] = recording.get("id", "")

            # Artist
            artist_credits = recording.get("artist-credit", [])
            if artist_credits:
                canonical_artist = ""
                for credit in artist_credits:
                    name = credit.get("name", "")
                    joinphrase = credit.get("joinphrase", "")
                    canonical_artist += name + joinphrase
                if canonical_artist:
                    result["artist"] = canonical_artist

            # Release (album) info
            releases = recording.get("releases", [])
            if releases:
                # Prefer the first release with a date
                best_release = releases[0]
                for rel in releases:
                    if rel.get("date"):
                        best_release = rel
                        break

                result["album"] = best_release.get("title", "")
                result["release_mbid"] = best_release.get("id", "")

                # Extract year from date
                date_str = best_release.get("date", "")
                if date_str:
                    year_match = re.match(r"(\d{4})", date_str)
                    if year_match:
                        result["year"] = year_match.group(1)

                # Release group for cover art
                release_group = best_release.get("release-group", {})
                if release_group:
                    result["release_group_mbid"] = release_group.get("id", "")

            # Tags / genres
            tags = recording.get("tags", [])
            if tags:
                sorted_tags = sorted(
                    tags, key=lambda t: t.get("count", 0), reverse=True
                )
                genre_names = [t["name"] for t in sorted_tags if t.get("name")]
                if genre_names:
                    result["genre"] = genre_names[0].title()
                    result["genres"] = [g.title() for g in genre_names[:5]]

            # If we got a release but no tags, try the release group for tags
            if not tags and result.get("release_group_mbid"):
                await asyncio.sleep(_MB_RATE_LIMIT)
                genre_info = await _lookup_release_group_tags(
                    client, result["release_group_mbid"], headers
                )
                if genre_info:
                    result.update(genre_info)

            logger.info(
                "🔍 MusicBrainz match: '{}' by '{}' → album='{}', year='{}'",
                result.get("title", "?"),
                result.get("artist", "?"),
                result.get("album", "?"),
                result.get("year", "?"),
            )
            return result

    except httpx.TimeoutException:
        logger.warning("⏱️ MusicBrainz lookup timed out for '{}' by '{}'", title, artist)
        return {}
    except httpx.HTTPStatusError as e:
        logger.warning(
            "⚠️ MusicBrainz HTTP error: {} {}",
            e.response.status_code,
            e.response.text[:200],
        )
        return {}
    except Exception as e:
        logger.warning("⚠️ MusicBrainz lookup error: {}", e)
        return {}


async def _lookup_release_group_tags(
    client: httpx.AsyncClient,
    release_group_mbid: str,
    headers: Dict[str, str],
) -> Dict[str, Any]:
    """Fetch genre tags from a release group."""
    try:
        url = f"{MUSICBRAINZ_API_URL}/release-group/{release_group_mbid}"
        params = {"inc": "tags", "fmt": "json"}
        resp = await client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        tags = data.get("tags", [])
        if tags:
            sorted_tags = sorted(tags, key=lambda t: t.get("count", 0), reverse=True)
            genre_names = [t["name"] for t in sorted_tags if t.get("name")]
            if genre_names:
                return {
                    "genre": genre_names[0].title(),
                    "genres": [g.title() for g in genre_names[:5]],
                }
    except Exception as e:
        logger.debug("Failed to fetch release group tags: {}", e)

    return {}


def _mb_escape(text: str) -> str:
    """Escape special Lucene query characters for MusicBrainz search."""
    # MusicBrainz uses Lucene query syntax
    specials = r'+-&|!(){}[]^"~*?:\/'
    escaped = ""
    for ch in text:
        if ch in specials:
            escaped += f"\\{ch}"
        else:
            escaped += ch
    return escaped


# ---------------------------------------------------------------------------
# Cover Art Archive
# ---------------------------------------------------------------------------


async def lookup_cover_art(
    release_mbid: str = "",
    release_group_mbid: str = "",
    timeout: float = 15.0,
) -> Optional[bytes]:
    """
    Download album cover art from the Cover Art Archive.

    Tries the release first, then falls back to the release group.
    Returns the image bytes (JPEG/PNG) or None on failure.
    """
    if not release_mbid and not release_group_mbid:
        return None

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        # Try release-level cover first
        if release_mbid:
            art = await _fetch_cover(client, "release", release_mbid)
            if art:
                return art

        # Fall back to release-group level
        if release_group_mbid:
            art = await _fetch_cover(client, "release-group", release_group_mbid)
            if art:
                return art

    return None


async def _fetch_cover(
    client: httpx.AsyncClient,
    entity_type: str,
    mbid: str,
) -> Optional[bytes]:
    """Fetch the front cover image from the Cover Art Archive."""
    url = f"{COVER_ART_ARCHIVE_URL}/{entity_type}/{mbid}/front-500"
    try:
        resp = await client.get(url)
        if resp.status_code == 200:
            content_type = resp.headers.get("content-type", "")
            if "image" in content_type or len(resp.content) > 1000:
                logger.info(
                    "🎨 Downloaded cover art from CAA ({}/{}, {} bytes)",
                    entity_type,
                    mbid[:8],
                    len(resp.content),
                )
                return resp.content
        elif resp.status_code == 404:
            logger.debug("No cover art at CAA {}/{}", entity_type, mbid[:8])
        else:
            logger.debug(
                "CAA returned {} for {}/{}",
                resp.status_code,
                entity_type,
                mbid[:8],
            )
    except Exception as e:
        logger.debug("Cover art fetch failed: {}", e)

    return None


# ---------------------------------------------------------------------------
# High-level helper: look up everything for a song
# ---------------------------------------------------------------------------


async def lookup_all(
    title: str,
    artist: str = "",
    download_art: bool = True,
) -> Dict[str, Any]:
    """
    Look up all available metadata for a song.

    Returns a dict with:
    - All fields from lookup_song_metadata()
    - 'cover_art_bytes': bytes or None (if download_art is True)
    - 'lookup_source': 'musicbrainz' or '' if no match

    This is the main entry point for the generator pipeline.
    """
    result: Dict[str, Any] = {
        "lookup_source": "",
        "cover_art_bytes": None,
    }

    # Step 1: MusicBrainz metadata
    mb_data = await lookup_song_metadata(title, artist)
    if mb_data:
        result.update(mb_data)
        result["lookup_source"] = "musicbrainz"

        # Step 2: Cover art (if we got a release ID)
        if download_art:
            release_mbid = mb_data.get("release_mbid", "")
            rg_mbid = mb_data.get("release_group_mbid", "")
            if release_mbid or rg_mbid:
                # Small delay to respect rate limits
                await asyncio.sleep(_MB_RATE_LIMIT)
                art_bytes = await lookup_cover_art(release_mbid, rg_mbid)
                result["cover_art_bytes"] = art_bytes

    return result
