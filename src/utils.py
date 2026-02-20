"""
Clone Hero Content Manager - Shared Utilities

Common helpers used across multiple modules to avoid duplication.
"""

import json
from typing import Any, Dict


def sanitize_filename(name: str) -> str:
    """
    Sanitize a string for use as a filename or remote directory name.

    Removes or replaces characters that are problematic on most file systems
    and in URLs.
    """
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
    result = name
    for old, new in replacements.items():
        result = result.replace(old, new)

    # Strip leading/trailing whitespace and dots
    result = result.strip(" .")

    return result or "unknown"


def parse_metadata_json(raw: Any) -> Dict[str, Any]:
    """
    Safely parse metadata that may be a JSON string or already a dict.

    Returns a dict in all cases (empty dict on parse failure).
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}
