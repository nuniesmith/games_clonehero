"""
Clone Hero Content Manager - Simple Session Auth

Single-user authentication using signed cookies.  The username and password
are read from environment variables (AUTH_USERNAME / AUTH_PASSWORD).

Usage:
    - Mount `login_page` and `login_post` on the pages router.
    - Add `require_auth` as a dependency on all protected routes,
      or use `auth_middleware` to protect the entire app.
    - Call `get_current_user(request)` to retrieve the logged-in username.
"""

import hashlib
import hmac
import json
import time
from typing import Any

from fastapi import Request, Response
from fastapi.responses import HTMLResponse

from src.config import (
    AUTH_PASSWORD,
    AUTH_USERNAME,
    CHARTER_NAME,
    SECRET_KEY,
    SESSION_COOKIE_NAME,
    SESSION_MAX_AGE,
)

# ---------------------------------------------------------------------------
# Cookie helpers
# ---------------------------------------------------------------------------


def _sign(payload: str) -> str:
    """Create an HMAC-SHA256 signature for a payload string."""
    return hmac.new(
        SECRET_KEY.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _create_session_cookie(username: str) -> str:
    """Create a signed session cookie value."""
    data = json.dumps(
        {
            "user": username,
            "charter": CHARTER_NAME,
            "ts": int(time.time()),
        }
    )
    sig = _sign(data)
    return f"{data}|{sig}"


def _parse_session_cookie(cookie_value: str) -> dict[str, Any] | None:
    """Parse and verify a session cookie.  Returns the session dict or None."""
    if not cookie_value or "|" not in cookie_value:
        return None

    try:
        data_part, sig_part = cookie_value.rsplit("|", 1)
        expected_sig = _sign(data_part)

        if not hmac.compare_digest(sig_part, expected_sig):
            return None

        session = json.loads(data_part)

        # Check expiry
        created = session.get("ts", 0)
        if time.time() - created > SESSION_MAX_AGE:
            return None

        return session
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_current_user(request: Request) -> str | None:
    """Return the logged-in username, or None if not authenticated."""
    cookie = request.cookies.get(SESSION_COOKIE_NAME, "")
    session = _parse_session_cookie(cookie)
    if session:
        return session.get("user")
    return None


def get_charter_name(request: Request) -> str:
    """Return the charter name for the current session."""
    cookie = request.cookies.get(SESSION_COOKIE_NAME, "")
    session = _parse_session_cookie(cookie)
    if session:
        return session.get("charter", CHARTER_NAME)
    return CHARTER_NAME


def is_authenticated(request: Request) -> bool:
    """Check whether the current request has a valid session."""
    return get_current_user(request) is not None


def set_session_cookie(response: Response, username: str) -> None:
    """Set the signed session cookie on a response."""
    value = _create_session_cookie(username)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=value,
        max_age=SESSION_MAX_AGE,
        httponly=True,
        samesite="lax",
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    """Remove the session cookie."""
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
    )


# ---------------------------------------------------------------------------
# Auth check — returns True if request should be allowed through
# ---------------------------------------------------------------------------

# Paths that don't require authentication
PUBLIC_PATHS = {
    "/login",
    "/api/health",
    "/static",
}


def _is_public(path: str) -> bool:
    """Return True if the path does not require authentication."""
    for pub in PUBLIC_PATHS:
        if path == pub or path.startswith(pub + "/") or path.startswith(pub + "?"):
            return True
    # Allow favicon and similar
    if path in ("/favicon.ico", "/robots.txt"):
        return True
    return False


def auth_required(request: Request) -> bool:
    """
    Return True if this request requires auth and the user is NOT logged in.
    (i.e. the request should be redirected to /login.)

    If AUTH_PASSWORD is empty, auth is disabled entirely (always returns False).
    """
    if not AUTH_PASSWORD:
        # Auth disabled — no password configured
        return False

    if _is_public(request.url.path):
        return False

    return not is_authenticated(request)


def verify_credentials(username: str, password: str) -> bool:
    """Verify login credentials against the configured values."""
    if not AUTH_PASSWORD:
        return False

    user_ok = hmac.compare_digest(username.lower(), AUTH_USERNAME.lower())
    pass_ok = hmac.compare_digest(password, AUTH_PASSWORD)
    return user_ok and pass_ok


# ---------------------------------------------------------------------------
# Login page HTML
# ---------------------------------------------------------------------------

LOGIN_PAGE_HTML = """\
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Login — Clone Hero Manager</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #ecf0f1;
        }

        .login-card {
            background: rgba(44, 62, 80, 0.92);
            border: 1px solid rgba(52, 152, 219, 0.3);
            border-radius: 16px;
            padding: 48px 40px 40px;
            width: 100%%;
            max-width: 420px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(10px);
        }

        .login-header {
            text-align: center;
            margin-bottom: 32px;
        }

        .login-header .icon {
            font-size: 3em;
            margin-bottom: 12px;
            display: block;
        }

        .login-header h1 {
            font-size: 1.5em;
            color: #3498db;
            margin-bottom: 6px;
        }

        .login-header p {
            color: #95a5a6;
            font-size: 0.9em;
        }

        .form-group {
            margin-bottom: 20px;
        }

        .form-group label {
            display: block;
            margin-bottom: 6px;
            color: #bdc3c7;
            font-size: 0.9em;
            font-weight: 600;
        }

        .form-group input {
            width: 100%%;
            padding: 12px 16px;
            background: rgba(255, 255, 255, 0.08);
            border: 2px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            color: #ecf0f1;
            font-size: 1em;
            transition: border-color 0.3s, background 0.3s;
        }

        .form-group input:focus {
            outline: none;
            border-color: #3498db;
            background: rgba(255, 255, 255, 0.12);
        }

        .form-group input::placeholder {
            color: rgba(255, 255, 255, 0.3);
        }

        .login-btn {
            width: 100%%;
            padding: 14px;
            background: linear-gradient(135deg, #3498db, #2980b9);
            border: none;
            border-radius: 8px;
            color: white;
            font-size: 1.05em;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            margin-top: 8px;
        }

        .login-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(52, 152, 219, 0.4);
        }

        .login-btn:active {
            transform: translateY(0);
        }

        .error-msg {
            background: rgba(231, 76, 60, 0.15);
            border: 1px solid rgba(231, 76, 60, 0.3);
            color: #e74c3c;
            padding: 10px 14px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-size: 0.9em;
            text-align: center;
        }

        .footer-note {
            text-align: center;
            margin-top: 24px;
            color: #7f8c8d;
            font-size: 0.8em;
        }
    </style>
</head>
<body>
    <div class="login-card">
        <div class="login-header">
            <span class="icon">🎸</span>
            <h1>Clone Hero Manager</h1>
            <p>Sign in to manage your song library</p>
        </div>

        %(error_html)s

        <form method="POST" action="/login">
            <div class="form-group">
                <label for="username">Username</label>
                <input
                    type="text"
                    id="username"
                    name="username"
                    placeholder="Enter your username"
                    autocomplete="username"
                    value="%(prefill_user)s"
                    required
                    autofocus
                />
            </div>

            <div class="form-group">
                <label for="password">Password</label>
                <input
                    type="password"
                    id="password"
                    name="password"
                    placeholder="Enter your password"
                    autocomplete="current-password"
                    required
                />
            </div>

            <button type="submit" class="login-btn">🔓 Sign In</button>
        </form>

        <p class="footer-note">🎵 Generate &bull; Sync &bull; Play</p>
    </div>
</body>
</html>
"""


def render_login_page(error: str = "", prefill_user: str = "") -> HTMLResponse:
    """Render the login page with an optional error message."""
    error_html = ""
    if error:
        error_html = f'<div class="error-msg">❌ {error}</div>'

    html = LOGIN_PAGE_HTML % {
        "error_html": error_html,
        "prefill_user": prefill_user,
    }
    return HTMLResponse(content=html)
