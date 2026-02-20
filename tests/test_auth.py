"""
Clone Hero Content Manager - Authentication Tests

Tests for the src/auth.py module. Validates:
- Session cookie creation and parsing (signed HMAC cookies)
- Cookie signature verification (tamper detection)
- Cookie expiry enforcement
- Credential verification (username/password matching)
- get_current_user / is_authenticated helpers
- auth_required middleware logic (public paths, protected paths)
- Login page rendering (with and without error messages)
- Session cookie set/clear on Response objects
- Edge cases: empty cookies, malformed cookies, expired sessions
"""

import hashlib
import hmac
import json
import os
import time
from unittest.mock import MagicMock, patch

import pytest

from src.auth import (
    _create_session_cookie,
    _parse_session_cookie,
    _sign,
    auth_required,
    clear_session_cookie,
    get_charter_name,
    get_current_user,
    is_authenticated,
    render_login_page,
    set_session_cookie,
    verify_credentials,
)
from src.config import (
    CHARTER_NAME,
    SECRET_KEY,
    SESSION_COOKIE_NAME,
    SESSION_MAX_AGE,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(cookies: dict | None = None, path: str = "/") -> MagicMock:
    """Create a mock FastAPI Request with optional cookies and URL path."""
    request = MagicMock()
    request.cookies = cookies or {}
    url_mock = MagicMock()
    url_mock.path = path
    request.url = url_mock
    return request


def _make_response() -> MagicMock:
    """Create a mock FastAPI Response with set_cookie and delete_cookie tracking."""
    response = MagicMock()
    response._cookies = {}

    def fake_set_cookie(**kwargs):
        response._cookies[kwargs.get("key", "")] = kwargs

    def fake_delete_cookie(**kwargs):
        key = kwargs.get("key", "")
        response._cookies.pop(key, None)

    response.set_cookie = MagicMock(side_effect=fake_set_cookie)
    response.delete_cookie = MagicMock(side_effect=fake_delete_cookie)
    return response


# ===========================================================================
# _sign
# ===========================================================================


class TestSign:
    """Test the HMAC signing helper."""

    def test_returns_hex_string(self):
        sig = _sign("hello")
        assert isinstance(sig, str)
        # HMAC-SHA256 hex digest is 64 chars
        assert len(sig) == 64

    def test_deterministic(self):
        """Same input should always produce the same signature."""
        sig1 = _sign("test payload")
        sig2 = _sign("test payload")
        assert sig1 == sig2

    def test_different_payloads_different_sigs(self):
        sig1 = _sign("payload_a")
        sig2 = _sign("payload_b")
        assert sig1 != sig2

    def test_empty_string(self):
        sig = _sign("")
        assert isinstance(sig, str)
        assert len(sig) == 64

    def test_unicode_payload(self):
        sig = _sign("héllo wörld ñ")
        assert isinstance(sig, str)
        assert len(sig) == 64

    def test_matches_manual_hmac(self):
        """Verify the signature matches a manually computed HMAC-SHA256."""
        payload = "test data"
        expected = hmac.new(
            SECRET_KEY.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        assert _sign(payload) == expected


# ===========================================================================
# _create_session_cookie / _parse_session_cookie
# ===========================================================================


class TestSessionCookie:
    """Test session cookie creation and parsing."""

    def test_create_returns_string(self):
        cookie = _create_session_cookie("testuser")
        assert isinstance(cookie, str)

    def test_create_contains_pipe_separator(self):
        cookie = _create_session_cookie("testuser")
        assert "|" in cookie

    def test_create_contains_username(self):
        cookie = _create_session_cookie("myuser")
        data_part = cookie.rsplit("|", 1)[0]
        payload = json.loads(data_part)
        assert payload["user"] == "myuser"

    def test_create_contains_charter(self):
        cookie = _create_session_cookie("user1")
        data_part = cookie.rsplit("|", 1)[0]
        payload = json.loads(data_part)
        assert payload["charter"] == CHARTER_NAME

    def test_create_contains_timestamp(self):
        before = int(time.time())
        cookie = _create_session_cookie("user1")
        after = int(time.time())
        data_part = cookie.rsplit("|", 1)[0]
        payload = json.loads(data_part)
        assert before <= payload["ts"] <= after

    def test_round_trip(self):
        """Create a cookie and parse it back — should recover the session."""
        cookie = _create_session_cookie("roundtrip_user")
        session = _parse_session_cookie(cookie)
        assert session is not None
        assert session["user"] == "roundtrip_user"
        assert session["charter"] == CHARTER_NAME

    def test_parse_valid_cookie(self):
        cookie = _create_session_cookie("valid_user")
        session = _parse_session_cookie(cookie)
        assert session is not None
        assert session["user"] == "valid_user"

    def test_parse_empty_string(self):
        session = _parse_session_cookie("")
        assert session is None

    def test_parse_none_like(self):
        session = _parse_session_cookie("")
        assert session is None

    def test_parse_no_pipe(self):
        session = _parse_session_cookie("no_pipe_separator")
        assert session is None

    def test_parse_invalid_json(self):
        session = _parse_session_cookie("not_json|abcdef1234567890")
        assert session is None

    def test_parse_tampered_data(self):
        """Modifying the data part should invalidate the signature."""
        cookie = _create_session_cookie("original_user")
        data_part, sig_part = cookie.rsplit("|", 1)
        # Tamper with the data
        tampered_data = data_part.replace("original_user", "hacker")
        tampered_cookie = f"{tampered_data}|{sig_part}"
        session = _parse_session_cookie(tampered_cookie)
        assert session is None

    def test_parse_tampered_signature(self):
        """Modifying the signature should invalidate the cookie."""
        cookie = _create_session_cookie("user1")
        data_part, sig_part = cookie.rsplit("|", 1)
        # Flip one char in the signature
        bad_sig = "0" + sig_part[1:] if sig_part[0] != "0" else "1" + sig_part[1:]
        tampered_cookie = f"{data_part}|{bad_sig}"
        session = _parse_session_cookie(tampered_cookie)
        assert session is None

    def test_parse_expired_cookie(self):
        """A cookie with an old timestamp should be rejected."""
        # Create cookie data with a timestamp way in the past
        old_ts = int(time.time()) - SESSION_MAX_AGE - 3600
        data = json.dumps({"user": "expired_user", "charter": "test", "ts": old_ts})
        sig = _sign(data)
        cookie = f"{data}|{sig}"
        session = _parse_session_cookie(cookie)
        assert session is None

    def test_parse_just_expired(self):
        """A cookie exactly at the expiry boundary should be rejected."""
        expired_ts = int(time.time()) - SESSION_MAX_AGE - 1
        data = json.dumps({"user": "edge_user", "charter": "test", "ts": expired_ts})
        sig = _sign(data)
        cookie = f"{data}|{sig}"
        session = _parse_session_cookie(cookie)
        assert session is None

    def test_parse_fresh_cookie(self):
        """A cookie created just now should be valid."""
        fresh_ts = int(time.time())
        data = json.dumps({"user": "fresh_user", "charter": "test", "ts": fresh_ts})
        sig = _sign(data)
        cookie = f"{data}|{sig}"
        session = _parse_session_cookie(cookie)
        assert session is not None
        assert session["user"] == "fresh_user"

    def test_multiple_pipes_in_data(self):
        """If the JSON data somehow contains pipes, rsplit should handle it."""
        # This shouldn't happen with normal JSON, but test robustness
        data = json.dumps(
            {"user": "pipe|user", "charter": "test", "ts": int(time.time())}
        )
        sig = _sign(data)
        cookie = f"{data}|{sig}"
        session = _parse_session_cookie(cookie)
        # Depending on the implementation, this may or may not work
        # since rsplit("|", 1) splits on the last pipe
        assert session is None or session.get("user") == "pipe|user"


# ===========================================================================
# verify_credentials
# ===========================================================================


class TestVerifyCredentials:
    """Test credential verification."""

    @patch("src.auth.AUTH_PASSWORD", "testpass123")
    @patch("src.auth.AUTH_USERNAME", "testuser")
    def test_correct_credentials(self):
        assert verify_credentials("testuser", "testpass123") is True

    @patch("src.auth.AUTH_PASSWORD", "testpass123")
    @patch("src.auth.AUTH_USERNAME", "testuser")
    def test_wrong_password(self):
        assert verify_credentials("testuser", "wrong_password") is False

    @patch("src.auth.AUTH_PASSWORD", "testpass123")
    @patch("src.auth.AUTH_USERNAME", "testuser")
    def test_wrong_username(self):
        assert verify_credentials("wronguser", "testpass123") is False

    @patch("src.auth.AUTH_PASSWORD", "testpass123")
    @patch("src.auth.AUTH_USERNAME", "testuser")
    def test_both_wrong(self):
        assert verify_credentials("wronguser", "wrongpass") is False

    @patch("src.auth.AUTH_PASSWORD", "")
    def test_empty_password_config_disables_auth(self):
        """When AUTH_PASSWORD is empty, verify_credentials always returns False."""
        assert verify_credentials("anyone", "anything") is False

    @patch("src.auth.AUTH_PASSWORD", "secret")
    @patch("src.auth.AUTH_USERNAME", "Admin")
    def test_case_insensitive_username(self):
        """Username comparison should be case-insensitive."""
        assert verify_credentials("admin", "secret") is True
        assert verify_credentials("ADMIN", "secret") is True
        assert verify_credentials("Admin", "secret") is True

    @patch("src.auth.AUTH_PASSWORD", "CaseSensitive")
    @patch("src.auth.AUTH_USERNAME", "user")
    def test_case_sensitive_password(self):
        """Password comparison should be case-sensitive."""
        assert verify_credentials("user", "CaseSensitive") is True
        assert verify_credentials("user", "casesensitive") is False
        assert verify_credentials("user", "CASESENSITIVE") is False

    @patch("src.auth.AUTH_PASSWORD", "pass")
    @patch("src.auth.AUTH_USERNAME", "user")
    def test_empty_username_input(self):
        assert verify_credentials("", "pass") is False

    @patch("src.auth.AUTH_PASSWORD", "pass")
    @patch("src.auth.AUTH_USERNAME", "user")
    def test_empty_password_input(self):
        assert verify_credentials("user", "") is False


# ===========================================================================
# get_current_user
# ===========================================================================


class TestGetCurrentUser:
    """Test the get_current_user() helper."""

    def test_returns_username_for_valid_session(self):
        cookie = _create_session_cookie("logged_in_user")
        request = _make_request(cookies={SESSION_COOKIE_NAME: cookie})
        user = get_current_user(request)
        assert user == "logged_in_user"

    def test_returns_none_for_no_cookie(self):
        request = _make_request(cookies={})
        user = get_current_user(request)
        assert user is None

    def test_returns_none_for_invalid_cookie(self):
        request = _make_request(cookies={SESSION_COOKIE_NAME: "garbage"})
        user = get_current_user(request)
        assert user is None

    def test_returns_none_for_expired_cookie(self):
        old_ts = int(time.time()) - SESSION_MAX_AGE - 100
        data = json.dumps({"user": "old_user", "charter": "test", "ts": old_ts})
        sig = _sign(data)
        cookie = f"{data}|{sig}"
        request = _make_request(cookies={SESSION_COOKIE_NAME: cookie})
        user = get_current_user(request)
        assert user is None

    def test_returns_none_for_tampered_cookie(self):
        cookie = _create_session_cookie("real_user")
        # Tamper
        tampered = cookie.replace("real_user", "fake_user")
        request = _make_request(cookies={SESSION_COOKIE_NAME: tampered})
        user = get_current_user(request)
        assert user is None

    def test_returns_none_for_empty_cookie_value(self):
        request = _make_request(cookies={SESSION_COOKIE_NAME: ""})
        user = get_current_user(request)
        assert user is None


# ===========================================================================
# get_charter_name
# ===========================================================================


class TestGetCharterName:
    """Test the get_charter_name() helper."""

    def test_returns_charter_from_session(self):
        cookie = _create_session_cookie("user1")
        request = _make_request(cookies={SESSION_COOKIE_NAME: cookie})
        charter = get_charter_name(request)
        assert charter == CHARTER_NAME

    def test_returns_default_when_no_session(self):
        request = _make_request(cookies={})
        charter = get_charter_name(request)
        assert charter == CHARTER_NAME

    def test_returns_default_when_invalid_session(self):
        request = _make_request(cookies={SESSION_COOKIE_NAME: "bad"})
        charter = get_charter_name(request)
        assert charter == CHARTER_NAME


# ===========================================================================
# is_authenticated
# ===========================================================================


class TestIsAuthenticated:
    """Test the is_authenticated() helper."""

    def test_true_for_valid_session(self):
        cookie = _create_session_cookie("auth_user")
        request = _make_request(cookies={SESSION_COOKIE_NAME: cookie})
        assert is_authenticated(request) is True

    def test_false_for_no_session(self):
        request = _make_request(cookies={})
        assert is_authenticated(request) is False

    def test_false_for_invalid_session(self):
        request = _make_request(cookies={SESSION_COOKIE_NAME: "invalid"})
        assert is_authenticated(request) is False

    def test_false_for_expired_session(self):
        old_ts = int(time.time()) - SESSION_MAX_AGE - 10
        data = json.dumps({"user": "old", "charter": "c", "ts": old_ts})
        sig = _sign(data)
        cookie = f"{data}|{sig}"
        request = _make_request(cookies={SESSION_COOKIE_NAME: cookie})
        assert is_authenticated(request) is False


# ===========================================================================
# set_session_cookie / clear_session_cookie
# ===========================================================================


class TestSetClearSessionCookie:
    """Test setting and clearing session cookies on responses."""

    def test_set_session_cookie_calls_set_cookie(self):
        response = _make_response()
        set_session_cookie(response, "new_user")
        response.set_cookie.assert_called_once()
        call_kwargs = response.set_cookie.call_args
        # Check key arguments
        if call_kwargs.kwargs:
            assert call_kwargs.kwargs.get("key") == SESSION_COOKIE_NAME
            assert call_kwargs.kwargs.get("httponly") is True
            assert call_kwargs.kwargs.get("path") == "/"
        else:
            # Positional args
            assert SESSION_COOKIE_NAME in str(call_kwargs)

    def test_set_session_cookie_value_is_parseable(self):
        response = _make_response()
        set_session_cookie(response, "parse_test_user")
        call_kwargs = response.set_cookie.call_args
        cookie_value = call_kwargs.kwargs.get("value", "")
        session = _parse_session_cookie(cookie_value)
        assert session is not None
        assert session["user"] == "parse_test_user"

    def test_set_session_cookie_max_age(self):
        response = _make_response()
        set_session_cookie(response, "user")
        call_kwargs = response.set_cookie.call_args
        assert call_kwargs.kwargs.get("max_age") == SESSION_MAX_AGE

    def test_set_session_cookie_samesite(self):
        response = _make_response()
        set_session_cookie(response, "user")
        call_kwargs = response.set_cookie.call_args
        assert call_kwargs.kwargs.get("samesite") == "lax"

    def test_clear_session_cookie_calls_delete(self):
        response = _make_response()
        clear_session_cookie(response)
        response.delete_cookie.assert_called_once()
        call_kwargs = response.delete_cookie.call_args
        if call_kwargs.kwargs:
            assert call_kwargs.kwargs.get("key") == SESSION_COOKIE_NAME


# ===========================================================================
# auth_required
# ===========================================================================


class TestAuthRequired:
    """Test the auth_required() middleware helper."""

    @patch("src.auth.AUTH_PASSWORD", "secret")
    def test_requires_auth_for_protected_path(self):
        """An unauthenticated request to a protected path should require auth."""
        request = _make_request(cookies={}, path="/songs")
        assert auth_required(request) is True

    @patch("src.auth.AUTH_PASSWORD", "secret")
    def test_does_not_require_auth_for_login(self):
        """The /login path should always be public."""
        request = _make_request(cookies={}, path="/login")
        assert auth_required(request) is False

    @patch("src.auth.AUTH_PASSWORD", "secret")
    def test_does_not_require_auth_for_health(self):
        """The /api/health path should always be public."""
        request = _make_request(cookies={}, path="/api/health")
        assert auth_required(request) is False

    @patch("src.auth.AUTH_PASSWORD", "secret")
    def test_does_not_require_auth_for_static(self):
        """Static file paths should be public."""
        request = _make_request(cookies={}, path="/static/css/style.css")
        assert auth_required(request) is False

    @patch("src.auth.AUTH_PASSWORD", "secret")
    def test_does_not_require_auth_for_favicon(self):
        request = _make_request(cookies={}, path="/favicon.ico")
        assert auth_required(request) is False

    @patch("src.auth.AUTH_PASSWORD", "")
    def test_auth_disabled_when_no_password(self):
        """When AUTH_PASSWORD is empty, auth should be disabled entirely."""
        request = _make_request(cookies={}, path="/songs")
        assert auth_required(request) is False

    @patch("src.auth.AUTH_PASSWORD", "secret")
    def test_authenticated_user_passes(self):
        """An authenticated request should not require auth."""
        cookie = _create_session_cookie("authed_user")
        request = _make_request(cookies={SESSION_COOKIE_NAME: cookie}, path="/songs")
        assert auth_required(request) is False

    @patch("src.auth.AUTH_PASSWORD", "secret")
    def test_expired_session_requires_auth(self):
        """An expired session should require re-authentication."""
        old_ts = int(time.time()) - SESSION_MAX_AGE - 100
        data = json.dumps({"user": "old", "charter": "c", "ts": old_ts})
        sig = _sign(data)
        cookie = f"{data}|{sig}"
        request = _make_request(cookies={SESSION_COOKIE_NAME: cookie}, path="/songs")
        assert auth_required(request) is True

    @patch("src.auth.AUTH_PASSWORD", "secret")
    def test_api_paths_require_auth(self):
        """API paths (other than /api/health) should require auth."""
        request = _make_request(cookies={}, path="/api/songs")
        assert auth_required(request) is True

    @patch("src.auth.AUTH_PASSWORD", "secret")
    def test_root_path_requires_auth(self):
        """The root path / should require auth."""
        request = _make_request(cookies={}, path="/")
        assert auth_required(request) is True

    @patch("src.auth.AUTH_PASSWORD", "secret")
    def test_generator_path_requires_auth(self):
        request = _make_request(cookies={}, path="/generator")
        assert auth_required(request) is True

    @patch("src.auth.AUTH_PASSWORD", "secret")
    def test_upload_path_requires_auth(self):
        request = _make_request(cookies={}, path="/upload")
        assert auth_required(request) is True


# ===========================================================================
# render_login_page
# ===========================================================================


class TestRenderLoginPage:
    """Test the login page HTML renderer."""

    def test_returns_html_response(self):
        response = render_login_page()
        assert response.status_code == 200
        assert "text/html" in response.media_type

    def test_contains_login_form(self):
        response = render_login_page()
        body = response.body.decode("utf-8")
        assert "<form" in body
        assert 'action="/login"' in body
        assert 'method="POST"' in body.lower() or 'method="post"' in body.lower()

    def test_contains_username_field(self):
        response = render_login_page()
        body = response.body.decode("utf-8")
        assert 'name="username"' in body

    def test_contains_password_field(self):
        response = render_login_page()
        body = response.body.decode("utf-8")
        assert 'name="password"' in body

    def test_contains_submit_button(self):
        response = render_login_page()
        body = response.body.decode("utf-8")
        assert 'type="submit"' in body

    def test_no_error_by_default(self):
        response = render_login_page()
        body = response.body.decode("utf-8")
        assert '<div class="error-msg">' not in body

    def test_shows_error_message(self):
        response = render_login_page(error="Invalid credentials")
        body = response.body.decode("utf-8")
        assert '<div class="error-msg">' in body
        assert "Invalid credentials" in body

    def test_prefills_username(self):
        response = render_login_page(prefill_user="myuser")
        body = response.body.decode("utf-8")
        assert 'value="myuser"' in body

    def test_empty_error_no_error_div(self):
        response = render_login_page(error="")
        body = response.body.decode("utf-8")
        assert '<div class="error-msg">' not in body

    def test_html_structure(self):
        response = render_login_page()
        body = response.body.decode("utf-8")
        assert "<!doctype html>" in body.lower() or "<!DOCTYPE html>" in body
        assert "</html>" in body
        assert "</head>" in body
        assert "</body>" in body

    def test_contains_branding(self):
        response = render_login_page()
        body = response.body.decode("utf-8")
        assert "Clone Hero" in body


# ===========================================================================
# Integration: Full login flow simulation
# ===========================================================================


class TestLoginFlowIntegration:
    """Integration tests simulating the full login/logout flow."""

    @patch("src.auth.AUTH_PASSWORD", "mypassword")
    @patch("src.auth.AUTH_USERNAME", "admin")
    def test_full_login_flow(self):
        """
        Simulate: unauthenticated → login → authenticated → logout → unauthenticated.
        """
        # Step 1: Unauthenticated request
        request = _make_request(cookies={}, path="/songs")
        assert auth_required(request) is True
        assert get_current_user(request) is None

        # Step 2: Verify correct credentials
        assert verify_credentials("admin", "mypassword") is True

        # Step 3: Set session cookie
        response = _make_response()
        set_session_cookie(response, "admin")
        cookie_value = response.set_cookie.call_args.kwargs.get("value", "")

        # Step 4: Authenticated request with the cookie
        request = _make_request(
            cookies={SESSION_COOKIE_NAME: cookie_value}, path="/songs"
        )
        assert auth_required(request) is False
        assert get_current_user(request) == "admin"
        assert is_authenticated(request) is True
        assert get_charter_name(request) == CHARTER_NAME

        # Step 5: Clear cookie (logout)
        response2 = _make_response()
        clear_session_cookie(response2)
        response2.delete_cookie.assert_called_once()

        # Step 6: Request without cookie is unauthenticated again
        request = _make_request(cookies={}, path="/songs")
        assert auth_required(request) is True
        assert get_current_user(request) is None

    @patch("src.auth.AUTH_PASSWORD", "secret")
    @patch("src.auth.AUTH_USERNAME", "user")
    def test_failed_login_flow(self):
        """Simulate a failed login attempt."""
        # Wrong password
        assert verify_credentials("user", "wrong") is False

        # The login page should show an error
        response = render_login_page(
            error="Invalid username or password", prefill_user="user"
        )
        body = response.body.decode("utf-8")
        assert "Invalid username or password" in body
        assert 'value="user"' in body

    @patch("src.auth.AUTH_PASSWORD", "")
    def test_auth_disabled_flow(self):
        """When auth is disabled, everything should be accessible."""
        request = _make_request(cookies={}, path="/songs")
        assert auth_required(request) is False

        request = _make_request(cookies={}, path="/")
        assert auth_required(request) is False

        request = _make_request(cookies={}, path="/generator")
        assert auth_required(request) is False

        request = _make_request(cookies={}, path="/api/songs")
        assert auth_required(request) is False

    @patch("src.auth.AUTH_PASSWORD", "pass")
    @patch("src.auth.AUTH_USERNAME", "user")
    def test_session_survives_page_navigation(self):
        """The same session cookie should work across different pages."""
        cookie = _create_session_cookie("user")

        pages = ["/", "/songs", "/generator", "/upload", "/browser", "/chart-viewer"]
        for page in pages:
            request = _make_request(cookies={SESSION_COOKIE_NAME: cookie}, path=page)
            assert auth_required(request) is False, f"Auth required for {page}"
            assert get_current_user(request) == "user", f"User not found for {page}"


# ===========================================================================
# Edge cases and security
# ===========================================================================


class TestSecurityEdgeCases:
    """Test security-relevant edge cases."""

    def test_cookie_with_only_pipe(self):
        session = _parse_session_cookie("|")
        assert session is None

    def test_cookie_with_many_pipes(self):
        session = _parse_session_cookie("|||||||")
        assert session is None

    def test_cookie_with_valid_json_but_no_user_key(self):
        data = json.dumps({"not_user": "value", "ts": int(time.time())})
        sig = _sign(data)
        cookie = f"{data}|{sig}"
        session = _parse_session_cookie(cookie)
        if session is not None:
            # Should at least have no user
            assert session.get("user") is None or session.get("user") == ""

    def test_cookie_with_future_timestamp(self):
        """A cookie with a far-future timestamp should still be valid
        (it hasn't expired yet)."""
        future_ts = int(time.time()) + 1000
        data = json.dumps({"user": "future_user", "charter": "c", "ts": future_ts})
        sig = _sign(data)
        cookie = f"{data}|{sig}"
        session = _parse_session_cookie(cookie)
        # The cookie is "from the future" but not expired — should be valid
        assert session is not None
        assert session["user"] == "future_user"

    def test_cookie_with_negative_timestamp(self):
        """A cookie with a negative timestamp should be expired."""
        data = json.dumps({"user": "neg_ts", "charter": "c", "ts": -1})
        sig = _sign(data)
        cookie = f"{data}|{sig}"
        session = _parse_session_cookie(cookie)
        assert session is None

    def test_cookie_with_zero_timestamp(self):
        """A cookie with timestamp 0 (epoch) should be expired."""
        data = json.dumps({"user": "epoch", "charter": "c", "ts": 0})
        sig = _sign(data)
        cookie = f"{data}|{sig}"
        session = _parse_session_cookie(cookie)
        assert session is None

    def test_very_long_username(self):
        """A very long username should still work in cookies."""
        long_user = "a" * 1000
        cookie = _create_session_cookie(long_user)
        session = _parse_session_cookie(cookie)
        assert session is not None
        assert session["user"] == long_user

    def test_special_chars_in_username(self):
        """Special characters in the username should round-trip."""
        special = "user<>\"'&;/\\|"
        cookie = _create_session_cookie(special)
        session = _parse_session_cookie(cookie)
        assert session is not None
        assert session["user"] == special

    def test_unicode_username(self):
        cookie = _create_session_cookie("usér_ñaме")
        session = _parse_session_cookie(cookie)
        assert session is not None
        assert session["user"] == "usér_ñaме"

    @patch("src.auth.AUTH_PASSWORD", "p@$$w0rd!#%^&*()")
    @patch("src.auth.AUTH_USERNAME", "user")
    def test_special_chars_in_password(self):
        """Special characters in the password should be handled correctly."""
        assert verify_credentials("user", "p@$$w0rd!#%^&*()") is True
        assert verify_credentials("user", "p@$$w0rd!#%^&*()x") is False

    def test_hmac_timing_safety(self):
        """
        verify_credentials uses hmac.compare_digest which is timing-safe.
        We just verify it's being used (indirectly, by checking the function
        reference).
        """
        import inspect

        source = inspect.getsource(verify_credentials)
        assert "compare_digest" in source
