"""Auth gate for hosted Distil. PRD FR14; ARCHITECTURE.md §8; TESTING T-A1..A4.

Rules (non-optional when hosting):
* **Fail closed:** if ``DISTIL_PUBLIC=true`` but ``DISTIL_AUTH_SECRET`` is unset, the app
  refuses to start (T-A1) — generating a public domain without auth would expose the API key
  and knowledge base.
* **Require auth on data routes** in public mode (T-A2); ``/health``, ``/login`` stay open.
* **Browser sessions:** a successful POST to ``/login`` sets a signed cookie so the browser
  doesn't need to send a Bearer header on every request.
* **Localhost convenience:** when not public, routes are reachable without the secret (T-A3).
* **Bind ``0.0.0.0:$PORT``** for platforms that inject the port (T-A4).
"""

from __future__ import annotations

import hashlib
import hmac
import os

from fastapi import Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

_OPEN_PATHS = {"/health", "/login"}
_COOKIE_NAME = "distil_session"


def is_public() -> bool:
    return os.environ.get("DISTIL_PUBLIC", "false").strip().lower() == "true"


def auth_secret() -> str | None:
    return os.environ.get("DISTIL_AUTH_SECRET") or None


def assert_startup_safe() -> None:
    """Fail closed: refuse to start in public mode without a secret (T-A1)."""
    if is_public() and not auth_secret():
        raise RuntimeError(
            "Refusing to start: DISTIL_PUBLIC=true but DISTIL_AUTH_SECRET is not set. "
            "Hosting without an auth secret would expose your API key and knowledge base. "
            "Set DISTIL_AUTH_SECRET before generating a public domain."
        )


def _sign(value: str, secret: str) -> str:
    """HMAC-SHA256 signature for cookie integrity."""
    return hmac.new(secret.encode(), value.encode(), hashlib.sha256).hexdigest()


def _make_session_cookie(secret: str) -> str:
    """Signed session token: 'authenticated:<hmac>'."""
    payload = "authenticated"
    sig = _sign(payload, secret)
    return f"{payload}:{sig}"


def _verify_session_cookie(cookie: str, secret: str) -> bool:
    """Return True if the cookie was signed with our secret."""
    try:
        payload, sig = cookie.rsplit(":", 1)
    except ValueError:
        return False
    expected = _sign(payload, secret)
    return hmac.compare_digest(expected, sig) and payload == "authenticated"


def request_is_authorized(request: Request) -> bool:
    """In public mode, accept a valid session cookie OR a matching Bearer token.
    On localhost, always allowed (T-A3)."""
    if not is_public():
        return True
    secret = auth_secret()
    if not secret:
        return False  # fail closed (defensive; startup check should have caught this)

    # 1. Session cookie (browser flow)
    cookie = request.cookies.get(_COOKIE_NAME)
    if cookie and _verify_session_cookie(cookie, secret):
        return True

    # 2. Bearer token (API / CLI flow)
    auth_header = request.headers.get("Authorization")
    if auth_header:
        token = auth_header.removeprefix("Bearer ").strip()
        if hmac.compare_digest(token, secret):
            return True

    return False


def path_is_open(path: str) -> bool:
    return path in _OPEN_PATHS


# ---- Login / logout HTML views -------------------------------------------------------

_LOGIN_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Distil — Login</title>
  <style>
    body {{ font-family: system-ui, sans-serif; display: flex; align-items: center;
            justify-content: center; min-height: 100vh; margin: 0; background: #f5f5f5; }}
    .card {{ background: white; padding: 2rem 2.5rem; border-radius: 8px;
             box-shadow: 0 2px 8px rgba(0,0,0,.12); min-width: 320px; }}
    h1 {{ margin: 0 0 1.5rem; font-size: 1.4rem; }}
    input {{ width: 100%; padding: .6rem .75rem; font-size: 1rem; border: 1px solid #ccc;
             border-radius: 4px; box-sizing: border-box; margin-bottom: 1rem; }}
    button {{ width: 100%; padding: .65rem; font-size: 1rem; background: #2563eb;
              color: white; border: none; border-radius: 4px; cursor: pointer; }}
    button:hover {{ background: #1d4ed8; }}
    .error {{ color: #dc2626; margin-bottom: 1rem; font-size: .9rem; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>🧪 Distil</h1>
    {error}
    <form method="post" action="/login">
      <input type="password" name="secret" placeholder="Enter your secret…" autofocus>
      <button type="submit">Sign in</button>
    </form>
  </div>
</body>
</html>"""


def login_page(error: bool = False) -> HTMLResponse:
    error_html = '<p class="error">Incorrect secret — try again.</p>' if error else ""
    return HTMLResponse(_LOGIN_HTML.format(error=error_html))


def login_response(secret_input: str) -> Response:
    """Validate the submitted secret; set cookie and redirect home, or show error."""
    secret = auth_secret()
    if secret and hmac.compare_digest(secret_input, secret):
        response = RedirectResponse(url="/", status_code=303)
        cookie_value = _make_session_cookie(secret)
        response.set_cookie(
            _COOKIE_NAME,
            cookie_value,
            httponly=True,
            samesite="lax",
            secure=is_public(),  # Secure flag only on HTTPS (Railway)
            max_age=60 * 60 * 24 * 30,  # 30 days
        )
        return response
    return login_page(error=True)


def logout_response() -> Response:
    """Clear the session cookie and redirect to login."""
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(_COOKIE_NAME)
    return response


def server_bind() -> tuple[str, int]:
    """Host/port the server must bind: 0.0.0.0 and $PORT when provided (T-A4)."""
    try:
        port = int(os.environ.get("PORT", "8000"))
    except ValueError:
        port = 8000
    return "0.0.0.0", port
