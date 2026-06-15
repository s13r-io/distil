"""Auth gate for hosted Distil. PRD FR14; ARCHITECTURE.md §8; TESTING T-A1..A4.

Rules (non-optional when hosting):
* **Fail closed:** if ``DISTIL_PUBLIC=true`` but ``DISTIL_AUTH_SECRET`` is unset, the app
  refuses to start (T-A1) — generating a public domain without auth would expose the API key
  and knowledge base.
* **Require auth on data routes** in public mode (T-A2); ``/health`` stays open.
* **Localhost convenience:** when not public, routes are reachable without the secret (T-A3).
* **Bind ``0.0.0.0:$PORT``** for platforms that inject the port (T-A4).
"""

from __future__ import annotations

import os

_OPEN_PATHS = {"/health"}


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


def request_is_authorized(authorization_header: str | None) -> bool:
    """In public mode, require a matching bearer secret. On localhost, always allowed (T-A3)."""
    if not is_public():
        return True
    secret = auth_secret()
    if not secret:
        return False  # fail closed (defensive; startup check should have caught this)
    if not authorization_header:
        return False
    token = authorization_header.removeprefix("Bearer ").strip()
    return token == secret


def path_is_open(path: str) -> bool:
    return path in _OPEN_PATHS


def server_bind() -> tuple[str, int]:
    """Host/port the server must bind: 0.0.0.0 and $PORT when provided (T-A4)."""
    try:
        port = int(os.environ.get("PORT", "8000"))
    except ValueError:
        port = 8000
    return "0.0.0.0", port
