"""DNS-resilience layer for the collector's HTTP calls to the server (see ``collector.py``'s
module docstring for the rest of the collector).

Observed repeatedly on the owner's Mac, always within a minute of a deploy: the system
resolver's cache holds a stale/broken answer for the server's hostname, so
``socket.getaddrinfo`` fails with ``socket.gaierror`` even though the name is genuinely
resolvable (``dig`` answers correctly) and the server itself is healthy (``curl --resolve``
succeeds). Clearing the cache needs ``sudo``, which the collector does not and must not have.
This module rides that out instead: when resolution fails, it substitutes the last IP address
that demonstrably worked for that hostname, and continues.

Two things this must never do: weaken TLS verification, or prefer the cached address over real
resolution. Both are structural here, not configuration:

* :func:`open_with_dns_fallback` always calls ``resolve`` first, on every single call — there is
  no "DNS is broken, stop trying" state anywhere. The pinned address is consulted only in the
  ``except`` branch, when resolution just failed. The moment the resolver starts working again,
  the very next call (the next poll tick, or the next retry attempt within one call) takes the
  normal path and the stored address is never even read.
* :class:`_PinnedHTTPSConnection` only ever substitutes the socket's destination address. The
  TLS handshake still asks for and verifies the original hostname (SNI's ``server_hostname`` and
  the certificate check are both ``self.host``, never the pinned address), and the HTTP ``Host``
  header is derived from ``self.host`` the same way an ordinary connection's would be. This is
  done by overriding only ``_create_connection`` (an instance attribute
  ``http.client.HTTPConnection`` already exposes for exactly this kind of substitution) rather
  than reimplementing ``connect()`` — so it inherits the stdlib's own tunnel/TCP_NODELAY/SNI
  handling unmodified.

A stored address is learned, never shipped: :class:`DNSFallbackStore` starts empty, and is
populated only by :func:`open_with_dns_fallback` recording the address a *real, completed*
request just succeeded through. It persists across restarts (the collector is relaunched
automatically by launchd) in a small JSON file beside no other collector state — this is the
first the collector has needed — keyed by hostname, default ``~/.distil/collector_dns_cache.json``
(override the directory with ``DISTIL_COLLECTOR_STATE_DIR``). A missing or corrupt file just
means "no fallback known yet", never an error.

Boundedness ("a stored address that stops working must not trap the collector retrying a dead
endpoint forever") falls out of the same per-call structure: a failure connecting to the pinned
address propagates as an ordinary connection failure (via ``urllib``'s existing
``URLError``/``HTTPError`` wrapping — nothing DNS-specific about it from here on), which
``collector.py``'s pre-existing bounded retry/backoff and ``run_collector``'s poll loop already
handle exactly as they would a real outage. Nothing in this module retries on its own; it never
holds onto "use the fallback" across calls.
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import socket
import ssl
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

_STATE_DIR_ENV = "DISTIL_COLLECTOR_STATE_DIR"
_DNS_CACHE_FILENAME = "collector_dns_cache.json"


def default_state_path() -> Path:
    """Where the last-good-address cache lives, absent an explicit override. Read at call time
    (same pattern as this repo's other ``DISTIL_*`` env vars) so a test or a differently
    configured machine never needs to touch ``$HOME``."""
    state_dir = os.environ.get(_STATE_DIR_ENV)
    base = Path(state_dir).expanduser() if state_dir else Path.home() / ".distil"
    return base / _DNS_CACHE_FILENAME


class DNSFallbackStore:
    """Persists, per hostname, the last IP address a real connection to it succeeded through.

    Written atomically (temp file in the same directory, then ``replace``) so a crash mid-write
    can't leave a value a later run would wrongly trust. A missing or unparsable file is treated
    identically to "no fallback known" — never raised.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    def get(self, hostname: str) -> str | None:
        try:
            raw = self._path.read_text(encoding="utf-8")
        except OSError:
            return None
        try:
            data = json.loads(raw)
        except ValueError:
            return None
        value = data.get(hostname) if isinstance(data, dict) else None
        return value if isinstance(value, str) and value else None

    def set(self, hostname: str, address: str) -> None:
        try:
            existing: dict[str, Any]
            try:
                existing = json.loads(self._path.read_text(encoding="utf-8"))
                if not isinstance(existing, dict):
                    existing = {}
            except (OSError, ValueError):
                existing = {}
            if existing.get(hostname) == address:
                return
            existing[hostname] = address
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(existing), encoding="utf-8")
            tmp_path.replace(self._path)
        except OSError as exc:
            # Losing this cache is degraded, not fatal: a future resolution failure with nothing
            # stored just fails honestly (see open_with_dns_fallback) — it never crashes the loop.
            _logger.warning("Could not persist DNS fallback address for %s: %s", hostname, exc)


def resolve_host(hostname: str, port: int) -> str:
    """Resolve ``hostname`` via the system resolver, exactly as an ordinary connection would.

    Deliberately does not catch anything: ``socket.getaddrinfo`` raises ``socket.gaierror`` (an
    ``OSError`` subclass) on a resolution failure and nothing else, so callers can distinguish
    "the name didn't resolve" from every other kind of connection failure by exception type
    alone — no string-matching a resolver's error text, which is exactly the class of bug
    ``distil/youtube.py``'s bot-check detection was burned by once already (see AGENTS.md).
    """
    infos = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    return infos[0][4][0]


def _pin_to_address(connection: http.client.HTTPConnection, address: str) -> None:
    """Make ``connection`` open its socket to ``address`` instead of resolving its own ``host``,
    while leaving every other part of ``connect()`` — including, for HTTPS, the SNI/certificate
    hostname check, which stays ``connection.host`` — untouched."""
    real_create_connection = socket.create_connection

    def _create_connection(addr: tuple[str, int], timeout: float, source_address: Any = None) -> Any:
        _, port = addr
        return real_create_connection((address, port), timeout, source_address)

    connection._create_connection = _create_connection


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, host: str, address: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(host, *args, **kwargs)
        _pin_to_address(self, address)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host: str, address: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(host, *args, **kwargs)
        _pin_to_address(self, address)


class _PinnedHTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, address: str) -> None:
        super().__init__()
        self._address = address

    def http_open(self, req: urllib.request.Request) -> Any:
        def build(host: str, **kwargs: Any) -> _PinnedHTTPConnection:
            return _PinnedHTTPConnection(host, self._address, **kwargs)

        return self.do_open(build, req)


class _PinnedHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, address: str, *, context: ssl.SSLContext | None = None) -> None:
        super().__init__(context=context)
        self._address = address

    def https_open(self, req: urllib.request.Request) -> Any:
        def build(host: str, **kwargs: Any) -> _PinnedHTTPSConnection:
            return _PinnedHTTPSConnection(host, self._address, **kwargs)

        return self.do_open(build, req, context=self._context)


def _build_pinned_opener(address: str, ssl_context: ssl.SSLContext | None) -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(
        _PinnedHTTPHandler(address), _PinnedHTTPSHandler(address, context=ssl_context)
    )


def open_with_dns_fallback(
    request: urllib.request.Request,
    timeout: float,
    *,
    dns_store: DNSFallbackStore,
    resolve: Callable[[str, int], str] = resolve_host,
    on_degraded: Callable[[str, str, Exception], None] | None = None,
    ssl_context: ssl.SSLContext | None = None,
) -> Any:
    """The collector's default opener. Resolution first, always; the stored fallback address is
    read only when ``resolve`` itself raises ``socket.gaierror``, and a successful request — via
    either path — records the address it used. See the module docstring for the full reasoning.
    """
    parsed = urllib.parse.urlsplit(request.full_url)
    hostname = parsed.hostname
    if hostname is None:
        return urllib.request.urlopen(request, timeout=timeout)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    try:
        address = resolve(hostname, port)
    except socket.gaierror as exc:
        fallback = dns_store.get(hostname)
        if fallback is None:
            _logger.error(
                "DNS resolution failed for %s (%s), and no fallback address is stored yet. "
                "This looks like a local name-resolution problem, not necessarily a server "
                "outage — will keep retrying real resolution.",
                hostname, exc,
            )
            raise urllib.error.URLError(exc) from exc
        _logger.warning(
            "DEGRADED: DNS resolution failed for %s (%s); falling back to the last address that "
            "worked, %s. This is a local resolver problem, not a server outage — normal "
            "resolution will resume automatically once it starts working again.",
            hostname, exc, fallback,
        )
        if on_degraded is not None:
            on_degraded(hostname, fallback, exc)
        address = fallback

    opener = _build_pinned_opener(address, ssl_context)
    response = opener.open(request, timeout=timeout)
    dns_store.set(hostname, address)
    return response
