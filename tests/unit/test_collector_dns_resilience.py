"""distil/collector_net.py: the owner's Mac has repeatedly had its system resolver cache a bad
answer for the server's hostname right after a deploy (``getaddrinfo`` fails while ``dig``/
``curl --resolve`` both prove the name and the server are fine). These tests prove
``open_with_dns_fallback`` rides that out without ever weakening TLS verification.

Resolution failure is simulated with a real ``socket.gaierror`` raised by a fake resolver for the
exact hostname under test — never by mocking out ``open_with_dns_fallback`` itself or stubbing
away the branch being tested. Certificate verification is proven against a genuine local TLS
server carrying a real (self-signed, freshly generated per test session) certificate, not asserted
by inspecting call arguments.
"""

from __future__ import annotations

import http.server
import shutil
import socket
import ssl
import subprocess
import threading
import urllib.error
import urllib.request

import pytest

from distil.collector_net import DNSFallbackStore, open_with_dns_fallback

_HOSTNAME = "distil-collector-test.internal"
_WRONG_HOSTNAME = "not-the-right-host.internal"

pytestmark = pytest.mark.skipif(
    shutil.which("openssl") is None, reason="openssl CLI not available to generate a test cert"
)


def _unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _OKHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # silence per-request logging in test output
        pass

    def do_GET(self) -> None:
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")


@pytest.fixture(scope="module")
def tls_cert(tmp_path_factory):
    """A real, freshly generated self-signed certificate whose SAN is _HOSTNAME — genuine
    material for a genuine TLS handshake, never hand-typed."""
    cert_dir = tmp_path_factory.mktemp("dns-resilience-cert")
    cert_path = cert_dir / "cert.pem"
    key_path = cert_dir / "key.pem"
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", str(key_path), "-out", str(cert_path),
            "-days", "2", "-nodes", "-subj", f"/CN={_HOSTNAME}",
            "-addext", f"subjectAltName=DNS:{_HOSTNAME}",
        ],
        check=True, capture_output=True,
    )
    return cert_path, key_path


@pytest.fixture(scope="module")
def tls_server(tls_cert):
    """A real local HTTPS server presenting the certificate above. Tests connect to it by literal
    IP address while asking for hostname verification against ``_HOSTNAME`` — exactly what a
    pinned collector connection does against the real server."""
    cert_path, key_path = tls_cert
    httpd = http.server.HTTPServer(("127.0.0.1", 0), _OKHandler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd.server_port
    finally:
        httpd.shutdown()
        thread.join(timeout=5)


@pytest.fixture
def trusting_ssl_context(tls_cert):
    """The client-side equivalent of a real browser trusting a real CA — except the CA here is
    our test cert itself, since it's self-signed. Verification stays genuine; only the trust
    anchor is test-local."""
    cert_path, _key_path = tls_cert
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.load_verify_locations(cafile=str(cert_path))
    return context


def _fake_resolver_failing_for(hostname_to_fail: str):
    """A resolver that actually raises socket.gaierror — the real exception type a broken system
    resolver produces — for one specific hostname, and defers to the real resolver otherwise."""

    def resolver(host: str, port: int) -> str:
        if host == hostname_to_fail:
            raise socket.gaierror(
                socket.EAI_NONAME, "nodename nor servname provided, or not known"
            )
        return socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)[0][4][0]

    return resolver


def _request_for(port: int, hostname: str = _HOSTNAME) -> urllib.request.Request:
    return urllib.request.Request(f"https://{hostname}:{port}/health")


@pytest.mark.unit
def test_resolution_failure_with_stored_fallback_continues_working(tls_server, trusting_ssl_context, tmp_path):
    store = DNSFallbackStore(tmp_path / "dns.json")
    store.set(_HOSTNAME, "127.0.0.1")
    response = open_with_dns_fallback(
        _request_for(tls_server),
        timeout=5.0,
        dns_store=store,
        resolve=_fake_resolver_failing_for(_HOSTNAME),
        ssl_context=trusting_ssl_context,
    )
    assert response.read() == b"ok"


@pytest.mark.unit
def test_resolution_failure_without_stored_fallback_fails_honestly_and_keeps_retrying(tmp_path):
    store = DNSFallbackStore(tmp_path / "dns.json")  # nothing ever written to it
    resolve = _fake_resolver_failing_for(_HOSTNAME)
    for _ in range(3):
        # Every call attempts real resolution again — never gives up or falls back to nothing.
        with pytest.raises(urllib.error.URLError):
            open_with_dns_fallback(
                _request_for(12345), timeout=5.0, dns_store=store, resolve=resolve,
            )
    assert store.get(_HOSTNAME) is None


@pytest.mark.unit
def test_successful_connection_records_the_address(tls_server, trusting_ssl_context, tmp_path):
    store = DNSFallbackStore(tmp_path / "dns.json")
    assert store.get(_HOSTNAME) is None

    def resolve(host: str, port: int) -> str:
        assert host == _HOSTNAME
        return "127.0.0.1"

    response = open_with_dns_fallback(
        _request_for(tls_server), timeout=5.0, dns_store=store, resolve=resolve,
        ssl_context=trusting_ssl_context,
    )
    assert response.read() == b"ok"
    assert store.get(_HOSTNAME) == "127.0.0.1"

    # Persists across "restart" — a fresh DNSFallbackStore over the same file sees it too.
    reloaded = DNSFallbackStore(tmp_path / "dns.json")
    assert reloaded.get(_HOSTNAME) == "127.0.0.1"


@pytest.mark.unit
def test_resolution_recovering_returns_to_normal_path_and_stops_using_fallback(
    tls_server, trusting_ssl_context, tmp_path
):
    store = DNSFallbackStore(tmp_path / "dns.json")
    store.set(_HOSTNAME, "127.0.0.1")
    degraded_calls = []

    # First call: resolution fails, fallback is used.
    response = open_with_dns_fallback(
        _request_for(tls_server), timeout=5.0, dns_store=store,
        resolve=_fake_resolver_failing_for(_HOSTNAME), ssl_context=trusting_ssl_context,
        on_degraded=lambda *args: degraded_calls.append(args),
    )
    assert response.read() == b"ok"
    assert len(degraded_calls) == 1

    # Second call: resolution now succeeds — the real path is taken, fallback is never consulted.
    resolve_calls = []

    def recovered_resolve(host: str, port: int) -> str:
        resolve_calls.append(host)
        return "127.0.0.1"

    response = open_with_dns_fallback(
        _request_for(tls_server), timeout=5.0, dns_store=store, resolve=recovered_resolve,
        ssl_context=trusting_ssl_context, on_degraded=lambda *args: degraded_calls.append(args),
    )
    assert response.read() == b"ok"
    assert resolve_calls == [_HOSTNAME]
    assert len(degraded_calls) == 1  # no new degraded event on the recovered call


@pytest.mark.unit
def test_stale_fallback_address_does_not_trap_the_collector(tls_server, trusting_ssl_context, tmp_path):
    store = DNSFallbackStore(tmp_path / "dns.json")
    dead_port = _unused_port()  # nothing listens here
    store.set(_HOSTNAME, "127.0.0.1")

    # Resolution keeps failing, and the stored fallback happens to be for the dead port's host —
    # simulate "the cached address no longer works" by pointing the request at the dead port.
    with pytest.raises(urllib.error.URLError):
        open_with_dns_fallback(
            _request_for(dead_port), timeout=5.0, dns_store=store,
            resolve=_fake_resolver_failing_for(_HOSTNAME), ssl_context=trusting_ssl_context,
        )

    # Bounded: the very next call (the collector's next poll tick) tries real resolution again
    # and succeeds — it never gets stuck retrying only the dead address.
    response = open_with_dns_fallback(
        _request_for(tls_server), timeout=5.0, dns_store=store,
        resolve=lambda host, port: "127.0.0.1", ssl_context=trusting_ssl_context,
    )
    assert response.read() == b"ok"


@pytest.mark.unit
def test_certificate_verification_rejects_hostname_mismatch_when_connecting_by_address(
    tls_server, trusting_ssl_context, tmp_path
):
    """Connecting by address must still validate the certificate against the request's hostname
    — a hostname the certificate was never issued for must fail, proving verification is genuine
    rather than skipped just because the destination was pinned to a literal address."""
    store = DNSFallbackStore(tmp_path / "dns.json")
    with pytest.raises(urllib.error.URLError) as excinfo:
        open_with_dns_fallback(
            _request_for(tls_server, hostname=_WRONG_HOSTNAME),
            timeout=5.0, dns_store=store,
            resolve=lambda host, port: "127.0.0.1", ssl_context=trusting_ssl_context,
        )
    assert isinstance(excinfo.value.reason, ssl.SSLCertVerificationError)


@pytest.mark.unit
def test_certificate_verification_accepts_matching_hostname_when_connecting_by_address(
    tls_server, trusting_ssl_context, tmp_path
):
    """The positive case for the test above: same pinned-by-address mechanism, correct hostname,
    genuine handshake succeeds."""
    store = DNSFallbackStore(tmp_path / "dns.json")
    response = open_with_dns_fallback(
        _request_for(tls_server), timeout=5.0, dns_store=store,
        resolve=lambda host, port: "127.0.0.1", ssl_context=trusting_ssl_context,
    )
    assert response.read() == b"ok"
