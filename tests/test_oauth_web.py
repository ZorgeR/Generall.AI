"""The OAuth callback endpoint (bot/oauth_web.py): settings, routes, headers, logs, TLS (admin files,
the generated self-signed pair, plain HTTP), startup failures. Real aiohttp servers on 127.0.0.1 with an
OS-chosen port; no other network."""
import asyncio
import logging
import ssl
import stat
from datetime import datetime, timedelta, timezone

import aiohttp
import pytest
from cryptography import x509
from cryptography.x509.oid import NameOID

from bot import oauth_flow, oauth_web

BASE = "https://gall.example.com"


@pytest.fixture(autouse=True)
def clean(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    oauth_flow.reset()
    oauth_web.reset()
    yield
    oauth_flow.reset()
    oauth_web.reset()


def settings(**env):
    return oauth_web.settings_from_env({"PUBLIC_BASE_URL": BASE, "OAUTH_LISTEN": "127.0.0.1:0", **env}, data_dir="data")


@pytest.fixture
async def served():
    server = await oauth_web.start(settings())
    yield f"http://127.0.0.1:{server.port}"
    await server.close()


async def get(url, method="GET", **kw):
    async with aiohttp.ClientSession() as session:
        async with session.request(method, url, allow_redirects=False, **kw) as response:
            return response.status, await response.text(), response.headers


# ---- settings -------------------------------------------------------------------------------------
def test_settings_defaults_and_the_tls_rule():
    assert oauth_web.settings_from_env({}) is None and oauth_web.settings_from_env({"PUBLIC_BASE_URL": " "}) is None
    default = oauth_web.settings_from_env({"PUBLIC_BASE_URL": BASE + "/"}, data_dir="data")
    assert (default.host, default.port, default.tls, default.public_base_url) == ("127.0.0.1", 8090, False, BASE)
    assert default.listen == "http://127.0.0.1:8090"  # plain HTTP on loopback: never public by default
    direct = oauth_web.settings_from_env({"PUBLIC_BASE_URL": BASE, "OAUTH_LISTEN": "0.0.0.0:443"}, data_dir="data")
    assert direct.tls is True and str(direct.cert) == "data/tls/gall.example.com.pem" and str(direct.key) == "data/tls/gall.example.com.key"
    assert str(direct.self_signed_cert) == "data/tls/gall.example.com.selfsigned.pem" and not direct.explicit_files
    assert settings(OAUTH_TLS="on").tls is True and settings(OAUTH_LISTEN="0.0.0.0:443", OAUTH_TLS="off").tls is False
    files = settings(OAUTH_TLS_CERT="/etc/x.pem", OAUTH_TLS_KEY="/etc/x.key")
    assert str(files.cert) == "/etc/x.pem" and files.explicit_files
    assert oauth_web.parse_listen("[::]:8443") == ("::", 8443) and oauth_web.parse_listen(":8090") == ("0.0.0.0", 8090)
    for bad in ({"OAUTH_LISTEN": "8090"}, {"OAUTH_LISTEN": "host:http"}, {"OAUTH_LISTEN": "h:70000"}, {"OAUTH_TLS": "maybe"},
                {"PUBLIC_BASE_URL": "gall.example.com"}, {"PUBLIC_BASE_URL": "https://gall.example.com/?x=1"}):
        with pytest.raises(ValueError):
            settings(**bad)


# ---- routes ---------------------------------------------------------------------------------------
async def test_health_and_everything_else(served):
    status, body, headers = await get(f"{served}/oauth/health")
    assert (status, body) == (200, "ok")
    for name, value in oauth_web.SECURITY_HEADERS.items():
        assert headers[name] == value
    assert headers["Server"] == "gall"
    for path in ("/", "/oauth", "/oauth/", "/oauth/callback/x", "/.env", "/oauth/../etc/passwd"):
        status, _, headers = await get(f"{served}{path}")
        assert status == 404 and headers["Cache-Control"] == "no-store", path
    for method in ("POST", "PUT", "DELETE", "HEAD"):
        status, _, _ = await get(f"{served}/oauth/callback?state=x&code=y", method=method)
        assert status == 405, method
    status, _, _ = await get(f"{served}/oauth/callback?state={'x' * 9000}")
    assert status in (400, 414)  # the request line limit


async def test_callback_resolves_once_and_unknown_expired_states_get_the_expired_page(served, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    oauth_web.set_bot_username("gall_bot")
    login = oauth_flow.register("STATE-SECRET", "42", "notion")
    status, page, headers = await get(f"{served}/oauth/callback?code=CODE-SECRET&state=STATE-SECRET&iss=https%3A%2F%2Fauth")
    assert status == 200 and "✅ Signed in" in page and 'href="tg://resolve?domain=gall_bot"' in page
    assert "CODE-SECRET" not in page and "STATE-SECRET" not in page
    assert headers["Content-Type"].startswith("text/html")
    result = login.future.result()
    assert (result.code, result.state, result.iss) == ("CODE-SECRET", "STATE-SECRET", "https://auth")
    status, page, _ = await get(f"{served}/oauth/callback?code=CODE-SECRET&state=STATE-SECRET")  # used
    assert status == 400 and "expired" in page
    status, page, _ = await get(f"{served}/oauth/callback?code=c&state=unknown")
    assert status == 400 and "Start again with /mcp" in page
    late = oauth_flow.register("STATE-LATE", "42", "linear")
    monkeypatch.setattr(oauth_flow, "LOGIN_TIMEOUT", 0)
    status, _, _ = await get(f"{served}/oauth/callback?code=c&state=STATE-LATE")
    assert status == 400 and isinstance(late.future.exception(), oauth_flow.LoginExpired)
    logged = "\n".join(f"{r.name} {r.getMessage()}" for r in caplog.records)
    assert "OAuth callback for notion of 42" in logged
    assert "CODE-SECRET" not in logged and "STATE-SECRET" not in logged and "STATE-LATE" not in logged


async def test_an_error_callback_reaches_the_flow(served):
    login = oauth_flow.register("S1", "42", "notion")
    status, page, _ = await get(f"{served}/oauth/callback?error=access_denied&error_description=no&state=S1")
    assert status == 200 and "not completed" in page
    error = login.future.exception()
    assert isinstance(error, oauth_flow.LoginRefused) and (error.error, error.description) == ("access_denied", "no")
    nocode = oauth_flow.register("S2", "42", "linear")
    await get(f"{served}/oauth/callback?state=S2")
    assert nocode.future.exception().error == "no_code"


async def test_without_a_bot_username_the_page_has_no_link(served):
    oauth_flow.register("S", "42", "notion")
    oauth_web.set_bot_username("bad name!")
    _, page, _ = await get(f"{served}/oauth/callback?code=c&state=S")
    assert "tg://" not in page


# ---- TLS --------------------------------------------------------------------------------------------
def test_the_self_signed_pair_is_generated_once_private_and_reused(tmp_path):
    s = settings(OAUTH_TLS="on")
    info = oauth_web.load_tls(s)
    assert info.self_signed and info.generated and info.cert == s.self_signed_cert
    assert stat.S_IMODE(s.self_signed_key.stat().st_mode) == 0o600
    assert stat.S_IMODE(s.self_signed_cert.parent.stat().st_mode) == 0o700
    cert = x509.load_pem_x509_certificate(s.self_signed_cert.read_bytes())
    assert cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value == "gall.example.com"
    san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert san.get_values_for_type(x509.DNSName) == ["gall.example.com"]
    lifetime = cert.not_valid_after_utc - datetime.now(timezone.utc)
    assert timedelta(days=3640) < lifetime <= timedelta(days=3650)
    assert info.describe().startswith(f"self-signed certificate, generated now {s.self_signed_cert}, valid until ")
    before = (s.self_signed_cert.read_bytes(), s.self_signed_key.read_bytes())
    again = oauth_web.load_tls(s)
    assert again.self_signed and not again.generated
    assert (s.self_signed_cert.read_bytes(), s.self_signed_key.read_bytes()) == before
    s.self_signed_cert.unlink()  # half a pair: regenerated, the key replaced too
    assert oauth_web.load_tls(s).generated and s.self_signed_key.read_bytes() != before[1]


def test_the_admins_files_win_and_are_never_touched(tmp_path, caplog):
    s = settings(OAUTH_TLS="on")
    oauth_web.ensure_self_signed("gall.example.com", s.cert, s.key)  # stands in for a Cloudflare origin certificate
    admin = (s.cert.read_bytes(), s.key.read_bytes())
    info = oauth_web.load_tls(s)
    assert not info.self_signed and info.cert == s.cert and not s.self_signed_cert.exists()
    assert info.describe().startswith(f"certificate {s.cert}, valid until ")
    assert (s.cert.read_bytes(), s.key.read_bytes()) == admin
    missing = settings(OAUTH_TLS="on", OAUTH_TLS_CERT=str(tmp_path / "no.pem"), OAUTH_TLS_KEY=str(tmp_path / "no.key"))
    assert oauth_web.load_tls(missing).self_signed
    assert any("not found: using a self-signed certificate" in r.getMessage() for r in caplog.records)


async def test_https_is_served_with_the_certificate(caplog):
    caplog.set_level(logging.INFO)
    server = await oauth_web.start(settings(OAUTH_TLS="on"))
    try:
        client = ssl.create_default_context(cafile=str(server.settings.self_signed_cert))
        client.check_hostname = False  # the certificate names gall.example.com, the test talks to 127.0.0.1
        status, body, _ = await get(f"https://127.0.0.1:{server.port}/oauth/health", ssl=client)
        assert (status, body) == (200, "ok")
        assert oauth_web.callback_url() == f"{BASE}/oauth/callback"
        old = ssl.create_default_context(cafile=str(server.settings.self_signed_cert))
        old.check_hostname = False
        old.maximum_version = ssl.TLSVersion.TLSv1_1
        with pytest.raises(aiohttp.ClientError):
            await get(f"https://127.0.0.1:{server.port}/oauth/health", ssl=old)
    finally:
        await server.close()
    assert oauth_web.callback_url() is None
    line = next(r.getMessage() for r in caplog.records if "OAuth callback endpoint listening" in r.getMessage())
    assert "self-signed certificate" in line and "valid until" in line and "PRIVATE KEY" not in caplog.text


async def test_plain_http_when_tls_is_off():
    server = await oauth_web.start(settings(OAUTH_TLS="off"))
    try:
        assert server.tls is None
        status, body, _ = await get(f"http://127.0.0.1:{server.port}/oauth/health")
        assert (status, body) == (200, "ok")
    finally:
        await server.close()


# ---- serve() --------------------------------------------------------------------------------------
async def test_serve_runs_until_cancelled(monkeypatch):
    monkeypatch.setenv("PUBLIC_BASE_URL", BASE)
    monkeypatch.setenv("OAUTH_LISTEN", "127.0.0.1:0")
    monkeypatch.delenv("OAUTH_TLS", raising=False)
    task = asyncio.create_task(oauth_web.serve())
    for _ in range(200):
        if oauth_web.callback_url():
            break
        await asyncio.sleep(0.01)
    assert oauth_web.callback_url() == f"{BASE}/oauth/callback" and oauth_flow.redirect_uri() == f"{BASE}/oauth/callback"
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert oauth_web.callback_url() is None and oauth_flow.redirect_uri() == oauth_flow.LOOPBACK_REDIRECT


async def test_serve_without_a_public_url_or_with_bad_settings_does_nothing(monkeypatch, caplog):
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    await asyncio.wait_for(oauth_web.serve(), 1)
    monkeypatch.setenv("PUBLIC_BASE_URL", BASE)
    monkeypatch.setenv("OAUTH_LISTEN", "nonsense")
    await asyncio.wait_for(oauth_web.serve(), 1)
    assert oauth_web.callback_url() is None
    assert any(r.levelno == logging.ERROR and "OAuth callback endpoint disabled" in r.getMessage() for r in caplog.records)


async def test_bind_failures_and_bad_tls_files_are_one_error_and_the_bot_runs_on(monkeypatch, caplog, tmp_path):
    blocker = await asyncio.start_server(lambda r, w: None, "127.0.0.1", 0)
    port = blocker.sockets[0].getsockname()[1]
    monkeypatch.setenv("PUBLIC_BASE_URL", BASE)
    monkeypatch.setenv("OAUTH_LISTEN", f"127.0.0.1:{port}")
    monkeypatch.setenv("OAUTH_TLS", "off")
    try:
        await asyncio.wait_for(oauth_web.serve(), 2)  # returns: the port is in use
    finally:
        blocker.close()
        await blocker.wait_closed()
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and "Could not start the OAuth callback endpoint" in errors[0].getMessage()
    assert oauth_web.callback_url() is None

    caplog.clear()
    s = settings(OAUTH_TLS="on")
    s.cert.parent.mkdir(parents=True, exist_ok=True)
    s.cert.write_text("not a certificate")
    s.key.write_text("not a key")
    monkeypatch.setenv("OAUTH_LISTEN", "127.0.0.1:0")
    monkeypatch.setenv("OAUTH_TLS", "on")
    await asyncio.wait_for(oauth_web.serve(), 2)
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and "SSLError" in errors[0].getMessage() and "not a key" not in caplog.text
    assert oauth_web.callback_url() is None
