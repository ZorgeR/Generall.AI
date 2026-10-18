"""The OAuth callback endpoint of MCP sign-ins: a small aiohttp server in the bot process.

Started by ``bot/app.py:on_startup`` as the background task "oauth-web" (``serve``) only when
``PUBLIC_BASE_URL`` is set; stopped with the other background tasks on shutdown. Configuration (read at
startup, ``settings_from_env``):

* ``PUBLIC_BASE_URL`` (``https://gall.example.com``): the redirect URI of sign-ins becomes
  ``PUBLIC_BASE_URL + /oauth/callback`` while the endpoint runs (``callback_url``); unset, or the endpoint
  could not start → the copy-paste fallback of ``bot.oauth_flow`` (``LOOPBACK_REDIRECT``).
* ``OAUTH_LISTEN`` (default ``127.0.0.1:8090``, plain HTTP: the recommended Cloudflare Tunnel or a reverse
  proxy forwards ``/oauth/`` there; never public by default).
* TLS when the port is 443 or ``OAUTH_TLS=on`` (``OAUTH_TLS=off`` forces plain HTTP): the admin's
  certificate (``OAUTH_TLS_CERT`` / ``OAUTH_TLS_KEY``, default ``data/tls/<host>.pem`` / ``.key``, e.g. a
  Cloudflare Origin Certificate for "Full (strict)") when both files exist, else a self-signed EC P-256
  certificate for the host, generated once into ``data/tls/<host>.selfsigned.pem`` / ``.key`` (key 0600,
  O_EXCL; reused; enough for Cloudflare "Full"). TLS 1.2 or newer. The startup log names the certificate
  (path, self-signed or not, expiry), never the key.

Routes: ``GET /oauth/callback`` (``code`` + ``state``, or ``error`` [+ ``error_description``]) resolves the
pending login of that state (``oauth_flow.resolve``: one-shot, bound to its user and server, 600 s) and
shows "✅ Signed in" with a ``tg://resolve`` link to the bot; an unknown, expired or used state shows the
expired page (400). ``GET /oauth/health`` → ``ok``. Anything else 404, other methods 405, no HEAD, small
header and body limits, no access log (the request line carries the code and state), every response with
``Cache-Control: no-store``, ``Referrer-Policy: no-referrer``, ``X-Content-Type-Options: nosniff``, a CSP
without external resources. Logs name the server and the user, never a code or a state. A bind failure
(port taken, no permission, bad TLS files) is one ERROR line; the bot runs on without the endpoint.
"""
from __future__ import annotations

import asyncio
import html
import ipaddress
import logging
import os
import re
import ssl
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping
from urllib.parse import urlsplit

from aiohttp import web

from bot import oauth_flow
from bot.config import config

logger = logging.getLogger(__name__)

CALLBACK_PATH = oauth_flow.CALLBACK_PATH
HEALTH_PATH = "/oauth/health"
DEFAULT_LISTEN = "127.0.0.1:8090"
TLS_DIR = "tls"
SELF_SIGNED_DAYS = 3650
MAX_LINE = 4096  # request line (the callback URL) and each header line
MAX_HEADERS = 50
MAX_BODY = 1024  # GET only; nothing reads a body
_USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{4,32}$")
SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
}

_callback_url: str | None = None
_bot_username: str | None = None


def callback_url() -> str | None:
    """The public redirect URI while the endpoint serves it, else None (the copy-paste fallback)."""
    return _callback_url


def set_bot_username(username: str | None) -> None:
    global _bot_username
    _bot_username = username if username and _USERNAME_RE.match(username) else None


# ---- settings -------------------------------------------------------------------------------
@dataclass(frozen=True)
class WebSettings:
    public_base_url: str
    hostname: str
    host: str
    port: int
    tls: bool
    cert: Path
    key: Path
    explicit_files: bool
    self_signed_cert: Path
    self_signed_key: Path

    @property
    def listen(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"{'https' if self.tls else 'http'}://{host}:{self.port}"


def parse_listen(value: str) -> tuple[str, int]:
    """``host:port`` (``[::]:443`` for IPv6; an empty host is 0.0.0.0) → (host, port); ValueError otherwise."""
    host, sep, port = str(value or "").strip().rpartition(":")
    if not sep or not port.isdigit() or not 0 <= int(port) < 65536:  # 0: any free port (tests)
        raise ValueError(f"OAUTH_LISTEN must be host:port, got {value!r}")
    return host.strip("[]") or "0.0.0.0", int(port)


def settings_from_env(env: Mapping[str, str] | None = None, data_dir: str | None = None) -> WebSettings | None:
    """None without ``PUBLIC_BASE_URL``; ValueError for a value that cannot work (logged by ``serve``)."""
    env = os.environ if env is None else env
    base = str(env.get("PUBLIC_BASE_URL") or "").strip().rstrip("/")
    if not base:
        return None
    parts = urlsplit(base)
    if parts.scheme not in ("https", "http") or not parts.hostname or parts.query or parts.fragment or parts.username:
        raise ValueError("PUBLIC_BASE_URL must look like https://gall.example.com")
    host, port = parse_listen(str(env.get("OAUTH_LISTEN") or "").strip() or DEFAULT_LISTEN)
    flag = str(env.get("OAUTH_TLS") or "").strip().lower()
    if flag in ("on", "true", "1", "yes"):
        tls = True
    elif flag in ("off", "false", "0", "no"):
        tls = False
    elif flag in ("", "auto"):
        tls = port == 443
    else:
        raise ValueError(f"OAUTH_TLS must be on or off, got {flag!r}")
    hostname = parts.hostname.lower()
    tls_dir = Path(data_dir or config.data_dir) / TLS_DIR
    cert_env, key_env = str(env.get("OAUTH_TLS_CERT") or "").strip(), str(env.get("OAUTH_TLS_KEY") or "").strip()
    return WebSettings(
        public_base_url=base, hostname=hostname, host=host, port=port, tls=tls,
        cert=Path(cert_env) if cert_env else tls_dir / f"{hostname}.pem",
        key=Path(key_env) if key_env else tls_dir / f"{hostname}.key",
        explicit_files=bool(cert_env or key_env),
        self_signed_cert=tls_dir / f"{hostname}.selfsigned.pem",
        self_signed_key=tls_dir / f"{hostname}.selfsigned.key",
    )


# ---- TLS ------------------------------------------------------------------------------------
@dataclass
class TlsInfo:
    context: ssl.SSLContext
    cert: Path
    self_signed: bool
    generated: bool
    not_after: datetime | None

    def describe(self) -> str:
        kind = "self-signed certificate" + (", generated now" if self.generated else "") if self.self_signed else "certificate"
        until = f", valid until {self.not_after:%Y-%m-%d}" if self.not_after else ""
        return f"{kind} {self.cert}{until}"


def _write_new(path: Path, data: bytes, mode: int) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    with os.fdopen(fd, "wb") as f:
        f.write(data)


def ensure_self_signed(hostname: str, cert_path: Path, key_path: Path) -> bool:
    """Make sure the bot's self-signed pair for ``hostname`` exists; True when generated now. Never touches
    the admin's files (other names). A half-written pair of the bot's own files is replaced."""
    if cert_path.is_file() and key_path.is_file():
        return False
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    cert_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    for path in (cert_path, key_path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hostname)])
    try:
        alt: x509.GeneralName = x509.IPAddress(ipaddress.ip_address(hostname))
    except ValueError:
        alt = x509.DNSName(hostname)
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=SELF_SIGNED_DAYS))
        .add_extension(x509.SubjectAlternativeName([alt]), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    _write_new(key_path, key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()), 0o600)
    _write_new(cert_path, cert.public_bytes(serialization.Encoding.PEM), 0o644)
    return True


def _not_after(cert_path: Path) -> datetime | None:
    try:
        from cryptography import x509

        return x509.load_pem_x509_certificate(cert_path.read_bytes()).not_valid_after_utc
    except Exception:  # noqa: BLE001 - only for the log line
        return None


def load_tls(settings: WebSettings) -> TlsInfo:
    """The server's TLS context: the admin's files when both exist, else the self-signed pair. Blocking."""
    if settings.cert.is_file() and settings.key.is_file():
        cert, key, self_signed, generated = settings.cert, settings.key, False, False
    else:
        if settings.explicit_files:
            logger.warning("OAUTH_TLS_CERT / OAUTH_TLS_KEY (%s, %s) not found: using a self-signed certificate",
                           settings.cert, settings.key)
        generated = ensure_self_signed(settings.hostname, settings.self_signed_cert, settings.self_signed_key)
        cert, key, self_signed = settings.self_signed_cert, settings.self_signed_key, True
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(str(cert), str(key))
    return TlsInfo(context=context, cert=cert, self_signed=self_signed, generated=generated, not_after=_not_after(cert))


# ---- the app --------------------------------------------------------------------------------
def _page(title: str, body: str, *, status: int = 200, link: bool = False) -> web.Response:
    extra = ""
    if link and _bot_username:
        extra = f'<p><a href="tg://resolve?domain={html.escape(_bot_username)}">Open Telegram</a></p>'
    text = (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>body{{font-family:system-ui,sans-serif;max-width:32rem;"
        "margin:15vh auto;padding:0 1rem;text-align:center;line-height:1.5}a{font-weight:600}</style></head>"
        f"<body><h1>{html.escape(title)}</h1><p>{html.escape(body)}</p>{extra}</body></html>"
    )
    return web.Response(text=text, status=status, content_type="text/html", charset="utf-8")


EXPIRED_PAGE = ("Sign-in link expired", "This sign-in link has expired. Start again with /mcp in Telegram.")
DONE_PAGE = ("✅ Signed in", "You can return to Telegram.")
NOT_COMPLETED_PAGE = ("Sign-in not completed", "The sign-in was cancelled or refused. You can return to Telegram.")


async def handle_callback(request: web.Request) -> web.Response:
    query = request.rel_url.query
    login = oauth_flow.resolve(query.get("state"), code=query.get("code"), error=query.get("error"),
                               error_description=query.get("error_description"), iss=query.get("iss"))
    if login is None:
        logger.info("OAuth callback with an unknown, expired or used state")
        return _page(*EXPIRED_PAGE, status=400)
    if query.get("error") or not query.get("code"):
        logger.info("OAuth callback for %s of %s: the sign-in was not completed", login.server, login.user_id)
        return _page(*NOT_COMPLETED_PAGE, link=True)
    logger.info("OAuth callback for %s of %s", login.server, login.user_id)
    return _page(*DONE_PAGE, link=True)


async def handle_health(request: web.Request) -> web.Response:
    return web.Response(text="ok", content_type="text/plain")


async def _harden(request: web.Request, response: web.StreamResponse) -> None:
    response.headers.update(SECURITY_HEADERS)
    response.headers["Server"] = "gall"


def build_app() -> web.Application:
    app = web.Application(client_max_size=MAX_BODY)
    app.router.add_get(CALLBACK_PATH, handle_callback, allow_head=False)
    app.router.add_get(HEALTH_PATH, handle_health, allow_head=False)
    app.on_response_prepare.append(_harden)
    return app


# ---- serving --------------------------------------------------------------------------------
@dataclass
class RunningServer:
    runner: web.AppRunner
    settings: WebSettings
    tls: TlsInfo | None

    @property
    def port(self) -> int:
        """The bound port (the configured one, or the one the OS picked for port 0 in tests)."""
        for address in self.runner.addresses:
            if isinstance(address, tuple):
                return int(address[1])
        return self.settings.port

    async def close(self) -> None:
        global _callback_url
        _callback_url = None
        await self.runner.cleanup()


async def start(settings: WebSettings) -> RunningServer:
    """Bind the endpoint (TLS loaded in a worker thread); raises when it cannot. Sets ``callback_url``."""
    global _callback_url
    tls = await asyncio.to_thread(load_tls, settings) if settings.tls else None
    runner = web.AppRunner(build_app(), access_log=None, handle_signals=False,
                           max_line_size=MAX_LINE, max_field_size=MAX_LINE, max_headers=MAX_HEADERS)
    await runner.setup()
    try:
        site = web.TCPSite(runner, settings.host, settings.port, ssl_context=tls.context if tls else None)
        await site.start()
    except BaseException:
        await runner.cleanup()
        raise
    server = RunningServer(runner=runner, settings=settings, tls=tls)
    _callback_url = settings.public_base_url + CALLBACK_PATH
    logger.info("OAuth callback endpoint listening on %s (%s); sign-ins return to %s",
                settings.listen, tls.describe() if tls else "plain HTTP", _callback_url)
    return server


async def serve() -> None:
    """The "oauth-web" background task: start the endpoint, then wait until cancelled (shutdown)."""
    try:
        settings = settings_from_env()
    except ValueError as e:
        logger.error("OAuth callback endpoint disabled: %s; MCP sign-ins use the copy-paste fallback", e)
        return
    if settings is None:
        logger.info("PUBLIC_BASE_URL is not set: MCP sign-ins use the copy-paste fallback")
        return
    try:
        server = await start(settings)
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001 - the bot runs on without the endpoint
        logger.error("Could not start the OAuth callback endpoint on %s: %s: %s; MCP sign-ins use the copy-paste fallback",
                     settings.listen, type(e).__name__, e)
        return
    try:
        await asyncio.Event().wait()
    finally:
        await server.close()


def reset() -> None:
    """Forget the running state (tests)."""
    global _callback_url, _bot_username
    _callback_url = None
    _bot_username = None
