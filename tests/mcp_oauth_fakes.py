"""A fake OAuth-protected MCP server for the sign-in tests: one httpx2 MockTransport handler serving the
MCP endpoint (401 without a valid bearer token, JSON-RPC otherwise), its protected resource metadata, and
an authorization server (metadata, dynamic client registration, token endpoint with PKCE and refresh-token
rotation). ``authorize`` plays the browser: it checks the authorization URL like a provider would and
returns the redirect target with a fresh code. No network anywhere."""
from __future__ import annotations

import base64
import hashlib
import json
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx2

MCP_URL = "https://mcp.example.com/mcp"
AUTH = "https://auth.example.com"
PRM_URL = "https://mcp.example.com/.well-known/oauth-protected-resource/mcp"


def json_response(data, status=200, headers=None):
    return httpx2.Response(status, json=data, headers=headers)


class FakeOAuthServer:
    def __init__(self, *, tools=None, auth_host=AUTH):
        self.auth = auth_host
        self.clients: dict[str, dict] = {}
        self.codes: dict[str, dict] = {}
        self.access: set[str] = set()
        self.refresh: dict[str, str] = {}
        self.requests: list[tuple[str, str]] = []
        self.tools = tools if tools is not None else [
            {"name": "search", "description": "Search pages", "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}}},
            {"name": "fetch", "description": "Fetch a page", "inputSchema": {"type": "object", "properties": {}}},
        ]
        self.expires_in: int | None = 3600
        self.refresh_fails = False
        self.registrations = 0
        self.tokens_issued = 0
        self.refreshes = 0
        self.calls: list[tuple[str, dict]] = []
        self.registration_status = 201
        self.token_body_extra: dict = {}

    # ---- the browser -------------------------------------------------------------------------
    def authorize(self, authorization_url: str) -> tuple[str, str, str]:
        """(redirect_uri, code, state) after checking the authorization request like a provider."""
        parts = urlsplit(authorization_url)
        assert f"{parts.scheme}://{parts.netloc}" == self.auth and parts.path == "/authorize"
        q = {k: v[0] for k, v in parse_qs(parts.query).items()}
        assert q["response_type"] == "code" and q["code_challenge_method"] == "S256"
        client = self.clients[q["client_id"]]
        assert q["redirect_uri"] in client["redirect_uris"], (q["redirect_uri"], client["redirect_uris"])
        code = f"code-{len(self.codes) + 1}-SECRETCODE"
        self.codes[code] = {"client_id": q["client_id"], "challenge": q["code_challenge"], "redirect_uri": q["redirect_uri"]}
        return q["redirect_uri"], code, q["state"]

    def redirect_target(self, authorization_url: str) -> str:
        redirect, code, state = self.authorize(authorization_url)
        return f"{redirect}?{urlencode({'code': code, 'state': state})}"

    # ---- tokens --------------------------------------------------------------------------------
    def _issue(self, client_id: str) -> dict:
        self.tokens_issued += 1
        access, refresh = f"at-{self.tokens_issued}-SECRETACCESS", f"rt-{self.tokens_issued}-SECRETREFRESH"
        self.access.add(access)
        self.refresh[refresh] = client_id
        body = {"access_token": access, "token_type": "bearer", "refresh_token": refresh, **self.token_body_extra}
        if self.expires_in is not None:
            body["expires_in"] = self.expires_in
        return body

    def expire_all_access_tokens(self) -> None:
        self.access.clear()

    # ---- the transport -------------------------------------------------------------------------
    def handler(self, request: httpx2.Request) -> httpx2.Response:
        url = request.url
        self.requests.append((request.method, f"{url.scheme}://{url.host}{url.path}"))
        origin = f"{url.scheme}://{url.host}"
        if origin == "https://mcp.example.com":
            if url.path.startswith("/.well-known/oauth-protected-resource"):
                return json_response({"resource": MCP_URL, "authorization_servers": [self.auth]})
            if url.path == "/mcp":
                return self._mcp(request)
        if origin == self.auth:
            if url.path in ("/.well-known/oauth-authorization-server", "/.well-known/openid-configuration"):
                return json_response({
                    "issuer": self.auth, "authorization_endpoint": f"{self.auth}/authorize",
                    "token_endpoint": f"{self.auth}/token", "registration_endpoint": f"{self.auth}/register",
                    "response_types_supported": ["code"], "grant_types_supported": ["authorization_code", "refresh_token"],
                    "code_challenge_methods_supported": ["S256"], "token_endpoint_auth_methods_supported": ["none"],
                })
            if url.path == "/register" and request.method == "POST":
                return self._register(request)
            if url.path == "/token" and request.method == "POST":
                return self._token(request)
        return httpx2.Response(404)

    def _register(self, request):
        if self.registration_status != 201:
            return json_response({"error": "invalid_client_metadata", "error_description": "registration closed"},
                                 status=self.registration_status)
        body = json.loads(request.content)
        self.registrations += 1
        client_id = f"client-{self.registrations}"
        record = {**body, "client_id": client_id, "client_id_issued_at": 1700000000}
        self.clients[client_id] = record
        return json_response(record, status=201)

    def _token(self, request):
        form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
        if form.get("grant_type") == "authorization_code":
            grant = self.codes.pop(form.get("code", ""), None)
            if grant is None or grant["client_id"] != form.get("client_id") or grant["redirect_uri"] != form.get("redirect_uri"):
                return json_response({"error": "invalid_grant"}, status=400)
            verifier = form.get("code_verifier", "")
            challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
            if challenge != grant["challenge"]:
                return json_response({"error": "invalid_grant", "error_description": "PKCE"}, status=400)
            return json_response(self._issue(grant["client_id"]))
        if form.get("grant_type") == "refresh_token":
            self.refreshes += 1
            client_id = self.refresh.pop(form.get("refresh_token", ""), None)
            if self.refresh_fails or client_id is None or client_id != form.get("client_id"):
                return json_response({"error": "invalid_grant"}, status=400)
            return json_response(self._issue(client_id))
        return json_response({"error": "unsupported_grant_type"}, status=400)

    def _mcp(self, request):
        header = request.headers.get("authorization", "")
        if not header.startswith("Bearer ") or header[len("Bearer "):] not in self.access:
            return httpx2.Response(401, headers={"WWW-Authenticate": f'Bearer resource_metadata="{PRM_URL}"'})
        if request.method != "POST":
            return httpx2.Response(405)
        message = json.loads(request.content)
        method, mid = message.get("method"), message.get("id")
        if mid is None:
            return httpx2.Response(202)
        if method == "initialize":
            result = {"protocolVersion": message["params"]["protocolVersion"], "capabilities": {"tools": {}},
                      "serverInfo": {"name": "fake", "version": "1"}}
        elif method == "tools/list":
            result = {"tools": self.tools}
        elif method == "tools/call":
            params = message.get("params") or {}
            self.calls.append((params.get("name"), params.get("arguments") or {}))
            result = {"content": [{"type": "text", "text": f"{params.get('name')} ok"}], "isError": False}
        else:
            return json_response({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": "Method not found"}})
        return json_response({"jsonrpc": "2.0", "id": mid, "result": result})
