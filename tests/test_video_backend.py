"""Veo on Google Cloud's Agent Platform (formerly Vertex AI): client choice, request flow, hints.

The Gemini API shut every Veo model down on 2026-10-22; Veo 3.1 GA (veo-3.1-generate-001) is served by
the Agent Platform, which the SDK calls with vertexai=True.
"""
import base64
import json
import os

import httpx
import pytest
from google import genai
from google.genai import types

os.environ.setdefault("GOOGLE_API_KEY", "x")  # the module builds its client at import

from agents import video_tools as vt  # noqa: E402


@pytest.fixture
def no_video_env(monkeypatch):
    for name in ("GOOGLE_VERTEX_API_KEY", "GOOGLE_VERTEX_CREDENTIALS", "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION",
                 "GOOGLE_GENAI_USE_VERTEXAI"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "gemini-key")


def service_account_json(project="gall-video"):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    return json.dumps({
        "type": "service_account", "project_id": project, "private_key_id": "1", "private_key": pem.decode(),
        "client_email": f"veo@{project}.iam.gserviceaccount.com", "client_id": "1",
        "token_uri": "https://oauth2.googleapis.com/token",
    })


def test_the_client_goes_where_veo_answers(no_video_env, monkeypatch):
    client, endpoint = vt.build_client()
    assert endpoint == "Gemini API" and not client._api_client.vertexai  # no Veo model there any more

    monkeypatch.setenv("GOOGLE_VERTEX_API_KEY", "AQ.agent-platform-key")
    client, endpoint = vt.build_client()
    assert endpoint == "Agent Platform (API key, express mode)"
    assert client._api_client.vertexai and client._api_client.api_key == "AQ.agent-platform-key"
    assert client._api_client._http_options.base_url == "https://aiplatform.googleapis.com/"


def test_a_project_sends_the_key_to_the_project_scoped_address(no_video_env, monkeypatch):
    """Express mode (no project) gets RESOURCE_PROJECT_INVALID from Veo; with a project the key goes regional."""
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"name": "projects/p/locations/us-central1/operations/1"})

    monkeypatch.setenv("GOOGLE_VERTEX_API_KEY", "AQ.key")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "gall-video")
    client, endpoint = vt.build_client()
    assert endpoint == "Agent Platform (API key, gall-video, us-central1)"
    api = client._api_client
    assert api.api_key == "AQ.key" and api.project == "gall-video" and api.location == "us-central1"

    client = genai.Client(vertexai=True, api_key="AQ.key", project="gall-video", location="us-central1",
                          http_options=types.HttpOptions(client_args={"transport": httpx.MockTransport(handler)}))
    client.models.generate_videos(model=vt.VEO_MODEL, prompt="x")
    assert seen[0].url.host == "us-central1-aiplatform.googleapis.com"
    assert seen[0].url.path == f"/v1beta1/projects/gall-video/locations/us-central1/publishers/google/models/{vt.VEO_MODEL}:predictLongRunning"
    assert seen[0].headers["x-goog-api-key"] == "AQ.key"


@pytest.mark.parametrize("encode", [lambda s: s, lambda s: base64.b64encode(s.encode()).decode()])
def test_a_service_account_key_wins_and_may_be_base64(no_video_env, monkeypatch, encode):
    monkeypatch.setenv("GOOGLE_VERTEX_API_KEY", "AQ.key")
    monkeypatch.setenv("GOOGLE_VERTEX_CREDENTIALS", encode(service_account_json()))
    client, endpoint = vt.build_client()
    assert endpoint == "Agent Platform (service account, gall-video, us-central1)"
    api = client._api_client
    assert api.vertexai and api.project == "gall-video" and api.location == "us-central1" and not api.api_key

    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "europe-west4")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "other")
    assert vt.build_client()[1] == "Agent Platform (service account, other, europe-west4)"


def test_a_broken_service_account_value_falls_back_and_is_logged(no_video_env, monkeypatch, caplog):
    monkeypatch.setenv("GOOGLE_VERTEX_CREDENTIALS", "not json, not base64 of json")
    monkeypatch.setenv("GOOGLE_VERTEX_API_KEY", "AQ.key")
    assert vt.build_client()[1] == "Agent Platform (API key, express mode)"
    assert "GOOGLE_VERTEX_CREDENTIALS is unusable" in caplog.text and "AQ.key" not in caplog.text

    monkeypatch.setenv("GOOGLE_VERTEX_CREDENTIALS", json.dumps({"type": "authorized_user"}))
    monkeypatch.delenv("GOOGLE_VERTEX_API_KEY")
    assert vt.build_client()[1] == "Gemini API"


def test_google_cloud_refusals_get_a_hint(monkeypatch):
    monkeypatch.setattr(vt, "VIDEO_ENDPOINT", "Agent Platform (API key, express mode)")
    project = vt.error_hint(Exception(
        "400 INVALID_ARGUMENT. {'error': {'message': 'Invalid resource field value in the request.', "
        "'details': [{'reason': 'RESOURCE_PROJECT_INVALID'}]}}"))
    assert "GOOGLE_CLOUD_PROJECT" in project and "service account" in project  # what the first live call returned
    assert "service account" in vt.error_hint(Exception("403 API_KEY_SERVICE_BLOCKED"))
    assert "service account" in vt.error_hint(Exception("401 UNAUTHENTICATED. API keys are not supported by this API."))
    assert "Vertex AI User" in vt.error_hint(Exception("403 PERMISSION_DENIED"))
    assert "VEO_MODEL" in vt.error_hint(Exception("404 NOT_FOUND. models/x is not found"))
    assert vt.error_hint(Exception("content policy")) == ""


class Sender:
    def __init__(self):
        self.sent = []

    async def send_text(self, text):
        self.sent.append(("text", text))

    async def send_video(self, path, caption=None):
        self.sent.append(("video", path))


VIDEO = b"\x00\x00\x00\x18ftypmp42" + b"0" * 16


def fake_agent_platform(seen, polls_before_done=1):
    state = {"polls": 0}

    def handler(request: httpx.Request):
        body = json.loads(request.content or b"{}")
        seen.append((request.url.path, request.headers.get("x-goog-api-key"), body))
        if request.url.path.endswith(":predictLongRunning"):
            return httpx.Response(200, json={"name": "publishers/google/models/veo-3.1-generate-001/operations/op1"})
        state["polls"] += 1
        if state["polls"] <= polls_before_done:
            return httpx.Response(200, json={"name": "publishers/google/models/veo-3.1-generate-001/operations/op1"})
        return httpx.Response(200, json={
            "name": "publishers/google/models/veo-3.1-generate-001/operations/op1", "done": True,
            "response": {"videos": [{"bytesBase64Encoded": base64.b64encode(VIDEO).decode(), "mimeType": "video/mp4"}]},
        })

    return genai.Client(vertexai=True, api_key="AQ.key", http_options=types.HttpOptions(
        client_args={"transport": httpx.MockTransport(handler)}))


@pytest.fixture
def agent_platform(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "7" / "images").mkdir(parents=True)
    seen = []
    monkeypatch.setattr(vt, "genai_client", fake_agent_platform(seen))
    monkeypatch.setattr(vt, "ON_AGENT_PLATFORM", True)
    monkeypatch.setattr(vt, "VIDEO_TOOLS_ENABLED", True)
    monkeypatch.setattr(vt, "POLL_INTERVAL_SECONDS", 0)
    return tmp_path, seen


async def test_image_to_video_runs_end_to_end_on_the_agent_platform(agent_platform):
    """Through the real SDK: start (predictLongRunning), poll (fetchPredictOperation), bytes inline."""
    from PIL import Image

    workspace, seen = agent_platform
    Image.new("RGB", (64, 96), (200, 100, 50)).save(workspace / "data" / "7" / "images" / "pic.jpg")
    sender = Sender()
    result = await vt.VideoTools("7", sender).execute_tool(
        "image_to_video_generator", {"prompt": "coffee pours", "image_path": "images/pic.jpg", "orientation": "portrait"})

    assert result.startswith("✅"), result
    start, *polls = seen
    assert start[0] == f"/v1beta1/publishers/google/models/{vt.VEO_MODEL}:predictLongRunning" and start[1] == "AQ.key"
    assert start[2]["parameters"]["generateAudio"] is True and start[2]["parameters"]["aspectRatio"] == "9:16"
    assert "image" in start[2]["instances"][0]  # the picture goes inline, no upload
    assert polls and all(p[0].endswith(":fetchPredictOperation") for p in polls)
    video = [p for kind, p in sender.sent if kind == "video"]
    assert len(video) == 1 and open(video[0], "rb").read() == VIDEO  # saved from the inline bytes


async def test_extension_sends_the_video_inline_on_the_agent_platform(agent_platform, monkeypatch):
    workspace, seen = agent_platform
    (workspace / "data" / "7" / "videos").mkdir(parents=True, exist_ok=True)
    (workspace / "data" / "7" / "videos" / "clip.mp4").write_bytes(VIDEO)

    def no_upload(**kw):
        raise AssertionError("files.upload exists only on the Gemini API")

    monkeypatch.setattr(vt.genai_client.files, "upload", no_upload)
    result = await vt.VideoTools("7", Sender()).execute_tool(
        "video_extension_generator", {"prompt": "and then it rains", "source_video_path": "videos/clip.mp4"})
    assert result.startswith("✅"), result
    assert "video" in seen[0][2]["instances"][0]


async def test_a_refused_api_key_is_explained_to_the_agent(agent_platform, monkeypatch):
    workspace, _ = agent_platform
    monkeypatch.setattr(vt, "VIDEO_ENDPOINT", "Agent Platform (API key, express mode)")

    def refuse(request):
        return httpx.Response(401, json={"error": {"code": 401, "status": "UNAUTHENTICATED",
                                                    "message": "API keys are not supported by this API."}})

    monkeypatch.setattr(vt, "genai_client", genai.Client(vertexai=True, api_key="AQ.key", http_options=types.HttpOptions(
        client_args={"transport": httpx.MockTransport(refuse)})))
    sender = Sender()
    result = await vt.VideoTools("7", sender).execute_tool("video_generator", {"prompt": "a cat"})
    assert result.startswith("Error generating video") and "GOOGLE_VERTEX_CREDENTIALS" in result
    assert any("failed" in text for kind, text in sender.sent if kind == "text")
