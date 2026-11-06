import asyncio
import logging
import os
import time
from typing import Dict, Any
from dotenv import load_dotenv
from pathlib import Path
import uuid
from google import genai
from google.genai import types
from google.genai.types import Image, VideoGenerationReferenceImage
from models import (
    VEO_DEFAULT_SECONDS,
    VEO_EXTENSION_SECONDS,
    VEO_GENERATE_AUDIO,
    VEO_MODEL,
    VIDEO_BACKEND,
    VIDEO_TOOLS_ENABLED,
    estimate_video_cost,
)

VIDEO_UNAVAILABLE = (
    "Error: video generation is not available: Veo now runs only on Google Cloud's Agent Platform, and the bot "
    "has no Agent Platform credentials (GOOGLE_VERTEX_API_KEY or GOOGLE_VERTEX_CREDENTIALS). "
    "Tell the user; for an animation use generate_animation."
)

load_dotenv()

logger = logging.getLogger(__name__)

CLOUD_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
DEFAULT_LOCATION = "us-central1"  # where Veo 3.1 GA is served


def _service_account_info(raw: str) -> dict:
    """GOOGLE_VERTEX_CREDENTIALS: the service-account key as JSON on one line, or base64 of it
    (the PROD .env renderer drops values that contain a newline)."""
    import base64
    import json

    text = raw.strip()
    if not text.startswith("{"):
        text = base64.b64decode(text).decode("utf-8")
    info = json.loads(text)
    if not isinstance(info, dict) or info.get("type") != "service_account":
        raise ValueError("not a service-account key")
    return info


def build_client():
    """The genai client and a label of where Veo calls go.

    Agent Platform (formerly Vertex AI) with a service account (every method accepts it), else
    with an Agent Platform API key (express mode), else the Gemini API (no Veo model answers
    there since 2026-10-22). A broken service-account value is logged and the next option used.
    """
    raw = (os.getenv("GOOGLE_VERTEX_CREDENTIALS") or "").strip()
    if raw:
        try:
            from google.oauth2 import service_account

            info = _service_account_info(raw)
            credentials = service_account.Credentials.from_service_account_info(info, scopes=[CLOUD_SCOPE])
            project = (os.getenv("GOOGLE_CLOUD_PROJECT") or "").strip() or info.get("project_id")
            location = (os.getenv("GOOGLE_CLOUD_LOCATION") or "").strip() or DEFAULT_LOCATION
            client = genai.Client(vertexai=True, project=project, location=location, credentials=credentials)
            return client, f"Agent Platform (service account, {project}, {location})"
        except Exception as e:  # noqa: BLE001 - one bad secret must not stop the bot
            logger.error("GOOGLE_VERTEX_CREDENTIALS is unusable (%s); trying the next video credential", e)
    api_key = (os.getenv("GOOGLE_VERTEX_API_KEY") or "").strip()
    if api_key:
        return genai.Client(vertexai=True, api_key=api_key), "Agent Platform (API key)"
    return genai.Client(api_key=os.getenv("GOOGLE_API_KEY")), "Gemini API"


genai_client, VIDEO_ENDPOINT = build_client()
ON_AGENT_PLATFORM = VIDEO_ENDPOINT.startswith("Agent Platform")
logger.info("Video tools: %s, model %s, listed: %s", VIDEO_ENDPOINT, VEO_MODEL, VIDEO_TOOLS_ENABLED)


def error_hint(error: BaseException) -> str:
    """What to do about a Google Cloud refusal, for the agent and the log."""
    text = str(error)
    low = text.lower()
    if "api keys are not supported" in low or ("401" in text and "Agent Platform (API key)" == VIDEO_ENDPOINT):
        return (
            " (The Agent Platform refused the API key for this call. Bind the key to a service account in the "
            "Google Cloud console, or give the bot a service-account key as GOOGLE_VERTEX_CREDENTIALS.)"
        )
    if "403" in text or "permission_denied" in low:
        return (
            " (Google Cloud refused access: enable the Agent Platform (Vertex AI) API and billing for the project, "
            "and allow the key or service account to use it, e.g. the role Vertex AI User.)"
        )
    if "404" in text or "not_found" in low:
        return (
            f" (The model {VEO_MODEL} is not available here: check VEO_MODEL (veo-3.1-generate-001, "
            "veo-3.1-fast-generate-001) and GOOGLE_CLOUD_LOCATION, or Veo access for the project.)"
        )
    if "only supported in the gemini developer client" in low:
        return " (This SDK call does not exist on the Agent Platform; the bot needs a code update.)"
    return ""

MAX_WAIT_SECONDS = 300  # 5 minutes max
POLL_INTERVAL_SECONDS = 20


class VideoTools:
    def __init__(self, user_id: str, sender):
        """
        Args:
            user_id: chat id of the user as a string
            sender: bot.sender.ChatSender bound to the user's chat
        """
        self.user_id = user_id
        self.sender = sender
        self.trace = None  # agents.trace.ToolTrace of the current turn, set by ChainOfThoughtAgent
        self.base_path = Path("./data") / str(user_id)
        # Create base directory if it doesn't exist
        self.base_path.mkdir(parents=True, exist_ok=True)
        # Create videos directory
        self.videos_path = self.base_path / "videos"
        self.videos_path.mkdir(parents=True, exist_ok=True)

        # Every tool, listed or not; the agent sees them only while VIDEO_TOOLS is on (models.py).
        self.all_tools_schema = [
            {
                "name": "video_generator",
                "description": "Generate high-quality videos from text descriptions using Google Veo 3.1 (state-of-the-art text-to-video generation with cinematic quality), save to videos directory, and send to user via Telegram. Supports detailed scene descriptions with camera movements, lighting, and actions. Use only when the user asks for a video; for an animation, a GIF, a sticker or a loop use generate_animation.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "A detailed description of the video you want to generate. Be specific with visual details, actions, camera movements, lighting, scene composition for best results. Example: 'a close-up shot of a golden retriever playing in a field of sunflowers'"
                        },
                        "orientation": {
                            "type": "string",
                            "description": "Video orientation: 'horizontal' for landscape (16:9, default), 'portrait' for vertical (9:16). Use 'portrait' when user asks for vertical/portrait video, otherwise use 'horizontal'.",
                            "enum": ["horizontal", "portrait"],
                            "default": "horizontal"
                        },
                        "quality": {
                            "type": "string",
                            "description": "Video quality/resolution: '720p' (default) or '1080p'. Use '720p' always, if user not specified otherwise.",
                            "enum": ["720p", "1080p"],
                            "default": "720p"
                        },
                        "negative_prompt": {
                            "type": "string",
                            "description": "Things you want to avoid in the video (e.g., 'blurry, low quality, static'). Optional parameter to exclude unwanted elements.",
                            "default": ""
                        },
                        "caption": {
                            "type": "string",
                            "description": "Caption for the video when sending to user",
                            "default": "Here is your generated video"
                        }
                    },
                    "required": ["prompt"]
                }
            },
            {
                "name": "image_to_video_generator",
                "description": "Generate a video (an MP4 clip of several seconds) from an existing image using Google Veo 3.1, following the prompt's motion, camera movements and scene evolution. Use only when the user explicitly asks for a video; for an animation, a GIF or 'animate this picture' use generate_animation.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "A detailed description of how the video should evolve from the starting image. Describe the motion, actions, camera movements, and changes you want to see. Example: 'The camera slowly zooms out while the dog starts running towards the camera, tail wagging'"
                        },
                        "image_path": {
                            "type": "string",
                            "description": "Path to the starting image file. This should be a full path to an image file in the user's directory (e.g., from images folder)."
                        },
                        "orientation": {
                            "type": "string",
                            "description": "Video orientation: 'horizontal' for landscape (16:9, default), 'portrait' for vertical (9:16). Use 'portrait' when user asks for vertical/portrait video, otherwise use 'horizontal'.",
                            "enum": ["horizontal", "portrait"],
                            "default": "horizontal"
                        },
                        "quality": {
                            "type": "string",
                            "description": "Video quality/resolution: '720p' (default) or '1080p'. Use '720p' always, if user not specified otherwise.",
                            "enum": ["720p", "1080p"],
                            "default": "720p"
                        },
                        "negative_prompt": {
                            "type": "string",
                            "description": "Things you want to avoid in the video (e.g., 'blurry, low quality, static, no motion'). Optional parameter to exclude unwanted elements.",
                            "default": ""
                        },
                        "caption": {
                            "type": "string",
                            "description": "Caption for the video when sending to user",
                            "default": "Here is your generated video from the image"
                        }
                    },
                    "required": ["prompt", "image_path"]
                }
            },
            {
                "name": "video_from_reference_images",
                "description": "Generate high-quality videos using reference images (assets like clothing, objects, characters) with Google Veo 3.1. Perfect for maintaining consistent visual elements across the video. NOTE: Currently only works with horizontal/16:9 aspect ratio.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "A detailed description of the video scene and how the reference images should be incorporated. Be specific about actions, camera movements, and scene composition."
                        },
                        "reference_image_paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Array of paths to reference images. These images represent assets (objects, clothing, characters) that should appear in the generated video. Each image will be used as a visual reference for the video generation."
                        },
                        "quality": {
                            "type": "string",
                            "description": "Video quality/resolution: '720p' (default) or '1080p'. Use '720p' always, if user not specified otherwise.",
                            "enum": ["720p", "1080p"],
                            "default": "720p"
                        },
                        "negative_prompt": {
                            "type": "string",
                            "description": "Things you want to avoid in the video (e.g., 'blurry, low quality, static'). Optional parameter to exclude unwanted elements.",
                            "default": ""
                        },
                        "caption": {
                            "type": "string",
                            "description": "Caption for the video when sending to user",
                            "default": "Here is your generated video with reference images"
                        }
                    },
                    "required": ["prompt", "reference_image_paths"]
                }
            },
            {
                "name": "video_interpolation_generator",
                "description": "Generate high-quality videos with specified first and last frames using Google Veo 3.1. The AI will interpolate the motion between the two frames. Perfect for creating smooth transitions between two specific moments.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "A detailed description of what should happen in the video between the first and last frame. Describe the motion, actions, transitions, and scene evolution."
                        },
                        "first_frame_path": {
                            "type": "string",
                            "description": "Path to the first frame image file. This will be the starting frame of the video."
                        },
                        "last_frame_path": {
                            "type": "string",
                            "description": "Path to the last frame image file. This will be the ending frame of the video."
                        },
                        "orientation": {
                            "type": "string",
                            "description": "Video orientation: 'horizontal' for landscape (16:9, default), 'portrait' for vertical (9:16).",
                            "enum": ["horizontal", "portrait"],
                            "default": "horizontal"
                        },
                        "quality": {
                            "type": "string",
                            "description": "Video quality/resolution: '720p' (default) or '1080p'. Use '720p' always, if user not specified otherwise.",
                            "enum": ["720p", "1080p"],
                            "default": "720p"
                        },
                        "negative_prompt": {
                            "type": "string",
                            "description": "Things you want to avoid in the video (e.g., 'blurry, low quality, static'). Optional parameter to exclude unwanted elements.",
                            "default": ""
                        },
                        "caption": {
                            "type": "string",
                            "description": "Caption for the video when sending to user",
                            "default": "Here is your interpolated video"
                        }
                    },
                    "required": ["prompt", "first_frame_path", "last_frame_path"]
                }
            },
            {
                "name": "video_extension_generator",
                "description": "Extend an existing video by generating additional content that continues from where the original video ends using Google Veo 3.1. NOTE: Currently only works with 720p resolution.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "A detailed description of what should happen in the extended portion of the video. Describe the continuation, new actions, camera movements, and scene evolution."
                        },
                        "source_video_path": {
                            "type": "string",
                            "description": "Path to the source video file that should be extended. The new content will continue from the end of this video."
                        },
                        "caption": {
                            "type": "string",
                            "description": "Caption for the video when sending to user",
                            "default": "Here is your extended video"
                        }
                    },
                    "required": ["prompt", "source_video_path"]
                }
            }
        ]
        self.tools_schema = self.all_tools_schema if VIDEO_TOOLS_ENABLED else []

    async def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        if not VIDEO_TOOLS_ENABLED and tool_name in {t["name"] for t in self.all_tools_schema}:
            return VIDEO_UNAVAILABLE
        if tool_name == "video_generator":
            return await self._video_generator(**tool_args)
        elif tool_name == "image_to_video_generator":
            return await self._image_to_video_generator(**tool_args)
        elif tool_name == "video_from_reference_images":
            return await self._video_from_reference_images(**tool_args)
        elif tool_name == "video_interpolation_generator":
            return await self._video_interpolation_generator(**tool_args)
        elif tool_name == "video_extension_generator":
            return await self._video_extension_generator(**tool_args)
        return f"Unknown tool: {tool_name}"

    # ------------------------------------------------------------------
    # helpers (every Veo call runs in a worker thread; polling is async)
    # ------------------------------------------------------------------
    def _resolve_path(self, path: str) -> Path:
        """Resolve a model-supplied path inside the user's workspace: relative, sandbox or host spelling, or a bare name in images/videos/downloads."""
        from agents.paths import resolve_under

        resolved = resolve_under(self.base_path, path, fallback_dirs=("images", "videos", "downloads"))
        return resolved if resolved is not None else self.videos_path / Path(path).name

    async def _notify(self, text: str) -> None:
        try:
            await self.sender.send_text(text)
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not send video notice: %s", e)

    async def _generate_and_wait(self, **kwargs):
        """Start a generate_videos operation and poll it without blocking the event loop."""
        config = kwargs.get("config")
        if ON_AGENT_PLATFORM and config is not None and getattr(config, "generate_audio", None) is None:
            config.generate_audio = VEO_GENERATE_AUDIO  # an Agent Platform option (the Gemini API refuses it)
        operation = await asyncio.to_thread(genai_client.models.generate_videos, model=VEO_MODEL, **kwargs)
        start = time.monotonic()
        while not operation.done:
            if time.monotonic() - start > MAX_WAIT_SECONDS:
                return None
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
            operation = await asyncio.to_thread(genai_client.operations.get, operation)
        if not operation.result or not operation.result.generated_videos:
            return False
        self._record_cost(kwargs, len(operation.result.generated_videos))
        return operation.result.generated_videos[0]

    def _record_cost(self, request: Dict[str, Any], videos: int) -> None:
        """Add one Veo operation's estimated cost to the turn's trace (never raises).

        Veo bills per generated second; the tools leave duration_seconds unset, so the
        default length applies (an extension adds a shorter clip to the source video).
        """
        if self.trace is None:
            return
        try:
            seconds = getattr(request.get("config"), "duration_seconds", None)
            if not seconds:
                seconds = VEO_EXTENSION_SECONDS if "video" in request else VEO_DEFAULT_SECONDS
            self.trace.add_usage(
                {"input_tokens": 0, "output_tokens": 0},
                model=VEO_MODEL,
                cost_usd=estimate_video_cost(VEO_MODEL, seconds * videos),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not record video cost for %s: %s", VEO_MODEL, e)

    async def _save_and_send(self, generated_video, filename: str, caption: str) -> Path:
        video_path = self.videos_path / filename

        def _download() -> None:
            # The Agent Platform returns the bytes inline (no Cloud Storage bucket is given); the
            # Gemini API returns a file to download (files.download exists only there).
            if not getattr(generated_video.video, "video_bytes", None):
                genai_client.files.download(file=generated_video.video)
            generated_video.video.save(str(video_path))

        await asyncio.to_thread(_download)
        await self.sender.send_video(str(video_path), caption=caption)
        return video_path

    async def _fail(self, user_text: str, error_message: str, error: BaseException | None = None) -> str:
        error_message += error_hint(error) if error is not None else ""
        logger.error(error_message)
        await self._notify(user_text)
        return error_message

    # ------------------------------------------------------------------
    # tools
    # ------------------------------------------------------------------
    async def _video_generator(self, prompt: str, orientation: str = "horizontal", quality: str = "720p", negative_prompt: str = "", caption: str = "Here is your generated video") -> str:
        """Generate a high-quality video from text using Google's Veo model"""
        logger.info("Generating video - prompt=%r orientation=%s quality=%s", prompt, orientation, quality)
        try:
            aspect_ratio = "16:9" if orientation == "horizontal" else "9:16"
            await self._notify(f"🎬 Video generation started ({aspect_ratio}, {quality})... This may take 1-2 minutes. Please wait.")
            config = types.GenerateVideosConfig(aspect_ratio=aspect_ratio, resolution=quality)
            if negative_prompt:
                config.negative_prompt = negative_prompt

            video = await self._generate_and_wait(prompt=prompt, config=config)
            if video is None:
                return "Error: Video generation timed out after 5 minutes. Please try again with a simpler prompt."
            if video is False:
                return "Error: No video was generated. Please try again with a different prompt."

            video_path = await self._save_and_send(video, f"veo3_video_{uuid.uuid4()}.mp4", caption)
            return (
                "✅ Video generated successfully!\n\n"
                f"Aspect Ratio: {aspect_ratio}\nResolution: {quality}\nFile saved to: {video_path}\n"
                "Video has been sent to user via Telegram.\n"
            )
        except Exception as e:
            return await self._fail("❌ Sorry, video generation failed. Please try again.", f"Error generating video: {str(e)}", e)

    async def _image_to_video_generator(self, prompt: str, image_path: str, orientation: str = "horizontal", quality: str = "720p", negative_prompt: str = "", caption: str = "Here is your generated video from the image") -> str:
        """Generate a high-quality video from an image using Google's Veo model"""
        logger.info("Generating video from image - prompt=%r image=%s", prompt, image_path)
        try:
            image_path_obj = self._resolve_path(image_path)
            if not image_path_obj.exists():
                return f"Error: The image at path {image_path} does not exist."
            aspect_ratio = "16:9" if orientation == "horizontal" else "9:16"
            source_image = await asyncio.to_thread(Image.from_file, location=str(image_path_obj))
            await self._notify(f"🎬 Video generation from image started ({aspect_ratio}, {quality})... This may take 1-2 minutes. Please wait.")
            config = types.GenerateVideosConfig(aspect_ratio=aspect_ratio, resolution=quality)
            if negative_prompt:
                config.negative_prompt = negative_prompt

            video = await self._generate_and_wait(prompt=prompt, image=source_image, config=config)
            if video is None:
                return "Error: Video generation timed out after 5 minutes. Please try again with a simpler prompt."
            if video is False:
                return "Error: No video was generated. Please try again with a different prompt or image."

            video_path = await self._save_and_send(video, f"veo3_from_image_{uuid.uuid4()}.mp4", caption)
            return (
                "✅ Video generated from image successfully!\n\n"
                f"Source image: {image_path}\nAspect Ratio: {aspect_ratio}\nResolution: {quality}\nVideo saved to: {video_path}\n"
                "Video has been sent to user via Telegram.\n"
            )
        except Exception as e:
            return await self._fail("❌ Sorry, video generation from image failed. Please try again.", f"Error generating video from image: {str(e)}", e)

    async def _video_from_reference_images(self, prompt: str, reference_image_paths: list, quality: str = "720p", negative_prompt: str = "", caption: str = "Here is your generated video with reference images") -> str:
        """Generate a high-quality video using reference images with Google's Veo model"""
        logger.info("Generating video with reference images - prompt=%r images=%s", prompt, reference_image_paths)
        try:
            reference_images = []
            for image_path in reference_image_paths:
                image_path_obj = self._resolve_path(image_path)
                if not image_path_obj.exists():
                    return f"Error: The reference image at path {image_path} does not exist."
                image = await asyncio.to_thread(Image.from_file, location=str(image_path_obj))
                reference_images.append(VideoGenerationReferenceImage(image=image, reference_type="asset"))

            aspect_ratio = "16:9"  # Currently only supports horizontal 16:9
            await self._notify(f"🎬 Video generation with {len(reference_images)} reference image(s) started ({aspect_ratio}, {quality})... This may take 1-2 minutes. Please wait.")
            config = types.GenerateVideosConfig(aspect_ratio=aspect_ratio, resolution=quality, reference_images=reference_images)
            if negative_prompt:
                config.negative_prompt = negative_prompt

            video = await self._generate_and_wait(prompt=prompt, config=config)
            if video is None:
                return "Error: Video generation timed out after 5 minutes. Please try again with a simpler prompt."
            if video is False:
                return "Error: No video was generated. Please try again with a different prompt or reference images."

            video_path = await self._save_and_send(video, f"veo3_reference_{uuid.uuid4()}.mp4", caption)
            return (
                "✅ Video with reference images generated successfully!\n\n"
                f"Reference images: {len(reference_images)}\nAspect Ratio: {aspect_ratio}\nResolution: {quality}\nFile saved to: {video_path}\n"
                "Video has been sent to user via Telegram.\n"
            )
        except Exception as e:
            return await self._fail("❌ Sorry, video generation with reference images failed. Please try again.", f"Error generating video with reference images: {str(e)}", e)

    async def _video_interpolation_generator(self, prompt: str, first_frame_path: str, last_frame_path: str, orientation: str = "horizontal", quality: str = "720p", negative_prompt: str = "", caption: str = "Here is your interpolated video") -> str:
        """Generate a high-quality video with specified first and last frames using Google's Veo model"""
        logger.info("Generating interpolated video - prompt=%r first=%s last=%s", prompt, first_frame_path, last_frame_path)
        try:
            first_frame_obj = self._resolve_path(first_frame_path)
            last_frame_obj = self._resolve_path(last_frame_path)
            if not first_frame_obj.exists():
                return f"Error: The first frame at path {first_frame_path} does not exist."
            if not last_frame_obj.exists():
                return f"Error: The last frame at path {last_frame_path} does not exist."
            first_image = await asyncio.to_thread(Image.from_file, location=str(first_frame_obj))
            last_image = await asyncio.to_thread(Image.from_file, location=str(last_frame_obj))
            aspect_ratio = "16:9" if orientation == "horizontal" else "9:16"
            await self._notify(f"🎬 Video interpolation started ({aspect_ratio}, {quality})... This may take 1-2 minutes. Please wait.")
            config = types.GenerateVideosConfig(aspect_ratio=aspect_ratio, resolution=quality, last_frame=last_image)
            if negative_prompt:
                config.negative_prompt = negative_prompt

            video = await self._generate_and_wait(prompt=prompt, image=first_image, config=config)
            if video is None:
                return "Error: Video generation timed out after 5 minutes. Please try again with a simpler prompt."
            if video is False:
                return "Error: No video was generated. Please try again with different frames or prompt."

            video_path = await self._save_and_send(video, f"veo3_interpolated_{uuid.uuid4()}.mp4", caption)
            return (
                "✅ Interpolated video generated successfully!\n\n"
                f"First frame: {first_frame_path}\nLast frame: {last_frame_path}\nAspect Ratio: {aspect_ratio}\nResolution: {quality}\nVideo saved to: {video_path}\n"
                "Video has been sent to user via Telegram.\n"
            )
        except Exception as e:
            return await self._fail("❌ Sorry, interpolated video generation failed. Please try again.", f"Error generating interpolated video: {str(e)}", e)

    async def _video_extension_generator(self, prompt: str, source_video_path: str, caption: str = "Here is your extended video") -> str:
        """Extend an existing video using Google's Veo model"""
        logger.info("Extending video - prompt=%r source=%s", prompt, source_video_path)
        try:
            video_path_obj = self._resolve_path(source_video_path)
            if not video_path_obj.exists():
                return f"Error: The source video at path {source_video_path} does not exist."
            if ON_AGENT_PLATFORM:  # the video goes inline; files.upload exists only on the Gemini API
                source_video = await asyncio.to_thread(types.Video.from_file, location=str(video_path_obj), mime_type="video/mp4")
            else:
                source_video = await asyncio.to_thread(genai_client.files.upload, file=str(video_path_obj))
            quality = "720p"  # Currently only supports 720p for extension
            await self._notify(f"🎬 Video extension started ({quality})... This may take 1-2 minutes. Please wait.")
            config = types.GenerateVideosConfig(number_of_videos=1, resolution=quality)

            video = await self._generate_and_wait(video=source_video, prompt=prompt, config=config)
            if video is None:
                return "Error: Video extension timed out after 5 minutes. Please try again with a simpler prompt."
            if video is False:
                return "Error: No extended video was generated. Please try again with a different prompt."

            video_path = await self._save_and_send(video, f"veo3_extended_{uuid.uuid4()}.mp4", caption)
            return (
                "✅ Video extended successfully!\n\n"
                f"Source video: {source_video_path}\nResolution: {quality}\nExtended video saved to: {video_path}\n"
                "Video has been sent to user via Telegram.\n"
            )
        except Exception as e:
            return await self._fail("❌ Sorry, video extension failed. Please try again.", f"Error extending video: {str(e)}", e)
