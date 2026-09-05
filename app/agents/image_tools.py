import asyncio
import logging
from openai import OpenAI
import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from pathlib import Path
import math
import uuid
import base64
from google import genai
from google.genai import types
import PIL.Image
import PIL.ImageOps
from image_utils import JPEG_FORMATS
load_dotenv()

logger = logging.getLogger(__name__)

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_api_key)
google_api_key = os.getenv("GOOGLE_API_KEY")
genai_client = genai.Client(api_key=google_api_key)
max_image_resolution_edit = int(os.getenv("MAX_IMAGE_RESOLUTION_EDIT", "4096") or "4096")

GEMINI_FLASH = "gemini-3.1-flash-image-preview"
GEMINI_PRO = "gemini-3-pro-image-preview"
GPT_IMAGE = "gpt-image-2-2026-04-21"

class TextPart:
    def __init__(self, text):
        self.text = str(text)

class InlineDataPart:
    def __init__(self, mime_type, data):
        self.mime_type = str(mime_type)
        self.data = bytes(data)

class ImageTools:
    def __init__(self, user_id: str, sender):
        """
        Args:
            user_id: chat id of the user as a string
            sender: bot.sender.ChatSender bound to the user's chat
        """
        self.user_id = user_id
        self.sender = sender
        self.base_path = Path("./data") / str(user_id)
        # Create base directory if it doesn't exist
        self.base_path.mkdir(parents=True, exist_ok=True)
        # Create images directory
        self.images_path = self.base_path / "images"
        self.images_path.mkdir(parents=True, exist_ok=True)

        self.tools_schema = [
            {
                "name": "image_generator",
                "description": "Generate high-quality images from text descriptions using AI image generation, save to images directory, and send to user via Telegram. ALWAYS use Normal mode by default unless user specifically requests Pro or GPT mode.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "A detailed description of the image you want to generate. Be specific with visual details, style, lighting, composition for best results."
                        },
                        "style": {
                            "type": "string",
                            "description": "Visual style for the image (e.g., 'photorealistic', 'digital art', 'cartoon', 'anime', 'impressionist', 'minimalist'). Used for Normal and Pro modes.",
                            "default": "photorealistic"
                        },
                        "model": {
                            "type": "string",
                            "description": "Model to use: 'Normal' (faster, standard quality, gemini-3.1-flash-image-preview), 'Pro' (higher quality, more control, gemini-3-pro-image-preview), or 'GPT' (OpenAI GPT Image 2, state-of-the-art quality, excellent text rendering, great for screenshot/UI mockup generation). ALWAYS use 'Normal' unless user specifically requests Pro or GPT mode.",
                            "enum": ["Normal", "Pro", "GPT"],
                            "default": "Normal"
                        },
                        "aspect_ratio": {
                            "type": "string",
                            "description": "Aspect ratio for Pro and GPT modes. Options: '1:1', '2:3', '3:2', '3:4', '4:3', '4:5', '5:4', '9:16', '16:9', '21:9'. Default is '16:9'.",
                            "enum": ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
                            "default": "16:9"
                        },
                        "resolution": {
                            "type": "string",
                            "description": "Resolution for Pro and GPT modes. Options: '1K', '2K', '4K'. Default is '2K'. Use '4K' ONLY if user explicitly requests 4K or ultra-high resolution.",
                            "enum": ["1K", "2K", "4K"],
                            "default": "2K"
                        },
                        "gpt_quality": {
                            "type": "string",
                            "description": "Quality for GPT mode only. Options: 'low' (fast drafts), 'medium', 'high' (best quality, slow and expensive), 'auto'. Default is 'auto'. Use 'high' ONLY if user explicitly requests highest quality. Only used when model is 'GPT'.",
                            "enum": ["low", "medium", "high", "auto"],
                            "default": "auto"
                        },
                        "gpt_output_format": {
                            "type": "string",
                            "description": "Output format for GPT mode only. 'jpeg' is faster than 'png'. Default is 'png'. Only used when model is 'GPT'.",
                            "enum": ["png", "jpeg", "webp"],
                            "default": "jpeg"
                        },
                        "variants": {
                            "type": "integer",
                            "description": "Number of image variants to generate. Default is 1. Set higher (up to 4) when user wants multiple options to choose from.",
                            "enum": [1, 2, 3, 4],
                            "default": 1
                        },
                        "caption": {
                            "type": "string",
                            "description": "Caption for the image when sending to user",
                            "default": "Here is your generated image"
                        }
                    },
                    "required": ["prompt"]
                }
            },
            {
                "name": "generate_image_dall_e",
                "description": "[OBSOLETE] Legacy DALL-E 3 image generator. Use 'image_generator' instead for better quality and modern capabilities. Kept for backward compatibility only.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "A detailed description of the image you want to generate"
                        },
                        "size": {
                            "type": "string",
                            "description": "Image size in format WIDTHxHEIGHT. Supported sizes: 1024x1024, 1024x1792, 1792x1024. Use by default '1024x1024', change only if user request other size.",
                            "enum": ["1024x1024", "1024x1792", "1792x1024"]
                        },
                        "quality": {
                            "type": "string",
                            "description": "The quality of the image. Use 'standard' for faster generation, 'hd' for more detailed images. Default is 'standard', change only if user request other quality.",
                            "enum": ["standard", "hd"]
                        },
                        "caption": {
                            "type": "string",
                            "description": "Caption for the image, when you send it to user",
                            "default": "Here is your image"
                        }
                    },
                    "required": ["prompt"]
                }
            },
            {
                "name": "generate_multimodal_image_and_text",
                "description": "Generate an image and text for each image (preferred for story generation, can generate single image or multiple images in single request and generate text for each image automatically) with next generation multimodal AI, answer with text and images in a single request using Google's Gemini model, save the images, and send both text and images to the user via Telegram",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "A detailed description of what you want to generate in text and images, must be written on English language for best results"
                        },
                        "style": {
                            "type": "string",
                            "description": "The visual style for the generated images (e.g., '3d digital art', 'photorealistic', 'cartoon', 'anime', .... any type of visual style)",
                            "default": "3d digital art"
                        }
                    },
                    "required": ["prompt"]
                }
            },
            {
                "name": "image_editing",
                "description": "Edit and transform existing images using advanced AI image editing capabilities. Supports adding, removing, or modifying elements, changing styles, adjusting colors, and mask-free editing. ALWAYS use Normal mode by default unless user specifically requests Pro or GPT mode.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Detailed instructions on how to edit the image (e.g., 'Add a red hat to the person', 'Change the background to a tropical beach', 'Remove the car from the scene', 'Make it look like a vintage photograph')"
                        },
                        "image_path": {
                            "type": "string",
                            "description": "Path to the image file to be edited. This should be a full path to an image file in the user's directory."
                        },
                        "model": {
                            "type": "string",
                            "description": "Model to use: 'Normal' (faster, standard quality, gemini-3.1-flash-image-preview), 'Pro' (higher quality, more control, gemini-3-pro-image-preview), or 'GPT' (OpenAI GPT Image 2, state-of-the-art editing, great for screenshot/UI mockup editing). ALWAYS use 'Normal' unless user specifically requests Pro or GPT mode.",
                            "enum": ["Normal", "Pro", "GPT"],
                            "default": "Normal"
                        },
                        "aspect_ratio": {
                            "type": "string",
                            "description": "Aspect ratio for Pro and GPT modes. Options: '1:1', '2:3', '3:2', '3:4', '4:3', '4:5', '5:4', '9:16', '16:9', '21:9'. Default is '16:9'.",
                            "enum": ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
                            "default": "16:9"
                        },
                        "resolution": {
                            "type": "string",
                            "description": "Resolution for Pro and GPT modes. Options: '1K', '2K', '4K'. Default is '2K'. Use '4K' ONLY if user explicitly requests 4K or ultra-high resolution.",
                            "enum": ["1K", "2K", "4K"],
                            "default": "2K"
                        },
                        "gpt_quality": {
                            "type": "string",
                            "description": "Quality for GPT mode only. Options: 'low', 'medium', 'high' (slow and expensive), 'auto'. Default is 'auto'. Use 'high' ONLY if user explicitly requests highest quality. Only used when model is 'GPT'.",
                            "enum": ["low", "medium", "high", "auto"],
                            "default": "auto"
                        },
                        "variants": {
                            "type": "integer",
                            "description": "Number of edit variants to generate. Default is 1. Set higher (up to 4) when user wants multiple options to choose from.",
                            "enum": [1, 2, 3, 4],
                            "default": 1
                        },
                        "caption": {
                            "type": "string",
                            "description": "Caption for the edited image when sending to user",
                            "default": "Here is your edited image"
                        }
                    },
                    "required": ["prompt", "image_path"]
                }
            },
            {
                "name": "image_composition",
                "description": "Compose a new image using multiple input images with advanced AI multi-image processing. Perfect for style transfer, combining elements from different images, or creating composite scenes. ALWAYS use Normal mode by default unless user specifically requests Pro or GPT mode. GPT mode supports up to 10 reference images.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Detailed instructions on how to compose the new image using the provided images (e.g., 'Take the dress from the first image and let the person from the second image wear it', 'Transfer the artistic style from the first image to the second image', 'Combine the foreground from image 1 with the background from image 2')"
                        },
                        "image_paths": {
                            "type": "array",
                            "description": "Array of paths to image files to be used in composition. For Normal/Pro: 2-3 images. For GPT: up to 10 images.",
                            "items": {
                                "type": "string"
                            },
                            "minItems": 2
                        },
                        "model": {
                            "type": "string",
                            "description": "Model to use: 'Normal' (faster, standard quality, gemini-3.1-flash-image-preview), 'Pro' (higher quality, more control, gemini-3-pro-image-preview), or 'GPT' (OpenAI GPT Image 2, state-of-the-art composition with up to 10 reference images, great for screenshot/UI mockup composition). ALWAYS use 'Normal' unless user specifically requests Pro or GPT mode.",
                            "enum": ["Normal", "Pro", "GPT"],
                            "default": "Normal"
                        },
                        "aspect_ratio": {
                            "type": "string",
                            "description": "Aspect ratio for Pro and GPT modes. Options: '1:1', '2:3', '3:2', '3:4', '4:3', '4:5', '5:4', '9:16', '16:9', '21:9'. Default is '16:9'.",
                            "enum": ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
                            "default": "16:9"
                        },
                        "resolution": {
                            "type": "string",
                            "description": "Resolution for Pro and GPT modes. Options: '1K', '2K', '4K'. Default is '2K'. Use '4K' ONLY if user explicitly requests 4K or ultra-high resolution.",
                            "enum": ["1K", "2K", "4K"],
                            "default": "2K"
                        },
                        "gpt_quality": {
                            "type": "string",
                            "description": "Quality for GPT mode only. Options: 'low', 'medium', 'high' (slow and expensive), 'auto'. Default is 'auto'. Use 'high' ONLY if user explicitly requests highest quality. Only used when model is 'GPT'.",
                            "enum": ["low", "medium", "high", "auto"],
                            "default": "auto"
                        },
                        "variants": {
                            "type": "integer",
                            "description": "Number of composition variants to generate. Default is 1. Set higher (up to 4) when user wants multiple options to choose from.",
                            "enum": [1, 2, 3, 4],
                            "default": 1
                        },
                        "caption": {
                            "type": "string",
                            "description": "Caption for the composed image when sending to user",
                            "default": "Here is your composed image"
                        }
                    },
                    "required": ["prompt", "image_paths"]
                }
            }
        ]

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _resolve_image_path(self, image_path: str) -> Path:
        """Resolve a model-supplied path inside the user's workspace (never outside it).

        Accepts a path relative to the workspace, the sandbox spelling
        (/home/runner/workspace/...), the host spelling (data/<uid>/...) or a bare
        file name looked up in images/, downloads/ and documents/. Returns a
        non-existent path under images/ when nothing matches, so callers report
        "does not exist" instead of touching a file elsewhere on the host.
        """
        from agents.paths import resolve_under

        resolved = resolve_under(self.base_path, image_path, fallback_dirs=("images", "downloads", "documents"))
        return resolved if resolved is not None else self.images_path / Path(image_path).name

    def _inline_hint(self, paths) -> str:
        """Tell the model how to show the saved image(s) inline in its answer."""
        rels: list[str] = []
        base = self.base_path.resolve()
        for raw in paths:
            try:
                rels.append(Path(raw).resolve().relative_to(base).as_posix())
            except ValueError:
                rels.append(f"images/{Path(raw).name}")
        if not rels:
            return ""
        examples = ", ".join(f"![caption]({r})" for r in rels[:3])
        return f"The user already received the file(s). To also show an image inline in your answer, write {examples}\n"

    def _prepare_image_for_edit(self, image_path: Path, temp_paths: List[Path]) -> Path:
        """Create a temporary JPEG for resized or non-JPEG edit inputs."""
        try:
            with PIL.Image.open(image_path) as img:
                image_format = (img.format or "").upper()
                needs_resizing = max_image_resolution_edit > 0 and max(img.size) > max_image_resolution_edit
                needs_conversion = image_format not in JPEG_FORMATS

                if needs_resizing or needs_conversion:
                    img = PIL.ImageOps.exif_transpose(img)

                    if needs_resizing:
                        img.thumbnail(
                            (max_image_resolution_edit, max_image_resolution_edit),
                            getattr(PIL.Image, "Resampling", PIL.Image).LANCZOS
                        )

                    if img.mode != "RGB":
                        img = img.convert("RGB")

                    temp_path = self.images_path / f"edit_input_{uuid.uuid4()}.jpg"
                    temp_paths.append(temp_path)
                    img.save(temp_path, format="JPEG", quality=90, optimize=True)
                    return temp_path
        except Exception as e:
            logger.warning("Failed to prepare image for edit, using original file: %s", e)

        return image_path

    def _cleanup_temp_images(self, temp_paths: List[Path]) -> None:
        for temp_path in temp_paths:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception as e:
                logger.warning("Failed to remove temporary image %s: %s", temp_path, e)

    def _resolve_gpt_size(self, resolution: str, aspect_ratio: str) -> str:
        """Translate resolution + aspect_ratio into a GPT Image 2 pixel size string.

        GPT Image 2 constraints: edges multiples of 16, max edge 3840,
        ratio ≤ 3:1, total pixels 655,360..8,294,400.
        """
        target_map = {"1K": 1024, "2K": 2048, "4K": 3840}
        target = target_map.get(resolution, 2048)

        w_r, h_r = (int(x) for x in aspect_ratio.split(":"))

        if w_r >= h_r:
            w = target
            h = round(target * h_r / w_r / 16) * 16
        else:
            h = target
            w = round(target * w_r / h_r / 16) * 16

        min_px, max_px = 655360, 8294400
        total = w * h
        if total < min_px:
            scale = math.ceil(math.sqrt(min_px / total) * 100) / 100
            w = round(w * scale / 16) * 16
            h = round(h * scale / 16) * 16
        if w * h > max_px:
            scale = math.sqrt(max_px / (w * h))
            w = int(w * scale / 16) * 16
            h = int(h * scale / 16) * 16

        w = max(min(w, 3840), 16)
        h = max(min(h, 3840), 16)
        return f"{w}x{h}"

    async def _send_text(self, text: str) -> None:
        """Send model-generated text (Markdown, split and with fallback handled by the sender)."""
        await self.sender.send_markdown(text)

    async def _send_image(self, path: Path, caption: str) -> None:
        await self.sender.send_document(str(path), caption=caption)

    async def _gemini(self, **kwargs):
        """generate_content in a worker thread so image generation never blocks the event loop."""
        return await asyncio.to_thread(genai_client.models.generate_content, **kwargs)

    @staticmethod
    def _gemini_config(model: str, aspect_ratio: str, resolution: str, with_text: bool = True):
        modalities = ["Text", "Image"] if with_text else None
        if model == "Pro":
            return types.GenerateContentConfig(
                response_modalities=modalities,
                image_config=types.ImageConfig(aspect_ratio=aspect_ratio, image_size=resolution)
            ) if with_text else types.GenerateContentConfig(
                image_config=types.ImageConfig(aspect_ratio=aspect_ratio, image_size=resolution)
            )
        return types.GenerateContentConfig(response_modalities=modalities) if with_text else None

    @staticmethod
    def _gemini_model_name(model: str) -> str:
        return GEMINI_PRO if model.lower() == "pro" else GEMINI_FLASH

    async def _deliver_parts(self, parts, prefix: str, text_label: str, all_saved_paths: List[str], variant_caption: str, result_message: str) -> str:
        """Send text parts and inline images from a Gemini response; returns the updated result message."""
        for part in parts:
            if 'text' in part.model_fields_set and part.text:
                await self._send_text(f"{text_label}\n\n{part.text}" if text_label else part.text)
                result_message += f"{text_label or 'Generated text:'} \n\n{part.text}\n\n"
            elif 'inline_data' in part.model_fields_set:
                extension = part.inline_data.mime_type.split('/')[-1]
                if extension == 'jpeg':
                    extension = 'jpg'
                image_path = self.images_path / f"{prefix}_{uuid.uuid4()}.{extension}"
                with open(image_path, "wb") as f:
                    f.write(part.inline_data.data)
                await self._send_image(image_path, variant_caption)
                all_saved_paths.append(str(image_path))
        return result_message

    # ------------------------------------------------------------------
    # dispatch
    # ------------------------------------------------------------------
    async def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        if tool_name == "image_generator":
            return await self._image_generator(**tool_args)
        elif tool_name == "generate_image_dall_e":
            return await self._generate_image_dall_e(**tool_args)
        elif tool_name == "generate_multimodal_image_and_text":
            return await self._generate_multimodal_image_and_text(**tool_args)
        elif tool_name == "image_editing":
            return await self._image_editing(**tool_args)
        elif tool_name == "image_composition":
            return await self._image_composition(**tool_args)
        return f"Unknown tool: {tool_name}"

    # ------------------------------------------------------------------
    # tools
    # ------------------------------------------------------------------
    async def _generate_image_dall_e(self, prompt: str, size: str = "1024x1024", quality: str = "standard", caption: str = "Here is your image") -> str:
        """Generate an image using DALL-E 3 and return the URL"""
        logger.info("Generating image with DALL-E 3 - prompt=%r size=%s quality=%s", prompt, size, quality)
        try:
            response = await asyncio.to_thread(
                openai_client.images.generate,
                model="dall-e-3",
                prompt=prompt,
                size=size,
                quality=quality,
                response_format="b64_json",
                n=1,
            )
            image_path = self.images_path / f"image_{uuid.uuid4()}.jpg"
            with open(image_path, "wb") as f:
                f.write(base64.b64decode(response.data[0].b64_json))
            await self._send_image(image_path, caption)
            return f"Image generated and sent to user via telegram successfully.\n\nFile also saved to: {image_path}\n\n" + self._inline_hint([image_path])
        except Exception as e:
            return f"Error generating image: {str(e)}"

    async def _generate_multimodal_image_and_text(self, prompt: str, style: str = "3d digital art") -> str:
        """Generate a story with images using Google's Gemini model"""
        logger.info("Generating multimodal story with Gemini - prompt=%r style=%s", prompt, style)
        try:
            formatted_prompt = f"Generate a story about {prompt} in a {style} style. For each scene, generate an image."
            response = await self._gemini(
                model=GEMINI_FLASH,
                contents=formatted_prompt,
                config=types.GenerateContentConfig(response_modalities=["Text", "Image"]),
            )
            contents = response.candidates[0].content.parts

            all_parts = []
            for content in contents:
                if 'text' in content.model_fields_set:
                    all_parts.append(TextPart(content.text))
                elif 'inline_data' in content.model_fields_set:
                    all_parts.append(InlineDataPart(content.inline_data.mime_type, content.inline_data.data))

            result_message = "Text and images were generated and successfully sent to user as telegram messages:\n\n"
            image_count = 0
            part_count = 0
            for part in all_parts:
                part_count += 1
                if isinstance(part, TextPart):
                    await self._send_text(f"Part {part_count}:\n\n{part.text}")
                    result_message += f"Part {part_count}:\n\n{part.text}\n\n"
                elif isinstance(part, InlineDataPart):
                    image_count += 1
                    image_path = self.images_path / f"story_image_{uuid.uuid4()}.jpg"
                    with open(image_path, "wb") as f:
                        f.write(part.data)
                    await self._send_image(image_path, f"Image {image_count}.")
                    result_message += f"Image for part {image_count}: {image_path}.\n\n"

            result_message += "\n\nEnd of messages.\nWrite answer to user using this text, remove under the hood details about styling info, formatting, or something like that, and translate it to user language if needed."
            return result_message
        except Exception as e:
            return f"Error generating multimodal story: {str(e)}"

    async def _image_editing(self, prompt: str, image_path: str, model: str = "Normal", aspect_ratio: str = "16:9", resolution: str = "2K", gpt_quality: str = "auto", variants: int = 1, caption: str = "Here is your edited image") -> str:
        """Edit an existing image using Gemini or GPT Image 2"""
        logger.info("Editing image - prompt=%r image=%s model=%s variants=%s", prompt, image_path, model, variants)
        temp_paths: List[Path] = []
        source_image = None
        try:
            image_path_obj = self._resolve_image_path(image_path)
            if not image_path_obj.exists():
                return f"Error: The image at path {image_path} does not exist."

            if model.lower() == "gpt":
                gpt_size = self._resolve_gpt_size(resolution, aspect_ratio)
                all_results = []
                for i in range(variants):
                    r = await self._gpt_image_edit(
                        prompt, [image_path_obj], gpt_size, gpt_quality,
                        f"{caption} (variant {i + 1}/{variants})" if variants > 1 else caption
                    )
                    all_results.append(r)
                return "\n".join(all_results)

            source_image_path = await asyncio.to_thread(self._prepare_image_for_edit, image_path_obj, temp_paths)
            source_image = PIL.Image.open(source_image_path)
            config = self._gemini_config(model, aspect_ratio, resolution)

            all_saved_paths: List[str] = []
            result_message = "Image transformation results:\n\n"
            for variant_idx in range(variants):
                response = await self._gemini(model=self._gemini_model_name(model), contents=(prompt, source_image), config=config)
                variant_caption = f"{caption} (variant {variant_idx + 1}/{variants})" if variants > 1 else caption
                result_message = await self._deliver_parts(
                    response.candidates[0].content.parts, "transformed", "Text explanation:", all_saved_paths, variant_caption, result_message
                )

            if all_saved_paths:
                paths_str = "\n".join(f"  - {p}" for p in all_saved_paths)
                result_message += f"Edited {len(all_saved_paths)} image(s) and sent to user.\nSaved to:\n{paths_str}\nImage transformation completed successfully.\n" + self._inline_hint(all_saved_paths)
            else:
                result_message += "Warning: No transformed image was generated. The model only provided text response.\n"
            return result_message
        except Exception as e:
            return f"Error editing image: {str(e)}"
        finally:
            if source_image:
                source_image.close()
            self._cleanup_temp_images(temp_paths)

    async def _gpt_image_edit(self, prompt: str, image_paths: List[Path], size: str = "2048x1152", quality: str = "auto", caption: str = "Here is your edited image") -> str:
        """Edit one or more images using OpenAI GPT Image 2"""
        logger.info("Editing image(s) with GPT Image 2 - prompt=%r images=%s size=%s", prompt, image_paths, size)
        temp_paths: List[Path] = []
        try:
            prepared = [await asyncio.to_thread(self._prepare_image_for_edit, p, temp_paths) for p in image_paths]

            def _edit():
                files = [open(str(p), "rb") for p in prepared]
                try:
                    kwargs = {"model": GPT_IMAGE, "prompt": prompt, "quality": quality, "size": size}
                    kwargs["image"] = files[0] if len(files) == 1 else files
                    return openai_client.images.edit(**kwargs)
                finally:
                    for f in files:
                        f.close()

            try:
                result = await asyncio.to_thread(_edit)
            finally:
                self._cleanup_temp_images(temp_paths)

            image_path = self.images_path / f"gpt_edited_{uuid.uuid4()}.png"
            with open(image_path, "wb") as f:
                f.write(base64.b64decode(result.data[0].b64_json))
            await self._send_image(image_path, caption)
            return f"Image edited with GPT Image 2 and saved to: {image_path} and sent to user.\nImage editing completed successfully.\n" + self._inline_hint([image_path])
        except Exception as e:
            return f"Error editing image with GPT Image 2: {str(e)}"

    async def _image_generator(self, prompt: str, style: str = "photorealistic", model: str = "Normal", aspect_ratio: str = "16:9", resolution: str = "2K", gpt_quality: str = "auto", gpt_output_format: str = "png", variants: int = 1, caption: str = "Here is your generated image") -> str:
        """Generate a high-quality image from text using Gemini or GPT Image 2"""
        logger.info("Generating image - prompt=%r model=%s variants=%s", prompt, model, variants)
        try:
            if model.lower() == "gpt":
                gpt_size = self._resolve_gpt_size(resolution, aspect_ratio)
                return await self._gpt_image_generate(prompt, gpt_size, gpt_quality, gpt_output_format, caption, n=variants)

            formatted_prompt = f"Create a {style} style image: {prompt}" if style and style != "photorealistic" else prompt
            config = self._gemini_config(model, aspect_ratio, resolution, with_text=False)

            all_saved_paths: List[str] = []
            result_message = ""
            for variant_idx in range(variants):
                kwargs = {"model": self._gemini_model_name(model), "contents": [formatted_prompt]}
                if config is not None:
                    kwargs["config"] = config
                response = await self._gemini(**kwargs)
                variant_caption = f"{caption} (variant {variant_idx + 1}/{variants})" if variants > 1 else caption
                result_message = await self._deliver_parts(
                    response.candidates[0].content.parts, "generated", "", all_saved_paths, variant_caption, result_message
                )

            if all_saved_paths:
                paths_str = "\n".join(f"  - {p}" for p in all_saved_paths)
                result_message += f"Generated {len(all_saved_paths)} image(s) and sent to user.\nSaved to:\n{paths_str}\nImage generation completed successfully.\n" + self._inline_hint(all_saved_paths)
            else:
                result_message += "Warning: No image was generated. The model only provided text response.\n"
            return result_message
        except Exception as e:
            return f"Error generating image: {str(e)}"

    async def _gpt_image_generate(self, prompt: str, size: str = "2048x1152", quality: str = "auto", output_format: str = "jpeg", caption: str = "Here is your generated image", n: int = 1) -> str:
        """Generate one or more images using OpenAI GPT Image 2"""
        logger.info("Generating image with GPT Image 2 - prompt=%r size=%s n=%s", prompt, size, n)
        try:
            result = await asyncio.to_thread(
                openai_client.images.generate,
                model=GPT_IMAGE, prompt=prompt, quality=quality, output_format=output_format, size=size, n=n,
            )
            ext = "jpg" if output_format == "jpeg" else output_format
            saved_paths = []
            for idx, item in enumerate(result.data):
                image_path = self.images_path / f"gpt_generated_{uuid.uuid4()}.{ext}"
                with open(image_path, "wb") as f:
                    f.write(base64.b64decode(item.b64_json))
                await self._send_image(image_path, f"{caption} (variant {idx + 1}/{n})" if n > 1 else caption)
                saved_paths.append(str(image_path))
            paths_str = "\n".join(f"  - {p}" for p in saved_paths)
            return f"Generated {len(saved_paths)} image(s) with GPT Image 2 and sent to user.\nSaved to:\n{paths_str}\nImage generation completed successfully.\n" + self._inline_hint(saved_paths)
        except Exception as e:
            return f"Error generating image with GPT Image 2: {str(e)}"

    async def _image_composition(self, prompt: str, image_paths: List[str], model: str = "Normal", aspect_ratio: str = "16:9", resolution: str = "2K", gpt_quality: str = "auto", variants: int = 1, caption: str = "Here is your composed image") -> str:
        """Compose a new image from multiple input images using Gemini or GPT Image 2"""
        logger.info("Composing image - prompt=%r images=%s model=%s variants=%s", prompt, image_paths, model, variants)
        temp_paths: List[Path] = []
        images: List[PIL.Image.Image] = []
        try:
            resolved_paths = []
            for image_path in image_paths:
                image_path_obj = self._resolve_image_path(image_path)
                if not image_path_obj.exists():
                    return f"Error: The image at path {image_path} does not exist."
                resolved_paths.append(image_path_obj)

            if len(resolved_paths) < 2:
                return "Error: At least 2 images are required for composition."

            if model.lower() == "gpt":
                gpt_size = self._resolve_gpt_size(resolution, aspect_ratio)
                all_results = []
                for i in range(variants):
                    r = await self._gpt_image_edit(
                        prompt, resolved_paths, gpt_size, gpt_quality,
                        f"{caption} (variant {i + 1}/{variants})" if variants > 1 else caption
                    )
                    all_results.append(r)
                return "\n".join(all_results)

            if len(resolved_paths) > 3:
                return "Error: Maximum 3 images are supported for composition with Normal/Pro mode. Use GPT mode for more images."

            prepared_paths = [await asyncio.to_thread(self._prepare_image_for_edit, p, temp_paths) for p in resolved_paths]
            images = [PIL.Image.open(p) for p in prepared_paths]
            input_contents = [*images, prompt]
            config = self._gemini_config(model, aspect_ratio, resolution)

            all_saved_paths: List[str] = []
            result_message = "Image composition results:\n\n"
            for variant_idx in range(variants):
                response = await self._gemini(model=self._gemini_model_name(model), contents=input_contents, config=config)
                variant_caption = f"{caption} (variant {variant_idx + 1}/{variants})" if variants > 1 else caption
                result_message = await self._deliver_parts(
                    response.candidates[0].content.parts, "composed", "Composition details:", all_saved_paths, variant_caption, result_message
                )

            if all_saved_paths:
                paths_str = "\n".join(f"  - {p}" for p in all_saved_paths)
                result_message += f"Composed {len(all_saved_paths)} image(s) and sent to user.\nSaved to:\n{paths_str}\nImage composition completed successfully.\n" + self._inline_hint(all_saved_paths)
            else:
                result_message += "Warning: No composed image was generated. The model only provided text response.\n"
            return result_message
        except Exception as e:
            return f"Error composing image: {str(e)}"
        finally:
            for image in images:
                image.close()
            self._cleanup_temp_images(temp_paths)
