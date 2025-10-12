import os
import time
from typing import Dict, Any
from dotenv import load_dotenv
from pathlib import Path
import uuid
import telegram
from google import genai
from google.genai import types
import PIL.Image

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
genai_client = genai.Client(api_key=google_api_key)

class VideoTools:
    def __init__(self, user_id: str, telegram_update: telegram.Update):
        self.user_id = user_id
        self.telegram_update = telegram_update
        self.base_path = Path("./data") / str(user_id)
        # Create base directory if it doesn't exist
        self.base_path.mkdir(parents=True, exist_ok=True)
        # Create videos directory
        self.videos_path = self.base_path / "videos"
        self.videos_path.mkdir(parents=True, exist_ok=True)

        self.tools_schema = [
            {
                "name": "video_generator",
                "description": "Generate high-quality videos from text descriptions using Google Veo 3.0 (state-of-the-art text-to-video generation with cinematic quality), save to videos directory, and send to user via Telegram. Supports detailed scene descriptions with camera movements, lighting, and actions.",
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
                            "description": "Video quality/resolution: '720p' (default, works with both orientations) or '1080p' (only for horizontal/16:9 videos). Use '1080p' only when user explicitly asks for high quality or 1080p video AND orientation is horizontal.",
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
                "description": "Generate high-quality videos from an existing image using Google Veo 3.0. Takes a starting image and animates it based on the prompt description. Perfect for bringing static images to life with motion, camera movements, and scene evolution.",
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
                            "description": "Video quality/resolution: '720p' (default, works with both orientations) or '1080p' (only for horizontal/16:9 videos). Use '1080p' only when user explicitly asks for high quality or 1080p video AND orientation is horizontal.",
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
            }
        ]

    async def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        if tool_name == "video_generator":
            return await self._video_generator(**tool_args)
        elif tool_name == "image_to_video_generator":
            return await self._image_to_video_generator(**tool_args)
        return f"Unknown tool: {tool_name}"

    async def _video_generator(self, prompt: str, orientation: str = "horizontal", quality: str = "720p", negative_prompt: str = "", caption: str = "Here is your generated video") -> str:
        """Generate a high-quality video from text using Google's Veo 3.0 model"""
        print(f"Generating video with Veo 3.0 - Prompt: {prompt}, Orientation: {orientation}, Quality: {quality}, Negative prompt: {negative_prompt}")
        try:
            # Convert orientation to aspect ratio
            aspect_ratio = "16:9" if orientation == "horizontal" else "9:16"
            
            # Validate quality and orientation combination
            if quality == "1080p" and orientation == "portrait":
                return "Error: 1080p resolution is only supported for horizontal (16:9) videos. Please use 720p for portrait videos or switch to horizontal orientation."
            
            # Notify user that video generation started (it can take some time)
            await self.telegram_update.message.reply_text(
                f"🎬 Video generation started ({aspect_ratio}, {quality})... This may take 1-2 minutes. Please wait.",
                parse_mode="markdown"
            )
            
            # Configure video generation
            config = types.GenerateVideosConfig(
                aspect_ratio=aspect_ratio,
                resolution=quality
            )
            if negative_prompt:
                config.negative_prompt = negative_prompt
            
            # Start video generation operation
            operation = genai_client.models.generate_videos(
                model="veo-3.0-generate-preview",
                prompt=prompt,
                config=config,
            )
            
            # Poll for completion
            max_wait_time = 300  # 5 minutes max
            start_time = time.time()
            poll_interval = 20  # Check every 20 seconds
            
            while not operation.done:
                if time.time() - start_time > max_wait_time:
                    return "Error: Video generation timed out after 5 minutes. Please try again with a simpler prompt."
                
                time.sleep(poll_interval)
                operation = genai_client.operations.get(operation)
            
            # Get the generated video
            if not operation.result or not operation.result.generated_videos:
                return "Error: No video was generated. Please try again with a different prompt."
            
            generated_video = operation.result.generated_videos[0]
            
            # Generate unique filename
            video_filename = f"veo3_video_{uuid.uuid4()}.mp4"
            video_path = self.videos_path / video_filename
            
            # Download the video file
            genai_client.files.download(file=generated_video.video)
            generated_video.video.save(str(video_path))
            
            # Send video to user via Telegram
            with open(video_path, 'rb') as video_file:
                await self.telegram_update.message.reply_video(
                    video=video_file,
                    caption=caption,
                    supports_streaming=True
                )
            
            result_message = "✅ Video generated successfully!\n\n"
            result_message += f"Aspect Ratio: {aspect_ratio}\n"
            result_message += f"Resolution: {quality}\n"
            result_message += f"File saved to: {video_path}\n"
            result_message += "Video has been sent to user via Telegram.\n"
            
            return result_message
            
        except Exception as e:
            error_message = f"Error generating video: {str(e)}"
            print(error_message)
            try:
                await self.telegram_update.message.reply_text(
                    "❌ Sorry, video generation failed. Please try again.",
                    parse_mode="markdown"
                )
            except Exception:
                pass
            return error_message

    async def _image_to_video_generator(self, prompt: str, image_path: str, orientation: str = "horizontal", quality: str = "720p", negative_prompt: str = "", caption: str = "Here is your generated video from the image") -> str:
        """Generate a high-quality video from an image using Google's Veo 3.0 model"""
        print(f"Generating video from image with Veo 3.0 - Prompt: {prompt}, Image: {image_path}, Orientation: {orientation}, Quality: {quality}, Negative prompt: {negative_prompt}")
        try:
            # Verify the image path exists
            image_path_obj = Path(image_path)
            if not image_path_obj.exists():
                return f"Error: The image at path {image_path} does not exist."
            
            # Convert orientation to aspect ratio
            aspect_ratio = "16:9" if orientation == "horizontal" else "9:16"
            
            # Validate quality and orientation combination
            if quality == "1080p" and orientation == "portrait":
                return "Error: 1080p resolution is only supported for horizontal (16:9) videos. Please use 720p for portrait videos or switch to horizontal orientation."
            
            # Open the image using PIL
            source_image = PIL.Image.open(image_path_obj)
            
            # Notify user that video generation started (it can take some time)
            await self.telegram_update.message.reply_text(
                f"🎬 Video generation from image started ({aspect_ratio}, {quality})... This may take 1-2 minutes. Please wait.",
                parse_mode="markdown"
            )
            
            # Configure video generation
            config = types.GenerateVideosConfig(
                aspect_ratio=aspect_ratio,
                resolution=quality
            )
            if negative_prompt:
                config.negative_prompt = negative_prompt
            
            # Start video generation operation with image
            operation = genai_client.models.generate_videos(
                model="veo-3.0-generate-preview",
                prompt=prompt,
                image=source_image,
                config=config,
            )
            
            # Poll for completion
            max_wait_time = 300  # 5 minutes max
            start_time = time.time()
            poll_interval = 20  # Check every 20 seconds
            
            while not operation.done:
                if time.time() - start_time > max_wait_time:
                    return "Error: Video generation timed out after 5 minutes. Please try again with a simpler prompt."
                
                time.sleep(poll_interval)
                operation = genai_client.operations.get(operation)
            
            # Get the generated video
            if not operation.result or not operation.result.generated_videos:
                return "Error: No video was generated. Please try again with a different prompt or image."
            
            generated_video = operation.result.generated_videos[0]
            
            # Generate unique filename
            video_filename = f"veo3_from_image_{uuid.uuid4()}.mp4"
            video_path = self.videos_path / video_filename
            
            # Download the video file
            genai_client.files.download(file=generated_video.video)
            generated_video.video.save(str(video_path))
            
            # Send video to user via Telegram
            with open(video_path, 'rb') as video_file:
                await self.telegram_update.message.reply_video(
                    video=video_file,
                    caption=caption,
                    supports_streaming=True
                )
            
            result_message = "✅ Video generated from image successfully!\n\n"
            result_message += f"Source image: {image_path}\n"
            result_message += f"Aspect Ratio: {aspect_ratio}\n"
            result_message += f"Resolution: {quality}\n"
            result_message += f"Video saved to: {video_path}\n"
            result_message += "Video has been sent to user via Telegram.\n"
            
            return result_message
            
        except Exception as e:
            error_message = f"Error generating video from image: {str(e)}"
            print(error_message)
            try:
                await self.telegram_update.message.reply_text(
                    "❌ Sorry, video generation from image failed. Please try again.",
                    parse_mode="markdown"
                )
            except Exception:
                pass
            return error_message

