import os
import time
from typing import Dict, Any
from dotenv import load_dotenv
from pathlib import Path
import uuid
import telegram
from google import genai
from google.genai import types
from google.genai.types import Image, VideoGenerationReferenceImage

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

    async def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
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

    async def _video_generator(self, prompt: str, orientation: str = "horizontal", quality: str = "720p", negative_prompt: str = "", caption: str = "Here is your generated video") -> str:
        """Generate a high-quality video from text using Google's Veo 3.0 model"""
        print(f"Generating video with Veo 3.1 - Prompt: {prompt}, Orientation: {orientation}, Quality: {quality}, Negative prompt: {negative_prompt}")
        try:
            # Convert orientation to aspect ratio
            aspect_ratio = "16:9" if orientation == "horizontal" else "9:16"
            
            # Notify user that video generation started (it can take some time)
            await self.telegram_update.message.reply_text(
                f"🎬 Video generation started ({aspect_ratio}, {quality})... This may take 1-2 minutes. Please wait.",
                parse_mode="markdown"
            )
            
            # Configure video generation
            config = types.GenerateVideosConfig(
                aspect_ratio=aspect_ratio, resolution=quality
            )
            if negative_prompt:
                config.negative_prompt = negative_prompt
            
            # Start video generation operation
            operation = genai_client.models.generate_videos(
                model="veo-3.1-generate-preview",
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
        print(f"Generating video from image with Veo 3.1 - Prompt: {prompt}, Image: {image_path}, Orientation: {orientation}, Quality: {quality}, Negative prompt: {negative_prompt}")
        try:
            # Verify the image path exists
            image_path_obj = Path(image_path)
            if not image_path_obj.exists():
                return f"Error: The image at path {image_path} does not exist."
            
            # Convert orientation to aspect ratio
            aspect_ratio = "16:9" if orientation == "horizontal" else "9:16"
            
            # Load image using the from_file method (handles bytes and mime_type automatically)
            source_image = Image.from_file(location=str(image_path_obj))
            
            # Notify user that video generation started (it can take some time)
            await self.telegram_update.message.reply_text(
                f"🎬 Video generation from image started ({aspect_ratio}, {quality})... This may take 1-2 minutes. Please wait.",
                parse_mode="markdown"
            )
            
            # Configure video generation
            config = types.GenerateVideosConfig(
                aspect_ratio=aspect_ratio, resolution=quality
            )

            if negative_prompt:
                config.negative_prompt = negative_prompt
            
            # Start video generation operation with image
            operation = genai_client.models.generate_videos(
                model="veo-3.1-generate-preview",
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

    async def _video_from_reference_images(self, prompt: str, reference_image_paths: list, quality: str = "720p", negative_prompt: str = "", caption: str = "Here is your generated video with reference images") -> str:
        """Generate a high-quality video using reference images with Google's Veo 3.1 model"""
        print(f"Generating video with reference images - Prompt: {prompt}, Reference images: {reference_image_paths}, Quality: {quality}")
        try:
            # Verify all reference image paths exist
            reference_images = []
            for image_path in reference_image_paths:
                image_path_obj = Path(image_path)
                if not image_path_obj.exists():
                    return f"Error: The reference image at path {image_path} does not exist."
                
                # Load each reference image and create VideoGenerationReferenceImage
                image = Image.from_file(location=str(image_path_obj))
                ref_image = VideoGenerationReferenceImage(
                    image=image,
                    reference_type="asset"
                )
                reference_images.append(ref_image)
            
            # Currently only supports horizontal 16:9
            aspect_ratio = "16:9"
            
            # Notify user that video generation started
            await self.telegram_update.message.reply_text(
                f"🎬 Video generation with {len(reference_images)} reference image(s) started ({aspect_ratio}, {quality})... This may take 1-2 minutes. Please wait.",
                parse_mode="markdown"
            )
            
            # Configure video generation
            config = types.GenerateVideosConfig(
                aspect_ratio=aspect_ratio,
                resolution=quality,
                reference_images=reference_images
            )
            if negative_prompt:
                config.negative_prompt = negative_prompt
            
            # Start video generation operation
            operation = genai_client.models.generate_videos(
                model="veo-3.1-generate-preview",
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
                return "Error: No video was generated. Please try again with a different prompt or reference images."
            
            generated_video = operation.result.generated_videos[0]
            
            # Generate unique filename
            video_filename = f"veo3_reference_{uuid.uuid4()}.mp4"
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
            
            result_message = "✅ Video with reference images generated successfully!\n\n"
            result_message += f"Reference images: {len(reference_images)}\n"
            result_message += f"Aspect Ratio: {aspect_ratio}\n"
            result_message += f"Resolution: {quality}\n"
            result_message += f"File saved to: {video_path}\n"
            result_message += "Video has been sent to user via Telegram.\n"
            
            return result_message
            
        except Exception as e:
            error_message = f"Error generating video with reference images: {str(e)}"
            print(error_message)
            try:
                await self.telegram_update.message.reply_text(
                    "❌ Sorry, video generation with reference images failed. Please try again.",
                    parse_mode="markdown"
                )
            except Exception:
                pass
            return error_message

    async def _video_interpolation_generator(self, prompt: str, first_frame_path: str, last_frame_path: str, orientation: str = "horizontal", quality: str = "720p", negative_prompt: str = "", caption: str = "Here is your interpolated video") -> str:
        """Generate a high-quality video with specified first and last frames using Google's Veo 3.1 model"""
        print(f"Generating interpolated video - Prompt: {prompt}, First frame: {first_frame_path}, Last frame: {last_frame_path}, Orientation: {orientation}, Quality: {quality}")
        try:
            # Verify both frame paths exist
            first_frame_obj = Path(first_frame_path)
            last_frame_obj = Path(last_frame_path)
            
            if not first_frame_obj.exists():
                return f"Error: The first frame at path {first_frame_path} does not exist."
            if not last_frame_obj.exists():
                return f"Error: The last frame at path {last_frame_path} does not exist."
            
            # Load both frames
            first_image = Image.from_file(location=str(first_frame_obj))
            last_image = Image.from_file(location=str(last_frame_obj))
            
            # Convert orientation to aspect ratio
            aspect_ratio = "16:9" if orientation == "horizontal" else "9:16"
            
            # Notify user that video generation started
            await self.telegram_update.message.reply_text(
                f"🎬 Video interpolation started ({aspect_ratio}, {quality})... This may take 1-2 minutes. Please wait.",
                parse_mode="markdown"
            )
            
            # Configure video generation with last frame
            config = types.GenerateVideosConfig(
                aspect_ratio=aspect_ratio,
                resolution=quality,
                last_frame=last_image
            )
            if negative_prompt:
                config.negative_prompt = negative_prompt
            
            # Start video generation operation with first frame as image parameter
            operation = genai_client.models.generate_videos(
                model="veo-3.1-generate-preview",
                prompt=prompt,
                image=first_image,
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
                return "Error: No video was generated. Please try again with different frames or prompt."
            
            generated_video = operation.result.generated_videos[0]
            
            # Generate unique filename
            video_filename = f"veo3_interpolated_{uuid.uuid4()}.mp4"
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
            
            result_message = "✅ Interpolated video generated successfully!\n\n"
            result_message += f"First frame: {first_frame_path}\n"
            result_message += f"Last frame: {last_frame_path}\n"
            result_message += f"Aspect Ratio: {aspect_ratio}\n"
            result_message += f"Resolution: {quality}\n"
            result_message += f"Video saved to: {video_path}\n"
            result_message += "Video has been sent to user via Telegram.\n"
            
            return result_message
            
        except Exception as e:
            error_message = f"Error generating interpolated video: {str(e)}"
            print(error_message)
            try:
                await self.telegram_update.message.reply_text(
                    "❌ Sorry, interpolated video generation failed. Please try again.",
                    parse_mode="markdown"
                )
            except Exception:
                pass
            return error_message

    async def _video_extension_generator(self, prompt: str, source_video_path: str, caption: str = "Here is your extended video") -> str:
        """Extend an existing video using Google's Veo 3.1 model"""
        print(f"Extending video - Prompt: {prompt}, Source video: {source_video_path}")
        try:
            # Verify the source video path exists
            video_path_obj = Path(source_video_path)
            if not video_path_obj.exists():
                return f"Error: The source video at path {source_video_path} does not exist."
            
            # Upload the source video to Google's servers first
            # The genai client requires videos to be uploaded before use
            uploaded_video = genai_client.files.upload(file=str(video_path_obj))
            source_video = uploaded_video
            
            # Currently only supports 720p for extension
            quality = "720p"
            
            # Notify user that video extension started
            await self.telegram_update.message.reply_text(
                f"🎬 Video extension started ({quality})... This may take 1-2 minutes. Please wait.",
                parse_mode="markdown"
            )
            
            # Configure video generation
            config = types.GenerateVideosConfig(
                number_of_videos=1,
                resolution=quality
            )
            
            # Start video extension operation
            operation = genai_client.models.generate_videos(
                model="veo-3.1-generate-preview",
                video=source_video,
                prompt=prompt,
                config=config,
            )
            
            # Poll for completion
            max_wait_time = 300  # 5 minutes max
            start_time = time.time()
            poll_interval = 20  # Check every 20 seconds
            
            while not operation.done:
                if time.time() - start_time > max_wait_time:
                    return "Error: Video extension timed out after 5 minutes. Please try again with a simpler prompt."
                
                time.sleep(poll_interval)
                operation = genai_client.operations.get(operation)
            
            # Get the generated video
            if not operation.result or not operation.result.generated_videos:
                return "Error: No extended video was generated. Please try again with a different prompt."
            
            generated_video = operation.result.generated_videos[0]
            
            # Generate unique filename
            video_filename = f"veo3_extended_{uuid.uuid4()}.mp4"
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
            
            result_message = "✅ Video extended successfully!\n\n"
            result_message += f"Source video: {source_video_path}\n"
            result_message += f"Resolution: {quality}\n"
            result_message += f"Extended video saved to: {video_path}\n"
            result_message += "Video has been sent to user via Telegram.\n"
            
            return result_message
            
        except Exception as e:
            error_message = f"Error extending video: {str(e)}"
            print(error_message)
            try:
                await self.telegram_update.message.reply_text(
                    "❌ Sorry, video extension failed. Please try again.",
                    parse_mode="markdown"
                )
            except Exception:
                pass
            return error_message
