from openai import OpenAI
import os
from typing import Dict, Any, List, Literal
from dotenv import load_dotenv
from pathlib import Path
import uuid
import base64
import telegram
import asyncio
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_api_key)

class ImageTools:
    def __init__(self, user_id: str, telegram_update: telegram.Update):
        self.user_id = user_id
        self.telegram_update = telegram_update
        self.base_path = Path("./data") / str(user_id)
        # Create base directory if it doesn't exist
        self.base_path.mkdir(parents=True, exist_ok=True)
        # Create images directory
        self.images_path = self.base_path / "images"
        self.images_path.mkdir(parents=True, exist_ok=True)

        self.tools_schema = [
            {
                "name": "generate_image",
                "description": "Generate an image using DALL-E 3 model, save it to the images directory, automatically send it to user via Telegram and return the image path",
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
            }
        ]

    async def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        if tool_name == "generate_image":
            return await self._generate_image(**tool_args)
        return f"Unknown tool: {tool_name}"

    async def _generate_image(self, prompt: str, size: str = "1024x1024", quality: str = "standard", caption: str = "Here is your image") -> str:
        """Generate an image using DALL-E 3 and return the URL"""
        print(f"Generating image with:\n\nPrompt: {prompt}\n\nSize: {size}\n\nQuality: {quality}\n\n")
        try:
            response = openai_client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size=size,
                quality=quality,
                response_format="b64_json",
                n=1,
            )
            image_in_base64_string = response.data[0].b64_json
            image_path = self.images_path / f"image_{uuid.uuid4()}.jpg"
            with open(image_path, "wb") as f:
                f.write(base64.b64decode(image_in_base64_string))
            
            with open(image_path, 'rb') as file:
                await self.telegram_update.message.reply_document(
                    document=file,
                    caption=caption
                )

            return f"Image generated and sent to user via telegram successfully.\n\nFile also saved to: {image_path}\n\n"
        except Exception as e:
            return f"Error generating image: {str(e)}" 