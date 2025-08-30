from openai import OpenAI
import os
from typing import Dict, Any, List, Literal
from dotenv import load_dotenv
from pathlib import Path
import uuid
import base64
import telegram
import asyncio
from google import genai
from google.genai import types
import PIL.Image
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_api_key)
google_api_key = os.getenv("GOOGLE_API_KEY")
genai_client = genai.Client(api_key=google_api_key)

class TextPart:
    def __init__(self, text):
        self.text = str(text)

class InlineDataPart:
    def __init__(self, mime_type, data):
        self.mime_type = str(mime_type)
        self.data = bytes(data)

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
                "description": "Generate an image using old image generation DALL-E 3 model (not multimodal, not preferred for story generation), save it to the images directory, automatically send it to user via Telegram and return the image path",
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
                "name": "transform_image",
                "description": "Transform an existing image using Google's Gemini model based on user instructions, save the new image, and send it to the user via Telegram",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Instructions on how to transform the image (e.g., 'Add a hat to the person', 'Change background to beach')"
                        },
                        "image_path": {
                            "type": "string",
                            "description": "Path to the image file to be transformed. This should be a full path to an image file in the user's directory."
                        },
                        "caption": {
                            "type": "string",
                            "description": "Caption for the transformed image when sending to user",
                            "default": "Here is your transformed image"
                        }
                    },
                    "required": ["prompt", "image_path"]
                }
            }
        ]

    async def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        if tool_name == "generate_image":
            return await self._generate_image(**tool_args)
        elif tool_name == "generate_multimodal_image_and_text":
            return await self._generate_multimodal_image_and_text(**tool_args)
        elif tool_name == "transform_image":
            return await self._transform_image(**tool_args)
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
            
    async def _generate_multimodal_image_and_text(self, prompt: str, style: str = "3d digital art") -> str:
        """Generate a story with images using Google's Gemini model"""
        print(f"Generating multimodal story with prompt: {prompt}, style: {style}")
        try:
            # Format the prompt to include style information
            formatted_prompt = f"Generate a story about {prompt} in a {style} style. For each scene, generate an image."
            
            response = genai_client.models.generate_content(
                model="gemini-2.5-flash-image-preview",
                contents=formatted_prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["Text", "Image"],
                    # media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH
                ),
            )
            
            # Wait all parts to be generated. WA where not all candidates are generated yet. Need to research how to do it better.
            # await asyncio.sleep(10)

            contents = response.candidates[0].content.parts
            
            all_parts = []
            for content in contents:
                if 'text' in content.model_fields_set:
                    all_parts.append(TextPart(content.text))
                elif 'inline_data' in content.model_fields_set:
                    all_parts.append(InlineDataPart(
                        content.inline_data.mime_type,
                        content.inline_data.data
                    ))
            
            result_message = "Text and images were generated and successfully sent to user as telegram messages:\n\n"
            image_count = 0
            part_count = 0
            for part in all_parts:
                part_count += 1
                if isinstance(part, TextPart):
                    try: # as markdown
                        await self.telegram_update.message.reply_text(f"Part {part_count}:\n\n{part.text}", parse_mode="markdown")
                    except:
                        # check length of text and split to chunks of 3000 characters
                        text_chunks = [part.text[i:i+3000] for i in range(0, len(part.text), 3000)]
                        sub_part_count = 0
                        for chunk in text_chunks:
                            sub_part_count += 1
                            try:
                                await self.telegram_update.message.reply_text(f"Part {part_count} subpart {sub_part_count}:\n\n{chunk}", parse_mode="markdown")
                            except:
                                await self.telegram_update.message.reply_text(f"Part {part_count} subpart {sub_part_count}:\n\n{chunk}")
                    result_message += f"Part {part_count}:\n\n{part.text}\n\n"
                elif isinstance(part, InlineDataPart):
                    image_count += 1
                    image_path = self.images_path / f"story_image_{uuid.uuid4()}.jpg"
                    with open(image_path, "wb") as f:
                        f.write(part.data)
                    
                    with open(image_path, 'rb') as file:
                        await self.telegram_update.message.reply_document(
                            document=file,
                            caption=f"Image {image_count}."
                        )
                    
                    result_message += f"Image for part {image_count}: {image_path}.\n\n"

            result_message += f"\n\nEnd of messages.\nWrite answer to user using this text, remove under the hood details about styling info, formatting, or something like that, and translate it to user language if needed."
            return result_message
        except Exception as e:
            return f"Error generating multimodal story: {str(e)}"
            
    async def _transform_image(self, prompt: str, image_path: str, caption: str = "Here is your transformed image") -> str:
        """Transform an existing image using Google's Gemini model"""
        print(f"Transforming image with prompt: {prompt}, image_path: {image_path}")
        try:
            # Verify the image path exists
            image_path_obj = Path(image_path)
            if not image_path_obj.exists():
                return f"Error: The image at path {image_path} does not exist."
            
            # Open the image using PIL
            source_image = PIL.Image.open(image_path_obj)
            
            # Call Gemini API to transform the image
            response = genai_client.models.generate_content(
                model="gemini-2.5-flash-image-preview",
                contents=(
                    prompt,
                    source_image
                ),
                config=types.GenerateContentConfig(
                    response_modalities=["Text", "Image"]
                ),
            )
            
            contents = response.candidates[0].content.parts
            
            all_parts = []
            for content in contents:
                if 'text' in content.model_fields_set:
                    all_parts.append(TextPart(content.text))
                elif 'inline_data' in content.model_fields_set:
                    all_parts.append(InlineDataPart(
                        content.inline_data.mime_type,
                        content.inline_data.data
                    ))
            
            result_message = "Image transformation results:\n\n"
            transformed_image_sent = False
            
            for part in all_parts:
                if isinstance(part, TextPart):
                    # Send any explanatory text from Gemini about the transformation
                    try: # as markdown
                        await self.telegram_update.message.reply_text(f"Text explanation:\n\n{part.text}", parse_mode="markdown")
                    except:
                        # check length of text and split to chunks of 3000 characters
                        text_chunks = [part.text[i:i+3000] for i in range(0, len(part.text), 3000)]
                        sub_part_count = 0
                        for chunk in text_chunks:
                            sub_part_count += 1
                            try:
                                await self.telegram_update.message.reply_text(f"Text explanation subpart {sub_part_count}:\n\n{chunk}", parse_mode="markdown")
                            except:
                                await self.telegram_update.message.reply_text(f"Text explanation subpart {sub_part_count}:\n\n{chunk}")
                    result_message += f"Text explanation sent to user: \n\n{part.text}\n\n"
                elif isinstance(part, InlineDataPart):
                    # Save and send the transformed image
                    extension = part.mime_type.split('/')[-1]
                    transformed_image_path = self.images_path / f"transformed_{uuid.uuid4()}.{extension}"
                    with open(transformed_image_path, "wb") as f:
                        f.write(part.data)
                    
                    with open(transformed_image_path, 'rb') as file:
                        await self.telegram_update.message.reply_document(
                            document=file,
                            caption=caption
                        )
                    
                    result_message += f"Transformed image saved to: {transformed_image_path} and sent to user.\n"
                    transformed_image_sent = True
            
            if transformed_image_sent:
                result_message += "Image transformation completed successfully.\n"
            else:
                result_message += "Warning: No transformed image was generated. The model only provided text response.\n"
                
            return result_message
        except Exception as e:
            return f"Error transforming image: {str(e)}"

