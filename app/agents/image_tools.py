from openai import OpenAI
import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from pathlib import Path
import uuid
import base64
import telegram
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
                "name": "image_generator",
                "description": "Generate high-quality images from text descriptions using Gemini image generation, save to images directory, and send to user via Telegram. ALWAYS use Normal mode by default unless user specifically requests Pro mode.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "A detailed description of the image you want to generate. Be specific with visual details, style, lighting, composition for best results."
                        },
                        "style": {
                            "type": "string",
                            "description": "Visual style for the image (e.g., 'photorealistic', 'digital art', 'cartoon', 'anime', 'impressionist', 'minimalist')",
                            "default": "photorealistic"
                        },
                        "model": {
                            "type": "string",
                            "description": "Model to use: 'Normal' (faster, standard quality, gemini-2.5-flash-image) or 'Pro' (higher quality, more control, gemini-3-pro-image-preview). ALWAYS use 'Normal' unless user specifically requests Pro mode.",
                            "enum": ["Normal", "Pro"],
                            "default": "Normal"
                        },
                        "aspect_ratio": {
                            "type": "string",
                            "description": "Aspect ratio for Pro mode only. Options: '1:1', '2:3', '3:2', '3:4', '4:3', '4:5', '5:4', '9:16', '16:9', '21:9'. Default is '16:9'. Only used when model is 'Pro'.",
                            "enum": ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
                            "default": "16:9"
                        },
                        "resolution": {
                            "type": "string",
                            "description": "Resolution for Pro mode only. Options: '1K', '2K', '4K'. Default is '2K'. Only used when model is 'Pro'.",
                            "enum": ["1K", "2K", "4K"],
                            "default": "2K"
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
                "description": "Edit and transform existing images using Gemini's advanced image editing capabilities. Supports adding, removing, or modifying elements, changing styles, adjusting colors, and mask-free editing. ALWAYS use Normal mode by default unless user specifically requests Pro mode.",
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
                            "description": "Model to use: 'Normal' (faster, standard quality, gemini-2.5-flash-image) or 'Pro' (higher quality, more control, gemini-3-pro-image-preview). ALWAYS use 'Normal' unless user specifically requests Pro mode.",
                            "enum": ["Normal", "Pro"],
                            "default": "Normal"
                        },
                        "aspect_ratio": {
                            "type": "string",
                            "description": "Aspect ratio for Pro mode only. Options: '1:1', '2:3', '3:2', '3:4', '4:3', '4:5', '5:4', '9:16', '16:9', '21:9'. Default is '16:9'. Only used when model is 'Pro'.",
                            "enum": ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
                            "default": "16:9"
                        },
                        "resolution": {
                            "type": "string",
                            "description": "Resolution for Pro mode only. Options: '1K', '2K', '4K'. Default is '2K'. Only used when model is 'Pro'.",
                            "enum": ["1K", "2K", "4K"],
                            "default": "2K"
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
                "description": "Compose a new image using multiple input images with Gemini's advanced multi-image processing. Perfect for style transfer, combining elements from different images, or creating composite scenes. ALWAYS use Normal mode by default unless user specifically requests Pro mode.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Detailed instructions on how to compose the new image using the provided images (e.g., 'Take the dress from the first image and let the person from the second image wear it', 'Transfer the artistic style from the first image to the second image', 'Combine the foreground from image 1 with the background from image 2')"
                        },
                        "image_paths": {
                            "type": "array",
                            "description": "Array of paths to image files to be used in composition. Should contain 2-3 images for best results.",
                            "items": {
                                "type": "string"
                            },
                            "minItems": 2,
                            "maxItems": 3
                        },
                        "model": {
                            "type": "string",
                            "description": "Model to use: 'Normal' (faster, standard quality, gemini-2.5-flash-image) or 'Pro' (higher quality, more control, gemini-3-pro-image-preview). ALWAYS use 'Normal' unless user specifically requests Pro mode.",
                            "enum": ["Normal", "Pro"],
                            "default": "Normal"
                        },
                        "aspect_ratio": {
                            "type": "string",
                            "description": "Aspect ratio for Pro mode only. Options: '1:1', '2:3', '3:2', '3:4', '4:3', '4:5', '5:4', '9:16', '16:9', '21:9'. Default is '16:9'. Only used when model is 'Pro'.",
                            "enum": ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"],
                            "default": "16:9"
                        },
                        "resolution": {
                            "type": "string",
                            "description": "Resolution for Pro mode only. Options: '1K', '2K', '4K'. Default is '2K'. Only used when model is 'Pro'.",
                            "enum": ["1K", "2K", "4K"],
                            "default": "2K"
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

    async def _generate_image_dall_e(self, prompt: str, size: str = "1024x1024", quality: str = "standard", caption: str = "Here is your image") -> str:
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
                model="gemini-2.5-flash-image",
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
                    except Exception:
                        # check length of text and split to chunks of 3000 characters
                        text_chunks = [part.text[i:i+3000] for i in range(0, len(part.text), 3000)]
                        sub_part_count = 0
                        for chunk in text_chunks:
                            sub_part_count += 1
                            try:
                                await self.telegram_update.message.reply_text(f"Part {part_count} subpart {sub_part_count}:\n\n{chunk}", parse_mode="markdown")
                            except Exception:
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

            result_message += "\n\nEnd of messages.\nWrite answer to user using this text, remove under the hood details about styling info, formatting, or something like that, and translate it to user language if needed."
            return result_message
        except Exception as e:
            return f"Error generating multimodal story: {str(e)}"
            
    async def _image_editing(self, prompt: str, image_path: str, model: str = "Normal", aspect_ratio: str = "16:9", resolution: str = "2K", caption: str = "Here is your edited image") -> str:
        """Edit an existing image using Google's Gemini model"""
        print(f"Editing image with prompt: {prompt}, image_path: {image_path}, Model: {model}")
        try:
            # Verify the image path exists
            image_path_obj = Path(image_path)
            if not image_path_obj.exists():
                return f"Error: The image at path {image_path} does not exist."
            
            # Open the image using PIL
            source_image = PIL.Image.open(image_path_obj)
            
            # Select model based on mode
            model_name = "gemini-3-pro-image-preview" if model.lower() == "pro" else "gemini-2.5-flash-image"
            
            # Build config based on mode
            if model == "Pro":
                config = types.GenerateContentConfig(
                    response_modalities=["Text", "Image"],
                    image_config=types.ImageConfig(
                        aspect_ratio=aspect_ratio,
                        image_size=resolution
                    )
                )
            else:
                config = types.GenerateContentConfig(
                    response_modalities=["Text", "Image"]
                )
            
            # Call Gemini API to transform the image
            response = genai_client.models.generate_content(
                model=model_name,
                contents=(
                    prompt,
                    source_image
                ),
                config=config,
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
                    except Exception:
                        # check length of text and split to chunks of 3000 characters
                        text_chunks = [part.text[i:i+3000] for i in range(0, len(part.text), 3000)]
                        sub_part_count = 0
                        for chunk in text_chunks:
                            sub_part_count += 1
                            try:
                                await self.telegram_update.message.reply_text(f"Text explanation subpart {sub_part_count}:\n\n{chunk}", parse_mode="markdown")
                            except Exception:
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
            return f"Error editing image: {str(e)}"
    
    async def _image_generator(self, prompt: str, style: str = "photorealistic", model: str = "Normal", aspect_ratio: str = "16:9", resolution: str = "2K", caption: str = "Here is your generated image") -> str:
        """Generate a high-quality image from text using Google's Gemini model"""
        print(f"Generating image with Gemini - Prompt: {prompt}, Style: {style}, Model: {model}")
        try:
            # Format the prompt to include style information
            if style and style != "photorealistic":
                formatted_prompt = f"Create a {style} style image: {prompt}"
            else:
                formatted_prompt = prompt
            
            # Select model based on mode
            model_name = "gemini-3-pro-image-preview" if model.lower() == "pro" else "gemini-2.5-flash-image"
            
            # Build config based on mode
            if model == "Pro":
                config = types.GenerateContentConfig(
                    image_config=types.ImageConfig(
                        aspect_ratio=aspect_ratio,
                        image_size=resolution
                    )
                )
                response = genai_client.models.generate_content(
                    model=model_name,
                    contents=[formatted_prompt],
                    config=config
                )
            else:
                response = genai_client.models.generate_content(
                    model=model_name,
                    contents=[formatted_prompt],
                )
            
            # Process the response and extract images
            contents = response.candidates[0].content.parts
            
            image_saved = False
            result_message = ""
            
            for part in contents:
                if 'text' in part.model_fields_set and part.text:
                    # Send any accompanying text
                    try:
                        await self.telegram_update.message.reply_text(part.text, parse_mode="markdown")
                    except Exception:
                        await self.telegram_update.message.reply_text(part.text)
                    result_message += f"Generated text: {part.text}\n\n"
                    
                elif 'inline_data' in part.model_fields_set:
                    # Save and send the generated image
                    extension = part.inline_data.mime_type.split('/')[-1]
                    if extension == 'jpeg':
                        extension = 'jpg'
                    image_path = self.images_path / f"generated_{uuid.uuid4()}.{extension}"
                    
                    with open(image_path, "wb") as f:
                        f.write(part.inline_data.data)
                    
                    with open(image_path, 'rb') as file:
                        await self.telegram_update.message.reply_document(
                            document=file,
                            caption=caption
                        )
                    
                    result_message += f"Image generated and saved to: {image_path} and sent to user.\n"
                    image_saved = True
            
            if image_saved:
                result_message += "Image generation completed successfully.\n"
            else:
                result_message += "Warning: No image was generated. The model only provided text response.\n"
                
            return result_message
        except Exception as e:
            return f"Error generating image: {str(e)}"
    
    async def _image_composition(self, prompt: str, image_paths: List[str], model: str = "Normal", aspect_ratio: str = "16:9", resolution: str = "2K", caption: str = "Here is your composed image") -> str:
        """Compose a new image from multiple input images using Google's Gemini model"""
        print(f"Composing image with prompt: {prompt}, image_paths: {image_paths}, Model: {model}")
        try:
            # Verify all image paths exist
            images = []
            for image_path in image_paths:
                image_path_obj = Path(image_path)
                if not image_path_obj.exists():
                    return f"Error: The image at path {image_path} does not exist."
                images.append(PIL.Image.open(image_path_obj))
            
            if len(images) < 2:
                return "Error: At least 2 images are required for composition."
            if len(images) > 3:
                return "Error: Maximum 3 images are supported for composition."
            
            # Build the content list with images and prompt
            contents = []
            for i, image in enumerate(images):
                contents.append(image)
            contents.append(prompt)
            
            # Select model based on mode
            model_name = "gemini-3-pro-image-preview" if model.lower() == "pro" else "gemini-2.5-flash-image"
            
            # Build config based on mode
            if model == "Pro":
                config = types.GenerateContentConfig(
                    response_modalities=["Text", "Image"],
                    image_config=types.ImageConfig(
                        aspect_ratio=aspect_ratio,
                        image_size=resolution
                    )
                )
            else:
                config = types.GenerateContentConfig(
                    response_modalities=["Text", "Image"]
                )
            
            # Call Gemini API for image composition
            response = genai_client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
            
            response_parts = response.candidates[0].content.parts
            
            result_message = "Image composition results:\n\n"
            composed_image_sent = False
            
            for part in response_parts:
                if 'text' in part.model_fields_set and part.text:
                    # Send any explanatory text from Gemini about the composition
                    try:
                        await self.telegram_update.message.reply_text(f"Composition details:\n\n{part.text}", parse_mode="markdown")
                    except Exception:
                        # Handle text length by splitting if needed
                        text_chunks = [part.text[i:i+3000] for i in range(0, len(part.text), 3000)]
                        for i, chunk in enumerate(text_chunks):
                            try:
                                await self.telegram_update.message.reply_text(f"Composition details (part {i+1}):\n\n{chunk}", parse_mode="markdown")
                            except Exception:
                                await self.telegram_update.message.reply_text(f"Composition details (part {i+1}):\n\n{chunk}")
                    result_message += f"Composition explanation sent to user: \n\n{part.text}\n\n"
                    
                elif 'inline_data' in part.model_fields_set:
                    # Save and send the composed image
                    extension = part.inline_data.mime_type.split('/')[-1]
                    if extension == 'jpeg':
                        extension = 'jpg'
                    composed_image_path = self.images_path / f"composed_{uuid.uuid4()}.{extension}"
                    
                    with open(composed_image_path, "wb") as f:
                        f.write(part.inline_data.data)
                    
                    with open(composed_image_path, 'rb') as file:
                        await self.telegram_update.message.reply_document(
                            document=file,
                            caption=caption
                        )
                    
                    result_message += f"Composed image saved to: {composed_image_path} and sent to user.\n"
                    composed_image_sent = True
            
            if composed_image_sent:
                result_message += "Image composition completed successfully.\n"
            else:
                result_message += "Warning: No composed image was generated. The model only provided text response.\n"
                
            return result_message
        except Exception as e:
            return f"Error composing image: {str(e)}"

