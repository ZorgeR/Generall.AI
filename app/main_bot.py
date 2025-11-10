import os
import logging
import uuid
import base64
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery, Message, Chat
from telegram.ext import Application, MessageHandler, CommandHandler, CallbackQueryHandler, filters, ContextTypes
from datetime import datetime, timezone, timedelta
from voice import VoiceManager
from elevenlabs.client import ElevenLabs
from pydub import utils, AudioSegment
from openai import OpenAI
import anthropic
import tempfile
import io
import asyncio
import json
import logging
from typing import Any
from pathlib import Path
# Import the secure container system
from secure_container.main import initialize_secure_containers, cleanup_containers
import shutil

# Check if running in Docker
in_docker = os.path.exists('/.dockerenv')

# First try to use system ffmpeg if available (especially in Docker)
if in_docker or shutil.which('ffmpeg'):
    ffmpeg_path = ""  # Use system path
    ffmpeg_bin = "ffmpeg"
elif os.name == "posix":
    ffmpeg_path = "./ffmpeg/ffmpeg"
    ffmpeg_bin = f"{ffmpeg_path}//ffmpeg"
else:
    ffmpeg_path = "C:\\Users\\ashamsiev\\AppData\\Local\\ffmpegio\\ffmpeg-downloader\\ffmpeg\\bin"
    ffmpeg_bin = f"{ffmpeg_path}//ffmpeg.exe"

AudioSegment.converter = f"{ffmpeg_bin}"

def get_prober_name():
    if in_docker or shutil.which('ffprobe'):
        return "ffprobe"  # Use system path
    elif os.name == "posix":
        return f"{ffmpeg_path}/ffprobe"
    else:
        return f"{ffmpeg_path}//ffprobe.exe"

utils.get_prober_name = get_prober_name

load_dotenv()

logger = logging.getLogger(__name__)

# Initialize the secure container system
success = initialize_secure_containers()
if success:
    logger.info("Secure container system initialized successfully!")
else:
    logger.warning("Secure container system initialization failed! Commands will run in the main container.")
    # If secure container initialization failed, log the error and exit
    logger.error("Secure container system initialization failed!")
    exit(1)

import agents.main as agents

anthropic_model = "claude-sonnet-4-5-20250929"
openai_model = "gpt-5-2025-08-07"

user_invite_limit = os.getenv("INVITE_LIMIT")
telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID").split(",")
telegram_admin_id = os.getenv("TELEGRAM_ADMIN_ID")  # Added for admin access
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_key_whisper = os.getenv("OPENAI_API_KEY_WHISPER")
openai_client_whisper = OpenAI(api_key=openai_api_key_whisper)
openai_client = OpenAI(api_key=openai_api_key)
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
send_reasoning = True

user_invites = {}
authorized_users = set(telegram_chat_id)
allow_all_users = os.getenv("TELEGRAM_ALLOWED_ALL_USERS", "false") == "true"

# Initialize voice manager
voice_manager = VoiceManager()

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Add these at the top level with other global variables
media_group_id = {}
media_group_photos = {}
media_group_captions = {}
media_group_waiting_message = {}
media_group_tasks = {}
MEDIA_GROUP_TIMEOUT = 10.0


# Initialize default settings structure
default_settings = {
    "summarization_history": {
        "enabled": True,
        "size": 5
    },
    "dialog_history": {
        "enabled": True,
        "size": 10
    },
    "reasoning_context": {
        "enabled": True
    },
    "short_term_memory": {
        "enabled": True
    },
    "critique": {
        "enabled": True,
        "max_iteration": 5
    },
    "judge": {
        "enabled": True,
        "max_iteration": 5
    },
    "tools": {
        "enabled": True,
        "max_iteration": 20
    },
    "semantic_search": {
        "enabled": True,
        "max_results": 3
    },
    "thinking": {
        "enabled": True
    },
    "system_prompt": {
        "type": "generall-ai-v2"  # Options: "generall-ai-v2", "perplexity-deep-research"
    }
}

class UserSettings:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.settings = {
            "summarization_history": {
                "enabled": True,
                "size": 5
            },
            "dialog_history": {
                "enabled": True,
                "size": 10
            },
            "reasoning_context": {
                "enabled": True
            },
            "short_term_memory": {
                "enabled": True
            },
            "critique": {
                "enabled": True,
                "max_iteration": 5
            },
            "judge": {
                "enabled": True,
                "max_iteration": 5
            },
            "tools": {
                "enabled": True,
                "max_iteration": 20
            },
            "semantic_search": {
                "enabled": True,
                "max_results": 3
            },
            "thinking": {
                "enabled": True
            },
            "system_prompt": {
                "type": "generall-ai-v2"  # Options: "generall-ai-v2", "perplexity-deep-research"
            }
        }
        self.load_settings()
    
    def get_setting(self, setting_name: str, sub_setting: str = None):
        if sub_setting:
            return self.settings.get(setting_name, {}).get(sub_setting)
        return self.settings.get(setting_name)

    def set_setting(self, setting_name: str, setting_value: Any, sub_setting: str = None):
        if sub_setting:
            if setting_name not in self.settings:
                self.settings[setting_name] = {}
            self.settings[setting_name][sub_setting] = setting_value
        else:
            self.settings[setting_name] = setting_value
        self.save_settings()

    def save_settings(self):
        os.makedirs(f"data/{self.user_id}", exist_ok=True)
        with open(f"data/{self.user_id}/settings.json", "w") as f:
            json.dump(self.settings, f, indent=2)

    def load_settings(self):
        if os.path.exists(f"data/{self.user_id}/settings.json"):
            with open(f"data/{self.user_id}/settings.json", "r") as f:
                loaded_settings = json.load(f)
                # Merge loaded settings with defaults
                for key, value in loaded_settings.items():
                    if isinstance(value, dict):
                        self.settings[key].update(value)
                    else:
                        self.settings[key] = value

    def validate_size(self, size: int) -> int:
        """Validate and constrain size values"""
        return max(1, min(50, size))

    def validate_iteration(self, iteration: int, type: str) -> int:
        """Validate and constrain iteration values"""
        if type == "critique":
            return max(1, min(300, iteration))
        elif type == "judge":
            return max(1, min(300, iteration))
        elif type == "tools":
            return max(1, min(300, iteration))

    def validate_semantic_max_results(self, max_results: int) -> int:
        """Validate and constrain semantic search max results value"""
        return max(1, min(20, max_results))

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def split_text_intelligently(text: str, max_length: int = 4000) -> list[str]:
    """
    Split text into chunks with a maximum length, trying to split at natural boundaries:
    1. Preferably at the last paragraph break (double newline) before max_length
    2. Otherwise at the last single newline before max_length
    3. Otherwise at the last space before max_length
    4. If none of the above is possible, split at exactly max_length
    
    Args:
        text: The text to split
        max_length: Maximum length of each chunk (default: 4000)
        
    Returns:
        List of text chunks
    """
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        if start + max_length >= len(text):
            chunks.append(text[start:])
            break
        
        # Try to find the last paragraph break (double newline) within the max_length
        end = start + max_length
        paragraph_break_pos = text.rfind('\n\n', start, end)
        
        if paragraph_break_pos != -1 and paragraph_break_pos > start:
            # Split at paragraph break
            chunks.append(text[start:paragraph_break_pos+2])
            start = paragraph_break_pos + 2
        else:
            # Try to find the last single newline within the max_length
            newline_pos = text.rfind('\n', start, end)
            
            if newline_pos != -1 and newline_pos > start:
                # Split at newline
                chunks.append(text[start:newline_pos+1])
                start = newline_pos + 1
            else:
                # Try to find the last space within the max_length
                space_pos = text.rfind(' ', start, end)
                
                if space_pos != -1 and space_pos > start:
                    # Split at space
                    chunks.append(text[start:space_pos+1])
                    start = space_pos + 1
                else:
                    # No natural breaking point, split at max_length
                    chunks.append(text[start:end])
                    start = end
    
    return chunks

async def describe_document_openai(question, file_path, file_mime_type):
    message_files = openai_client.files.create(
        file=open(file_path, "rb"),
        purpose="assistants"
    )
    message = openai_client.chat.completions.create(
        model=openai_model,
        messages=[
            {"role": "user", "content": question}
        ],
        files=[message_files]
    )
    return message.choices[0].message.content

async def describe_document_anthropic(question, file_path, document_type, file_mime_type):
    document_in_base64 = None
    try:
        with open(file_path, "rb") as document_file:
            document_in_base64 = base64.b64encode(document_file.read()).decode("utf-8")
    except FileNotFoundError:
        return "Error: Document file not found"

    if document_in_base64 is None:
        return "Error: Document file not found"

    message = anthropic_client.messages.create(
        model=anthropic_model,
        messages=[
            {
            "role": "user",
            "content": [
                {
                    "type": document_type,
                    "source": {
                        "type": "base64",
                        "media_type": file_mime_type,
                        "data": document_in_base64
                    }
                },
                {
                    "type": "text",
                    "text": question
                }
                ]
            }],
        system="You are a very professional document analyze specialist. Analyze the given document in a detailed way, to answer user's question.",
        max_tokens=4096,
    )

    return message.content[0].text

async def describe_txt(question, txt_path):
    """Analyze text files and answer questions about them"""
    try:
        with open(txt_path, "r", encoding="utf-8") as txt_file:
            txt_content = txt_file.read()
    except FileNotFoundError:
        return "Error: TXT file not found"
    except UnicodeDecodeError:
        # Try alternative encoding if UTF-8 fails
        try:
            with open(txt_path, "r", encoding="latin-1") as txt_file:
                txt_content = txt_file.read()
        except Exception as e:
            return f"Error reading TXT file: {str(e)}"

    # Handle large text files by splitting them if needed
    if len(txt_content) > 100000:  # Approximately 100KB
        return await process_large_text(txt_content, question, "text document")
    
    message = anthropic_client.messages.create(
        model=anthropic_model,
        messages=[
            {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Here is the content of a text document:\n\n<document>\n{txt_content}\n</document>\n\n<user_question>\n{question}\n</user_question>"
                }]
            }],
        system="You are a very professional document analyze specialist. Analyze the given text document in a detailed way, to answer user's question.",
        max_tokens=4096,
    )

    return message.content[0].text

async def describe_json(question, json_path):
    """Analyze JSON files and answer questions about them"""
    try:
        with open(json_path, "r", encoding="utf-8") as json_file:
            json_content = json_file.read()
            # Validate it's proper JSON by parsing it
            json.loads(json_content)
    except FileNotFoundError:
        return "Error: JSON file not found"
    except json.JSONDecodeError:
        return "Error: Invalid JSON format"
    except Exception as e:
        return f"Error reading JSON file: {str(e)}"

    # Handle large JSON files by splitting them if needed
    if len(json_content) > 100000:  # Approximately 100KB
        return await process_large_text(json_content, question, "JSON document")
    
    message = anthropic_client.messages.create(
        model=anthropic_model,
        messages=[
            {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Here is the content of a JSON document:\n\n<json_document>\n{json_content}\n</json_document>\n\n<user_question>\n{question}\n</user_question>"
                }]
            }],
        system="You are a very professional data analyst specializing in JSON. Analyze the given JSON document in a detailed way, to answer user's question. Format your insights clearly.",
        max_tokens=4096,
    )

    return message.content[0].text

async def describe_docx(question, docx_path):
    """Analyze DOCX files and answer questions about them"""
    try:
        try:
            import docx2txt
        except ImportError:
            return "Error: docx2txt library not installed. Please install it with 'pip install docx2txt'."
            
        docx_content = docx2txt.process(docx_path)
    except FileNotFoundError:
        return "Error: DOCX file not found"
    except Exception as e:
        return f"Error reading DOCX file: {str(e)}"

    # Handle large DOCX files by splitting them if needed
    if len(docx_content) > 100000:  # Approximately 100KB
        return await process_large_text(docx_content, question, "Word document")
    
    message = anthropic_client.messages.create(
        model=anthropic_model,
        messages=[
            {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Here is the content of a Word document:\n\n{docx_content}\n\n{question}"
                }]
            }],
        system="You are a very professional document analyst specializing in Word documents. Analyze the given document in a detailed way, to answer user's question.",
        max_tokens=4096,
    )

    return message.content[0].text

async def describe_xlsx(question, xlsx_path):
    """Analyze Excel (XLSX) files and answer questions about them"""
    try:
        try:
            import pandas as pd
        except ImportError:
            return "Error: pandas library not installed. Please install it with 'pip install pandas openpyxl'."
        
        # Read all sheets
        xlsx_content = ""
        excel_file = pd.ExcelFile(xlsx_path)
        sheet_names = excel_file.sheet_names
        
        for sheet in sheet_names:
            df = pd.read_excel(xlsx_path, sheet_name=sheet)
            xlsx_content += f"\n\nSheet: {sheet}\n"
            xlsx_content += df.to_string(index=True) + "\n"
            
    except FileNotFoundError:
        return "Error: XLSX file not found"
    except Exception as e:
        return f"Error reading XLSX file: {str(e)}"

    # Handle large Excel files by splitting them if needed
    if len(xlsx_content) > 100000:  # Approximately 100KB
        return await process_large_text(xlsx_content, question, "Excel spreadsheet")
    
    message = anthropic_client.messages.create(
        model=anthropic_model,
        messages=[
            {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Here is the content of an Excel spreadsheet:\n\n{xlsx_content}\n\n{question}"
                }]
            }],
        system="You are a very professional data analyst specializing in Excel spreadsheets. Analyze the given spreadsheet in a detailed way, to answer user's question. Format numeric insights clearly.",
        max_tokens=4096,
    )

    return message.content[0].text

async def process_large_text(content, question, content_type):
    """Process large text content by splitting it into chunks and analyzing each chunk"""
    # Split the content into chunks of approximately 50KB each
    chunk_size = 50000
    chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
    
    summaries = []
    for i, chunk in enumerate(chunks):
        message = anthropic_client.messages.create(
            model=anthropic_model,
            messages=[
                {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Here is part {i+1} of {len(chunks)} of a {content_type}:\n\n{chunk}\n\nSummarize this part of the document concisely."
                    }]
                }],
            system="You are a very professional document analyst. Summarize this part of the document concisely.",
            max_tokens=1000,
        )
        summaries.append(message.content[0].text)
    
    # Combine the summaries and answer the question
    combined_summary = "\n\n".join([f"Part {i+1} summary: {summary}" for i, summary in enumerate(summaries)])
    
    final_message = anthropic_client.messages.create(
        model=anthropic_model,
        messages=[
            {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"I have analyzed a large {content_type} in parts. Here are the summaries of each part:\n\n{combined_summary}\n\nBased on these summaries, please answer the following question: {question}"
                }]
            }],
        system="You are a very professional document analyst. Based on the provided summaries, answer the user's question thoroughly.",
        max_tokens=4096,
    )
    
    return final_message.content[0].text

async def describe_document(question, file_path):
    """Generic function to handle document analysis based on file extension"""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        return await describe_document_anthropic(question, file_path, "document", "application/pdf")
    elif file_extension in ['.txt', '.csv', '.py', '.sh', '.bat', '.md', '.ps1', '.js', '.css', '.html', '.php', '.sql', '.xml', '.yaml', '.yml', '.toml', '.ini', '.conf', '.log', '.jsonl']:
        return await describe_txt(question, file_path)
    elif file_extension == '.json':
        return await describe_json(question, file_path)
    elif file_extension == '.docx':
        return await describe_docx(question, file_path)
    elif file_extension in ['.xlsx', '.xls']:
        return await describe_xlsx(question, file_path)
    else:
        return f"Error: Unsupported file type {file_extension}"

async def describe_image_anthropic(question, image_path):
    base64_image = encode_image(image_path)
    
    image_media_type = "image/jpeg"
    # image_data = base64.standard_b64encode(image_content).decode("utf-8")

    message = anthropic_client.messages.create(
        model=anthropic_model,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_media_type,
                            "data": base64_image,
                        },
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ],
            }
        ],
    )

    return message.content[0].text

async def describe_image_openai(question, image_path):
    base64_image = encode_image(image_path)

    response = openai_client.chat.completions.create(
        model=openai_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
    )

    return response.choices[0].message.content

async def transcribe_audio(audio_file_path):
    with open(audio_file_path, "rb") as audio:
        response = openai_client_whisper.audio.transcriptions.create(
            model="whisper-1",
            file=audio, 
            response_format="text"
        )
    if response:
        return response
    else:
        return False

async def get_answer(user_message, user_id, update_status=None, update=None, context=None):
    # Handle special cases directly
    if user_message.lower().strip() in ["current time", "time", "what time is it"]:
        current_time = datetime.now()
        return f"Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}"
    
    # Get user settings
    user_settings = UserSettings(user_id).settings
    
    # Check and initialize missing settings
    for category, settings in default_settings.items():
        if category not in user_settings:
            user_settings[category] = {}
        
        if isinstance(settings, dict):
            for key, value in settings.items():
                if key not in user_settings[category]:
                    user_settings[category][key] = value
    
    # Save updated settings
    UserSettings(user_id).settings = user_settings
    UserSettings(user_id).save_settings()
    
    # For all other queries, use the agent
    agent = agents.ChainOfThoughtAgent(model_type="anthropic", model=anthropic_model, user_id=user_id, telegram_update=update, user_settings=user_settings)
    response, messages = await agent.generate_response(user_message, update_status)
    print("\nFinal Response with Step-by-Step Reasoning:")
    print(response)
    return response, messages

async def sendvoice_to_user(audio_stream):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        for chunk in audio_stream:
            temp_audio.write(chunk)
        temp_audio_path = temp_audio.name

    return temp_audio_path

async def send_response_to_user(update: Update, thinking_message, response: str):
    """Helper function to send bot's response to the user, handling long messages and markdown formatting"""
    try:
        # Try sending with markdown
        if len(response) > 4000:
            chunks = split_text_intelligently(response)
            for chunk in chunks:
                await update.message.reply_text(chunk, parse_mode="markdown")
            await thinking_message.edit_text(text="Сообщение было длинным, поэтому разбито на несколько частей...", parse_mode="markdown")
        else:
            if response == "" or response == None:
                response = "🤖 *No response from the AI.*"
            await thinking_message.edit_text(text=response, parse_mode="markdown")
    except Exception as e:
        # Fallback to plain text if markdown fails
        if len(response) > 4000:
            chunks = split_text_intelligently(response)
            for chunk in chunks:
                await update.message.reply_text(chunk)
            await thinking_message.edit_text(text="Сообщение было длинным, поэтому разбито на несколько частей...")
        else:
            if response == "" or response == None:
                response = "🤖 No response from the AI."
            await thinking_message.edit_text(text=response)

async def send_reasoning_file(update: Update, messages, user_id: str):
    """Helper function to send reasoning file if enabled"""
    user_settings = UserSettings(user_id)
    if not user_settings.get_setting("reasoning_context", "enabled") or not messages:
        return
        
    uuid_reasoning = str(uuid.uuid4())
    file_path = f"reasoning_{user_id}_{uuid_reasoning}.txt"
    
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            for msg in messages:
                f.write(msg["content"][0]["text"] + "\n\n========\n\n")
        
        await update.message.reply_document(
            document=open(file_path, "rb"), 
            caption="Reasoning history."
        )
    except Exception as e:
        print(f"Error sending reasoning file: {str(e)}")
    finally:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error removing reasoning file: {str(e)}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.message.chat_id)
    
    logger.info(f"Received message from user {user_id}: {update.message.text[:50]}...")

    # Check if the user is authorized
    if not is_user_authorized(user_id):
        logger.warning(f"Unauthorized access attempt from user {user_id}")
        await update.message.reply_text("Unauthorized. You need an invite to use this bot.")
        return
    
    # Get user's message
    user_message = update.message.text
    
    # Show typing status and send initial message
    await context.bot.send_chat_action(chat_id=update.message.chat_id, action='typing')
    thinking_message = await update.message.reply_text("💭 *Thinking...*", parse_mode="markdown")
    
    async def update_thinking_message(step: str, details: str, iteration: int, critique: int):
        if step == "saving":
            iteration = "final"
            critique = "end"
        try:
            await thinking_message.edit_text(
                f"💭 *Thinking...*\n"
                f"- - - - \n"
                f"📝 *Step:* _{step.replace('_', '-')}_\n"
                f"📋 *Details:* _{details.replace('_', '-')}_\n"
                f"🔄 *Iterations:* _{iteration}_\n"
                f"🎯 *Critiques:* _{critique}_",
                parse_mode="markdown"
            )
        except Exception as e:
            logger.warning(f"Failed to update thinking message: {str(e)}")
    
    try:
        # Get response
        logger.info(f"Processing message for user {user_id}")
        response, messages = await get_answer(user_message, user_id, update_thinking_message, update, context)
        logger.info(f"Got response for user {user_id}, sending to user")
        await send_response_to_user(update, thinking_message, response)
        await send_reasoning_file(update, messages, user_id)
        logger.info(f"Successfully processed message for user {user_id}")
    except Exception as e:
        # Edit thinking message with error if something goes wrong
        trace_id = str(uuid.uuid4())
        await thinking_message.edit_text(text=f"❌ An error occurred. Trace ID: {trace_id}")
        logger.error(f"An error occurred with trace ID {trace_id}: {str(e)}")
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")

async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.message.chat_id)
    
    # Check if the user is authorized
    if not is_user_authorized(user_id):
        await update.message.reply_text("Unauthorized. You need an invite to use this bot.")
        return
    
    voice_message = update.message.voice
    
    # Create temporary files in a directory we know exists
    temp_dir = "temp_audio"
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_ogg = os.path.join(temp_dir, f"voice_{uuid.uuid4()}.oga")
    temp_mp3 = os.path.join(temp_dir, f"voice_{uuid.uuid4()}.mp3")
    
    # Send initial status message
    status_message = await update.message.reply_text("🎙️ *Transcribing audio...*", parse_mode="markdown")
    
    try:
        # Download voice message
        voice_file = await context.bot.get_file(voice_message.file_id)
        await voice_file.download_to_drive(temp_ogg)
        
        print(f"Downloaded voice file to: {temp_ogg}")
        
        # Convert OGA to MP3
        audio = AudioSegment.from_file(temp_ogg, format="ogg")
        audio.export(temp_mp3, format="mp3")
        
        print(f"Converted to MP3: {temp_mp3}")
        
        # Transcribe MP3 file
        transcription = await transcribe_audio(temp_mp3)
        
        if transcription:
            print(f"Transcription: {transcription}")
            await status_message.edit_text(f"🎙️ *Transcription:*\n{transcription}", parse_mode="markdown")
            
            # Process transcription like a regular message
            await context.bot.send_chat_action(chat_id=update.message.chat_id, action='typing')
            thinking_message = await update.message.reply_text("💭 *Thinking...*", parse_mode="markdown")
            
            async def update_thinking_message(step: str, details: str, iteration: int, critique: int):
                if step == "saving":
                    iteration = "final"
                    critique = "end"
                await thinking_message.edit_text(
                    f"💭 *Thinking...*\n"
                    f"- - - - \n"
                    f"📝 *Step:* _{step.replace('_', '-')}_\n"
                    f"📋 *Details:* _{details.replace('_', '-')}_\n"
                    f"🔄 *Iterations:* _{iteration}_\n"
                    f"🎯 *Critiques:* _{critique}_",
                    parse_mode="markdown"
                )
            
            try:
                # Get response using the same logic as handle_message
                response, messages = await get_answer(transcription, user_id, update_thinking_message, update, context)
                
                # Generate and send audio response
                await thinking_message.edit_text("🎙️ *Generating audio...*", parse_mode="markdown")
                voice_id = voice_manager.get_user_voice(user_id)
                
                try:
                    client = ElevenLabs(api_key=elevenlabs_api_key)
                    audio_stream = client.generate(
                        text=response,
                        voice=voice_id,
                        model="eleven_multilingual_v2"
                    )
                    temp_audio_path = await sendvoice_to_user(audio_stream)
                    audio_file = io.BytesIO(open(temp_audio_path, "rb").read())
                    await update.message.reply_voice(audio_file)
                except Exception as e:
                    print(f"Error generating audio: {str(e)}")
                
                # Send text response
                await send_response_to_user(update, thinking_message, response)
                await send_reasoning_file(update, messages, user_id)
                
                return transcription
            except Exception as e:
                trace_id = str(uuid.uuid4())
                await thinking_message.edit_text(text=f"❌ An error occurred. Trace ID: {trace_id}")
                logging.error(f"An error occurred with trace ID {trace_id}: {str(e)}")
        else:
            await status_message.edit_text("❌ Failed to transcribe audio", parse_mode="markdown")
            return False

    except Exception as e:
        print(f"Error transcribing voice message: {str(e)}")
        await status_message.edit_text("❌ Error processing voice message", parse_mode="markdown")
        return None
        
    finally:
        # Clean up temporary files
        try:
            if os.path.exists(temp_ogg):
                os.remove(temp_ogg)
            if os.path.exists(temp_mp3):
                os.remove(temp_mp3)
        except Exception as e:
            print(f"Error cleaning up temporary files: {str(e)}")

async def voice_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /voice command"""
    user_id = str(update.message.chat_id)
    
    # Check if the user is authorized
    if not is_user_authorized(user_id):
        await update.message.reply_text("Unauthorized. You need an invite to use this bot.")
        return
    
    # Get available voices and create keyboard
    voices = voice_manager.get_available_voices()
    current_voice_id = voice_manager.get_user_voice(user_id)
    current_voice = voice_manager.get_voice_name(current_voice_id)
    
    # Create keyboard with voice options
    keyboard = []
    row = []
    for i, voice_name in enumerate(voices.keys(), 1):
        # Add checkmark to current voice
        button_text = f"✓ {voice_name}" if voice_name == current_voice else voice_name
        row.append(InlineKeyboardButton(button_text, callback_data=f"voice_{voice_name}"))
        
        # Create new row every 2 buttons
        if i % 2 == 0 or i == len(voices):
            keyboard.append(row)
            row = []
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "*Choose a voice:*",
        reply_markup=reply_markup,
        parse_mode="markdown"
    )

async def voice_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle voice selection button presses"""
    query = update.callback_query
    user_id = str(query.message.chat_id)
    
    # Check if the user is authorized
    if not is_user_authorized(user_id):
        await query.answer("Unauthorized. You need an invite to use this bot.")
        return
    
    await query.answer()  # Acknowledge the button press
    
    # Extract voice name from callback data
    voice_name = query.data.replace("voice_", "")
    
    if voice_manager.set_user_voice(user_id, voice_name):
        # Update keyboard with new selection
        voices = voice_manager.get_available_voices()
        keyboard = []
        row = []
        for i, name in enumerate(voices.keys(), 1):
            button_text = f"✓ {name}" if name == voice_name else name
            row.append(InlineKeyboardButton(button_text, callback_data=f"voice_{name}"))
            if i % 2 == 0 or i == len(voices):
                keyboard.append(row)
                row = []
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            "*Choose a voice:*\n\n_Voice set to:_ " + voice_name,
            reply_markup=reply_markup,
            parse_mode="markdown"
        )
    else:
        await query.edit_message_text(
            f"Error setting voice to: {voice_name}\nPlease try again.",
            parse_mode="markdown"
        )

async def process_media_group(update: Update, context: ContextTypes.DEFAULT_TYPE, user_id: str):
    """Process all photos in a media group"""
    try:
        if not user_id in media_group_photos:
            print(f"User {user_id} is waiting for media group, but it is not in media_group_photos")
            return

        if user_id in media_group_waiting_message and media_group_waiting_message[user_id]:
            await media_group_waiting_message[user_id].edit_text("🖼️ *Processing media group...*", parse_mode="markdown")
        
        photos = media_group_photos[user_id]

        print(f"User {user_id} has {len(photos)} photos in media group")

        if media_group_captions[user_id] == None:
            caption = "Describe what is in this image in user language."
            describe_question = caption
        else:
            caption = media_group_captions[user_id]
            describe_question = f"Describe what is in this image and answer to this question: {caption}"

        
        # Send initial status message
        status_message = await update.message.reply_text("🖼️ *Analyzing images...*", parse_mode="markdown")
        
        temp_photos = []  # Keep track of temporary files for cleanup
        all_descriptions = []  # Store descriptions for all photos
        image_paths = []  # Store paths to downloaded images
        
        try:
            # Process each photo
            for i, photo_group in enumerate(photos, 1):
                try:
                    photo = photo_group[0]  # Get the photo from the group
                    # Download the photo
                    photo_file = await context.bot.get_file(photo.file_id)
                    temp_dir = f"./data/{user_id}/temp_photos"
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_photo = os.path.join(temp_dir, f"photo_{uuid.uuid4()}.jpg")
                    temp_photos.append(temp_photo)
                    await photo_file.download_to_drive(temp_photo)
                    
                    # Save the permanent copy to user's directory
                    user_images_dir = os.path.join("data", user_id, "images")
                    os.makedirs(user_images_dir, exist_ok=True)
                    permanent_image_path = os.path.join(user_images_dir, f"image_{uuid.uuid4()}.jpg")
                    # Copy the image to the permanent location
                    shutil.copy(temp_photo, permanent_image_path)
                    image_paths.append(permanent_image_path)
                    
                    # Get descriptions from both services
                    await status_message.edit_text(f"🤖 *Getting Anthropic description for image {i}...*", parse_mode="markdown")
                    anthropic_description = await describe_image_anthropic(question=describe_question, image_path=temp_photo)
                    
                    await status_message.edit_text(f"🤖 *Getting OpenAI description for image {i}...*", parse_mode="markdown")
                    openai_description = await describe_image_openai(question=describe_question, image_path=temp_photo)
                    
                    all_descriptions.append({
                        'anthropic': anthropic_description,
                        'openai': openai_description,
                        'path': permanent_image_path
                    })
                    
                except Exception as e:
                    logging.error(f"Error processing photo {i}: {str(e)}")
                    all_descriptions.append({
                        'anthropic': f"Error processing image {i}",
                        'openai': f"Error processing image {i}",
                        'path': "error_path"
                    })
            
            # Craft the user question combining caption and all descriptions
            descriptions_text = "\n\n".join([
                f"Image {i+1} (path: {desc['path']}):\n"
                f"Anthropic description: {desc['anthropic']}\n"
                f"OpenAI description: {desc['openai']}"
                for i, desc in enumerate(all_descriptions)
            ])
            
            user_question = f"{caption}\n\nUser attached {len(all_descriptions)} image(s) to this message. Here are the details about each image from Anthropic and OpenAI:\n\n{descriptions_text}"
            
            await status_message.edit_text("🤖 *Processing...*", parse_mode="markdown")
            # Process like a regular message
            await context.bot.send_chat_action(chat_id=update.message.chat_id, action='typing')
            thinking_message = await update.message.reply_text("💭 *Thinking...*", parse_mode="markdown")
            
            async def update_thinking_message(step: str, details: str, iteration: int, critique: int):
                if step == "saving":
                    iteration = "final"
                    critique = "end"
                await thinking_message.edit_text(
                    f"💭 *Thinking...*\n"
                    f"- - - - \n"
                    f"📝 *Step:* _{step.replace('_', '-')}_\n"
                    f"📋 *Details:* _{details.replace('_', '-')}_\n"
                    f"🔄 *Iterations:* _{iteration}_\n"
                    f"🎯 *Critiques:* _{critique}_",
                    parse_mode="markdown"
                )
            
            # Get response using the same logic as handle_message
            response, messages = await get_answer(user_question, user_id, update_thinking_message, update, context)
            await send_response_to_user(update, thinking_message, response)
            await send_reasoning_file(update, messages, user_id)
            await status_message.edit_text("🤖 *Done!*", parse_mode="markdown")
            
        finally:
            # Clean up all temporary files
            for temp_photo in temp_photos:
                try:
                    if os.path.exists(temp_photo):
                        os.remove(temp_photo)
                except Exception as e:
                    print(f"Error cleaning up temporary photo file {temp_photo}: {str(e)}")
    finally:
        # Clean up media group data
        media_group_id[user_id] = None
        media_group_photos[user_id] = []
        media_group_captions[user_id] = None
        if user_id in media_group_waiting_message and media_group_waiting_message[user_id]:
            await media_group_waiting_message[user_id].delete()
        media_group_waiting_message[user_id] = None
        if user_id in media_group_tasks:
            del media_group_tasks[user_id]

async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming photo messages with support for multiple photos"""
    user_id = str(update.message.chat_id)

    # Check if the user is authorized
    if not is_user_authorized(user_id):
        await update.message.reply_text("Unauthorized. You need an invite to use this bot.")
        return
    
    if update.message.media_group_id:
        # Cancel any existing delayed task for this user
        if user_id in media_group_tasks and media_group_tasks[user_id]:
            media_group_tasks[user_id].cancel()
        
        # Initialize media group data if needed
        if user_id not in media_group_id or media_group_id[user_id] != update.message.media_group_id:
            media_group_id[user_id] = update.message.media_group_id
            media_group_photos[user_id] = []
            media_group_captions[user_id] = None
            media_group_waiting_message[user_id] = await update.message.reply_text(
                "🖼️ *Image media group received... Waiting for images...*",
                parse_mode="markdown"
            )
        
        # Update caption if provided
        if update.message.caption and media_group_captions[user_id] == None:
            media_group_captions[user_id] = update.message.caption

        # Add photo to the group
        photos = [update.message.photo[-1]]  # Get the largest photo
        if photos not in media_group_photos[user_id]:
            media_group_photos[user_id].append(photos)
            await media_group_waiting_message[user_id].edit_text(
                f"🖼️ *Image {len(media_group_photos[user_id])} received... Waiting for other images...*",
                parse_mode="markdown"
            )
        
        async def process_media_group_with_timeout():
            try:
                await asyncio.sleep(MEDIA_GROUP_TIMEOUT)
                await process_media_group(update, context, user_id)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logging.error(f"Error in process_media_group_with_timeout: {str(e)}")
        
        task = asyncio.create_task(process_media_group_with_timeout())
        media_group_tasks[user_id] = task
        return
    
    # Handle single photo message
    photos = [update.message.photo[-1]]  # Get the largest photo
    if update.message.caption == None:
        caption = "Describe what is in this image in user language."
        describe_question = caption
    else:
        caption = update.message.caption
        describe_question = f"Describe what is in this image and answer to this question: {caption}"
    
    # Send initial status message
    status_message = await update.message.reply_text("🖼️ *Analyzing images...*", parse_mode="markdown")
    
    temp_photos = []  # Keep track of temporary files for cleanup
    all_descriptions = []  # Store descriptions for all photos
    image_paths = []  # Store paths to downloaded images
    
    try:
        # Process single photo (rest of the existing code for single photo processing)
        for i, photo in enumerate(photos, 1):  # Process the single photo
            try:
                # Download the photo
                photo_file = await context.bot.get_file(photo.file_id)
                temp_dir = "temp_photos"
                os.makedirs(temp_dir, exist_ok=True)
                temp_photo = os.path.join(temp_dir, f"photo_{uuid.uuid4()}.jpg")
                temp_photos.append(temp_photo)
                await photo_file.download_to_drive(temp_photo)
                
                # Save the permanent copy to user's directory
                user_images_dir = os.path.join("data", user_id, "images")
                os.makedirs(user_images_dir, exist_ok=True)
                permanent_image_path = os.path.join(user_images_dir, f"image_{uuid.uuid4()}.jpg")
                # Copy the image to the permanent location
                shutil.copy(temp_photo, permanent_image_path)
                image_paths.append(permanent_image_path)
                
                # Get descriptions from both services
                await status_message.edit_text(f"🤖 *Getting Anthropic description...*", parse_mode="markdown")
                anthropic_description = await describe_image_anthropic(question=describe_question, image_path=temp_photo)
                
                await status_message.edit_text(f"🤖 *Getting OpenAI description...*", parse_mode="markdown")
                openai_description = await describe_image_openai(question=describe_question, image_path=temp_photo)
                
                all_descriptions.append({
                    'anthropic': anthropic_description,
                    'openai': openai_description,
                    'path': permanent_image_path
                })
                
            except Exception as e:
                logging.error(f"Error processing photo {i}: {str(e)}")
                all_descriptions.append({
                    'anthropic': f"Error processing image {i}",
                    'openai': f"Error processing image {i}",
                    'path': "error_path"
                })
        
        # Craft the user question combining caption and all descriptions
        descriptions_text = "\n\n".join([
            f"Image {i+1} (path: {desc['path']}):\n"
            f"Anthropic description: {desc['anthropic']}\n"
            f"OpenAI description: {desc['openai']}"
            for i, desc in enumerate(all_descriptions)
        ])
        
        user_question = f"{caption}\n\nUser attached {len(all_descriptions)} image(s) to this message. Here are the details about each image from Anthropic and OpenAI:\n\n{descriptions_text}"
        
        await status_message.edit_text("🤖 *Processing...*", parse_mode="markdown")
        # Process like a regular message
        await context.bot.send_chat_action(chat_id=update.message.chat_id, action='typing')
        thinking_message = await update.message.reply_text("💭 *Thinking...*", parse_mode="markdown")
        
        async def update_thinking_message(step: str, details: str, iteration: int, critique: int):
            if step == "saving":
                iteration = "final"
                critique = "end"
            await thinking_message.edit_text(
                f"💭 *Thinking...*\n"
                f"- - - - \n"
                f"📝 *Step:* _{step.replace('_', '-')}_\n"
                f"📋 *Details:* _{details.replace('_', '-')}_\n"
                f"🔄 *Iterations:* _{iteration}_\n"
                f"🎯 *Critiques:* _{critique}_",
                parse_mode="markdown"
            )
        
        # Get response using the same logic as handle_message
        response, messages = await get_answer(user_question, user_id, update_thinking_message, update, context)
        await send_response_to_user(update, thinking_message, response)
        await send_reasoning_file(update, messages, user_id)
        await status_message.edit_text("🤖 *Done!*", parse_mode="markdown")
    except Exception as e:
        trace_id = str(uuid.uuid4())
        await status_message.edit_text(text=f"❌ An error occurred while analyzing the images. Trace ID: {trace_id}")
        logging.error(f"An error occurred with trace ID {trace_id}: {str(e)}")
    
    finally:
        # Clean up all temporary files
        for temp_photo in temp_photos:
            try:
                if os.path.exists(temp_photo):
                    os.remove(temp_photo)
            except Exception as e:
                print(f"Error cleaning up temporary photo file {temp_photo}: {str(e)}")

async def handle_document_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming document messages (PDF, TXT, JSON, DOCX, XLSX)"""
    user_id = str(update.message.chat_id)

    # Check if the user is authorized
    if not is_user_authorized(user_id):
        await update.message.reply_text("Unauthorized. You need an invite to use this bot.")
        return
    
    # Check if the document has a supported file extension
    file_name = update.message.document.file_name.lower()
    supported_extensions = ['.pdf', '.txt', '.json', '.docx', '.xlsx', '.xls', '.csv', '.py', '.sh', '.bat', '.md', '.ps1', '.js', '.css', '.html', '.php', '.sql', '.xml', '.yaml', '.yml', '.toml', '.ini', '.conf', '.log', '.jsonl']
    file_extension = os.path.splitext(file_name)[1].lower()
    
    if file_extension not in supported_extensions:
        supported_formats = ", ".join([ext.replace(".", "").upper() for ext in supported_extensions])
        await update.message.reply_text(f"❌ Only {supported_formats} documents are supported.")
        return

    # Send initial status message
    doc_type = file_extension.replace(".", "").upper()
    status_message = await update.message.reply_text(f"📄 *Processing {doc_type} document...*", parse_mode="markdown")
    
    temp_file = None
    try:
        # Download the document
        document = update.message.document
        document_file = await context.bot.get_file(document.file_id)
        temp_dir = "temp_docs"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, f"doc_{uuid.uuid4()}{file_extension}")
        await document_file.download_to_drive(temp_file)

        # Get the caption or use default question
        if update.message.caption == None:
            caption = "Analyze this document and describe its contents in detail."
            describe_question = caption
        else:
            caption = update.message.caption
            describe_question = f"Analyze this document and describe its contents in detail. When you are done, answer the following question: {caption}"

        # Get document description
        await status_message.edit_text(f"🤖 *Analyzing {doc_type} content...*", parse_mode="markdown")
        document_description = await describe_document(describe_question, temp_file)

        # Craft the user question combining caption and document description
        user_question = f"{caption}\n\nUser attached a {doc_type} document to this message. Here is the analysis of document contents:\n\n{document_description}"

        await status_message.edit_text("🤖 *Processing...*", parse_mode="markdown")
        # Process like a regular message
        await context.bot.send_chat_action(chat_id=update.message.chat_id, action='typing')
        thinking_message = await update.message.reply_text("💭 *Thinking...*", parse_mode="markdown")

        async def update_thinking_message(step: str, details: str, iteration: int, critique: int):
            if step == "saving":
                iteration = "final"
                critique = "end"
            await thinking_message.edit_text(
                f"💭 *Thinking...*\n"
                f"- - - - \n"
                f"📝 *Step:* _{step.replace('_', '-')}_\n"
                f"📋 *Details:* _{details.replace('_', '-')}_\n"
                f"🔄 *Iterations:* _{iteration}_\n"
                f"🎯 *Critiques:* _{critique}_",
                parse_mode="markdown"
            )

        # Get response using the same logic as handle_message
        response, messages = await get_answer(user_question, user_id, update_thinking_message, update, context)
        await send_response_to_user(update, thinking_message, response)
        await send_reasoning_file(update, messages, user_id)
        await status_message.edit_text("🤖 *Done!*", parse_mode="markdown")

    except Exception as e:
        trace_id = str(uuid.uuid4())
        await status_message.edit_text(text=f"❌ An error occurred while analyzing the document. Trace ID: {trace_id}")
        logging.error(f"An error occurred with trace ID {trace_id}: {str(e)}")

    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception as e:
                print(f"Error cleaning up temporary file {temp_file}: {str(e)}")

async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /settings command"""
    user_id = str(update.message.chat_id)
    
    # Check if the user is authorized
    if not is_user_authorized(user_id):
        await update.message.reply_text("Unauthorized. You need an invite to use this bot.")
        return
    
    # Initialize or get user settings
    user_settings = UserSettings(user_id)
    
    # Create status overview text
    status_text = (
        "*Settings Overview*\n\n"
        "📚 *Summarization*: " + 
        f"{'✅' if user_settings.get_setting('summarization_history', 'enabled') else '❌'} | " +
        f"Size: {user_settings.get_setting('summarization_history', 'size')}\n"
        
        "💬 *Dialog History*: " + 
        f"{'✅' if user_settings.get_setting('dialog_history', 'enabled') else '❌'} | " +
        f"Size: {user_settings.get_setting('dialog_history', 'size')}\n"
        
        "🧠 *Reasoning Context*: " + 
        f"{'✅' if user_settings.get_setting('reasoning_context', 'enabled') else '❌'}\n"
        
        "💭 *Short Term Memory*: " + 
        f"{'✅' if user_settings.get_setting('short_term_memory', 'enabled') else '❌'}\n"
        
        "🎯 *Critique*: " + 
        f"{'✅' if user_settings.get_setting('critique', 'enabled') else '❌'} | " +
        f"Max: {user_settings.get_setting('critique', 'max_iteration')}\n"
        
        "⚖️ *Judge*: " + 
        f"{'✅' if user_settings.get_setting('judge', 'enabled') else '❌'} | " +
        f"Max: {user_settings.get_setting('judge', 'max_iteration')}\n"
        
        "🛠️ *Tools*: " + 
        f"{'✅' if user_settings.get_setting('tools', 'enabled') else '❌'} | " +
        f"Max: {user_settings.get_setting('tools', 'max_iteration')}\n"
        
        "🔍 *Semantic Search*: " + 
        f"{'✅' if user_settings.get_setting('semantic_search', 'enabled') else '❌'} | " +
        f"Max Results: {user_settings.get_setting('semantic_search', 'max_results')}\n"
        
        "🤔 *Thinking Status*: " + 
        f"{'✅' if user_settings.get_setting('thinking', 'enabled') else '❌'}\n"
        
        "🧩 *System Prompt*: " + 
        f"{user_settings.get_setting('system_prompt', 'type')}\n\n"
        
        "Select a setting to configure:"
    )
    
    # Create main settings menu
    keyboard = [
        [
            InlineKeyboardButton("📚 Summarization", callback_data="settings_summarization"),
            InlineKeyboardButton("💬 Dialog", callback_data="settings_dialog")
        ],
        [
            InlineKeyboardButton("🧠 Reasoning", callback_data="settings_reasoning"),
            InlineKeyboardButton("💭 Memory", callback_data="settings_memory")
        ],
        [
            InlineKeyboardButton("🎯 Critique", callback_data="settings_critique"),
            InlineKeyboardButton("⚖️ Judge", callback_data="settings_judge")
        ],
        [
            InlineKeyboardButton("🛠️ Tools", callback_data="settings_tools"),
            InlineKeyboardButton("🔍 Semantic", callback_data="settings_semantic")
        ],
        [
            InlineKeyboardButton("🤔 Thinking", callback_data="settings_thinking"),
            InlineKeyboardButton("🧩 System Prompt", callback_data="settings_system_prompt")
        ]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    try:
        await update.message.reply_text(status_text, reply_markup=reply_markup, parse_mode="markdown")
    except Exception as e:
        # Fallback without markdown if formatting fails
        await update.message.reply_text(
            status_text.replace('*', '').replace('✅', '+').replace('❌', '-'),
            reply_markup=reply_markup
        )

async def settings_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle settings menu button presses"""
    query = update.callback_query
    user_id = str(query.message.chat_id)
    
    # Check if the user is authorized
    if not is_user_authorized(user_id):
        await query.answer("Unauthorized. You need an invite to use this bot.")
        return
    
    await query.answer()
    user_settings = UserSettings(user_id)
    
    print(f"Received callback data: {query.data}")  # Debug log
    
    # Handle special cases first
    if query.data.startswith("set_semantic_max_"):
        max_results = int(query.data.split("_")[-1])
        user_settings.set_setting("semantic_search", max_results, "max_results")
        await show_semantic_menu(query, user_settings)
        return
    elif query.data == "toggle_semantic_enabled":
        current = user_settings.get_setting("semantic_search", "enabled")
        user_settings.set_setting("semantic_search", not current, "enabled")
        await show_semantic_menu(query, user_settings)
        return
    elif query.data == "semantic_max_results":
        await show_semantic_max_results_menu(query)
        return
    elif query.data == "settings_system_prompt":  # Special case for system prompt menu
        await show_system_prompt_menu(query, user_settings)
        return
    elif query.data.startswith("settings_system_prompt_set_"):  # Special case for system prompt type selection
        prompt_type = query.data.replace("settings_system_prompt_set_", "")
        print(f"Setting system prompt type to: {prompt_type}")  # Debug log
        user_settings.set_setting("system_prompt", prompt_type, "type")
        await show_system_prompt_menu(query, user_settings)
        return
    
    # Handle regular settings navigation
    data = query.data.split("_")
    category = data[1] if len(data) > 1 else None
    action = data[2] if len(data) > 2 else None
    value = data[3] if len(data) > 3 else None
    
    print(f"Parsed data - category: {category}, action: {action}, value: {value}")  # Debug log
    
    if category == "summarization":
        if action == "toggle":
            current = user_settings.get_setting("summarization_history", "enabled")
            user_settings.set_setting("summarization_history", not current, "enabled")
            await show_summarization_menu(query, user_settings)
        elif action == "size":
            if value:
                size = user_settings.validate_size(int(value))
                user_settings.set_setting("summarization_history", size, "size")
                await show_summarization_menu(query, user_settings)
            else:
                await show_size_input_menu(query, "summarization")
        else:
            await show_summarization_menu(query, user_settings)
    
    elif category == "dialog":
        if action == "toggle":
            current = user_settings.get_setting("dialog_history", "enabled")
            user_settings.set_setting("dialog_history", not current, "enabled")
            await show_dialog_menu(query, user_settings)
        elif action == "size":
            if value:
                size = user_settings.validate_size(int(value))
                user_settings.set_setting("dialog_history", size, "size")
                await show_dialog_menu(query, user_settings)
            else:
                await show_size_input_menu(query, "dialog")
        else:
            await show_dialog_menu(query, user_settings)
    
    elif category == "reasoning":
        if action == "toggle":
            current = user_settings.get_setting("reasoning_context", "enabled")
            user_settings.set_setting("reasoning_context", not current, "enabled")
            await show_reasoning_menu(query, user_settings)
        else:
            await show_reasoning_menu(query, user_settings)
    
    elif category == "memory":
        if action == "toggle":
            current = user_settings.get_setting("short_term_memory", "enabled")
            user_settings.set_setting("short_term_memory", not current, "enabled")
            await show_memory_menu(query, user_settings)
        else:
            await show_memory_menu(query, user_settings)
    
    elif category == "critique":
        if action == "toggle":
            current = user_settings.get_setting("critique", "enabled")
            user_settings.set_setting("critique", not current, "enabled")
            await show_critique_menu(query, user_settings)
        elif action == "iteration":
            if value:
                iteration = user_settings.validate_iteration(int(value), "critique")
                user_settings.set_setting("critique", iteration, "max_iteration")
                await show_critique_menu(query, user_settings)
            else:
                await show_iteration_input_menu(query, "critique")
        else:
            await show_critique_menu(query, user_settings)
    
    elif category == "judge":
        if action == "toggle":
            current = user_settings.get_setting("judge", "enabled")
            user_settings.set_setting("judge", not current, "enabled")
            await show_judge_menu(query, user_settings)
        elif action == "iteration":
            if value:
                iteration = user_settings.validate_iteration(int(value), "judge")
                user_settings.set_setting("judge", iteration, "max_iteration")
                await show_judge_menu(query, user_settings)
            else:
                await show_iteration_input_menu(query, "judge")
        else:
            await show_judge_menu(query, user_settings)
    
    elif category == "tools":
        if action == "toggle":
            current = user_settings.get_setting("tools", "enabled")
            user_settings.set_setting("tools", not current, "enabled")
            await show_tools_menu(query, user_settings)
        elif action == "iteration":
            if value:
                iteration = user_settings.validate_iteration(int(value), "tools")
                user_settings.set_setting("tools", iteration, "max_iteration")
                await show_tools_menu(query, user_settings)
            else:
                await show_iteration_input_menu(query, "tools")
        else:
            await show_tools_menu(query, user_settings)
    
    elif category == "main":
        # Create status overview text - same as in settings_command
        status_text = (
            "*Settings Overview*\n\n"
            "📚 *Summarization*: " + 
            f"{'✅' if user_settings.get_setting('summarization_history', 'enabled') else '❌'} | " +
            f"Size: {user_settings.get_setting('summarization_history', 'size')}\n"
            
            "💬 *Dialog History*: " + 
            f"{'✅' if user_settings.get_setting('dialog_history', 'enabled') else '❌'} | " +
            f"Size: {user_settings.get_setting('dialog_history', 'size')}\n"
            
            "🧠 *Reasoning Context*: " + 
            f"{'✅' if user_settings.get_setting('reasoning_context', 'enabled') else '❌'}\n"
            
            "💭 *Short Term Memory*: " + 
            f"{'✅' if user_settings.get_setting('short_term_memory', 'enabled') else '❌'}\n"
            
            "🎯 *Critique*: " + 
            f"{'✅' if user_settings.get_setting('critique', 'enabled') else '❌'} | " +
            f"Max: {user_settings.get_setting('critique', 'max_iteration')}\n"
            
            "⚖️ *Judge*: " + 
            f"{'✅' if user_settings.get_setting('judge', 'enabled') else '❌'} | " +
            f"Max: {user_settings.get_setting('judge', 'max_iteration')}\n"
            
            "🛠️ *Tools*: " + 
            f"{'✅' if user_settings.get_setting('tools', 'enabled') else '❌'} | " +
            f"Max: {user_settings.get_setting('tools', 'max_iteration')}\n"
            
            "🔍 *Semantic Search*: " + 
            f"{'✅' if user_settings.get_setting('semantic_search', 'enabled') else '❌'} | " +
            f"Max Results: {user_settings.get_setting('semantic_search', 'max_results')}\n"
            
            "🤔 *Thinking Status*: " + 
            f"{'✅' if user_settings.get_setting('thinking', 'enabled') else '❌'}\n"
            
            "🧩 *System Prompt*: " + 
            f"{user_settings.get_setting('system_prompt', 'type')}\n\n"
            
            "Select a setting to configure:"
        )

        keyboard = [
            [
                InlineKeyboardButton("📚 Summarization", callback_data="settings_summarization"),
                InlineKeyboardButton("💬 Dialog", callback_data="settings_dialog")
            ],
            [
                InlineKeyboardButton("🧠 Reasoning", callback_data="settings_reasoning"),
                InlineKeyboardButton("💭 Memory", callback_data="settings_memory")
            ],
            [
                InlineKeyboardButton("🎯 Critique", callback_data="settings_critique"),
                InlineKeyboardButton("⚖️ Judge", callback_data="settings_judge")
            ],
            [
                InlineKeyboardButton("🛠️ Tools", callback_data="settings_tools"),
                InlineKeyboardButton("🔍 Semantic", callback_data="settings_semantic")
            ],
            [
                InlineKeyboardButton("🤔 Thinking", callback_data="settings_thinking"),
                InlineKeyboardButton("🧩 System Prompt", callback_data="settings_system_prompt")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        try:
            await query.edit_message_text(
                status_text,
                reply_markup=reply_markup,
                parse_mode="markdown"
            )
        except Exception as e:
            # Fallback without markdown if formatting fails
            await query.edit_message_text(
                status_text.replace('*', '').replace('✅', '+').replace('❌', '-'),
                reply_markup=reply_markup
            )
    elif category == "semantic":
        if action == "toggle":
            current = user_settings.get_setting("semantic_search", "enabled")
            user_settings.set_setting("semantic_search", not current, "enabled")
            await show_semantic_menu(query, user_settings)
        elif action == "max":
            if value:
                max_results = user_settings.validate_semantic_max_results(int(value))
                user_settings.set_setting("semantic_search", max_results, "max_results")
                await show_semantic_menu(query, user_settings)
            else:
                await show_semantic_max_results_menu(query)
        else:
            await show_semantic_menu(query, user_settings)
    elif category == "thinking":
        if action == "toggle":
            current = user_settings.get_setting("thinking", "enabled")
            user_settings.set_setting("thinking", not current, "enabled")
            await show_thinking_menu(query, user_settings)
        else:
            await show_thinking_menu(query, user_settings)
    elif category == "system_prompt":
        if action == "set":
            if value:
                user_settings.set_setting("system_prompt", value, "type")
                await show_system_prompt_menu(query, user_settings)
            else:
                await show_system_prompt_menu(query, user_settings)
        else:
            await show_system_prompt_menu(query, user_settings)

async def show_summarization_menu(query: CallbackQuery, user_settings: UserSettings):
    """Show summarization history settings menu"""
    enabled = user_settings.get_setting("summarization_history", "enabled")
    size = user_settings.get_setting("summarization_history", "size")
    
    keyboard = [
        [InlineKeyboardButton(
            f"{'✅' if enabled else '❌'} Enabled",
            callback_data="settings_summarization_toggle"
        )],
        [InlineKeyboardButton(
            f"📊 History Size: {size}",
            callback_data="settings_summarization_size"
        )],
        [InlineKeyboardButton("⬅️ Back", callback_data="settings_main")]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(
        "*Summarization History Settings*\n"
        f"Status: {'Enabled' if enabled else 'Disabled'}\n"
        f"History Size: {size} entries",
        reply_markup=reply_markup,
        parse_mode="markdown"
    )

async def show_dialog_menu(query: CallbackQuery, user_settings: UserSettings):
    """Show dialog history settings menu"""
    enabled = user_settings.get_setting("dialog_history", "enabled")
    size = user_settings.get_setting("dialog_history", "size")
    
    keyboard = [
        [InlineKeyboardButton(
            f"{'✅' if enabled else '❌'} Enabled",
            callback_data="settings_dialog_toggle"
        )],
        [InlineKeyboardButton(
            f"📊 History Size: {size}",
            callback_data="settings_dialog_size"
        )],
        [InlineKeyboardButton("⬅️ Back", callback_data="settings_main")]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(
        "*Dialog History Settings*\n"
        f"Status: {'Enabled' if enabled else 'Disabled'}\n"
        f"History Size: {size} entries",
        reply_markup=reply_markup,
        parse_mode="markdown"
    )

async def show_reasoning_menu(query: CallbackQuery, user_settings: UserSettings):
    """Show reasoning context settings menu"""
    enabled = user_settings.get_setting("reasoning_context", "enabled")
    
    keyboard = [
        [InlineKeyboardButton(
            f"{'✅' if enabled else '❌'} Enabled",
            callback_data="settings_reasoning_toggle"
        )],
        [InlineKeyboardButton("⬅️ Back", callback_data="settings_main")]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(
        "*Reasoning Context Settings*\n"
        f"Status: {'Enabled' if enabled else 'Disabled'}",
        reply_markup=reply_markup,
        parse_mode="markdown"
    )

async def show_memory_menu(query: CallbackQuery, user_settings: UserSettings):
    """Show short term memory settings menu"""
    enabled = user_settings.get_setting("short_term_memory", "enabled")
    
    keyboard = [
        [InlineKeyboardButton(
            f"{'✅' if enabled else '❌'} Enabled",
            callback_data="settings_memory_toggle"
        )],
        [InlineKeyboardButton("⬅️ Back", callback_data="settings_main")]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(
        "*Short Term Memory Settings*\n"
        f"Status: {'Enabled' if enabled else 'Disabled'}",
        reply_markup=reply_markup,
        parse_mode="markdown"
    )

async def show_size_input_menu(query: CallbackQuery, category: str):
    """Show size input menu"""
    sizes = [1, 5, 10, 20, 30, 50]
    keyboard = []
    row = []
    
    for size in sizes:
        row.append(InlineKeyboardButton(
            str(size),
            callback_data=f"settings_{category}_size_{size}"
        ))
        if len(row) == 3:
            keyboard.append(row)
            row = []
    
    if row:  # Add any remaining buttons
        keyboard.append(row)
    
    keyboard.append([InlineKeyboardButton("⬅️ Back", callback_data=f"settings_{category}")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(
        f"*Select {category.title()} History Size*\n"
        "Choose the number of entries to keep:",
        reply_markup=reply_markup,
        parse_mode="markdown"
    )

async def show_critique_menu(query: CallbackQuery, user_settings: UserSettings):
    """Show critique settings menu"""
    enabled = user_settings.get_setting("critique", "enabled")
    max_iteration = user_settings.get_setting("critique", "max_iteration")
    
    keyboard = [
        [InlineKeyboardButton(
            f"{'✅' if enabled else '❌'} Enabled",
            callback_data="settings_critique_toggle"
        )],
        [InlineKeyboardButton(
            f"🔄 Max Iterations: {max_iteration}",
            callback_data="settings_critique_iteration"
        )],
        [InlineKeyboardButton("⬅️ Back", callback_data="settings_main")]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(
        "*Critique Settings*\n"
        f"Status: {'Enabled' if enabled else 'Disabled'}\n"
        f"Max Iterations: {max_iteration}",
        reply_markup=reply_markup,
        parse_mode="markdown"
    )

async def show_judge_menu(query: CallbackQuery, user_settings: UserSettings):
    """Show judge settings menu"""
    enabled = user_settings.get_setting("judge", "enabled")
    max_iteration = user_settings.get_setting("judge", "max_iteration")
    
    keyboard = [
        [InlineKeyboardButton(
            f"{'✅' if enabled else '❌'} Enabled",
            callback_data="settings_judge_toggle"
        )],
        [InlineKeyboardButton(
            f"🔄 Max Iterations: {max_iteration}",
            callback_data="settings_judge_iteration"
        )],
        [InlineKeyboardButton("⬅️ Back", callback_data="settings_main")]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(
        "*Judge Settings*\n"
        f"Status: {'Enabled' if enabled else 'Disabled'}\n"
        f"Max Iterations: {max_iteration}",
        reply_markup=reply_markup,
        parse_mode="markdown"
    )

async def show_tools_menu(query: CallbackQuery, user_settings: UserSettings):
    """Show tools settings menu"""
    enabled = user_settings.get_setting("tools", "enabled")
    max_iteration = user_settings.get_setting("tools", "max_iteration")
    
    keyboard = [
        [InlineKeyboardButton(
            f"{'✅' if enabled else '❌'} Enabled",
            callback_data="settings_tools_toggle"
        )],
        [InlineKeyboardButton(
            f"🔄 Max Iterations: {max_iteration}",
            callback_data="settings_tools_iteration"
        )],
        [InlineKeyboardButton("⬅️ Back", callback_data="settings_main")]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(
        "*Tools Settings*\n"
        f"Status: {'Enabled' if enabled else 'Disabled'}\n"
        f"Max Iterations: {max_iteration}",
        reply_markup=reply_markup,
        parse_mode="markdown"
    )

async def show_iteration_input_menu(query: CallbackQuery, category: str):
    """Show iteration input menu"""
    iterations = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 80, 100, 150, 200, 300]
    keyboard = []
    row = []
    
    for iteration in iterations:
        row.append(InlineKeyboardButton(
            str(iteration),
            callback_data=f"settings_{category}_iteration_{iteration}"
        ))
        if len(row) == 3:
            keyboard.append(row)
            row = []
    
    if row:  # Add any remaining buttons
        keyboard.append(row)
    
    keyboard.append([InlineKeyboardButton("⬅️ Back", callback_data=f"settings_{category}")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(
        f"*Select {category.title()} Max Iterations*\n"
        "Choose the maximum number of iterations:",
        reply_markup=reply_markup,
        parse_mode="markdown"
    )

async def show_semantic_menu(query: CallbackQuery, user_settings: UserSettings):
    """Show semantic search settings menu"""
    semantic_enabled = user_settings.get_setting("semantic_search", "enabled")
    max_results = user_settings.get_setting("semantic_search", "max_results")
    
    keyboard = [
        [InlineKeyboardButton(
            f"{'✅' if semantic_enabled else '❌'} Enabled",
            callback_data="settings_semantic_toggle"
        )],
        [InlineKeyboardButton(
            f"🔍 Max Results: {max_results}",
            callback_data="settings_semantic_max"
        )],
        [InlineKeyboardButton("⬅️ Back", callback_data="settings_main")]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(
        "*Semantic Search Settings*\n\n"
        f"Status: {'Enabled' if semantic_enabled else 'Disabled'}\n"
        f"Max Results: {max_results}\n\n"
        "This feature enables semantic search over your conversation history\n"
        "to find relevant past interactions.",
        reply_markup=reply_markup,
        parse_mode="markdown"
    )

async def show_semantic_max_results_menu(query: CallbackQuery):
    """Show menu for setting max results in semantic search"""
    max_results = [1, 3, 5, 7, 10, 15, 20]
    keyboard = []
    row = []
    
    for size in max_results:
        row.append(InlineKeyboardButton(
            str(size),
            callback_data=f"settings_semantic_max_{size}"
        ))
        if len(row) == 3:
            keyboard.append(row)
            row = []
    
    if row:  # Add any remaining buttons
        keyboard.append(row)
    
    keyboard.append([InlineKeyboardButton("⬅️ Back", callback_data="settings_semantic")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(
        "*Select Maximum Results*\n\n"
        "Choose the maximum number of past conversations\n"
        "to include in semantic search:",
        reply_markup=reply_markup,
        parse_mode="markdown"
    )

async def show_thinking_menu(query: CallbackQuery, user_settings: UserSettings):
    """Show thinking status settings menu"""
    enabled = user_settings.get_setting("thinking", "enabled")
    
    keyboard = [
        [InlineKeyboardButton(
            f"{'✅' if enabled else '❌'} Enabled",
            callback_data="settings_thinking_toggle"
        )],
        [InlineKeyboardButton("⬅️ Back", callback_data="settings_main")]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text(
        "*Thinking Status Settings*\n\n"
        f"Status: {'Enabled' if enabled else 'Disabled'}\n\n"
        "This setting controls whether the bot shows detailed\n"
        "thinking steps during processing.",
        reply_markup=reply_markup,
        parse_mode="markdown"
    )

async def show_system_prompt_menu(query: CallbackQuery, user_settings: UserSettings):
    """Show system prompt settings menu"""
    print("Showing system prompt menu")  # Debug log
    current_type = user_settings.get_setting("system_prompt", "type")
    print(f"Current system prompt type: {current_type}")  # Debug log
    
    keyboard = [
        [InlineKeyboardButton(
            f"{'✅' if current_type == 'generall-ai-v2' else '○'} generall-ai-v2",
            callback_data="settings_system_prompt_set_generall-ai-v2"
        )],
        [InlineKeyboardButton(
            f"{'✅' if current_type == 'generall-ai-v1' else '○'} generall-ai-v1",
            callback_data="settings_system_prompt_set_generall-ai-v1"
        )],
        [InlineKeyboardButton(
            f"{'✅' if current_type == 'perplexity-deep-research' else '○'} perplexity-deep-research",
            callback_data="settings_system_prompt_set_perplexity-deep-research"
        )],
        [InlineKeyboardButton(
            f"{'✅' if current_type == 'perplexity-r1' else '○'} perplexity-r1",
            callback_data="settings_system_prompt_set_perplexity-r1"
        )],
        [InlineKeyboardButton("⬅️ Back", callback_data="settings_main")]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    descriptions = {
        "generall-ai-v2": "Generall.AI v2 system prompt.",
        "generall-ai-v1": "Generall.AI v1 system prompt.",
        "perplexity-deep-research": "Perplexity Deep Research system prompt.",
        "perplexity-r1": "Perplexity R1 system prompt."
    }
    
    try:
        print("Attempting to edit message text")  # Debug log
        await query.edit_message_text(
            "*System Prompt Settings*\n\n"
            f"Current Type: *{current_type}*\n\n"
            f"Description: _{descriptions.get(current_type, 'No description available')}_\n\n"
            "Choose a system prompt type:",
            reply_markup=reply_markup,
            parse_mode="markdown"
        )
        print("Successfully edited message text")  # Debug log
    except Exception as e:
        print(f"Error editing message: {str(e)}")  # Debug log
        # Try without markdown if markdown fails
        try:
            await query.edit_message_text(
                "System Prompt Settings\n\n"
                f"Current Type: {current_type}\n\n"
                f"Description: {descriptions.get(current_type, 'No description available')}\n\n"
                "Choose a system prompt type:",
                reply_markup=reply_markup
            )
            print("Successfully edited message text without markdown")  # Debug log
        except Exception as e:
            print(f"Error editing message without markdown: {str(e)}")  # Debug log

def calculate_next_trigger(reminder_time: str, period_type: str, period_interval: int) -> datetime:
    """Calculate the next trigger time for a periodic reminder"""
    current_time = datetime.now(timezone.utc)
    last_trigger = datetime.fromisoformat(reminder_time)
    
    if period_type == "hourly":
        next_trigger = last_trigger + timedelta(hours=period_interval)
    elif period_type == "daily":
        next_trigger = last_trigger + timedelta(days=period_interval)
    elif period_type == "weekly":
        next_trigger = last_trigger + timedelta(weeks=period_interval)
    elif period_type == "monthly":
        # Add months by calculating days (approximate)
        next_trigger = last_trigger + timedelta(days=30 * period_interval)
    
    # If next trigger is in the past, keep adding intervals until it's in the future
    while next_trigger <= current_time:
        if period_type == "hourly":
            next_trigger += timedelta(hours=period_interval)
        elif period_type == "daily":
            next_trigger += timedelta(days=period_interval)
        elif period_type == "weekly":
            next_trigger += timedelta(weeks=period_interval)
        elif period_type == "monthly":
            next_trigger += timedelta(days=30 * period_interval)
    
    return next_trigger

async def check_and_send_reminders(context: ContextTypes.DEFAULT_TYPE):
    """Check for due reminders and send them to users"""
    try:
        # Get all user directories in data folder
        data_path = Path("./data")
        if not data_path.exists():
            return

        current_time = datetime.now(timezone.utc)
        logging.info(f"Checking reminders at {current_time.isoformat()}")
        
        for user_dir in data_path.iterdir():
            if not user_dir.is_dir():
                continue
                
            user_id = user_dir.name
            reminders_path = user_dir / "reminders" / "reminders.json"
            
            if not reminders_path.exists():
                continue
                
            try:
                with open(reminders_path, "r", encoding="utf-8") as f:
                    reminders = json.load(f)
            except Exception as e:
                logging.error(f"Error reading reminders for user {user_id}: {str(e)}")
                continue
            
            updated_reminders = []
            reminders_changed = False
            
            for reminder in reminders:
                if (reminder["status"] != "pending" or 
                    reminder.get("type") != "user" or 
                    not reminder.get("enabled", True)):  # Skip disabled reminders
                    updated_reminders.append(reminder)
                    continue
                    
                try:
                    # Parse the ISO format string with timezone info
                    reminder_time = datetime.fromisoformat(reminder["time"])
                    if reminder_time.tzinfo is None:
                        reminder_time = reminder_time.replace(tzinfo=timezone.utc)
                    
                    logging.debug(f"Checking reminder: {reminder['text']} scheduled for {reminder_time.isoformat()}")
                    
                    # Check if reminder is due
                    if reminder_time <= current_time:
                        # Handle reminder based on type
                        try:
                            if reminder.get("type", "user") == "user":
                                try:
                                    # Send user reminder
                                    await context.bot.send_message(
                                        chat_id=user_id,
                                        text=f"🔔 *Reminder*\n\n{reminder['text']}",
                                        parse_mode='Markdown'
                                    )
                                except Exception as e:
                                    logging.error(f"Error sending Markdown reminder to user {user_id}: {str(e)}")
                                    await context.bot.send_message(
                                        chat_id=user_id,
                                        text=f"🔔 *Reminder*\n\n{reminder['text']}"
                                    )

                            # Handle periodic reminders
                            if reminder.get("is_periodic", False):
                                # Calculate next trigger time
                                next_trigger = calculate_next_trigger(
                                    reminder_time.isoformat(),
                                    reminder["period_type"],
                                    reminder["period_interval"]
                                )
                                
                                # Update reminder for next trigger
                                reminder["last_triggered"] = current_time.isoformat()
                                reminder["time"] = next_trigger.isoformat()
                                reminder["next_trigger"] = next_trigger.isoformat()
                                reminders_changed = True
                            else:
                                # Mark one-time reminder as completed
                                reminder["status"] = "completed"
                                reminder["completed_at"] = current_time.isoformat()
                                reminders_changed = True
                                
                            logging.info(f"Processed {reminder.get('type', 'user')} reminder for user {user_id}: {reminder['text']}")
                        except Exception as e:
                            logging.error(f"Error processing reminder for user {user_id}: {str(e)}")
                            
                    updated_reminders.append(reminder)
                    
                except Exception as e:
                    logging.error(f"Error processing reminder: {str(e)}")
                    updated_reminders.append(reminder)
            
            # Save updated reminders if any were completed or updated
            if reminders_changed:
                try:
                    with open(reminders_path, "w", encoding="utf-8") as f:
                        json.dump(updated_reminders, f, indent=2, ensure_ascii=False)
                    logging.info(f"Updated reminders saved for user {user_id}")
                except Exception as e:
                    logging.error(f"Error saving updated reminders for user {user_id}: {str(e)}")
                    
    except Exception as e:
        logging.error(f"Error in reminder checker: {str(e)}")

async def check_and_process_agent_reminders(context: ContextTypes.DEFAULT_TYPE):
    """Check for due agent reminders and process them"""
    try:
        # Get all user directories in data folder
        data_path = Path("./data")
        if not data_path.exists():
            return

        current_time = datetime.now(timezone.utc)
        logging.info(f"Checking agent reminders at {current_time.isoformat()}")
        
        for user_dir in data_path.iterdir():
            if not user_dir.is_dir():
                continue
                
            user_id = user_dir.name
            reminders_path = user_dir / "reminders" / "reminders.json"
            
            if not reminders_path.exists():
                continue
                
            try:
                with open(reminders_path, "r", encoding="utf-8") as f:
                    reminders = json.load(f)
            except Exception as e:
                logging.error(f"Error reading reminders for user {user_id}: {str(e)}")
                continue
            
            updated_reminders = []
            reminders_changed = False
            
            for reminder in reminders:
                if (reminder["status"] != "pending" or 
                    reminder.get("type") != "agent" or 
                    not reminder.get("enabled", True)):  # Skip disabled reminders
                    updated_reminders.append(reminder)
                    continue
                    
                try:
                    # Parse the ISO format string with timezone info
                    reminder_time = datetime.fromisoformat(reminder["time"])
                    if reminder_time.tzinfo is None:
                        reminder_time = reminder_time.replace(tzinfo=timezone.utc)
                    
                    logging.debug(f"Checking agent reminder: {reminder['text']} scheduled for {reminder_time.isoformat()}")
                    
                    # Check if reminder is due
                    if reminder_time <= current_time:
                        try:
                            print(f"Processing agent reminder: {reminder['text']}")

                            print("Sending initial message to user")
                            try:
                                initial_message = await context.bot.send_message(
                                    chat_id=user_id,
                                    text="🤖 *Agent Reminder Task*\n\nProcessing scheduled task:\n" + reminder['text'],
                                    parse_mode="markdown"
                                )
                                print(f"Initial message sent: {initial_message.message_id}")
                            except Exception as e:
                                try:
                                    initial_message = await context.bot.send_message(
                                        chat_id=user_id,
                                        text="🤖 *Agent Reminder Task*\n\nProcessing scheduled task:\n" + reminder['text']
                                    )
                                    print(f"Initial message sent: {initial_message.message_id}")
                                except Exception as e:
                                    logging.error(f"Error sending initial message: {str(e)}")

                            # Send thinking message to user
                            thinking_message = await context.bot.send_message(
                                chat_id=user_id,
                                text="💭 *Processing Agent Task...*",
                                parse_mode="markdown"
                            )
                            print(f"Thinking message sent: {thinking_message.message_id}")

                            # Create a mock Update object for the agent
                            mock_message = Message(
                                message_id=thinking_message.message_id,
                                date=datetime.now(),
                                chat=Chat(id=int(user_id), type="private"),
                                text=reminder['text'],
                                from_user=None
                            )
                            print(f"Mock message created: {mock_message}")
                            mock_update = Update(
                                update_id=thinking_message.message_id,
                                message=mock_message
                            )
                            print(f"Mock update created: {mock_update}")
                            
                            print("Creating update_thinking_message function")
                            async def update_thinking_message(step: str, details: str, iteration: int, critique: int):
                                if step == "saving":
                                    iteration = "final"
                                    critique = "end"
                                await thinking_message.edit_text(
                                    f"💭 *Processing Agent Task...*\n"
                                    f"- - - - \n"
                                    f"📝 *Step:* _{step.replace('_', '-')}_\n"
                                    f"📋 *Details:* _{details.replace('_', '-')}_\n"
                                    f"🔄 *Iterations:* _{iteration}_\n"
                                    f"🎯 *Critiques:* _{critique}_",
                                    parse_mode="markdown"
                                )

                            print("Calling get_answer with mock update")
                            # Process the reminder using get_answer with mock Update
                            response, messages = await get_answer(
                                reminder['text'], 
                                user_id, 
                                update_thinking_message,
                                mock_update,  # Pass the mock Update object
                                context
                            )

                            print(f"get_answer returned response: {response}")
                            # Send the response
                            if len(response) > 4000:
                                await thinking_message.edit_text("Processing complete. Response is long, sending in parts...")
                                chunks = split_text_intelligently(response)
                                for chunk in chunks:
                                    try:
                                        await context.bot.send_message(
                                            chat_id=user_id,
                                            text=chunk,
                                            parse_mode="markdown"
                                        )
                                    except Exception as e:
                                        logging.error(f"Error sending Markdown response part to user {user_id}: {str(e)}")
                                        await context.bot.send_message(
                                            chat_id=user_id,
                                            text=chunk
                                        )
                            else:
                                try:
                                    await thinking_message.edit_text(text=response, parse_mode="markdown")
                                except Exception as e:
                                    logging.error(f"Error sending response to user {user_id}: {str(e)}")
                                    await thinking_message.edit_text(text=response)

                            print("Sending reasoning file if available")
                            # Send reasoning file if available
                            if messages:
                                uuid_reasoning = str(uuid.uuid4())
                                file_path = f"reasoning_{user_id}_{uuid_reasoning}.txt"
                                try:
                                    with open(file_path, "w", encoding="utf-8") as f:
                                        for msg in messages:
                                            f.write(msg["content"][0]["text"] + "\n\n========\n\n")
                                    
                                    await context.bot.send_document(
                                        chat_id=user_id,
                                        document=open(file_path, "rb"),
                                        caption="Agent task reasoning history."
                                    )
                                except Exception as e:
                                    logging.error(f"Error sending reasoning file: {str(e)}")
                                finally:
                                    try:
                                        if os.path.exists(file_path):
                                            os.remove(file_path)
                                    except Exception as e:
                                        logging.error(f"Error removing reasoning file: {str(e)}")

                            # Handle periodic reminders
                            if reminder.get("is_periodic", False):
                                # Calculate next trigger time
                                next_trigger = calculate_next_trigger(
                                    reminder_time.isoformat(),
                                    reminder["period_type"],
                                    reminder["period_interval"]
                                )
                                
                                # Update reminder for next trigger
                                reminder["last_triggered"] = current_time.isoformat()
                                reminder["time"] = next_trigger.isoformat()
                                reminder["next_trigger"] = next_trigger.isoformat()
                                reminder["agent_response"] = response
                                reminders_changed = True
                            else:
                                # Mark one-time reminder as completed
                                reminder["status"] = "completed"
                                reminder["completed_at"] = current_time.isoformat()
                                reminder["agent_response"] = response
                                reminders_changed = True
                                
                            logging.info(f"Processed agent reminder for user {user_id}")
                            
                        except Exception as e:
                            error_trace = str(uuid.uuid4())
                            logging.error(f"Error processing agent reminder for user {user_id} (Trace: {error_trace}): {str(e)}")
                            await context.bot.send_message(
                                chat_id=user_id,
                                text=f"❌ Error processing agent task. Trace ID: {error_trace}",
                                parse_mode="markdown"
                            )
                            
                    updated_reminders.append(reminder)
                    
                except Exception as e:
                    logging.error(f"Error processing reminder: {str(e)}")
                    updated_reminders.append(reminder)
            
            # Save updated reminders if any were completed or updated
            if reminders_changed:
                try:
                    with open(reminders_path, "w", encoding="utf-8") as f:
                        json.dump(updated_reminders, f, indent=2, ensure_ascii=False)
                    logging.info(f"Updated reminders saved for user {user_id}")
                except Exception as e:
                    logging.error(f"Error saving updated reminders for user {user_id}: {str(e)}")
                    
    except Exception as e:
        logging.error(f"Error in agent reminder checker: {str(e)}")

async def reminders_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /reminders command"""
    user_id = str(update.message.chat_id)
    
    # Check if the user is authorized
    if not is_user_authorized(user_id):
        await update.message.reply_text("Unauthorized. You need an invite to use this bot.")
        return
    
    await show_reminders_menu(update.message, user_id)

async def show_reminders_menu(message: Message, user_id: str, page: int = 1):
    """Show main reminders menu with list of active reminders"""
    reminders_path = Path("./data") / user_id / "reminders" / "reminders.json"
    if not reminders_path.exists():
        await message.reply_text("No reminders found.")
        return
    
    try:
        with open(reminders_path, "r", encoding="utf-8") as f:
            all_reminders = json.load(f)
    except Exception as e:
        await message.reply_text("Error loading reminders.")
        return
    
    # Filter active reminders (status is pending)
    active_reminders = [r for r in all_reminders if r["status"] == "pending"]
    completed_reminders = [r for r in all_reminders if r["status"] == "completed"]
    
    # Pagination
    items_per_page = 5
    start_idx = (page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    
    current_reminders = active_reminders[start_idx:end_idx]
    total_pages = (len(active_reminders) + items_per_page - 1) // items_per_page
    
    if not current_reminders and page > 1:
        # If current page is empty but there are previous pages, show last page
        return await show_reminders_menu(message, user_id, total_pages)
    
    # Create keyboard with reminder controls
    keyboard = []
    for reminder in current_reminders:
        # Format time
        reminder_time = datetime.fromisoformat(reminder["time"])
        formatted_time = reminder_time.strftime("%H:%M %d/%m")  # More compact time format
        
        # Add periodic info if needed
        if reminder.get("is_periodic"):
            formatted_time += f" ↻{reminder['period_interval']}{reminder['period_type'][0]}"
        
        # Get preview of reminder text (first 35 chars)
        text_preview = reminder['text'][:35] + ('...' if len(reminder['text']) > 35 else '')
        
        # Create main info button (larger)
        keyboard.append([
            InlineKeyboardButton(
                f"📅 {formatted_time} : {text_preview}",
                callback_data=f"reminder_info_{reminder['id']}"
            )
        ])

        # Add delete and toggle buttons
        keyboard.append([
            InlineKeyboardButton(" ", callback_data="noop"),
            InlineKeyboardButton("❌ Delete", callback_data=f"reminder_delete_{reminder['id']}"),
            InlineKeyboardButton("⏸️ Disable" if reminder.get("enabled", True) else "▶️ Enable", callback_data=f"reminder_toggle_{reminder['id']}"),
            InlineKeyboardButton(" ", callback_data="noop"),
        ])
    
    # Add navigation buttons if needed
    nav_buttons = []
    if page > 1:
        nav_buttons.append(InlineKeyboardButton("⬅️", callback_data=f"reminders_page_{page-1}"))
    if total_pages > 1:
        nav_buttons.append(InlineKeyboardButton(f"{page}/{total_pages}", callback_data="noop"))
    if page < total_pages:
        nav_buttons.append(InlineKeyboardButton("➡️", callback_data=f"reminders_page_{page+1}"))
    if nav_buttons:
        keyboard.append(nav_buttons)
    
    # Add summary button
    keyboard.append([
        InlineKeyboardButton("📊 Summary", callback_data="reminders_summary")
    ])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Create status message
    status_text = (
        "*Your Reminders*\n\n"
        f"Active: {len(active_reminders)} | "
        f"Completed: {len(completed_reminders)}\n\n"
        "Select a reminder to manage:"
    )
    
    try:
        await message.reply_text(status_text, reply_markup=reply_markup, parse_mode="markdown")
    except Exception as e:
        # Try without markdown
        await message.reply_text(
            status_text.replace('*', ''),
            reply_markup=reply_markup
        )

async def reminder_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle reminder menu button presses"""
    query = update.callback_query
    user_id = str(query.message.chat_id)
    
    # Check if authorized
    if user_id not in telegram_chat_id:
        await query.answer("Unauthorized chat")
        return
    
    # Don't show alert for noop actions
    if query.data == "noop":
        await query.answer()
        return
    
    await query.answer()
    
    data = query.data.split("_")
    action = data[1] if len(data) > 1 else None
    
    if action == "page":
        page = int(data[2])
        await show_reminders_menu(query.message, user_id, page)
        await query.message.delete()
    
    elif action == "info":
        reminder_id = data[2]
        await show_reminder_info(query, user_id, reminder_id)
    
    elif action == "delete":
        reminder_id = data[2]
        await delete_reminder(query, user_id, reminder_id)
    
    elif action == "toggle":
        reminder_id = data[2]
        await toggle_reminder(query, user_id, reminder_id)
    
    elif action == "summary":
        await show_reminders_summary(query, user_id)

async def show_reminder_info(query: CallbackQuery, user_id: str, reminder_id: str):
    """Show detailed information about a specific reminder"""
    reminders_path = Path("./data") / user_id / "reminders" / "reminders.json"
    
    try:
        with open(reminders_path, "r", encoding="utf-8") as f:
            reminders = json.load(f)
            
        reminder = next((r for r in reminders if r["id"] == reminder_id), None)
        if not reminder:
            await query.edit_message_text("Reminder not found.")
            return
        
        reminder_time = datetime.fromisoformat(reminder["time"])
        formatted_time = reminder_time.strftime("%Y-%m-%d %H:%M:%S UTC")
        
        next_trigger = ""
        if reminder.get("is_periodic") and reminder.get("next_trigger"):
            next_time = datetime.fromisoformat(reminder["next_trigger"])
            next_trigger = f"\nNext Trigger: {next_time.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        
        last_triggered = ""
        if reminder.get("last_triggered"):
            last_time = datetime.fromisoformat(reminder["last_triggered"])
            last_triggered = f"\nLast Triggered: {last_time.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        
        periodic_info = ""
        if reminder.get("is_periodic"):
            periodic_info = f"\nRepeats every: {reminder['period_interval']} {reminder['period_type']}"
        
        info_text = (
            f"*Reminder Details*\n\n"
            f"Text: {reminder['text']}\n"
            f"Type: {reminder['type']}\n"
            f"Status: {reminder['status']}\n"
            f"Created: {datetime.fromisoformat(reminder['created_at']).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"Scheduled: {formatted_time}"
            f"{periodic_info}"
            f"{next_trigger}"
            f"{last_triggered}"
        )
        
        keyboard = [[InlineKeyboardButton("⬅️ Back", callback_data="reminders_page_1")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        try:
            await query.edit_message_text(
                info_text,
                reply_markup=reply_markup,
                parse_mode="markdown"
            )
        except Exception as e:
            logging.error(f"Error showing reminder info as markdown: {str(e)}")
            await query.edit_message_text(
                info_text,
                reply_markup=reply_markup
            )
    except Exception as e:
        logging.error(f"Error showing reminder info: {str(e)}")
        await query.edit_message_text(
            "Error showing reminder information. Please try again.",
            parse_mode="markdown"
        )

async def delete_reminder(query: CallbackQuery, user_id: str, reminder_id: str):
    """Delete a specific reminder"""
    reminders_path = Path("./data") / user_id / "reminders" / "reminders.json"
    
    try:
        with open(reminders_path, "r", encoding="utf-8") as f:
            reminders = json.load(f)
        
        # Find and remove the reminder
        reminders = [r for r in reminders if r["id"] != reminder_id]
        
        with open(reminders_path, "w", encoding="utf-8") as f:
            json.dump(reminders, f, indent=2, ensure_ascii=False)
        
        # Show updated menu
        await show_reminders_menu(query.message, user_id)
        await query.message.delete()
        
    except Exception as e:
        logging.error(f"Error deleting reminder: {str(e)}")
        await query.edit_message_text(
            "Error deleting reminder. Please try again.",
            parse_mode="markdown"
        )

async def toggle_reminder(query: CallbackQuery, user_id: str, reminder_id: str):
    """Toggle a reminder's enabled status"""
    reminders_path = Path("./data") / user_id / "reminders" / "reminders.json"
    
    try:
        with open(reminders_path, "r", encoding="utf-8") as f:
            reminders = json.load(f)
        
        # Find and toggle the reminder
        for reminder in reminders:
            if reminder["id"] == reminder_id:
                reminder["enabled"] = not reminder.get("enabled", True)
                break
        
        with open(reminders_path, "w", encoding="utf-8") as f:
            json.dump(reminders, f, indent=2, ensure_ascii=False)
        
        # Show updated menu
        await show_reminders_menu(query.message, user_id)
        await query.message.delete()
        
    except Exception as e:
        logging.error(f"Error toggling reminder: {str(e)}")
        await query.edit_message_text(
            "Error toggling reminder. Please try again.",
            parse_mode="markdown"
        )

async def show_reminders_summary(query: CallbackQuery, user_id: str):
    """Show a summary of all reminders"""
    reminders_path = Path("./data") / user_id / "reminders" / "reminders.json"
    
    try:
        with open(reminders_path, "r", encoding="utf-8") as f:
            reminders = json.load(f)
        
        # Count different types of reminders
        active_user = len([r for r in reminders if r["status"] == "pending" and r["type"] == "user" and r.get("enabled", True)])
        active_agent = len([r for r in reminders if r["status"] == "pending" and r["type"] == "agent" and r.get("enabled", True)])
        disabled = len([r for r in reminders if r["status"] == "pending" and not r.get("enabled", True)])
        completed = len([r for r in reminders if r["status"] == "completed"])
        periodic = len([r for r in reminders if r.get("is_periodic", False)])
        
        summary_text = (
            "*Reminders Summary*\n\n"
            f"Active User Reminders: {active_user}\n"
            f"Active Agent Tasks: {active_agent}\n"
            f"Disabled Reminders: {disabled}\n"
            f"Completed Reminders: {completed}\n"
            f"Periodic Reminders: {periodic}\n"
        )
        
        keyboard = [[InlineKeyboardButton("⬅️ Back", callback_data="reminders_page_1")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            summary_text,
            reply_markup=reply_markup,
            parse_mode="markdown"
        )
    except Exception as e:
        logging.error(f"Error showing reminders summary: {str(e)}")
        await query.edit_message_text(
            "Error showing reminders summary. Please try again.",
            parse_mode="markdown"
        )

def ensure_data_directory():
    """Ensure the data directory exists"""
    os.makedirs("data", exist_ok=True)

def load_authorized_users():
    """Load authorized users from the userlist.json file"""
    global authorized_users
    try:
        if os.path.exists("data/userlist.json"):
            with open("data/userlist.json", "r") as f:
                user_data = json.load(f)
                if "users" in user_data:
                    # Combine environment chat IDs with saved users
                    authorized_users = set(telegram_chat_id) | set(user_data["users"])
                if "invites" in user_data:
                    global user_invites
                    user_invites = user_data["invites"]
            print(f"Loaded {len(authorized_users)} authorized users from userlist.json")
        else:
            # Create the file if it doesn't exist
            ensure_data_directory()
            save_authorized_users()
            print("Created new userlist.json file")
    except Exception as e:
        logging.error(f"Error loading authorized users: {str(e)}")

def save_authorized_users():
    """Save authorized users to the userlist.json file"""
    ensure_data_directory()
    try:
        with open("data/userlist.json", "w") as f:
            json.dump({
                "users": list(authorized_users),
                "invites": user_invites
            }, f, indent=2)
    except Exception as e:
        logging.error(f"Error saving authorized users: {str(e)}")

def is_user_authorized(user_id):
    """Check if a user is authorized to use the bot"""
    if allow_all_users:
        return True
    return user_id in authorized_users

def generate_invite_code(user_id):
    """Generate a unique invite code for a user"""
    code = str(uuid.uuid4())[:8]  # Use first 8 chars of a UUID
    
    # Initialize user's invites if not exists
    if user_id not in user_invites:
        user_invites[user_id] = {}
    
    # Store the invite
    user_invites[user_id][code] = {
        "created_at": datetime.now().isoformat(),
        "used_by": None
    }
    
    # Save updated invites
    save_authorized_users()
    
    return code

def get_user_invite_count(user_id):
    """Get the number of unused invites a user has created"""
    if user_id not in user_invites:
        return 0
    
    # Count invites that haven't been used
    return sum(1 for invite in user_invites[user_id].values() if invite["used_by"] is None)

def is_invite_code_valid(code):
    """Check if an invite code is valid"""
    for user_id, invites in user_invites.items():
        if code in invites and invites[code]["used_by"] is None:
            return user_id
    return None

def use_invite_code(code, used_by):
    """Mark an invite code as used"""
    for user_id, invites in user_invites.items():
        if code in invites and invites[code]["used_by"] is None:
            user_invites[user_id][code]["used_by"] = used_by
            authorized_users.add(used_by)
            save_authorized_users()
            return user_id
    return None

async def invite_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /invite command"""
    user_id = str(update.message.chat_id)
    
    # Check if the user is authorized
    if not is_user_authorized(user_id):
        await update.message.reply_text("Unauthorized. You need an invite to use this bot.")
        return
    
    # Check if args contain a code (someone is trying to use an invite)
    if context.args and len(context.args) > 0:
        invite_code = context.args[0]
        inviter_id = is_invite_code_valid(invite_code)
        
        if not inviter_id:
            await update.message.reply_text("❌ Invalid or already used invite code.")
            return
            
        # Check if user is trying to use their own invite
        if inviter_id == user_id:
            await update.message.reply_text(
                "🔄 This is your own invite code! Share it with others instead.\n\n"
                "Forward this message to friends who want access to the bot:",
                parse_mode="HTML"
            )
            
            # Send a separate message with the invite link for easy forwarding
            invite_link = f"https://t.me/{context.bot.username}?start=invite_{invite_code}"
            keyboard = [
                [InlineKeyboardButton("📩 Share Invite", url=invite_link)]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                f"<b>Invite Link</b>\n\n"
                f"Use this link to join: {invite_link}\n\n"
                f"Or use this command: <code>/invite {invite_code}</code>",
                reply_markup=reply_markup,
                parse_mode="HTML"
            )
            return
            
        # Normal invite usage flow
        use_invite_code(invite_code, user_id)
        await update.message.reply_text("✅ Invite accepted! You now have access to the bot.")
        
        # Notify admin about the new user
        total_users = len(authorized_users)
        if telegram_admin_id:
            try:
                await context.bot.send_message(
                    chat_id=telegram_admin_id,
                    text=f"🔔 New user joined!\n"
                         f"User ID: <code>{user_id}</code>\n"
                         f"Invited by: <code>{inviter_id}</code>\n"
                         f"Total users: {total_users}",
                    parse_mode="HTML"
                )
            except Exception as e:
                logging.error(f"Failed to notify admin: {str(e)}")
        return
    
    # Check if user can create invites
    is_admin = user_id == telegram_admin_id
    invite_count = get_user_invite_count(user_id)
    
    if not is_admin and invite_count >= user_invite_limit:
        await update.message.reply_text(f"❌ You've reached your invite limit ({user_invite_limit}).")
        return
    
    # Generate invite code
    invite_code = generate_invite_code(user_id)
    invite_link = f"https://t.me/{context.bot.username}?start=invite_{invite_code}"
    
    # Create inline button for the invite link
    keyboard = [
        [InlineKeyboardButton("📩 Activate invite!", url=invite_link)]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Use HTML format instead of markdown to avoid parsing issues
    remaining = "unlimited" if is_admin else str(user_invite_limit - invite_count - 1)
    
    await update.message.reply_text(
        f"🎟️ <b>New Invite Created</b>\n\n"
        f"Share this link: {invite_link}\n\n"
        f"Or use this command:\n<code>/invite {invite_code}</code>\n\n"
        f"Invites remaining: {remaining}/{user_invite_limit}",
        reply_markup=reply_markup,
        parse_mode="HTML"
    )

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /start command"""
    user_id = str(update.message.chat_id)
    
    # Check if this is an invite link with a parameter
    if context.args and len(context.args) > 0 and context.args[0].startswith("invite_"):
        invite_code = context.args[0][7:]  # Remove 'invite_' prefix
        inviter_id = is_invite_code_valid(invite_code)
        
        if not inviter_id:
            await update.message.reply_text("❌ Invalid or already used invite code.")
            return
            
        # Check if user is trying to use their own invite
        if inviter_id == user_id:
            await update.message.reply_text(
                "🔄 This is your own invite code! Share it with others instead.\n\n"
                "Forward this message to friends who want access to the bot:",
                parse_mode="HTML"
            )
            
            # Send a separate message with the invite link for easy forwarding
            invite_link = f"https://t.me/{context.bot.username}?start=invite_{invite_code}"
            keyboard = [
                [InlineKeyboardButton("📩 Share Invite", url=invite_link)]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                f"<b>Invite Link</b>\n\n"
                f"Use this link to join: {invite_link}\n\n"
                f"Or use this command: <code>/invite {invite_code}</code>",
                reply_markup=reply_markup,
                parse_mode="HTML"
            )
            return
            
        # Normal invite usage flow
        use_invite_code(invite_code, user_id)
        await update.message.reply_text("✅ Invite accepted! You now have access to the bot.")
        
        # Notify admin about the new user
        total_users = len(authorized_users)
        if telegram_admin_id:
            try:
                await context.bot.send_message(
                    chat_id=telegram_admin_id,
                    text=f"🔔 New user joined!\n"
                         f"User ID: <code>{user_id}</code>\n"
                         f"Invited by: <code>{inviter_id}</code>\n"
                         f"Total users: {total_users}",
                    parse_mode="HTML"
                )
            except Exception as e:
                logging.error(f"Failed to notify admin: {str(e)}")
        return
    
    # Regular start command
    if is_user_authorized(user_id):
        await update.message.reply_text(
            "👋 Welcome to Generall.AI bot! Use me to get AI assistance.\n\n"
            "You can send me messages, voices, images, media groups, pdfs, and more to analyze. I have memory, access to the internet and a wide range of tools to help you."
        )
    else:
        await update.message.reply_text(
            "👋 Welcome! This bot requires an invitation to use.\n\n"
            "If you have an invite code, please use:\n"
            "<code>/invite YOUR_CODE_HERE</code>",
            parse_mode="HTML"
        )

async def list_users_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /listusers command - admin only"""
    user_id = str(update.message.chat_id)
    
    # Only admin can list users
    if user_id != telegram_admin_id:
        await update.message.reply_text("Unauthorized. Only admin can use this command.")
        return
    
    # Get user data
    user_count = len(authorized_users)
    invite_count = sum(1 for user_invites_dict in user_invites.values() 
                     for invite in user_invites_dict.values() 
                     if invite["used_by"] is None)
    
    # Format user list
    user_list = "\n".join([f"- <code>{user}</code>" for user in authorized_users])
    
    await update.message.reply_text(
        f"📊 <b>Bot Users Summary</b>\n\n"
        f"Total users: {user_count}\n"
        f"Active invites: {invite_count}\n\n"
        f"<b>User IDs:</b>\n{user_list}",
        parse_mode="HTML"
    )

def main():
    # Load authorized users from file
    load_authorized_users()
    
    # Validate bot token
    if not telegram_bot_token or telegram_bot_token == "":
        logger.error("TELEGRAM_BOT_TOKEN is not set in environment variables!")
        print("ERROR: TELEGRAM_BOT_TOKEN is not set!")
        exit(1)
    
    logger.info(f"Bot token loaded (length: {len(telegram_bot_token)} chars)")
    
    try:
        # Create application with job queue enabled
        app = Application.builder().token(telegram_bot_token).build()
        logger.info("Telegram application created successfully")
    except Exception as e:
        logger.error(f"Failed to create Telegram application: {str(e)}")
        print(f"ERROR: Failed to create Telegram application: {str(e)}")
        exit(1)
    
    # Add command handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("invite", invite_command))
    app.add_handler(CommandHandler("listusers", list_users_command))  # Admin command to list users
    
    # Add message handler
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Add voice message handler
    app.add_handler(MessageHandler(filters.VOICE, handle_voice_message))
    
    # Add photo message handler
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))

    # Add document message handler
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document_message))
    
    # Add voice command handler
    app.add_handler(CommandHandler("voice", voice_command))
    
    # Add settings command handler
    app.add_handler(CommandHandler("settings", settings_command))
    
    # Add reminders command handler
    app.add_handler(CommandHandler("reminders", reminders_command))
    
    # Add callback query handlers
    app.add_handler(CallbackQueryHandler(voice_button, pattern="^voice_"))
    app.add_handler(CallbackQueryHandler(settings_button, pattern="^settings_"))
    app.add_handler(CallbackQueryHandler(reminder_button, pattern="^reminder_"))
    app.add_handler(CallbackQueryHandler(reminder_button, pattern="^reminders_"))
    
    # Set up job queue for reminders
    if app.job_queue:
        # Schedule reminder checks
        app.job_queue.run_repeating(check_and_send_reminders, interval=10, first=1)
        app.job_queue.run_repeating(check_and_process_agent_reminders, interval=10, first=1)
        print("Reminder schedulers started...")
    else:
        print("Warning: Job queue is not available. Please install 'python-telegram-bot[job-queue]' for reminder functionality.")
    
    # Register a shutdown handler to clean up containers
    if hasattr(app, 'post_shutdown') and callable(app.post_shutdown):
        app.post_shutdown(shutdown_handler)
    else:
        # Alternative: use signal handlers or atexit for cleanup
        import atexit
        atexit.register(shutdown_handler, app)
    
    # Start polling with error handling
    print("Bot is running...")
    logger.info("Starting bot polling...")
    try:
        app.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,  # Drop pending updates to start fresh
            pool_timeout=30,
            connect_timeout=30,
            read_timeout=30
        )
    except Exception as e:
        logger.error(f"Bot polling failed: {str(e)}")
        print(f"ERROR: Bot polling failed: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)

def shutdown_handler(application=None):
    """Clean up containers when the application shuts down"""
    logging.info("Application shutting down, cleaning up containers...")
    cleanup_containers()
    logging.info("Container cleanup complete")

if __name__ == "__main__":
    main()
