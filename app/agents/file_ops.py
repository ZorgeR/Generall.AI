import os
from pathlib import Path
from typing import List, Dict, Any
import requests
from bs4 import BeautifulSoup
import boto3
from botocore.client import Config
from dotenv import load_dotenv
import zipfile
from datetime import datetime
import telegram
import asyncio

load_dotenv()

# Initialize Telegram bot
telegram_bot = telegram.Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))

class FileOperations:
    def __init__(self, user_id: str, telegram_update: telegram.Update):
        self.user_id = user_id
        self.telegram_update = telegram_update
        self.base_path = Path("./data") / str(user_id)
        # Create base directory if it doesn't exist
        self.base_path.mkdir(parents=True, exist_ok=True)
        # Create downloads directory
        self.downloads_path = self.base_path / "downloads"
        self.downloads_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize S3 client
        self.s3_client = self._init_s3_client()

    def _init_s3_client(self):
        """Initialize S3 client with environment variables"""
        try:
            s3_client = boto3.client(
                's3',
                endpoint_url=os.getenv('S3_HOST'),
                aws_access_key_id=os.getenv('S3_ACCESS_KEY'),
                aws_secret_access_key=os.getenv('S3_SECRET_KEY')
            )
            print(f"S3 client initialized with endpoint: {os.getenv('S3_HOST')}")
            return s3_client
        except Exception as e:
            print(f"Warning: Failed to initialize S3 client: {str(e)}")
            return None

    @property
    def tools_schema(self) -> List[Dict[str, Any]]:
        """Return the schema for all file operation tools"""
        return [
            {
                "name": "list_files",
                "description": "List all files in the user's data directory",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "create_file",
                "description": "Create a new text file with given content. Use \\n for line breaks in content.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Name of the file to create",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write to the file. Use \\n for line breaks.",
                        },
                    },
                    "required": ["filename", "content"],
                },
            },
            {
                "name": "read_file",
                "description": "Read content from a text file",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Name of the file to read",
                        },
                    },
                    "required": ["filename"],
                },
            },
            {
                "name": "create_directory",
                "description": "Create a new directory in the user's data space",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "dirname": {
                            "type": "string",
                            "description": "Name of the directory to create",
                        },
                    },
                    "required": ["dirname"],
                },
            },
            {
                "name": "delete_file",
                "description": "Delete a file from the user's data space",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Name of the file to delete",
                        },
                    },
                    "required": ["filename"],
                },
            },
            {
                "name": "download_file",
                "description": "Download a file from a URL and save it to the downloads directory",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL of the file to download",
                        },
                        "filename": {
                            "type": "string",
                            "description": "Name to save the file as (optional - will use URL filename if not provided)",
                        }
                    },
                    "required": ["url"],
                },
            },
            {
                "name": "download_webpage",
                "description": "Download a webpage and extract its text content",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL of the webpage to download",
                        },
                        "save_to_file": {
                            "type": "boolean",
                            "description": "Whether to save the content to a file (optional, defaults to False). The whole content will be saved to a file if this is True, and the preview will be returned in the response limited by max_chars parameter.",
                        },
                        "filename": {
                            "type": "string",
                            "description": "Name of the file to save content to (optional, only used if save_to_file is True)",
                        },
                        "max_chars": {
                            "type": "integer",
                            "description": "Maximum number of characters to return in the preview, defaults to 5000",
                            "default": 5000
                        }
                    },
                    "required": ["url"],
                },
            },
            {
                "name": "upload_to_s3",
                "description": "Upload a file to S3 storage using configured credentials",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to upload, relative to user's data directory",
                        },
                        "s3_path": {
                            "type": "string",
                            "description": "Path in S3 where to save the file (optional, will use aibot/files/ by default)",
                        }
                    },
                    "required": ["file_path"],
                },
            },
            {
                "name": "create_zip_archive",
                "description": "Create a ZIP archive from one or multiple files",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "files": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "List of file paths to include in the archive, relative to user's data directory",
                        },
                        "archive_name": {
                            "type": "string",
                            "description": "Name for the ZIP archive (optional, will generate automatically if not provided)",
                        }
                    },
                    "required": ["files"],
                },
            },
            {
                "name": "send_file_path_to_user_via_telegram",
                "description": "Send a file to a user via Telegram using filepath",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to send, relative to user's data directory. If you want to send file to user via telegram, use this tool with file path like this: data/user_id/folder_name/filename.jpg",
                        },
                        "caption": {
                            "type": "string",
                            "description": "Optional caption for the file, when you send it to user via telegram",
                            "default": "Here is your file"
                        }
                    },
                    "required": ["file_path"],
                },
            }
        ]

    async def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Execute a tool by name with given arguments"""
        if tool_name == "list_files":
            return self.list_directory()
        elif tool_name == "create_file":
            return self.create_text_file(tool_args["filename"], tool_args["content"])
        elif tool_name == "read_file":
            return self.read_text_file(tool_args["filename"])
        elif tool_name == "create_directory":
            return self.create_directory(tool_args["dirname"])
        elif tool_name == "delete_file":
            return self.delete_file(tool_args["filename"])
        elif tool_name == "download_file":
            return self.download_file(tool_args["url"], tool_args.get("filename"))
        elif tool_name == "download_webpage":
            return self.download_webpage(
                tool_args["url"],
                tool_args.get("save_to_file", False),
                tool_args.get("filename"),
                tool_args.get("max_chars", 5000)
            )
        elif tool_name == "upload_to_s3":
            return self.upload_to_s3(
                tool_args["file_path"],
                tool_args.get("s3_path")
            )
        elif tool_name == "create_zip_archive":
            return self.create_zip_archive(
                tool_args["files"],
                tool_args.get("archive_name")
            )
        elif tool_name == "send_file_path_to_user_via_telegram":
            return await self.send_file_to_user(
                tool_args["file_path"],
                tool_args.get("caption")
            )
        else:
            return f"Unknown tool: {tool_name}"

    def list_directory(self) -> str:
        """List all files in the user's data directory"""
        try:
            files = [str(f.relative_to(self.base_path)) for f in self.base_path.glob("**/*") if f.is_file()]
            return f"Files found: {files}"
        except Exception as e:
            return f"Error listing directory: {str(e)}"

    def create_text_file(self, filename: str, content: str) -> str:
        """Create a text file with given content"""
        try:
            file_path = self.base_path / filename
            # Ensure the parent directories exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Replace literal \n with actual newlines and write
            content = content.replace('\\n', '\n')
            
            with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
                f.write(content)
            return f"File {filename} created successfully. Path: {file_path}. Content: {content}"
        except Exception as e:
            return f"Error creating file: {str(e)}"

    def read_text_file(self, filename: str) -> str:
        """Read content from a text file"""
        try:
            file_path = self.base_path / filename
            if not file_path.exists():
                return f"File {filename} does not exist"
            
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def create_directory(self, dirname: str) -> str:
        """Create a new directory in the user's data space"""
        try:
            dir_path = self.base_path / dirname
            dir_path.mkdir(parents=True, exist_ok=True)
            return f"Directory {dirname} created successfully"
        except Exception as e:
            return f"Error creating directory: {str(e)}"

    def delete_file(self, filename: str) -> str:
        """Delete a file from the user's data space"""
        try:
            file_path = self.base_path / filename
            if not file_path.exists():
                return f"File {filename} does not exist"
            
            if not file_path.is_file():
                return f"{filename} is not a file"
            
            file_path.unlink()
            return f"File {filename} deleted successfully"
        except Exception as e:
            return f"Error deleting file: {str(e)}"

    def download_file(self, url: str, filename: str = None) -> str:
        """Download a file from a URL and save it to the downloads directory"""
        try:
            # Get the filename from the URL if not provided
            if not filename:
                filename = url.split('/')[-1]
                if not filename:
                    return "Could not determine filename from URL"

            file_path = self.downloads_path / filename
            
            # Download the file
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an error for bad status codes
            
            # Save the file
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            rel_path = str(file_path.relative_to(self.base_path))

            return f"File downloaded successfully to {rel_path}"
        except Exception as e:
            return f"Error downloading file: {str(e)}"

    def download_webpage(self, url: str, save_to_file: bool = False, filename: str = None, max_chars: int = 5000) -> str:
        """Download a webpage and extract its text content"""
        try:
            # Download the webpage
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            # Parse with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text content
            text = soup.get_text()
            
            # Clean up text: remove blank lines and excessive whitespace
            lines = (line.strip() for line in text.splitlines())
            text = '\n'.join(line for line in lines if line)

            if save_to_file:
                if not filename:
                    # Create filename from URL
                    from urllib.parse import urlparse
                    parsed_url = urlparse(url)
                    filename = f"{parsed_url.netloc.replace('.', '_')}_{parsed_url.path.replace('/', '_')}.txt"
                    if len(filename) > 100:  # Truncate if too long
                        filename = filename[:100] + ".txt"
                
                file_path = self.downloads_path / filename
                with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
                    f.write(text)
                
                rel_path = str(file_path.relative_to(self.base_path))

                return f"Webpage content saved to {rel_path}\n\n. You can read full content of a file using it path. First {max_chars} characters of content for preview:\n{text[:max_chars]}..."
            
            # If not saving to file, also limit the returned text to max_chars
            if len(text) > max_chars:
                return f"{text[:max_chars]}..."
            return text

        except Exception as e:
            return f"Error downloading webpage: {str(e)}"

    def upload_to_s3(self, file_path: str, s3_path: str = None) -> str:
        """Upload a file to S3 storage"""
        try:
            if not self.s3_client:
                return "S3 client not initialized. Check your S3 credentials in environment variables."

            # Get the full local file path
            local_file_path = self.base_path / file_path
            if not local_file_path.exists():
                return f"File {file_path} does not exist"
            
            if not local_file_path.is_file():
                return f"{file_path} is not a file"

            # Get bucket name from env
            bucket_name = os.getenv('S3_BUCKET_NAME')
            if not bucket_name:
                return "S3_BUCKET_NAME environment variable not set"

            # Get S3 path
            if not s3_path:
                s3_path = os.getenv('S3_PATH_TO_STORE', '')
            
            # Combine S3 path with filename
            s3_key = f"{s3_path.rstrip('/')}/{local_file_path.name}"

            # Upload file
            self.s3_client.upload_file(
                str(local_file_path),
                bucket_name,
                s3_key
            )

            # Generate a pre-signed URL for temporary access
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket_name, 'Key': s3_key},
                ExpiresIn=3600  # URL valid for 1 hour
            )

            return f"File uploaded successfully to S3\nBucket: {bucket_name}\nPath: {s3_key}\nTemporary access URL (valid for 1 hour): {url}"

        except Exception as e:
            return f"Error uploading file to S3: {str(e)}"

    def create_zip_archive(self, files: List[str], archive_name: str = None) -> str:
        """Create a ZIP archive from one or multiple files"""
        try:
            # Generate archive name if not provided
            if not archive_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                archive_name = f"archive_{timestamp}.zip"
            
            # Ensure archive has .zip extension
            if not archive_name.lower().endswith('.zip'):
                archive_name += '.zip'

            # Create archive in downloads directory
            archive_path = self.downloads_path / archive_name

            # Check if all files exist first
            files_to_zip = []
            for file_path in files:
                full_path = self.base_path / file_path
                if not full_path.exists():
                    return f"File {file_path} does not exist"
                if not full_path.is_file():
                    return f"{file_path} is not a file"
                files_to_zip.append((full_path, file_path))  # Store both full path and relative path

            # Create ZIP archive
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for full_path, rel_path in files_to_zip:
                    # Add file to archive using relative path as name in archive
                    zipf.write(full_path, rel_path)

            # Get archive size
            archive_size = archive_path.stat().st_size
            size_mb = archive_size / (1024 * 1024)  # Convert to MB

            # List files included in archive
            included_files = "\n".join([f"- {rel_path}" for _, rel_path in files_to_zip])

            archive_rel_path = str(archive_path.relative_to(self.base_path))

            return f"""ZIP archive created successfully
Path: {archive_rel_path}
Size: {size_mb:.2f} MB
Files included:
{included_files}"""

        except Exception as e:
            return f"Error creating ZIP archive: {str(e)}"

    async def send_file_to_user(self, file_path: str, caption: str = "Here is your file") -> str:
        """Send a file to a user via Telegram"""
        try:

            if Path(self.base_path / file_path).exists():
                full_path = str(Path(self.base_path / file_path))
            elif Path(self.downloads_path / file_path).exists():
                full_path = str(Path(self.downloads_path / file_path))
            elif Path(file_path).exists():
                full_path = str(Path(file_path))
            else:
                return f"File {file_path} not found"

            print(f"Sending file to user {self.user_id}: {full_path}")

            if not Path(full_path).exists():
                return f"Error: File {file_path} does not exist"
            
            with open(full_path, 'rb') as file:
                await self.telegram_update.message.reply_document(
                    document=file,
                    caption=caption
                )

            return f"Successfully sent file {file_path} to chat ID {self.user_id}.\n"
            
        except Exception as e:
            return f"Error sending file: {str(e)}"
