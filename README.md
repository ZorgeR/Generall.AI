# 🤖 GALL.AI (prev. GenerALL.AI) - Multimodal Gen AI Agent System

> **Note:** GenerALL.AI refers to a "General Purpose AI", not a military general.

## 📋 Overview

GALL.AI (General AI) is a sophisticated multimodal agent system with telegram bot communication interface powered by advanced AI models like Claude Sonnet 5 and GPT-5.6. It provides a seamless interface for users to interact with AI capabilities through Telegram, supporting text, voice, images, documents, and more.

## 🔎 At a Glance

- **Multimodal Agent System**: Process and respond to text, voice, images, and documents within a single conversation
- **Tool-Based Architecture**: Powered by specialized modules for tasks like web search, file operations, code execution, and more
- **Extensible Framework**: Built on a modular system that can be expanded with new capabilities
- **Natural Interface**: Communicate with advanced AI models through the familiar Telegram messaging platform
- **Secure Execution**: All operations run in isolated environments with proper permission controls

## ✨ Features

- **🔄 Multimodal Support**: Process and respond to text, voice messages, images, PDFs, and other document types
- **🎤 Voice Processing**: Transcribe voice messages and generate voice responses with customizable voices via ElevenLabs
- **🖼️ Image Analysis**: Describe and analyze images using state-of-the-art vision models
- **📸 Multiple Photo Processing**: Analyze multiple images simultaneously to describe, compare differences, and redraw content
- **🎨 Image Generation**: Create new images based on text descriptions or modifications of existing images
  - **🔄 Generate with DALL-E 3**: Create single images with OpenAI's DALL-E 3 model
  - **✨ Multimodal Generation with Gemini**: Generate stories with matching images using Google's Gemini model
  - **🖌️ Image-to-Image Transformation**: Edit and transform existing images with powerful AI editing capabilities
- **📄 Document Processing**: Extract and analyze content from PDFs and other documents with deep understanding of the content
- **⏰ Reminder System**: Create and manage reminders with natural language processing for both specific times and contextual events
- **📅 Task Scheduling**: Schedule tasks to execute based on specific times or event triggers like "after a match ends"
- **🔒 Secure Container System**: Run unsafe commands in isolated Docker containers
- **💻 Terminal Access**: Execute system commands securely through the bot interface
- **📦 Package Management**: Install and run packages in a secure containerized environment
- **📁 File Sharing**: Send files directly to users through Telegram
- **📱 SMS Messaging**: Send SMS text messages to phone numbers via Twilio integration
- **⚙️ Customizable Settings**: Fine-tune the assistant's behavior through an interactive settings menu
- **🧠 Persistent Memory**: Maintain conversation context across sessions
- **🔌 Advanced Agent Architecture**: Powered by a modular agent system with specialized tools for different tasks
- **💾 Memory Management**: Smart conversation history handling with summarization capabilities
- **🔍 Web Search**: Search the web for up-to-date information using multiple providers (Perplexity, Tavily)
- **🧩 Reasoning Systems**: Multiple stages of reasoning with critique and judge mechanisms
- **☁️ Cloud Storage**: Upload and manage files on S3-compatible storage
- **💻 Code Execution**: Write and execute Python and Bash scripts in secure environments
- **💬 In-Process Communication**: Send Telegram messages, files, or reactions directly during thinking operations through tool-based interactions, not just as final bot output
- **🔒 SSH Connections**: Securely connect to remote servers via SSH to perform system administration tasks and retrieve information
- **🔍 Shodan API Integration**: Leverage Shodan's powerful internet scanning capabilities to gather intelligence on networks, devices, and potential security issues
- **🛠️ Advanced Tools & Techniques**: Utilize specialized cybersecurity, networking, and data analysis tools for comprehensive assessments and operations
- **🎟️ Invite System**: Control access through user invites with admin oversight
  - Users can create and share limited invite codes (5 per user)
  - Admins have unlimited invite generation capability
  - Track invite usage and user access via admin commands
  - Automatically notify admins when new users join

## ⚙️ How memory work

![schema](static/schema/genai_tg_schema_v2.png)

- Control all memory aspect
- Long term memory
- Short term memory
- Previous conversation full thinking context
- RAG automated memory
- Summarization memory

## 🚀 Example Scenarios

Here are some advanced scenarios that demonstrate the power and versatility of GenerALL.AI:

### 📊 Deep Research with File Export

```
User: "Research the latest developments in quantum computing in 2023-2024. Save the results as a PDF and send it to me."

Bot: "I'll research the latest developments in quantum computing and prepare a comprehensive report for you."
     [Conducts research using Perplexity and Tavily integrations]
     [Generates a structured report in Markdown format]
     [Converts to PDF]
     [Uploads to S3 storage]
     [Sends file to user via Telegram]
     "Here's your research report on quantum computing advancements in 2023-2024. I've included sections on hardware breakthroughs, algorithm developments, and industry applications."
```

### 🖥️ Remote Server Management

```
User: "Connect to my production server at ssh://user@example.com and check disk usage on all volumes."

Bot: "I'll connect to your production server and check the disk usage."
     [Establishes secure SSH connection]
     [Executes df -h command]
     [Formats results in a readable table]
     "Here's the current disk usage on your production server:
      Filesystem    Size    Used    Avail   Use%    Mounted on
      /dev/sda1     50G     23G     25G     48%     /
      /dev/sdb1     1T      750G    230G    77%     /data"
```

### 🔍 Security Vulnerability Scanning

```
User: "Scan my website https://example.com for common security vulnerabilities."

Bot: "I'll perform a security scan of your website. This might take a few minutes."
     [Sets up a secure container for scanning]
     [Installs and runs security tools]
     [Analyzes results and compiles a security report]
     "I've completed the security scan of your website. I found:
      - 2 potential XSS vulnerabilities in your contact form
      - Outdated SSL certificate (expires in 15 days)
      - Missing HTTP security headers
      I've attached a detailed report with recommendations for fixing these issues."
```

### 🎨 Advanced Image Generation and Transformation

```
User: "Create a story about a space explorer discovering an ancient alien civilization. Include images for each key scene."

Bot: "I'll create a multimodal story about a space explorer discovering an ancient alien civilization."
     [Generates a complete story with matching images for each key scene]
     [Sends text and images together in a coherent narrative]
     "The Stellar Archaeologist - A space explorer's journey to the lost civilization of Proxima Centauri IV..."
     [Image of a space explorer approaching an alien planet]
     "As Captain Elara approached the crystalline structures, their surfaces shimmered with an otherworldly light..."
     [Image of alien crystal structures with mysterious glowing patterns]
     
User: "This is great! Can you transform the second image to add more alien symbols and make it look more mysterious?"

Bot: "I'll transform that image to add more alien symbols and enhance the mysterious atmosphere."
     [Takes the existing image and applies the requested transformations]
     [Sends the modified image with new alien symbols and enhanced mysterious elements]
     "Here's the transformed image with additional alien symbols and a more mysterious atmosphere. I've added ancient hieroglyphs that glow with an eerie blue light and enhanced the shadowy elements to create a sense of the unknown."
```

### ⏱️ Complex Event-Based Reminders

```
User: "Monitor the NASA website and alert me when the next SpaceX launch is scheduled. Remind me 1 day before and again 1 hour before the launch."

Bot: "I'll monitor the NASA and SpaceX websites for upcoming launch information and set up the reminders for you."
     [Sets up regular monitoring of relevant websites]
     [Identifies next launch date]
     [Creates two context-based reminders]
     "I've set up monitoring for the next SpaceX launch. Currently, the next launch is scheduled for June 15th at 3:00 PM EDT. I'll remind you on June 14th at 3:00 PM and again on June 15th at 2:00 PM."
```

### 👨‍💻 Custom Code Development and Execution

```
User: "Write a Python script that analyzes a CSV file of stock prices, calculates moving averages, and generates a visualization. Use pandas and matplotlib."

Bot: "I'll create a Python script for stock price analysis with moving averages visualization."
     [Writes Python code using pandas and matplotlib]
     [Installs required libraries in a secure container]
     [Executes the code with sample data]
     [Generates visualization]
     [Sends the resulting graph and script file to the user]
     "I've created and executed a stock analysis script that:
      1. Loads CSV data using pandas
      2. Calculates 7-day and 30-day moving averages
      3. Generates a visualization with original prices and both moving averages
      4. Highlights potential buy/sell signals
      
      I've attached both the visualization and the Python script. You can modify the script 
      to use your own data by changing the file path in line 12."
```

### 🎟️ Invite-Based Access Control

```
User: "/invite"

Bot: "🎟️ New Invite Created

Share this link: https://t.me/YourBotName?start=invite_a1b2c3d4

Or use this command:
/invite a1b2c3d4

Invites remaining: 4/5"

[Later, when someone uses the invite]

New User: "/start invite_a1b2c3d4"

Bot: "✅ Invite accepted! You now have access to the bot.
     
     👋 Welcome to Generall.AI bot! Use me to get AI assistance.
     
     You can send me messages, voice recordings, or images to analyze."

[Admin receives notification]

Bot to Admin: "🔔 New user joined!
               User ID: `123456789`
               🌟 Invited by: `987654321`
               🌟 Total users: 3"
```

## 🔄 Advanced Features

### ✨ Rich Messages

Answers are sent as **rich Telegram messages** (Bot API 10.1): real GitHub-flavored Markdown with
headings, tables, task lists, fenced code blocks, block quotes and LaTeX math, rendered natively by
the Telegram app. While the answer is being generated (with `STREAMING_ENABLED=true`) the draft
streams with live formatting and the model's thinking shows in Telegram's "Thinking…" block.

Pictures can appear inside the answer: the agent writes `![caption](images/photo.jpg)` for an image
it generated, a photo you sent or a downloaded file (or an https image URL) and it renders inline at
that spot; with rich messages off the picture is sent as a separate photo after the text.

Rich messages are on by default and can be switched per chat in `/settings` → **Rich Messages**.
Turn them off if your Telegram app is too old and shows "unsupported message"; the bot then uses
classic Markdown formatting. If the Bot API server (the `telegram-bot-api` sidecar) is older than
10.1 the bot detects it once and falls back to MarkdownV2 formatting (tables as monospace blocks).

### 🎤 Voice Settings
Customize voice parameters including voice model selection, stability, clarity, and style. The bot can both listen to your voice messages and respond with generated voice using ElevenLabs.

### 📄 PDF Processing
Upload PDFs to extract and analyze content. The bot can understand complex documents, summarize contents, answer questions about the document, and provide insights.

### 🎨 Advanced Image Generation and Transformation

GenerALL.AI offers multiple powerful image generation and transformation capabilities:

#### 📸 DALL-E 3 Image Generation
Generate high-quality single images with OpenAI's DALL-E 3 model, with customizable parameters:
- Control image size and quality
- Detailed prompt capabilities
- Supports art styles, photography styles, and concept visualization

#### ✨ Multimodal Generation with Gemini
Create rich stories with matching images using Google's advanced Gemini model:
- Generate text and images simultaneously within a single cohesive experience
- Create multiple images that match the narrative flow
- Support for various artistic styles (3D digital art, photorealistic, cartoon, anime, etc.)
- Perfect for storytelling, educational content, and creative projects

#### 🖌️ Image-to-Image Transformation
Transform and edit existing images with AI:
- Add or remove elements from images
- Change styles, colors, or artistic approaches
- Apply creative modifications based on text instructions
- Useful for design iterations, creative exploration, and visual problem-solving
- Works with images the user has sent previously

### 📁 Supported Text File Types
GenerALL.AI supports processing and analysis of various file formats including:
- **Documents**: PDF, TXT, DOCX, MD
- **Data Files**: JSON, JSONL, CSV, XLSX, XLS
- **Code Files**: PY, JS, HTML, CSS, PHP, SQL
- **Configuration Files**: XML, YAML, YML, TOML, INI, CONF
- **Shell Scripts**: SH, BAT, PS1
- **System Files**: LOG

The bot can extract content, analyze structure, and help you understand the information contained within these supported file types.

### 🎬 Supported Media File Types
GenerALL.AI supports processing and analysis of various media formats including:
- **Images**: JPG, JPEG, PNG, GIF*, BMP*, WEBP*
- **Audio**: Telegram Voice Message, mp3*, ogg*
- **Video**: -/-

The bot can:
- **Images**: Analyze content, detect objects, read text (OCR), and describe scenes
- **Audio**: Transcribe speech, analyze audio content, and detect language and answer using voice generation
- **Video**: -/-


### 💻 Code Execution Capabilities
- **🐍 Python Development**: Create, edit, and execute Python scripts in a secure environment
- **🔧 Bash Scripting**: Run Bash scripts and system commands safely
- **📦 Package Installation**: Install Python libraries and dependencies as needed
- **📊 Data Analysis**: Process and visualize data with popular libraries like pandas, numpy, and matplotlib
- **🔁 Automated Workflows**: Create scripts for repetitive tasks and automated data processing
- **🔒 Secure Execution**: All code runs in isolated containers for security
- **📋 Code Editing**: Iteratively improve code based on requirements and feedback

### ⏰ Reminder and Task System

## 💻 System Requirements

### 🛠️ For Local Development

- Python 3.12+
- Docker and Docker Compose
- FFmpeg (for audio processing)
- Git

### 🔑 API Keys Required

- Telegram Bot Token
- Anthropic API Key (for Claude Sonnet 5 and Haiku 4.5)
- OpenAI API Key (for GPT-5.6, Whisper and embeddings)
- ElevenLabs API Key (for voice synthesis)
- Google API Key (for Gemini image generation/transformation)

## 📥 Installation

### 🐳 Using Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/generall.ai.git
   cd generall.ai
   ```

2. Create an `.env` file based on the provided `.env.example`:
   ```bash
   cp .env.example .env
   ```

3. Edit the `.env` file with your API keys and configuration.

4. Build and start the Docker container:
   ```bash
   docker-compose up --build
   ```

### 🔧 Manual Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/generall.ai.git
   cd generall.ai
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r app/requirements.txt
   ```

4. Create and configure your `.env` file.

5. Install FFmpeg system-wide or use the included ffmpeg-downloader.

6. Run the bot:
   ```bash
   python app/main_bot.py
   ```

## 🔄 Compatibility

- **💻 Operating Systems**: Linux (recommended for production), macOS, Windows
- **🚀 Deployment**: Docker-based deployment supported across all major platforms
- **🐍 Python Version**: 3.12+ required
- **📱 Telegram clients**: rich message rendering needs a Telegram app released after Bot API 10.1 (older apps show "unsupported message"; switch Rich Messages off in `/settings`)
- **🖥️ Hardware Requirements**: 
  - Minimum: 4GB RAM, 2 CPU cores
  - Recommended: 8GB+ RAM, 4+ CPU cores (especially for handling multiple conversations)
- **🌐 Network**: Requires internet connection for API access

## ⚙️ Configuration

The application is configured via environment variables in the `.env` file. Key configuration options include:

### 🔑 API Keys

- `ANTHROPIC_API_KEY`: API key for Anthropic Claude models
- `OPENAI_API_KEY`: API key for OpenAI GPT models
- `GOOGLE_API_KEY`: API key for Gemini AI models
- `OPENAI_API_KEY_WHISPER`: API key for OpenAI Whisper (voice transcription)
- `TAVILY_API_KEY`: API key for Tavily search integration
- `PERPLEXITY_API_KEY`: API key for Perplexity search integration
- `ELEVENLABS_API_KEY`: API key for ElevenLabs voice synthesis

### 🤖 Telegram Configuration

- `TELEGRAM_BOT_TOKEN`: Your Telegram bot token from BotFather
- `TELEGRAM_CHAT_ID`: Comma-separated list of initially allowed Telegram chat IDs
- `TELEGRAM_ADMIN_ID`: Telegram chat ID for the admin user (has unlimited invites and can list all users)

### 🧠 Agent Configuration

- Tool, critique and judge iteration limits are per-user settings (see `/settings`); the
  `MAX_AGENT_*_ITERATIONS` variables are obsolete and ignored.

### 🧩 Models

Every model name lives in `app/models.py` and can be overridden from `.env` (defaults in
parentheses; see the commented block in `.env.example`):

- `ANTHROPIC_MODEL` (`claude-sonnet-5`): agent loop, judge, final compile, document and image
  description. `ANTHROPIC_EFFORT` (`high`) is sent as `output_config.effort` on every call.
- `ANTHROPIC_MODEL_FAST` (`claude-haiku-4-5`): topic/summary, complexity classifier, quick answers.
- `OPENAI_MODEL` (`gpt-5.6-terra`): critique and GPT vision on photos. `VIDEO_FRAMES_MODEL`
  (`gpt-5.6-luna`): video frame description. Both are reasoning models and get
  `OPENAI_REASONING_EFFORT` (`high`).
- `WHISPER_MODEL` (`whisper-1`), `EMBEDDING_MODEL` (`text-embedding-ada-002`, with
  `EMBEDDING_DIMENSION` `1536`), `TTS_MODEL` (`eleven_multilingual_v2`).
- `GEMINI_IMAGE_MODEL_FLASH`, `GEMINI_IMAGE_MODEL_PRO`, `GPT_IMAGE_MODEL`, `DALLE_MODEL`,
  `VEO_MODEL`, `PERPLEXITY_MODEL`: image, video and deep-research models.

### ⚙️ Concurrency

- `MAX_CONCURRENT_TURNS`: agent turns running at the same time across all users (default: 8)
- `MAX_SANDBOX_CONTAINERS`: sandbox containers running at the same time (default: 4)
- `THREAD_POOL_SIZE`: worker threads for blocking work such as Docker waits and HTTP SDK calls (default: 32)
- `TURN_TIMEOUT_SECONDS`: hard limit for one message or scheduled task (default: 1800)
- `DROP_PENDING_UPDATES`: discard messages that arrived while the bot was down (default: true)

### ☁️ S3 Storage Configuration

- `S3_HOST`: S3-compatible storage host URL
- `S3_ACCESS_KEY`: S3 access key for authentication
- `S3_SECRET_KEY`: S3 secret key for authentication
- `S3_BUCKET_NAME`: Name of the S3 bucket to use
- `S3_PATH_TO_STORE`: Path within the bucket to store files

### 📱 Twilio Configuration

- `TWILIO_ACCOUNT_SID`: Twilio account SID for SMS messaging
- `TWILIO_AUTH_TOKEN`: Twilio authentication token
- `TWILIO_FROM_NUMBER`: Twilio phone number to send messages from

## 🔐 Security Considerations

- The application uses secure Docker containers to run potentially unsafe commands
- User access is restricted to specified Telegram chat IDs
- API keys are stored securely in environment variables
- Docker socket is mounted to allow container management

## 📝 Usage

1. Start a conversation with your bot on Telegram
2. Send text messages, voice recordings, images, or documents
3. The bot will process your input and respond accordingly
4. Use `/settings` to customize the bot's behavior
5. Use `/reminders` to manage your reminders
6. Use `/cancel` to stop the task that is running and clear your queue
7. Use `/settings` → Rich Messages to switch between rich and classic formatting
8. While the agent works, the "Thinking…" message lists the tool calls as they run (✅ done, ⏳ running,
   ❌ failed, with a short argument preview and duration); tools requested together run in parallel.
   With rich messages on, a one-line summary of the tool calls stays above the answer.

### ⏳ Queues and isolation

Every chat has its own queue. Messages you send while the bot is still working on a previous one
are answered immediately with a short "still working on…" notice (with a Stop button) and are then
processed in order, each as its own turn. Queues are fully isolated: a long task for one user, such
as video generation or a sandboxed command, never delays other users. `MAX_CONCURRENT_TURNS`,
`MAX_SANDBOX_CONTAINERS` and `TURN_TIMEOUT_SECONDS` in `.env` bound the whole process.

## 🚀 Deploying to production

Production is deployed by a manual GitHub Actions run (`.github/workflows/deploy.yml` plus the
`deploy/` helpers) instead of the hand-typed `ssh … && git pull && docker compose …` routine. All of
its configuration lives in the GitHub environment named **`PROD`**.

**The Deploy button**: Actions → *Deploy to production* → *Run workflow*. Pick the git ref to deploy
(branch, tag or full commit SHA; default `main`) and, optionally, tick *dry run*. If `PROD` has
*required reviewers* configured, the run pauses for their approval before it touches the server.

**Server connection** — four `SERVER_*` entries in `PROD` (secrets or variables; they never reach
the server's `.env`): `SERVER_ADDR` (IP or host name), `SERVER_USER` (`root`), `SERVER_SSH_KEY`
(an OpenSSH private key whose public half is in the server's `authorized_keys`) and
`SERVER_CODE_ROOT_PATH` (the checkout, e.g. `/opt/generall.ai/Generall.AI`). Optional:
`SERVER_SSH_PORT` (default 22) and `SERVER_KNOWN_HOSTS` (a `known_hosts` line that pins the host
key; without it the key is fetched with `ssh-keyscan` on every run).

**Application configuration**: every *other* secret and variable visible to `PROD` becomes one
`KEY=value` line of the server's `.env` (secrets win over variables with the same name; keys must
look like `TELEGRAM_BOT_TOKEN`, i.e. match `^[A-Z][A-Z0-9_]*$`). To add or change a setting, edit it
in `PROD` and deploy — no workflow change is needed, and the server's `.env` should not be edited by
hand any more (the previous file is kept as `.env.bak`). `deploy/render_env.py` quotes values as
needed and skips values containing newlines; the job log shows key names only, never values.

**First run**: start with a *dry run*. It renders the file, connects to the server and lists the keys
missing there (would be added), the keys that exist only in the server's current `.env` (would be
dropped — add them to `PROD` first) and the keys present in both. Nothing is changed on the server.

**What a real deploy does** (`deploy/remote_deploy.sh`, streamed over SSH): refuses to run if
tracked files are modified on the server; `git fetch` and checkout of the chosen ref; installs the
uploaded `.env`; `docker compose pull telegram-bot-api` (best effort), `docker compose build --pull
bot`, `docker compose up -d --remove-orphans`; then waits up to 90 s for the `bot` container to be
running and to log `is running`. If it does not, the job fails and prints `docker compose ps` and the
last log lines (the previous commit and `.env.bak` are named for a manual rollback). `data/` is never
touched, and a concurrency group runs deploys one at a time.

## ❓ Troubleshooting

- **🤖 Bot not responding**: Check your Telegram token and allowed chat IDs
- **🎤 Voice features not working**: Ensure FFmpeg is properly installed
- **🐳 Container issues**: Verify Docker is running and the user has appropriate permissions
- **🔑 API errors**: Check your API keys and network connection
- **✨ Answers arrive as "unsupported message"**: update the Telegram app, or turn Rich Messages off in `/settings`
- **✨ Answers have no headings/tables and the log says rich messages are not supported**: the
  `telegram-bot-api` sidecar is older than Bot API 10.1; run `docker compose pull telegram-bot-api && docker compose up -d`

## 📄 License

GenerALL.AI is released under a custom license with the following terms:

- ✅ **Personal Use**: You may use this software for personal, non-commercial purposes.
- ✅ **Modification**: You may modify the software and create derivative works.
- ✅ **Distribution**: You may distribute copies of the original or modified software.
- ✅ **Attribution**: You must give appropriate credit to the original authors.
- ❌ **Commercial Use**: Commercial use requires explicit permission from the copyright holders.

See the [LICENSE](./LICENSE) file for complete details.

## 🧪 Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## 👏 Acknowledgments

- Built with [aiogram](https://github.com/aiogram/aiogram)
- Uses AI models from Anthropic and OpenAI
- Voice synthesis powered by ElevenLabs 


## ❤️ Donation:

Become a Patron: https://www.patreon.com/zorg
Donate via Paypal: http://paypal.me/mezorg/15

##### ₿ BTC:
![btc](static/donations/btc.png)
`bc1qyymy3ufvq3c3uq2q4927ll6x4rhvdw8gxlydwc`

##### Ξ ETH:
![eth](static/donations/eth.png)
`0x0213A705065B193D14f1A3cd075977e28Da8F9B3`

##### 💵 USDT-TRC20:
![usdt](static/donations/usdt.png)
`TG2efcamZ1767TkBfeGUn8QWaRGrLpHUxD`

##### 💵 SOL:
![sol](static/donations/sol.png)
`B7faayiFUqM64Dgt4iUtpSfWbo4VANBs8bknSXv3e53E`

##### 💎 TON:
![ton](static/donations/ton.png)
`UQDg07heLBcWdYO_sP6_Hc9hCu24E3v05sBJuRqc_DyWKreq`

##### 🐕 DOGE:
![doge](static/donations/doge.png)
`D9RbkgazaGhkT4FHkJtHoh4hDxkzAZwQnK`


You can make a donation / subscription, or say thanks in Telegram: https://t.me/ZorgeR

Boosty (Donate / Subscription) : https://boosty.to/zorgg
