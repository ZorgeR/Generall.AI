from typing import List, Dict, Any, Union
from tavily import TavilyClient
import requests
import os
from datetime import datetime, timezone
from pathlib import Path
import re
import json
from models import PERPLEXITY_MODEL, PERPLEXITY_MODELS, PERPLEXITY_REASONING_MODEL
from agents.transcript import is_context_block, is_tool_result_message, message_text

MEMORY_SEARCH_MAX_RESULTS = 20
MEMORY_SEARCH_SNIPPET = 120  # characters kept on each side of a match
_ALTERNATIVES_RE = re.compile(r"\s+\|\s+")  # "lisbon | lisboa": any alternative matches
_CONTEXT_TIME_RE = re.compile(r"Current time in UTC\+0: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
_CONVERSATION_TS_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$")
_NOTE_DATE_RE = re.compile(r"^-\s*\[(\d{4}-\d{2}-\d{2})\]")  # a dated profile note

class SearchTools:
    def __init__(self, user_id: str = None):
        self.client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        self.perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
        self.perplexity_url = "https://api.perplexity.ai/chat/completions"
        self.base_path = Path("./data") / str(user_id) # Base path for memory files
        self.user_id = user_id  # Store user_id as instance variable

    @property
    def tools_schema(self) -> List[Dict[str, Any]]:
        """Return the schema for all search operation tools"""
        return [
            {
                "name": "memory_search",
                "description": (
                    "Search your long-term memory: past conversation summaries, questions and answers, "
                    "the user's profile and the current transcript. Use it before saying you do not remember "
                    "something. Plain substring search (no regex); separate alternatives with ' | ', e.g. "
                    "'Lisbon | Lisboa', and search in the user's language and in English. Returns a JSON "
                    "array of hits, newest first: source (conversation | profile | transcript), timestamp, "
                    "topic, field and a snippet around the match."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The word or phrase to look for; several alternatives separated by ' | '.",
                        },
                        "case_sensitive": {
                            "type": "boolean",
                            "description": "Match case exactly (default: no)",
                            "default": False
                        },
                        "max_results": {
                            "type": "integer",
                            "description": f"Hits to return, newest first (default {MEMORY_SEARCH_MAX_RESULTS})",
                            "minimum": 1,
                            "maximum": 100,
                            "default": MEMORY_SEARCH_MAX_RESULTS
                        }
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "search_web",
                "description": "Search the web for current information on a topic. Returns relevant search results and summaries.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to find information about. Send just a what you want to find, in natural language, not a url or anything else.",
                        },
                        "search_depth": {
                            "type": "string",
                            "description": "Either 'basic' for quick searches or 'deep' for more thorough research",
                            "enum": ["basic", "deep"]
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of search results to return (default: 5)",
                            "minimum": 1,
                            "maximum": 50
                        },
                        "topic": {
                            "type": "string",
                            "description": "The category of the search. Determines which agents will be used",
                            "enum": ["general", "news"],
                            "default": "general"
                        },
                        "days": {
                            "type": "integer",
                            "description": "Number of days back to include in search results (only for 'news' topic)",
                            "minimum": 1,
                            "default": 3
                        },
                        "time_range": {
                            "type": "string",
                            "description": "Time range for search results",
                            "enum": ["day", "week", "month", "year", "d", "w", "m", "y"]
                        },
                        "include_images": {
                            "type": "boolean",
                            "description": "Include query-related images in response",
                            "default": False
                        },
                        "include_image_descriptions": {
                            "type": "boolean",
                            "description": "Include query-related images and their descriptions",
                            "default": False
                        },
                        "include_raw_content": {
                            "type": "boolean",
                            "description": "Include cleaned and parsed HTML content of each result",
                            "default": False
                        },
                        "include_answer": {
                            "type": "string",
                            "description": "Include AI-generated answer (False/'basic'/'advanced')",
                            "enum": ["basic", "advanced"]
                        },
                        "include_domains": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of domains to include in search results"
                        },
                        "exclude_domains": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of domains to exclude from search results"
                        }
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "deep_research",
                "description": "Perform in-depth research using Perplexity AI. Best for complex topics requiring detailed analysis.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The research question or topic to analyze"
                        },
                        "model": {
                            "type": "string",
                            "description": "Model to use for research: sonar-reasoning-pro (step by step reasoning, slower but more thorough), sonar-pro (faster advanced research, but without step by step reasoning), sonar (faster, cheaper, but for most basic research)",
                            "enum": list(PERPLEXITY_MODELS),
                            "default": PERPLEXITY_MODEL
                        },
                        "mode": {
                            "type": "string",
                            "description": "Research mode determining depth and focus",
                            "enum": ["concise", "copilot", "academic"],
                            "default": "copilot"
                        },
                        "focus": {
                            "type": "string",
                            "description": "Specific aspect to focus on",
                            "enum": ["writing", "analysis", "math", "coding"],
                            "default": "analysis"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]

    def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Execute a search tool by name with given arguments"""
        if tool_name == "search_web":
            return self.search_web(
                query=tool_args["query"],
                search_depth=tool_args.get("search_depth", "basic"),
                max_results=tool_args.get("max_results", 10),
                topic=tool_args.get("topic", "general"),
                days=tool_args.get("days", 3),
                time_range=tool_args.get("time_range"),
                include_images=tool_args.get("include_images", False),
                include_image_descriptions=tool_args.get("include_image_descriptions", False),
                include_raw_content=tool_args.get("include_raw_content", False),
                include_answer=tool_args.get("include_answer"),
                include_domains=tool_args.get("include_domains"),
                exclude_domains=tool_args.get("exclude_domains")
            )
        elif tool_name == "memory_search":
            return self.memory_search(
                query=tool_args["query"],
                case_sensitive=tool_args.get("case_sensitive", False),
                max_results=tool_args.get("max_results", MEMORY_SEARCH_MAX_RESULTS),
            )
        elif tool_name == "deep_research":
            return self.deep_research(
                query=tool_args["query"],
                model=tool_args.get("model", PERPLEXITY_MODEL),
                mode=tool_args.get("mode", "copilot"),
                focus=tool_args.get("focus", "analysis")
            )
        else:
            return f"Unknown tool: {tool_name}"

    def search_web(
        self,
        query: str,
        search_depth: str = "basic",
        max_results: int = 10,
        topic: str = "general",
        days: int = 3,
        time_range: str = None,
        include_images: bool = False,
        include_image_descriptions: bool = False,
        include_raw_content: bool = False,
        include_answer: Union[bool, str] = False,  # Added include_answer parameter
        include_domains: List[str] = None,
        exclude_domains: List[str] = None
    ) -> str:
        """Search the web using Tavily's API
        
        Args:
            query (str): The search query to execute
            search_depth (str): Either 'basic' or 'advanced' (default: 'basic')
            max_results (int): Maximum number of results (default: 5)
            topic (str): Either 'general' or 'news' (default: 'general')
            days (int): Days back for news search (default: 3, only for news topic)
            time_range (str): Time range filter ('day'/'d', 'week'/'w', 'month'/'m', 'year'/'y')
            include_images (bool): Include query-related images (default: False)
            include_image_descriptions (bool): Add descriptions for images (default: False)
            include_raw_content (bool): Include parsed HTML content (default: False)
            include_answer (Union[bool, str]): Include AI-generated answer (False/'basic'/'advanced')
            include_domains (List[str]): Domains to include (default: None)
            exclude_domains (List[str]): Domains to exclude (default: None)
        """
        try:
            # Prepare API parameters with validation
            params = {
                "query": query,
                "search_depth": search_depth if search_depth in ["basic", "advanced"] else "basic",
                "max_results": min(max(1, max_results), 50),  # Ensure between 1 and 50
                "include_images": include_images,
                "include_image_descriptions": include_image_descriptions,
                "include_raw_content": False ##include_raw_content,
            }

            # Add include_answer if specified
            if include_answer:
                if isinstance(include_answer, str) and include_answer in ["basic", "advanced"]:
                    params["include_answer"] = include_answer
                elif isinstance(include_answer, bool):
                    params["include_answer"] = "basic" if include_answer else False

            # Only add optional parameters if they have valid values
            if topic in ["general", "news"]:
                params["topic"] = topic
                if topic == "news" and days:
                    params["days"] = min(max(1, days), 30)  # Ensure between 1 and 30

            if time_range in ["day", "week", "month", "year", "d", "w", "m", "y"]:
                params["time_range"] = time_range

            if include_domains and isinstance(include_domains, list):
                params["include_domains"] = include_domains

            if exclude_domains and isinstance(exclude_domains, list):
                params["exclude_domains"] = exclude_domains

            # Make API call with validated parameters
            search_result = self.client.search(**params)
            
            # Format the results
            formatted_results = []

            formatted_results.append(f"Search results: (This is just summarize of page content, you must use the download page content tool to read a content of a page in a detailed way!):")
            
            # Add AI answer if present
            if include_answer and "answer" in search_result:
                formatted_results.append(f"AI Answer:\n{search_result['answer']}\n")
            
            # Add search results
            formatted_results.append("Search Results:")
            for result in search_result['results']:
                formatted_result = f"- {result['title']}\n  {result['content']}\n  URL: {result['url']}"
                
                if include_raw_content and 'raw_content' in result:
                    formatted_result += f"\n  Raw Content: {result['raw_content'][:3000]}..."
                
                formatted_results.append(formatted_result)
            
            # Add image results if present
            if include_images and 'images' in search_result:
                formatted_results.append("\nImage Results:")
                for image in search_result['images']:
                    if include_image_descriptions:
                        formatted_results.append(f"- {image['description']}\n  URL: {image['url']}")
                    else:
                        formatted_results.append(f"- URL: {image['url']}")
            
            return "\n\n".join(formatted_results)
        except Exception as e:
            return f"Error performing search: {str(e)}. Parameters sent: {params}"

    def memory_search(self, query: str, case_sensitive: bool = False, max_results: int = MEMORY_SEARCH_MAX_RESULTS) -> str:
        """Substring search over the user's long-term memory, in the bot process.

        Sources under ``data/<uid>/``: the conversation files (summary, topic, question, response),
        ``profile.md`` (one hit per matching entry, field = its section) and the text view of the
        transcripts (user and assistant text blocks only: no context blocks, tool calls, tool
        results or base64 media). ``query`` may hold several alternatives separated by ``" | "``.
        Returns a JSON array of hits ``{source, timestamp, topic, field, snippet}`` newest first,
        at most ``max_results``, or a "No matches ..." sentence.
        """
        terms = [t for t in (s.strip() for s in _ALTERNATIVES_RE.split(query or "")) if t]
        if not terms:
            return "Error: memory_search needs a word or phrase to look for."
        try:
            max_results = max(1, min(int(max_results or MEMORY_SEARCH_MAX_RESULTS), 100))
        except (TypeError, ValueError):
            max_results = MEMORY_SEARCH_MAX_RESULTS
        flags = 0 if case_sensitive else re.IGNORECASE
        pattern = re.compile("|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)), flags)
        try:
            hits = self._conversation_hits(pattern) + self._profile_hits(pattern) + self._transcript_hits(pattern)
        except Exception as e:  # noqa: BLE001 - a tool result, never an exception for the turn
            return f"Error performing memory search: {e}"
        if not hits:
            return f"No matches for {' | '.join(terms)} in your memory (conversations, profile, transcript)."
        hits.sort(key=lambda h: h["timestamp"], reverse=True)
        return json.dumps(hits[:max_results], ensure_ascii=False)

    # ---- memory_search internals ----------------------------------------------
    @staticmethod
    def _snippet(text: str, match: re.Match) -> str:
        start, end = max(0, match.start() - MEMORY_SEARCH_SNIPPET), min(len(text), match.end() + MEMORY_SEARCH_SNIPPET)
        return ("…" if start else "") + text[start:end] + ("…" if end < len(text) else "")

    def _hit(self, pattern: re.Pattern, text, source: str, timestamp: str, topic: str, field: str) -> dict | None:
        text = " ".join(str(text or "").split())
        match = pattern.search(text) if text else None
        if not match:
            return None
        return {"source": source, "timestamp": timestamp, "topic": topic, "field": field, "snippet": self._snippet(text, match)}

    @staticmethod
    def _conversation_timestamp(data: dict, path: Path) -> str:
        raw = str(data.get("timestamp") or "")
        m = _CONVERSATION_TS_RE.match(raw) or _CONVERSATION_TS_RE.match("_".join(path.stem.split("_")[1:3]))
        if m:
            y, mo, d, h, mi, s = m.groups()
            return f"{y}-{mo}-{d} {h}:{mi}:{s}"
        return raw

    def _conversation_hits(self, pattern: re.Pattern) -> list[dict]:
        folder = self.base_path / "conversations"
        hits: list[dict] = []
        if not folder.exists():
            return hits
        for path in folder.glob("conversation_*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            timestamp = self._conversation_timestamp(data, path)
            topic = str(data.get("topic") or "general")
            for field in ("summary", "topic", "question", "response"):
                hit = self._hit(pattern, data.get(field), "conversation", timestamp, topic, field)
                if hit:
                    hits.append(hit)
        return hits

    def _profile_hits(self, pattern: re.Pattern) -> list[dict]:
        path = self.base_path / "profile.md"
        if not path.exists():
            return []
        try:
            text = path.read_text(encoding="utf-8")
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        except (OSError, UnicodeDecodeError):
            return []
        hits: list[dict] = []
        section = "About"
        for line in text.splitlines():
            if line.startswith("## "):
                section = line[3:].strip() or section
                continue
            dated = _NOTE_DATE_RE.match(line)
            hit = self._hit(pattern, line, "profile", f"{dated.group(1)} 00:00:00" if dated else modified, "profile", section)
            if hit:
                hits.append(hit)
        return hits

    def _transcript_hits(self, pattern: re.Pattern) -> list[dict]:
        folder = self.base_path / "transcripts"
        hits: list[dict] = []
        if not folder.exists():
            return hits
        for path in folder.glob("*transcript.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
                continue
            topic = "current conversation" if path.name == "transcript.json" else path.name.replace("_transcript.json", "").replace("_", " ")
            timestamp = str(data.get("updated") or "")[:19].replace("T", " ")
            for message in data["messages"]:
                if not isinstance(message, dict) or is_tool_result_message(message):
                    continue
                content = message.get("content")
                if message.get("role") == "user" and isinstance(content, list):
                    for block in content:  # the turn's own context block carries its time
                        if is_context_block(block):
                            m = _CONTEXT_TIME_RE.search(block["text"])
                            if m:
                                timestamp = m.group(1)
                field = "user" if message.get("role") == "user" else "assistant"
                hit = self._hit(pattern, message_text(message), "transcript", timestamp, topic, field)
                if hit:
                    hits.append(hit)
        return hits

    def deep_research(
        self,
        query: str,
        model: str = PERPLEXITY_MODEL,
        mode: str = "copilot",
        focus: str = "analysis"
    ) -> str:
        """Perform in-depth research using Perplexity AI
        
        Args:
            query (str): The research question or topic to analyze
            model (str): Model to use - 'sonar-reasoning-pro' (step by step reasoning) or 'sonar-pro' (faster research)
            mode (str): Research mode - 'concise', 'copilot', or 'academic'
            focus (str): Focus area - 'writing', 'analysis', 'math', 'coding'
            
        Returns:
            str: Formatted research results including analysis and sources
        """
        try:
            # Configure system prompt based on mode and focus
            system_prompt = f"You are an AI assistant specialized in {focus}. "
            if mode == "academic":
                system_prompt += "Provide detailed academic analysis with citations."
            elif mode == "concise":
                system_prompt += "Be precise and concise."
            elif mode == "copilot":
                system_prompt += "Help guide the research process with relevant insights."

            # Add reasoning instruction for sonar-reasoning-pro
            if model == PERPLEXITY_REASONING_MODEL:
                system_prompt += " Break down your analysis into clear steps, showing your reasoning process."

            # Prepare the API request payload
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                "max_tokens": 4000 if model == PERPLEXITY_REASONING_MODEL else 2000,  # More tokens for reasoning model
                "temperature": 0.7,
                "top_p": 0.9,
                "return_images": False,
                "return_related_questions": True,
                "stream": False
            }

            # Set up headers with API key
            headers = {
                "Authorization": f"Bearer {self.perplexity_api_key}",
                "Content-Type": "application/json"
            }

            # Make the API request
            response = requests.post(
                self.perplexity_url,
                json=payload,
                headers=headers
            )

            # Check if request was successful
            response.raise_for_status()
            result = response.json()

            # Format the results
            formatted_results = []
            
            # Add model info
            formatted_results.append(f"Model: {model}")
            
            # Add the main analysis
            if "choices" in result and result["choices"]:
                main_response = result["choices"][0]["message"]["content"]
                formatted_results.append(f"\nAnalysis:\n{main_response}\n")
            
            # Add related questions if available
            if result.get("related_questions"):
                formatted_results.append("\nRelated Questions:")
                for question in result["related_questions"]:
                    formatted_results.append(f"- {question}")
            
            # Add any additional metadata
            if "usage" in result:
                formatted_results.append("\nUsage Information:")
                for key, value in result["usage"].items():
                    formatted_results.append(f"- {key}: {value}")
            
            return "\n\n".join(formatted_results)
            
        except Exception as e:
            return f"Error performing deep research: {str(e)}. Parameters sent: {payload}" 