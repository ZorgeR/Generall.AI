from typing import List, Dict, Any, Union
from tavily import TavilyClient
import requests
import os
from pathlib import Path
import re
import json

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
                "description": "Search through long term memory files for specific words or phrases",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The word or phrase to search for in memory files. Language specific. For best result, make a 2 search, on user language and English.", 
                        },
                        "case_sensitive": {
                            "type": "boolean",
                            "description": "Whether to perform case-sensitive search",
                            "default": False
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
                            "enum": ["sonar-reasoning-pro", "sonar-pro", "sonar"],
                            "default": "sonar"
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
                case_sensitive=tool_args.get("case_sensitive", False)
            )
        elif tool_name == "deep_research":
            return self.deep_research(
                query=tool_args["query"],
                model=tool_args.get("model", "sonar"),
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

    def memory_search(self, query: str, case_sensitive: bool = False) -> str:
        """Search through memory files for specific words or phrases.
        First searches for the complete phrase, then searches for individual words.
        
        Returns:
            str: JSON-formatted string containing:
                - full_match: list of file paths containing the complete phrase
                - word_match: list of dicts with word found and file path
        """
        try:
            # Determine search path based on user_id
            search_path = self.base_path
            
            # Compile search patterns
            flags = 0 if case_sensitive else re.IGNORECASE
            full_pattern = re.compile(re.escape(query), flags=flags)
            
            # Split query into individual words and create patterns
            words = query.split()
            word_patterns = {word: re.compile(re.escape(word), flags=flags) for word in words}
            
            # Initialize results
            full_matches = []
            word_matches = []
            
            # Search through files
            for file_path in search_path.glob("**/*"):
                if file_path.is_file() and file_path.suffix in ['.txt', '.json', '.md']:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            rel_path = str(file_path.relative_to(self.base_path))
                            
                            # Check for full phrase match
                            if full_pattern.search(content):
                                full_matches.append(rel_path)
                            
                            # Check for individual word matches
                            for word, pattern in word_patterns.items():
                                if pattern.search(content):
                                    word_matches.append({
                                        "word_find": word,
                                        "file_path": rel_path
                                    })
                    except Exception as e:
                        print(f"Error reading file {file_path}: {str(e)}")
                        continue
            
            # Format results
            result = {
                "full_match": full_matches,
                "word_match": word_matches
            }
            
            return str(result)
                
        except Exception as e:
            return f"Error performing memory search: {str(e)}"

    def deep_research(
        self,
        query: str,
        model: str = "sonar",
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
            if model == "sonar-reasoning-pro":
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
                "max_tokens": 4000 if model == "sonar-reasoning-pro" else 2000,  # More tokens for reasoning model
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