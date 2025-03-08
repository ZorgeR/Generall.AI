import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from .container_manager import ContainerManager

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class SecureToolWrapper:
    """
    Wrapper for tools that need to be run in a secure container.
    This class intercepts tool calls and runs them in a secure container.
    """
    
    # List of tool types that should be run in a secure container
    SECURE_TOOL_TYPES = [
        "terminal_tools",  # Terminal commands
        "code_tools",      # Code execution
        "file_ops",        # File operations
        "image_tools",     # Image generation and manipulation
        "search_tools",    # Web search and file search operations
        "embeddings",      # Vector embeddings operations
        "system_tools"     # System administration tools
    ]
    
    # List of specific tool methods that should be run in a secure container
    SECURE_TOOL_METHODS = {
        # Terminal operations
        "run_command",     # Terminal command execution
        
        # Code operations
        "execute_python",  # Python code execution
        
        # File operations
        "list_files",      # File listing
        "read_file",       # File reading
        "create_file",     # File creation
        "delete_file",     # File deletion
        "create_directory", # Directory creation
        "delete_directory", # Directory deletion
        
        # Search operations
        "memory_search",   # Search in memory files
        
        # Image operations
        "_generate_image", # Image generation
        
        # Embedding operations
        "add_conversation", # Add conversation to vector store
        "search_conversations", # Search in vector store
        "clear",           # Clear vector store
        
        # System operations
        "install_package", # Install a package
        "run_shell_script" # Run a shell script
    }
    
    def __init__(self, base_data_path="./data"):
        """
        Initialize the secure tool wrapper.
        
        Args:
            base_data_path: Base path for user data directories
        """
        self.container_manager = ContainerManager(base_data_path)
        logger.info("Secure tool wrapper initialized")
    
    def needs_secure_execution(self, tool_type: str, method_name: str) -> bool:
        """
        Check if a tool method needs to be run in a secure container.
        
        Args:
            tool_type: Type of the tool (e.g., terminal_tools, code_tools)
            method_name: Name of the method to check
            
        Returns:
            True if the method needs to be run in a secure container, False otherwise
        """
        # Check if the tool type is in the list of secure tool types
        if any(secure_type in tool_type.lower() for secure_type in self.SECURE_TOOL_TYPES):
            return True
        
        # Check if the method name is in the list of secure tool methods
        if method_name in self.SECURE_TOOL_METHODS:
            return True
        
        return False
    
    def wrap_terminal_command(self, user_id: str, command: str, timeout: int = 60, network_enabled: bool = False) -> str:
        """
        Wrap a terminal command to run in a secure container.
        
        Args:
            user_id: User ID
            command: Command to run
            timeout: Timeout in seconds
            network_enabled: Whether to enable network access for the container
            
        Returns:
            Command output
        """
        logger.info(f"Running terminal command in secure container for user {user_id}: {command}")
        return self.container_manager.run_command(user_id, command, timeout, network_enabled)
    
    def wrap_python_execution(self, user_id: str, code: str, timeout: int = 60, network_enabled: bool = False) -> str:
        """
        Wrap Python code execution to run in a secure container.
        
        Args:
            user_id: User ID
            code: Python code to run
            timeout: Timeout in seconds
            network_enabled: Whether to enable network access for the container
            
        Returns:
            Code execution output
        """
        logger.info(f"Running Python code in secure container for user {user_id}")
        return self.container_manager.run_python_code(user_id, code, timeout, network_enabled)
    
    def wrap_file_operation(self, user_id: str, operation: str, args: Dict[str, Any], timeout: int = 30) -> str:
        """
        Wrap a file operation to run in a secure container.
        
        Args:
            user_id: User ID
            operation: Operation name (list_files, read_file, etc.)
            args: Operation arguments
            timeout: Timeout in seconds
            
        Returns:
            Operation result
        """
        logger.info(f"Running file operation {operation} in secure container for user {user_id}")
        return self.container_manager.run_file_operation(user_id, operation, args, timeout)
    
    def wrap_search_operation(self, user_id: str, operation: str, args: Dict[str, Any], timeout: int = 60) -> str:
        """
        Wrap a search operation to run in a secure container.
        
        Args:
            user_id: User ID
            operation: Operation name (memory_search, etc.)
            args: Operation arguments
            timeout: Timeout in seconds
            
        Returns:
            Operation result
        """
        logger.info(f"Running search operation {operation} in secure container for user {user_id}")
        
        # Create a Python script to perform the search operation
        script = f"""
import os
import json
import sys
from pathlib import Path
import re

def memory_search(query, case_sensitive=False):
    base_path = Path('.')
    results = []
    
    # Search through all text files in the directory
    for file_path in base_path.glob('**/*.txt'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Perform search based on case sensitivity
            if case_sensitive:
                matches = re.finditer(re.escape(query), content)
            else:
                matches = re.finditer(re.escape(query), content, re.IGNORECASE)
            
            # Extract matches with context
            for match in matches:
                start = max(0, match.start() - 100)
                end = min(len(content), match.end() + 100)
                context = content[start:end]
                
                results.append({{
                    "file": str(file_path),
                    "context": context,
                    "position": match.start()
                }})
        except Exception as e:
            results.append({{
                "file": str(file_path),
                "error": str(e)
            }})
    
    return results

# Parse arguments
operation = "{operation}"
args = {json.dumps(args)}

# Execute the operation
if operation == "memory_search":
    result = memory_search(
        args.get("query", ""),
        args.get("case_sensitive", False)
    )
else:
    result = f"Unknown operation: {{operation}}"

# Print the result as JSON
print(json.dumps(result))
"""
        
        return self.container_manager.run_python_code(user_id, script, timeout)
    
    def wrap_image_operation(self, user_id: str, operation: str, args: Dict[str, Any], timeout: int = 120) -> str:
        """
        Wrap an image operation to run in a secure container.
        
        Args:
            user_id: User ID
            operation: Operation name (_generate_image, etc.)
            args: Operation arguments
            timeout: Timeout in seconds
            
        Returns:
            Operation result
        """
        logger.info(f"Running image operation {operation} in secure container for user {user_id}")
        
        # For image generation, we'll need to use the OpenAI API which requires network access
        # This should be handled by the main application, not in the secure container
        return f"Image operations like {operation} should be handled by the main application with proper API access"
    
    def wrap_embedding_operation(self, user_id: str, operation: str, args: Dict[str, Any], timeout: int = 60) -> str:
        """
        Wrap an embedding operation to run in a secure container.
        
        Args:
            user_id: User ID
            operation: Operation name (add_conversation, search_conversations, clear)
            args: Operation arguments
            timeout: Timeout in seconds
            
        Returns:
            Operation result
        """
        logger.info(f"Running embedding operation {operation} in secure container for user {user_id}")
        
        # For embedding operations, we'll need to use the OpenAI API which requires network access
        # This should be handled by the main application, not in the secure container
        return f"Embedding operations like {operation} should be handled by the main application with proper API access"
    
    def wrap_shell_script(self, user_id: str, script_content: str, timeout: int = 60, network_enabled: bool = False) -> str:
        """
        Wrap a shell script to run in a secure container.
        
        Args:
            user_id: User ID
            script_content: Content of the shell script
            timeout: Timeout in seconds
            network_enabled: Whether to enable network access for the container
            
        Returns:
            Script execution output
        """
        logger.info(f"Running shell script in secure container for user {user_id}")
        return self.container_manager.run_shell_script(user_id, script_content, timeout, network_enabled)
    
    def wrap_package_installation(self, user_id: str, package_name: str, timeout: int = 300) -> str:
        """
        Wrap a package installation to run in a secure container.
        
        Args:
            user_id: User ID
            package_name: Name of the package to install
            timeout: Timeout in seconds
            
        Returns:
            Installation output
        """
        logger.info(f"Installing package {package_name} in secure container for user {user_id}")
        return self.container_manager.install_package(user_id, package_name, timeout)
    
    def cleanup(self):
        """Clean up all containers created by this wrapper"""
        return self.container_manager.cleanup_all_containers() 