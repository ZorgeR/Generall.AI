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
        
        # NOTE: image generation, embeddings and memory_search run in the bot process (they need
        # API access or the user's memory files); only the names above are ever sandboxed.
        
        # System operations
        "install_package", # Install a package
        "run_shell_script" # Run a shell script
    }
    
    def __init__(self, base_data_path="./data", host_data_path=None):
        """
        Initialize the secure tool wrapper.
        
        Args:
            base_data_path: Base path for user data directories (internal path)
            host_data_path: Base path on the host machine for Docker volume mounting
        """
        self.container_manager = ContainerManager(base_data_path, host_data_path)
        logger.info("Secure tool wrapper initialized")
    
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
    
    def wrap_python_package_installation(self, user_id: str, package_name: str, timeout: int = 300) -> str:
        """
        Wrap a pip package installation (``wrap_package_installation(..., manager="pip")``).

        Args:
            user_id: User ID
            package_name: Name of the pip package to install
            timeout: Timeout in seconds

        Returns:
            Installation output
        """
        return self.wrap_package_installation(user_id, package_name, timeout=timeout, manager="pip")

    def wrap_package_installation(self, user_id: str, package_name: str, timeout: int = 300,
                                  manager: str = "apt") -> str:
        """
        Wrap a package installation: apt or pip, in the user's sandbox session, saved for replay.

        Args:
            user_id: User ID
            package_name: Package name(s), separated by spaces
            timeout: Timeout in seconds
            manager: "apt" (default) or "pip"

        Returns:
            Installation output
        """
        logger.info(f"Installing {manager} package {package_name} in secure container for user {user_id}")
        return self.container_manager.install_package(user_id, package_name, timeout=timeout, manager=manager)
    
    def cleanup(self):
        """Clean up all containers created by this wrapper"""
        return self.container_manager.cleanup_all_containers() 