import os
import sys
import inspect
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from .secure_tool_wrapper import SecureToolWrapper

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class ToolIntegrator:
    """
    Integrates the secure container wrapper with existing tools.
    This class patches the existing tool methods to run in a secure container.
    """
    
    def __init__(self, base_data_path="./data", host_data_path=None):
        """
        Initialize the tool integrator.
        
        Args:
            base_data_path: Base path for user data directories (internal path)
            host_data_path: Base path on the host machine for Docker volume mounting
        """
        self.secure_wrapper = SecureToolWrapper(base_data_path, host_data_path)
        logger.info("Tool integrator initialized")
    
    def patch_terminal_tools(self, terminal_tools_class):
        """
        Patch the terminal tools class to run commands in a secure container.
        
        Args:
            terminal_tools_class: The terminal tools class to patch
        """
        original_run_command = terminal_tools_class.run_command
        
        def secure_run_command(self, command, timeout=30, network_enabled=False):
            """Secure version of run_command that runs in a container"""
            logger.info(f"Intercepted terminal command for user {self.user_id}: {command}")
            return self.secure_wrapper.wrap_terminal_command(self.user_id, command, timeout, network_enabled)
        
        # Add secure wrapper to the class
        terminal_tools_class.secure_wrapper = self.secure_wrapper
        
        # Replace the original method with the secure version
        terminal_tools_class.run_command = secure_run_command
        logger.info("Successfully replaced TerminalTools.run_command with secure version")
        
        # Verify that the method was actually replaced
        if terminal_tools_class.run_command.__name__ == 'secure_run_command':
            logger.info("Verified that run_command was replaced with secure_run_command")
        else:
            logger.error(f"Failed to replace run_command! Current method: {terminal_tools_class.run_command.__name__}")
        
        # Add new methods to the terminal tools class
        def run_shell_script(self, script_content, timeout=60, network_enabled=False):
            """Run a shell script in a secure container"""
            logger.info(f"Running shell script for user {self.user_id}")
            return self.secure_wrapper.wrap_shell_script(self.user_id, script_content, timeout, network_enabled)
        
        def install_package(self, package_name, timeout=300):
            """Install a package in a secure container"""
            logger.info(f"Installing package {package_name} for user {self.user_id}")
            return self.secure_wrapper.wrap_package_installation(self.user_id, package_name, timeout)
        
        # Add the new methods to the class
        terminal_tools_class.run_shell_script = run_shell_script
        terminal_tools_class.install_package = install_package
        
        # Update the tools schema to include the new methods
        original_tools_schema = terminal_tools_class.tools_schema
        
        @property
        def enhanced_tools_schema(self):
            schema = original_tools_schema.fget(self)
            
            # Add run_shell_script to the schema
            schema.append({
                "name": "run_shell_script",
                "description": "Execute a shell script in a secure container. The script will be run with bash.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "script_content": {
                            "type": "string",
                            "description": "The content of the shell script to execute.",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds (default: 60)",
                            "default": 60
                        },
                        "network_enabled": {
                            "type": "boolean",
                            "description": "Whether to enable network access for the container (default: false)",
                            "default": False
                        }
                    },
                    "required": ["script_content"],
                },
            })
            
            # Add install_package to the schema
            schema.append({
                "name": "install_package",
                "description": "Install a package in the secure container using apt-get.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "package_name": {
                            "type": "string",
                            "description": "The name of the package to install.",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds (default: 300)",
                            "default": 300
                        }
                    },
                    "required": ["package_name"],
                },
            })
            
            return schema
        
        # Replace the original tools_schema property with the enhanced one
        terminal_tools_class.tools_schema = enhanced_tools_schema
        
        # Update the execute_tool method to handle the new tools
        original_execute_tool = terminal_tools_class.execute_tool
        
        def enhanced_execute_tool(self, tool_name, tool_args):
            if tool_name == "run_shell_script":
                return self.run_shell_script(
                    tool_args["script_content"],
                    tool_args.get("timeout", 60),
                    tool_args.get("network_enabled", False)
                )
            elif tool_name == "install_package":
                return self.install_package(
                    tool_args["package_name"],
                    tool_args.get("timeout", 300)
                )
            else:
                return original_execute_tool(self, tool_name, tool_args)
        
        # Replace the original execute_tool method with the enhanced one
        terminal_tools_class.execute_tool = enhanced_execute_tool
        
        logger.info("Terminal tools patched to use secure container with enhanced functionality")
    
    def patch_code_tools(self, code_tools_class):
        """
        Patch the code tools class to run Python code in a secure container.
        
        Args:
            code_tools_class: The code tools class to patch
        """
        original_execute_python = code_tools_class.execute_python
        
        def secure_execute_python(self, code, timeout=120, network_enabled=False):
            """Secure version of execute_python that runs in a container"""
            logger.info(f"Intercepted Python code execution for user {self.user_id}")
            return self.secure_wrapper.wrap_python_execution(self.user_id, code, timeout, network_enabled)
        
        # Add secure wrapper to the class
        code_tools_class.secure_wrapper = self.secure_wrapper
        
        # Replace the original method with the secure version
        code_tools_class.execute_python = secure_execute_python
        
        # Update the tools schema to include network_enabled parameter
        original_tools_schema = code_tools_class.tools_schema
        
        @property
        def enhanced_tools_schema(self):
            schema = original_tools_schema.fget(self)
            
            # Find the execute_python tool and add network_enabled parameter
            for tool in schema:
                if tool["name"] == "execute_python":
                    tool["input_schema"]["properties"]["network_enabled"] = {
                        "type": "boolean",
                        "description": "Whether to enable network access for the container (default: false)",
                        "default": False
                    }
            
            return schema
        
        # Replace the original tools_schema property with the enhanced one
        code_tools_class.tools_schema = enhanced_tools_schema
        
        # Update the execute_tool method to handle the network_enabled parameter
        original_execute_tool = code_tools_class.execute_tool
        
        def enhanced_execute_tool(self, tool_name, tool_args):
            if tool_name == "execute_python":
                return self.execute_python(
                    tool_args["code"],
                    tool_args.get("timeout", 120),
                    tool_args.get("network_enabled", False)
                )
            else:
                return original_execute_tool(self, tool_name, tool_args)
        
        # Replace the original execute_tool method with the enhanced one
        code_tools_class.execute_tool = enhanced_execute_tool
        
        logger.info("Code tools patched to use secure container with network option")
    
    def patch_file_operations(self, file_ops_class):
        """
        Patch the file operations class to run file operations in a secure container.
        
        Args:
            file_ops_class: The file operations class to patch
        """
        # Get all methods that need to be patched
        methods_to_patch = [
            method_name for method_name in dir(file_ops_class)
            if not method_name.startswith('_') and 
            callable(getattr(file_ops_class, method_name)) and
            method_name in self.secure_wrapper.SECURE_TOOL_METHODS
        ]
        
        # Create a dictionary to store original methods
        original_methods = {}
        
        for method_name in methods_to_patch:
            # Store the original method
            original_method = getattr(file_ops_class, method_name)
            original_methods[method_name] = original_method
            
            # Create a secure version of the method
            def create_secure_method(method_name, original_method):
                def secure_method(self, *args, **kwargs):
                    """Secure version of the method that runs in a container"""
                    logger.info(f"Intercepted file operation {method_name} for user {self.user_id}")
                    
                    # Extract arguments from the method call
                    method_args = {}
                    if method_name == "list_files":
                        method_args = {"path": kwargs.get("path", ".")}
                    elif method_name == "read_file":
                        method_args = {"filename": args[0] if args else kwargs.get("filename")}
                    elif method_name == "create_file":
                        method_args = {
                            "filename": args[0] if args else kwargs.get("filename"),
                            "content": args[1] if len(args) > 1 else kwargs.get("content", "")
                        }
                    elif method_name == "delete_file":
                        method_args = {"filename": args[0] if args else kwargs.get("filename")}
                    elif method_name == "create_directory":
                        method_args = {"dirname": args[0] if args else kwargs.get("dirname")}
                    elif method_name == "delete_directory":
                        method_args = {"dirname": args[0] if args else kwargs.get("dirname")}
                    
                    return self.secure_wrapper.wrap_file_operation(
                        self.user_id, method_name, method_args
                    )
                
                return secure_method
            
            # Replace the original method with the secure version
            setattr(file_ops_class, method_name, create_secure_method(method_name, original_method))
        
        # Add secure wrapper to the class
        file_ops_class.secure_wrapper = self.secure_wrapper
        
        logger.info(f"File operations patched to use secure container: {methods_to_patch}")
    
    def patch_search_tools(self, search_tools_class):
        """
        Patch the search tools class to run search operations in a secure container.
        
        Args:
            search_tools_class: The search tools class to patch
        """
        # Only patch memory_search as it operates on local files
        if hasattr(search_tools_class, "memory_search"):
            original_memory_search = search_tools_class.memory_search
            
            def secure_memory_search(self, query, case_sensitive=False):
                """Secure version of memory_search that runs in a container"""
                logger.info(f"Intercepted memory search for user {self.user_id}: {query}")
                return self.secure_wrapper.wrap_search_operation(
                    self.user_id, 
                    "memory_search", 
                    {"query": query, "case_sensitive": case_sensitive}
                )
            
            # Add secure wrapper to the class
            search_tools_class.secure_wrapper = self.secure_wrapper
            
            # Replace the original method with the secure version
            search_tools_class.memory_search = secure_memory_search
            
            logger.info("Search tools patched to use secure container")
    
    def patch_image_tools(self, image_tools_class):
        """
        Patch the image tools class to handle image operations securely.
        
        Args:
            image_tools_class: The image tools class to patch
        """
        # For image generation, we need API access, so we don't run it in the container
        # But we can still log the operations for security monitoring
        if hasattr(image_tools_class, "_generate_image"):
            original_generate_image = image_tools_class._generate_image
            
            async def secure_generate_image(self, prompt, size="1024x1024", quality="standard", caption="Here is your image"):
                """Secure version of _generate_image that logs the operation"""
                logger.info(f"Intercepted image generation for user {self.user_id}: {prompt}")
                # We don't run this in a container as it needs API access
                return await original_generate_image(self, prompt, size, quality, caption)
            
            # Replace the original method with the secure version
            image_tools_class._generate_image = secure_generate_image
            
            logger.info("Image tools patched for security monitoring")
    
    def patch_embeddings(self, embeddings_class):
        """
        Patch the embeddings class to handle embedding operations securely.
        
        Args:
            embeddings_class: The embeddings class to patch
        """
        # For embeddings, we need API access, so we don't run it in the container
        # But we can still log the operations for security monitoring
        methods_to_monitor = ["add_conversation", "search_conversations", "clear"]
        
        for method_name in methods_to_monitor:
            if hasattr(embeddings_class, method_name):
                original_method = getattr(embeddings_class, method_name)
                
                def create_secure_method(method_name, original_method):
                    def secure_method(self, *args, **kwargs):
                        """Secure version of the method that logs the operation"""
                        logger.info(f"Intercepted embedding operation {method_name} for user {self.user_id}")
                        # We don't run this in a container as it needs API access
                        return original_method(self, *args, **kwargs)
                    
                    return secure_method
                
                # Replace the original method with the secure version
                setattr(embeddings_class, method_name, create_secure_method(method_name, original_method))
        
        logger.info("Embeddings patched for security monitoring")
    
    def patch_system_tools(self, system_tools_class):
        """
        Patch the system tools class to run operations in a secure container.
        
        Args:
            system_tools_class: The system tools class to patch
        """
        # Add secure wrapper to the class
        system_tools_class.secure_wrapper = self.secure_wrapper
        
        # Patch install_package method
        original_install_package = system_tools_class.install_package
        
        def secure_install_package(self, package_name, timeout=300):
            """Secure version of install_package that runs in a container"""
            logger.info(f"Intercepted package installation for user {self.user_id}: {package_name}")
            return self.secure_wrapper.wrap_package_installation(self.user_id, package_name, timeout)
        
        system_tools_class.install_package = secure_install_package
        
        # Patch run_shell_script method
        original_run_shell_script = system_tools_class.run_shell_script
        
        def secure_run_shell_script(self, script_content, timeout=60, network_enabled=False):
            """Secure version of run_shell_script that runs in a container"""
            logger.info(f"Intercepted shell script execution for user {self.user_id}")
            return self.secure_wrapper.wrap_shell_script(self.user_id, script_content, timeout, network_enabled)
        
        system_tools_class.run_shell_script = secure_run_shell_script
        
        # Patch check_system_info method
        original_check_system_info = system_tools_class.check_system_info
        
        def secure_check_system_info(self, info_type="all"):
            """Secure version of check_system_info that runs in a container"""
            logger.info(f"Intercepted system info check for user {self.user_id}: {info_type}")
            
            script_content = f"""#!/bin/bash
echo "System Information:"
echo "==================="

if [[ "{info_type}" == "os" || "{info_type}" == "all" ]]; then
    echo -e "\nOS Information:"
    echo "--------------"
    cat /etc/os-release
    echo -e "\nKernel Information:"
    uname -a
fi

if [[ "{info_type}" == "cpu" || "{info_type}" == "all" ]]; then
    echo -e "\nCPU Information:"
    echo "--------------"
    lscpu | grep -E 'Model name|Socket|Core|Thread|CPU MHz'
fi

if [[ "{info_type}" == "memory" || "{info_type}" == "all" ]]; then
    echo -e "\nMemory Information:"
    echo "-----------------"
    free -h
fi

if [[ "{info_type}" == "disk" || "{info_type}" == "all" ]]; then
    echo -e "\nDisk Information:"
    echo "---------------"
    df -h
fi

if [[ "{info_type}" == "network" || "{info_type}" == "all" ]]; then
    echo -e "\nNetwork Information:"
    echo "------------------"
    ip addr
    echo -e "\nNetwork Connections:"
    ss -tuln
fi
"""
            return self.secure_wrapper.wrap_shell_script(self.user_id, script_content, 30, False)
        
        system_tools_class.check_system_info = secure_check_system_info
        
        # Patch manage_service method
        original_manage_service = system_tools_class.manage_service
        
        def secure_manage_service(self, service_name, action="status", timeout=60):
            """Secure version of manage_service that runs in a container"""
            logger.info(f"Intercepted service management for user {self.user_id}: {service_name} ({action})")
            
            script_content = f"""#!/bin/bash
echo "Service Management: {service_name} ({action})"
echo "==================="

if command -v systemctl &> /dev/null; then
    sudo systemctl {action} {service_name}
    sudo systemctl status {service_name}
elif command -v service &> /dev/null; then
    sudo service {service_name} {action}
    sudo service {service_name} status
else
    echo "No service management system found"
fi
"""
            return self.secure_wrapper.wrap_shell_script(self.user_id, script_content, timeout, False)
        
        system_tools_class.manage_service = secure_manage_service
        
        # Patch monitor_process method
        original_monitor_process = system_tools_class.monitor_process
        
        def secure_monitor_process(self, process_name="", show_details=False):
            """Secure version of monitor_process that runs in a container"""
            logger.info(f"Intercepted process monitoring for user {self.user_id}: {process_name}")
            
            script_content = f"""#!/bin/bash
echo "Process Monitoring:"
echo "=================="

if [[ -z "{process_name}" ]]; then
    echo -e "\nRunning Processes:"
    ps aux | head -n 20
else
    echo -e "\nProcesses matching '{process_name}':"
    if [[ "{str(show_details).lower()}" == "true" ]]; then
        ps aux | grep "{process_name}" | grep -v grep
        echo -e "\nDetailed information:"
        pgrep -f "{process_name}" | xargs -r ps -o pid,ppid,user,%cpu,%mem,vsz,rss,tty,stat,start,time,command -p
    else
        ps aux | grep "{process_name}" | grep -v grep
    fi
fi
"""
            return self.secure_wrapper.wrap_shell_script(
                self.user_id, 
                script_content, 
                30, 
                False
            )
        
        system_tools_class.monitor_process = secure_monitor_process
        
        # Patch network_diagnostics method
        original_network_diagnostics = system_tools_class.network_diagnostics
        
        def secure_network_diagnostics(self, target="localhost", diagnostic_type="ping", timeout=60):
            """Secure version of network_diagnostics that runs in a container"""
            logger.info(f"Intercepted network diagnostics for user {self.user_id}: {target} ({diagnostic_type})")
            
            script_content = f"""#!/bin/bash
echo "Network Diagnostics:"
echo "==================="
TARGET="{target}"
TYPE="{diagnostic_type}"

if [[ "$TYPE" == "ping" || "$TYPE" == "all" ]]; then
    echo -e "\nPing Test to $TARGET:"
    echo "-------------------"
    ping -c 4 "$TARGET"
fi

if [[ "$TYPE" == "traceroute" || "$TYPE" == "all" ]]; then
    echo -e "\nTraceroute to $TARGET:"
    echo "--------------------"
    if command -v traceroute &> /dev/null; then
        traceroute "$TARGET"
    else
        echo "traceroute not installed"
    fi
fi

if [[ "$TYPE" == "dns" || "$TYPE" == "all" ]]; then
    echo -e "\nDNS Lookup for $TARGET:"
    echo "---------------------"
    nslookup "$TARGET"
    dig "$TARGET"
fi

if [[ "$TYPE" == "port_scan" || "$TYPE" == "all" ]]; then
    echo -e "\nPort Scan for $TARGET:"
    echo "-------------------"
    nc -zv "$TARGET" 20-25 80 443 2>&1
fi
"""
            # Network diagnostics requires network access
            return self.secure_wrapper.wrap_shell_script(
                self.user_id, 
                script_content, 
                timeout, 
                True  # Enable network access
            )
        
        system_tools_class.network_diagnostics = secure_network_diagnostics
        
        # Patch list_installed_packages method
        original_list_installed_packages = system_tools_class.list_installed_packages
        
        def secure_list_installed_packages(self):
            """Secure version of list_installed_packages that runs in a container"""
            logger.info(f"Intercepted list installed packages for user {self.user_id}")
            
            script_content = """#!/bin/bash
echo "Installed Packages:"
echo "=================="

# Read the installed_packages.txt file if it exists
if [ -f "/home/runner/workspace/installed_packages.txt" ]; then
    echo "User-installed packages:"
    cat /home/runner/workspace/installed_packages.txt | sort
    echo ""
    
    # Check if these packages are actually installed
    echo "Verification of installed packages:"
    while read package; do
        if dpkg -l | grep -q "\\b$package\\b"; then
            echo "✓ $package - installed"
        else
            echo "✗ $package - not installed"
        fi
    done < /home/runner/workspace/installed_packages.txt
else
    echo "No user-installed packages found."
fi

echo ""
echo "System packages (pre-installed in the container):"
dpkg-query -W -f='${Status} ${Package}\\n' | grep "^install ok installed" | cut -d" " -f4 | grep -v "^lib" | head -n 20
echo "... (showing only first 20 system packages)"
"""
            return self.secure_wrapper.wrap_shell_script(self.user_id, script_content, 30, False)
        
        system_tools_class.list_installed_packages = secure_list_installed_packages
        
        # Patch remove_package method
        original_remove_package = system_tools_class.remove_package
        
        def secure_remove_package(self, package_name):
            """Secure version of remove_package that runs in a container"""
            logger.info(f"Intercepted remove package for user {self.user_id}: {package_name}")
            
            script_content = f"""#!/bin/bash
echo "Removing package from installed list: {package_name}"

# Check if the packages file exists
if [ ! -f "/home/runner/workspace/installed_packages.txt" ]; then
    echo "No installed packages list found."
    exit 0
fi

# Create a temporary file
TEMP_FILE=$(mktemp)

# Filter out the package
grep -v "^{package_name}$" /home/runner/workspace/installed_packages.txt > "$TEMP_FILE" || echo "Package not found in list"

# Replace the original file
mv "$TEMP_FILE" /home/runner/workspace/installed_packages.txt

echo "Package {package_name} removed from installed packages list."
"""
            return self.secure_wrapper.wrap_shell_script(self.user_id, script_content, 30, False)
        
        system_tools_class.remove_package = secure_remove_package
        
        logger.info("System tools patched to use secure container")
    
    def patch_all_tools(self, agent_module):
        """
        Patch all tools in the agent module to use secure containers.
        
        Args:
            agent_module: The agent module containing the tools
            
        Returns:
            True if all tools were patched successfully, False otherwise
        """
        try:
            logger.info("Starting to patch all tools to use secure containers")
            
            # Track success of patching each tool type
            patched_tools = {
                "terminal_tools": False,
                "code_tools": False,
                "file_ops": False,
                "search_tools": False,
                "image_tools": False,
                "embeddings": False,
                "system_tools": False
            }
            
            # Get the terminal tools class
            try:
                from importlib import import_module
                
                # Import the terminal tools module
                terminal_tools_module = import_module(".terminal_tools", agent_module.__name__)
                terminal_tools_class = terminal_tools_module.TerminalTools
                logger.info(f"Found TerminalTools class: {terminal_tools_class}")
                self.patch_terminal_tools(terminal_tools_class)
                patched_tools["terminal_tools"] = True
            except (AttributeError, ImportError) as e:
                logger.error(f"Error patching terminal tools: {str(e)}")
            
            # Get the code tools class
            try:
                code_tools_module = import_module(".code_tools", agent_module.__name__)
                code_tools_class = code_tools_module.CodeTools
                logger.info(f"Found CodeTools class: {code_tools_class}")
                self.patch_code_tools(code_tools_class)
                patched_tools["code_tools"] = True
            except (AttributeError, ImportError) as e:
                logger.error(f"Error patching code tools: {str(e)}")
            
            # Get the file operations class
            try:
                file_ops_module = import_module(".file_ops", agent_module.__name__)
                file_ops_class = file_ops_module.FileOperations
                logger.info(f"Found FileOperations class: {file_ops_class}")
                self.patch_file_operations(file_ops_class)
                patched_tools["file_ops"] = True
            except (AttributeError, ImportError) as e:
                logger.error(f"Error patching file operations: {str(e)}")
            
            # Get the search tools class
            try:
                search_tools_module = import_module(".search_tools", agent_module.__name__)
                search_tools_class = search_tools_module.SearchTools
                logger.info(f"Found SearchTools class: {search_tools_class}")
                self.patch_search_tools(search_tools_class)
                patched_tools["search_tools"] = True
            except (AttributeError, ImportError) as e:
                logger.warning(f"Could not patch search tools: {str(e)}")
            
            # Get the image tools class
            try:
                image_tools_module = import_module(".image_tools", agent_module.__name__)
                image_tools_class = image_tools_module.ImageTools
                logger.info(f"Found ImageTools class: {image_tools_class}")
                self.patch_image_tools(image_tools_class)
                patched_tools["image_tools"] = True
            except (AttributeError, ImportError) as e:
                logger.warning(f"Could not patch image tools: {str(e)}")
            
            # Get the embeddings class
            try:
                embeddings_module = import_module(".embeddings", agent_module.__name__)
                embeddings_class = embeddings_module.ConversationEmbeddings
                logger.info(f"Found ConversationEmbeddings class: {embeddings_class}")
                self.patch_embeddings(embeddings_class)
                patched_tools["embeddings"] = True
            except (AttributeError, ImportError) as e:
                logger.warning(f"Could not patch embeddings: {str(e)}")
            
            # Get the system tools class
            try:
                system_tools_module = import_module(".system_tools", agent_module.__name__)
                system_tools_class = system_tools_module.SystemTools
                logger.info(f"Found SystemTools class: {system_tools_class}")
                self.patch_system_tools(system_tools_class)
                patched_tools["system_tools"] = True
            except (AttributeError, ImportError) as e:
                logger.warning(f"Could not patch system tools: {str(e)}")
            
            # Check if any tools were patched successfully
            if any(patched_tools.values()):
                logger.info(f"Patched tools summary: {patched_tools}")
                return True
            else:
                logger.error("Failed to patch any tools!")
                return False
        except Exception as e:
            logger.error(f"Error in patch_all_tools: {str(e)}")
            return False 