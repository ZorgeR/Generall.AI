import os
import logging
import json
from typing import Dict, Any, List, Optional

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class SystemTools:
    """
    Tools for system administration tasks.
    These tools allow for package management, service control, and system monitoring.
    All operations are executed in a secure container to prevent system compromise.
    """
    
    def __init__(self, user_id: str):
        """
        Initialize the system tools.
        
        Args:
            user_id: User ID for tracking and security
        """
        self.user_id = user_id
        logger.info(f"SystemTools initialized for user {user_id}")
    
    @property
    def tools_schema(self) -> List[Dict[str, Any]]:
        """
        Get the schema for the system tools.
        
        Returns:
            List of tool schemas
        """
        return [
            {
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
            },
            {
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
            },
            {
                "name": "check_system_info",
                "description": "Get information about the system environment in the secure container.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "info_type": {
                            "type": "string",
                            "description": "Type of information to retrieve (os, cpu, memory, disk, network, all)",
                            "enum": ["os", "cpu", "memory", "disk", "network", "all"],
                            "default": "all"
                        }
                    },
                    "required": [],
                },
            },
            {
                "name": "manage_service",
                "description": "Manage a system service in the secure container.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "service_name": {
                            "type": "string",
                            "description": "Name of the service to manage.",
                        },
                        "action": {
                            "type": "string",
                            "description": "Action to perform on the service.",
                            "enum": ["start", "stop", "restart", "status"],
                            "default": "status"
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds (default: 60)",
                            "default": 60
                        }
                    },
                    "required": ["service_name", "action"],
                },
            },
            {
                "name": "monitor_process",
                "description": "Monitor a process or list running processes in the secure container.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "process_name": {
                            "type": "string",
                            "description": "Name of the process to monitor. Leave empty to list all processes.",
                            "default": ""
                        },
                        "show_details": {
                            "type": "boolean",
                            "description": "Whether to show detailed information about the process.",
                            "default": False
                        }
                    },
                    "required": [],
                },
            },
            {
                "name": "network_diagnostics",
                "description": "Run network diagnostics in the secure container.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "Target host or IP address for diagnostics.",
                            "default": "localhost"
                        },
                        "diagnostic_type": {
                            "type": "string",
                            "description": "Type of diagnostic to run.",
                            "enum": ["ping", "traceroute", "dns", "port_scan", "all"],
                            "default": "ping"
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds (default: 60)",
                            "default": 60
                        }
                    },
                    "required": ["target"],
                },
            },
            {
                "name": "list_installed_packages",
                "description": "List all packages that have been installed for this user.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "remove_package",
                "description": "Remove a package from the user's installed packages list.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "package_name": {
                            "type": "string",
                            "description": "The name of the package to remove.",
                        }
                    },
                    "required": ["package_name"],
                },
            }
        ]
    
    def install_package(self, package_name: str, timeout: int = 300) -> str:
        """
        Install a package in the secure container.
        
        The package will be installed in the current container and also saved to a list
        of user-installed packages. Future container operations for this user will
        automatically install all packages from this list before executing the command.
        This provides persistence for installed packages across container runs.
        
        Args:
            package_name: Name of the package to install
            timeout: Timeout in seconds
            
        Returns:
            Installation output
        """
        logger.info(f"Installing package {package_name} for user {self.user_id}")
        
        # This will be replaced by the secure wrapper
        script = f"""#!/bin/bash
set -e
echo "Installing package: {package_name}"
sudo apt-get update
sudo apt-get install -y {package_name}
if [ $? -eq 0 ]; then
    echo "Package {package_name} installed successfully"
    exit 0
else
    echo "Failed to install package {package_name}"
    exit 1
fi
"""
        return f"Package {package_name} installation requested"
    
    def run_shell_script(self, script_content: str, timeout: int = 60, network_enabled: bool = False) -> str:
        """
        Run a shell script in a secure container.
        
        Args:
            script_content: Content of the shell script
            timeout: Timeout in seconds
            network_enabled: Whether to enable network access for the container
            
        Returns:
            Script execution output
        """
        logger.info(f"Running shell script for user {self.user_id}")
        
        # This will be replaced by the secure wrapper
        return "Shell script execution requested"
    
    def check_system_info(self, info_type: str = "all") -> str:
        """
        Get information about the system environment in the secure container.
        
        Args:
            info_type: Type of information to retrieve (os, cpu, memory, disk, network, all)
            
        Returns:
            System information
        """
        logger.info(f"Checking system info ({info_type}) for user {self.user_id}")
        
        script_content = """#!/bin/bash
echo "System Information:"
echo "==================="

if [[ "$1" == "os" || "$1" == "all" ]]; then
    echo -e "\nOS Information:"
    echo "--------------"
    cat /etc/os-release
    echo -e "\nKernel Information:"
    uname -a
fi

if [[ "$1" == "cpu" || "$1" == "all" ]]; then
    echo -e "\nCPU Information:"
    echo "--------------"
    lscpu | grep -E 'Model name|Socket|Core|Thread|CPU MHz'
fi

if [[ "$1" == "memory" || "$1" == "all" ]]; then
    echo -e "\nMemory Information:"
    echo "-----------------"
    free -h
fi

if [[ "$1" == "disk" || "$1" == "all" ]]; then
    echo -e "\nDisk Information:"
    echo "---------------"
    df -h
fi

if [[ "$1" == "network" || "$1" == "all" ]]; then
    echo -e "\nNetwork Information:"
    echo "------------------"
    ip addr
    echo -e "\nNetwork Connections:"
    ss -tuln
fi
"""
        
        # This will be replaced by the secure wrapper
        return f"System information ({info_type}) requested"
    
    def manage_service(self, service_name: str, action: str = "status", timeout: int = 60) -> str:
        """
        Manage a system service in the secure container.
        
        Args:
            service_name: Name of the service to manage
            action: Action to perform on the service (start, stop, restart, status)
            timeout: Timeout in seconds
            
        Returns:
            Service management output
        """
        logger.info(f"Managing service {service_name} ({action}) for user {self.user_id}")
        
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
        
        # This will be replaced by the secure wrapper
        return f"Service {service_name} {action} requested"
    
    def monitor_process(self, process_name: str = "", show_details: bool = False) -> str:
        """
        Monitor a process or list running processes in the secure container.
        
        Args:
            process_name: Name of the process to monitor. Leave empty to list all processes.
            show_details: Whether to show detailed information about the process.
            
        Returns:
            Process monitoring output
        """
        logger.info(f"Monitoring process {process_name} for user {self.user_id}")
        
        script_content = """#!/bin/bash
echo "Process Monitoring:"
echo "=================="

if [[ -z "$1" ]]; then
    echo -e "\nRunning Processes:"
    ps aux | head -n 20
else
    echo -e "\nProcesses matching '$1':"
    if [[ "$2" == "true" ]]; then
        ps aux | grep "$1" | grep -v grep
        echo -e "\nDetailed information:"
        pgrep -f "$1" | xargs -r ps -o pid,ppid,user,%cpu,%mem,vsz,rss,tty,stat,start,time,command -p
    else
        ps aux | grep "$1" | grep -v grep
    fi
fi
"""
        
        # This will be replaced by the secure wrapper
        return f"Process monitoring for {process_name or 'all processes'} requested"
    
    def network_diagnostics(self, target: str = "localhost", diagnostic_type: str = "ping", timeout: int = 60) -> str:
        """
        Run network diagnostics in the secure container.
        
        Args:
            target: Target host or IP address for diagnostics
            diagnostic_type: Type of diagnostic to run (ping, traceroute, dns, port_scan, all)
            timeout: Timeout in seconds
            
        Returns:
            Network diagnostics output
        """
        logger.info(f"Running network diagnostics ({diagnostic_type}) for target {target} for user {self.user_id}")
        
        script_content = """#!/bin/bash
echo "Network Diagnostics:"
echo "==================="
TARGET="$1"
TYPE="$2"

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
        
        # This will be replaced by the secure wrapper
        return f"Network diagnostics ({diagnostic_type}) for {target} requested"
    
    def list_installed_packages(self) -> str:
        """
        List all packages that have been installed for this user.
        
        These packages are automatically installed in each container
        before executing any command.
        
        Returns:
            List of installed packages
        """
        logger.info(f"Listing installed packages for user {self.user_id}")
        
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
dpkg-query -W -f='${Status} ${Package}\n' | grep "^install ok installed" | cut -d" " -f4 | grep -v "^lib" | head -n 20
echo "... (showing only first 20 system packages)"
"""
        
        # This will be replaced by the secure wrapper
        return "Listing installed packages"
    
    def remove_package(self, package_name: str) -> str:
        """
        Remove a package from the user's installed packages list.
        
        This doesn't uninstall the package from the current container,
        but prevents it from being automatically installed in future containers.
        
        Args:
            package_name: Name of the package to remove
            
        Returns:
            Removal result
        """
        logger.info(f"Removing package {package_name} from user {self.user_id}'s list")
        
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
grep -v "^{package_name}$" /home/runner/workspace/installed_packages.txt > "$TEMP_FILE"

# Replace the original file
mv "$TEMP_FILE" /home/runner/workspace/installed_packages.txt

echo "Package {package_name} removed from installed packages list."
"""
        
        # This will be replaced by the secure wrapper
        return f"Package {package_name} removal requested"
    
    def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """
        Execute a system tool.
        
        Args:
            tool_name: Name of the tool to execute
            tool_args: Tool arguments
            
        Returns:
            Tool execution output
        """
        logger.info(f"Executing system tool {tool_name} for user {self.user_id}")
        
        if tool_name == "install_package":
            return self.install_package(
                tool_args["package_name"],
                tool_args.get("timeout", 300)
            )
        elif tool_name == "list_installed_packages":
            return self.list_installed_packages()
        elif tool_name == "remove_package":
            return self.remove_package(
                tool_args["package_name"]
            )
        elif tool_name == "run_shell_script":
            return self.run_shell_script(
                tool_args["script_content"],
                tool_args.get("timeout", 60),
                tool_args.get("network_enabled", False)
            )
        elif tool_name == "check_system_info":
            return self.check_system_info(
                tool_args.get("info_type", "all")
            )
        elif tool_name == "manage_service":
            return self.manage_service(
                tool_args["service_name"],
                tool_args.get("action", "status"),
                tool_args.get("timeout", 60)
            )
        elif tool_name == "monitor_process":
            return self.monitor_process(
                tool_args.get("process_name", ""),
                tool_args.get("show_details", False)
            )
        elif tool_name == "network_diagnostics":
            return self.network_diagnostics(
                tool_args.get("target", "localhost"),
                tool_args.get("diagnostic_type", "ping"),
                tool_args.get("timeout", 60)
            )
        else:
            return f"Unknown tool: {tool_name}" 