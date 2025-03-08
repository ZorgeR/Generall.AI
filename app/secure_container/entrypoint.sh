#!/bin/bash
set -e

# This script serves as the entrypoint for the secure container
# It receives a command to execute and runs it in a controlled environment

# Print container information
echo "Secure container environment initialized"
echo "Available tools:"
echo "- Python $(python --version 2>&1)"
echo "- Git $(git --version 2>&1)"
echo "- Busybox $(busybox --help | head -n 1)"
echo "- SSH $(ssh -V 2>&1)"
echo "- And many other standard Linux utilities"
echo ""

# The command to run is passed as arguments to the script
if [ $# -eq 0 ]; then
    echo "Error: No command specified"
    echo "Usage: Pass a command to execute in the secure container"
    echo "Examples:"
    echo "  - Run a Python script: python script.py"
    echo "  - Run a shell command: ls -la"
    echo "  - Run multiple commands: bash -c 'cd /tmp && ls -la'"
    exit 1
fi

echo "Executing command: $@"
echo "Working directory: $(pwd)"
echo "-----------------------------------"

# Execute the command
exec "$@" 