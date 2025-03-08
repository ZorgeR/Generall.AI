from typing import List, Dict, Any
import subprocess
import os
import telegram
from dotenv import load_dotenv

load_dotenv()

class TerminalTools:
    def __init__(self, user_id: str, telegram_update: telegram.Update):
        self.user_id = user_id
        self.telegram_update = telegram_update

    @property
    def tools_schema(self) -> List[Dict[str, Any]]:
        """Return the schema for terminal operation tools"""

        return [
            {
                "name": "run_command",
                "description": "Execute a terminal/shell command and return the output.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The command to execute.",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds (default: 120)",
                            "default": 120
                        }
                    },
                    "required": ["command"],
                },
            }
        ]

    def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Execute a terminal tool by name with given arguments"""
        if tool_name == "run_command":
            return self.run_command(
                tool_args["command"],
                tool_args.get("timeout", 30)
            )
        else:
            return f"Unknown tool: {tool_name}"

    def run_command(self, command: str, timeout: int = 30) -> str:
        """Execute a terminal command and return the output"""

        try:
            # Create a new process to run the command
            process = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            output = []
            output.append(f"\nCommand: {command}")
            if process.stdout:
                output.append(f"Output:\n{process.stdout}")
            if process.stderr:
                output.append(f"Errors:\n{process.stderr}")
            if process.returncode != 0:
                output.append(f"Exit code: {process.returncode}")
                
            return "\n".join(output) if output else "Command completed successfully with no output"
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout} seconds"
        except Exception as e:
            return f"Error executing command: {str(e)}" 