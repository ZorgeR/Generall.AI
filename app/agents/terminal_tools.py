from typing import List, Dict, Any
import subprocess
import os
from dotenv import load_dotenv

load_dotenv()

class TerminalTools:
    def __init__(self, user_id: str, sender=None):
        self.user_id = user_id
        self.sender = sender

    @property
    def tools_schema(self) -> List[Dict[str, Any]]:
        """Return the schema for terminal operation tools"""

        return [
            {
                "name": "run_command",
                "description": (
                    "Run a shell command in the user's persistent Linux sandbox session and return its "
                    "output (stdout and stderr together). The command runs with bash (bash -lc) in "
                    "/home/runner/workspace, the user's files, so pipes, &&, cd, globs and redirects work. "
                    "Each call is a new bash process (cd and variables do not carry over), but files, "
                    "installed packages and background processes stay while the session is open. The "
                    "sandbox has internet access; private networks are blocked. Long jobs: start them in "
                    "the background (nohup <command> > job.log 2>&1 &) and check job.log later."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The bash command line to run.",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Seconds before the command is stopped (default: 30, at most 3600)",
                            "default": 30
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