from typing import List, Dict, Any
import subprocess
import sys
import os

class CodeTools:
    def __init__(self, user_id: str = "default"):
        """
        Initialize CodeTools with user_id.
        
        Args:
            user_id: User ID for tracking code execution
        """
        self.user_id = user_id
    
    @property
    def tools_schema(self) -> List[Dict[str, Any]]:
        """Return the schema for code execution tools"""
        return [
            {
                "name": "execute_python",
                "description": "Execute Python code and return the output. Always store useful code in codebase folder.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds (default: 120)",
                            "default": 120
                        }
                    },
                    "required": ["code"],
                },
            }
        ]

    def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Execute a code tool by name with given arguments"""
        if tool_name == "execute_python":
            return self.execute_python(tool_args["code"])
        else:
            return f"Unknown tool: {tool_name}"

    def execute_python(self, code: str, timeout: int = 120) -> str:
        """Execute Python code and return the output"""
        try:
            # Add codebase directory to Python path for imports
            env = os.environ.copy()
            env["PYTHONPATH"] = os.path.join(os.getcwd()) + os.pathsep + env.get("PYTHONPATH", "")
            
            # Create a new process to run the code
            process = subprocess.run(
                [sys.executable, '-c', code],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env
            )
            
            if process.returncode == 0:
                return f"Output:\n{process.stdout}"
            else:
                return f"Error:\n{process.stderr}"
        except Exception as e:
            return f"Error executing code: {str(e)}" 