from typing import List, Dict, Any
from datetime import datetime
import pytz

class TimeTools:
    @property
    def tools_schema(self) -> List[Dict[str, Any]]:
        """Return the schema for time operation tools"""
        return [
            {
                "name": "get_time_in_timezone",
                "description": "Get the current time in a specific timezone",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "The timezone to get time for (e.g. 'America/New_York', 'Europe/London', 'Asia/Tokyo')",
                        },
                        "format": {
                            "type": "string",
                            "description": "Optional datetime format string (default: '%Y-%m-%d %H:%M:%S %Z')",
                        }
                    },
                    "required": ["timezone"],
                },
            },
            {
                "name": "list_timezones",
                "description": "List all available timezone names",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filter": {
                            "type": "string",
                            "description": "Optional filter string to search for specific timezones",
                        }
                    }
                },
            }
        ]

    def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Execute a time tool by name with given arguments"""
        if tool_name == "get_time_in_timezone":
            return self.get_time_in_timezone(
                tool_args["timezone"],
                tool_args.get("format", "%Y-%m-%d %H:%M:%S %Z")
            )
        elif tool_name == "list_timezones":
            return self.list_timezones(tool_args.get("filter", ""))
        else:
            return f"Unknown tool: {tool_name}"

    def get_time_in_timezone(self, timezone: str, format: str = "%Y-%m-%d %H:%M:%S %Z") -> str:
        """Get current time in specified timezone"""
        try:
            tz = pytz.timezone(timezone)
            current_time = datetime.now(tz)
            return current_time.strftime(format)
        except pytz.exceptions.UnknownTimeZoneError:
            return f"Unknown timezone: {timezone}. Use list_timezones to see available options."

    def list_timezones(self, filter: str = "") -> str:
        """List all available timezones, optionally filtered by search string"""
        all_timezones = pytz.all_timezones
        if filter:
            filtered_zones = [tz for tz in all_timezones if filter.lower() in tz.lower()]
            return "\n".join(filtered_zones) if filtered_zones else f"No timezones found matching '{filter}'"
        return "\n".join(all_timezones) 