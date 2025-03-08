from twilio.rest import Client
import os
from typing import Dict, Any
from pydantic import BaseModel

class SMSTools:
    def __init__(self):
        # Initialize Twilio client with credentials from environment variables
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.from_number = os.getenv("TWILIO_FROM_NUMBER")
        self.client = Client(self.account_sid, self.auth_token) if self.account_sid and self.auth_token else None
        # Twilio SMS length limits
        self.MAX_SMS_LENGTH = 1600 # max is 1600 and will be split into multiple segments

    @property
    def tools_schema(self):
        """Return the schema for available SMS tools"""
        return [
            {
                "name": "send_sms",
                "description": "Send an SMS message using Twilio. Messages longer than 1600 characters will be automatically split into multiple segments. The maximum total message length is 1600 characters.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "to_number": {
                                "type": "string",
                                "description": "The phone number to send the SMS to (in E.164 format, e.g., +1234567890)"
                            },
                            "message": {
                                "type": "string",
                                "description": "The message content to send. Maximum length is 1600 characters. Messages longer than 1600 characters will be automatically split into multiple segments."
                            }
                        },
                    "required": ["to_number", "message"]
                }
            }
        ]

    def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Execute the specified SMS tool"""
        if tool_name == "send_sms":
            return self._send_sms(tool_args["to_number"], tool_args["message"])
        return f"Unknown tool: {tool_name}"

    def _send_sms(self, to_number: str, message: str) -> str:
        """Send an SMS message using Twilio"""
        if not self.client:
            return "Error: Twilio credentials not configured"
        
        # Check message length
        if len(message) > self.MAX_SMS_LENGTH:
            return f"Error: Message exceeds maximum length of {self.MAX_SMS_LENGTH} characters. Current length: {len(message)}"
        
        try:
            # Send message using Twilio client
            message = self.client.messages.create(
                body=message,
                from_=self.from_number,
                to=to_number
            )
            segments = (len(message.body) + 159) // 160  # Calculate number of segments
            return f"SMS sent successfully. Message SID: {message.sid}. Message length: {len(message.body)} characters ({segments} segment{'s' if segments > 1 else ''})"
        except Exception as e:
            return f"Error sending SMS: {str(e)}" 
