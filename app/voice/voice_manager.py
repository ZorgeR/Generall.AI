import json
from pathlib import Path
from typing import Dict, Optional

class VoiceManager:
    def __init__(self):
        self.voices_path = Path(__file__).parent / "voices.json"
        self.config_path = Path(__file__).parent / "config.json"
        self.voices = self._load_voices()
        self.config = self._load_config()
    
    def _load_voices(self) -> Dict:
        """Load available voices"""
        with open(self.voices_path, 'r') as f:
            return {voice["name"]: voice["id"] for voice in json.load(f)}
    
    def _load_config(self) -> Dict:
        """Load voice configuration"""
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def _save_config(self):
        """Save voice configuration"""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get_available_voices(self) -> Dict[str, str]:
        """Get list of available voices"""
        return self.voices
    
    def get_user_voice(self, user_id: str) -> str:
        """Get voice ID for a user"""
        return self.config["users"].get(user_id, self.config["default_voice"])
    
    def set_user_voice(self, user_id: str, voice_name: str) -> bool:
        """Set voice for a user"""
        if voice_name not in self.voices:
            return False
        
        self.config["users"][user_id] = self.voices[voice_name]
        self._save_config()
        return True
    
    def get_voice_name(self, voice_id: str) -> Optional[str]:
        """Get voice name by ID"""
        for name, vid in self.voices.items():
            if vid == voice_id:
                return name
        return None 