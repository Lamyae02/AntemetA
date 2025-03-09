import os

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Whisper model settings
# Options: "tiny", "base", "small", "medium", "large"
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")