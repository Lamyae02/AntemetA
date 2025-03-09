import whisper
import os
from pathlib import Path

# Load the model once at module import time to avoid reloading for each request
# Use "tiny" or "base" for faster but less accurate results, or "medium"/"large" for better accuracy
MODEL = None

def get_model():
    """
    Load the Whisper model if not already loaded
    """
    global MODEL
    if MODEL is None:
        # Use smaller model (tiny or base) for faster inference
        # or larger model (medium or large) for better accuracy
        MODEL = whisper.load_model("base")
    return MODEL

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe an audio file to text using OpenAI's Whisper model.
    
    Parameters:
    - audio_path: Path to the audio file
    
    Returns:
    - Transcribed text
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Get the model
    model = get_model()
    
    # Perform the transcription
    result = model.transcribe(audio_path)
    
    # Return the transcribed text
    return result["text"]