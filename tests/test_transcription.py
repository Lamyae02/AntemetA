import os
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
import wave
import numpy as np

from app.main import app
from app.transcription import transcribe_audio

client = TestClient(app)

# Mock the transcribe_audio function to avoid actual model loading during tests
import app.transcription
original_transcribe_audio = app.transcription.transcribe_audio

def create_test_wav_file(duration=1, sample_rate=16000):
    """
    Create a simple .wav file for testing purposes
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_file_path = f.name
    
    # Generate silence
    silence = np.zeros(int(duration * sample_rate), dtype=np.int16)
    
    # Write the WAV file
    with wave.open(temp_file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(silence.tobytes())
    
    return temp_file_path

@pytest.fixture
def test_wav_file():
    """
    Fixture to create and clean up test WAV file
    """
    file_path = create_test_wav_file()
    yield file_path
    # Clean up
    if os.path.exists(file_path):
        os.unlink(file_path)

@pytest.fixture(autouse=True)
def mock_transcribe_audio(monkeypatch):
    """
    Mock the transcribe_audio function for testing
    """
    def mock_function(audio_path):
        # Instead of actually transcribing, return a fixed test string
        return "This is a test transcription."
    
    monkeypatch.setattr(app.transcription, "transcribe_audio", mock_function)
    yield
    # Restore the original function after the test
    monkeypatch.setattr(app.transcription, "transcribe_audio", original_transcribe_audio)

def test_root_endpoint():
    """
    Test the root endpoint of the API
    """
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_transcription_endpoint(test_wav_file):
    """
    Test the transcription endpoint with a test WAV file
    """
    with open(test_wav_file, "rb") as f:
        response = client.post(
            "/transcription",
            files={"file": ("test.wav", f, "audio/wav")}
        )
    
    assert response.status_code == 200
    assert "transcription" in response.json()
    assert response.json()["transcription"] == "This is a test transcription."

def test_transcription_endpoint_no_file():
    """
    Test the transcription endpoint without providing a file
    """
    response = client.post("/transcription")
    assert response.status_code == 422  # FastAPI's validation error

def test_transcription_endpoint_wrong_extension():
    """
    Test the transcription endpoint with a file that has wrong extension
    """
    with tempfile.NamedTemporaryFile(suffix=".txt") as f:
        f.write(b"This is not a WAV file")
        f.seek(0)
        response = client.post(
            "/transcription",
            files={"file": ("test.txt", f, "text/plain")}
        )
    
    assert response.status_code == 400
    assert "Only .wav files are supported" in response.json()["detail"]