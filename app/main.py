from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import tempfile
from pathlib import Path

from app.transcription import transcribe_audio

app = FastAPI(
    title="Speech-to-Text API",
    description="API for transcribing .wav audio files to text",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {"message": "Welcome to the Speech-to-Text API"}

@app.post("/transcription")
async def transcribe(file: UploadFile = File(...)):
    """
    Transcribe a .wav audio file to text.
    
    Parameters:
    - file: Audio file in .wav format
    
    Returns:
    - JSON with the transcribed text
    """
    # Check if file is provided
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file extension
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        # Write the uploaded file content to the temporary file
        temp_file.write(await file.read())
        temp_file_path = temp_file.name
    
    try:
        # Process the audio file using the transcription module
        transcription = transcribe_audio(temp_file_path)
        
        # Return the transcription as a JSON response
        return JSONResponse(content={"transcription": transcription})
    except Exception as e:
        # Handle exceptions
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)