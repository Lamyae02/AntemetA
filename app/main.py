from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
import os
import tempfile
from pathlib import Path

from app.transcription import transcribe_audio

app = FastAPI(
    title="Speech-to-Text API",
    description="API for transcribing .wav audio files to text",
    version="1.0.0"
)

@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Serve the home page with information about the API and a button to the documentation
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>API de Transcription Audio</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 0;
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
            }
            .container {
                max-width: 800px;
                background-color: white;
                padding: 2rem;
                border-radius: 10px;
                box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
                margin: 2rem;
            }
            header {
                text-align: center;
                margin-bottom: 2rem;
            }
            h1 {
                color: #1a365d;
                margin-bottom: 0.5rem;
            }
            .tagline {
                color: #4a5568;
                font-size: 1.2rem;
                margin-bottom: 1.5rem;
            }
            .author {
                font-style: italic;
                color: #718096;
                margin-bottom: 2rem;
            }
            .description {
                margin-bottom: 2rem;
                color: #2d3748;
            }
            .features {
                margin-bottom: 2rem;
            }
            .feature-item {
                display: flex;
                align-items: flex-start;
                margin-bottom: 1rem;
            }
            .feature-icon {
                margin-right: 1rem;
                color: #4299e1;
                font-size: 1.2rem;
                font-weight: bold;
            }
            .btn-container {
                text-align: center;
                margin-top: 2rem;
            }
            .btn {
                display: inline-block;
                background-color: #3182ce;
                color: white;
                padding: 12px 24px;
                border-radius: 5px;
                text-decoration: none;
                font-weight: bold;
                transition: background-color 0.3s;
            }
            .btn:hover {
                background-color: #2b6cb0;
            }
            footer {
                text-align: center;
                margin-top: 2rem;
                color: #4a5568;
                font-size: 0.9rem;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>API de Transcription Audio</h1>
                <p class="tagline">Convertissez facilement vos fichiers audio en texte</p>
                <p class="author">Développé par Lamyae Tabli</p>
            </header>
            
            <section class="description">
                <p>
                    Bienvenue sur notre service de transcription audio basé sur l'intelligence artificielle.
                    Cette API REST vous permet de convertir vos fichiers audio au format .wav en texte
                    avec une grande précision grâce au modèle Whisper d'OpenAI.
                </p>
            </section>
            
            <section class="features">
                <h2>Caractéristiques</h2>
                <div class="feature-item">
                    <span class="feature-icon">✓</span>
                    <div>
                        <strong>Haute précision</strong>: Utilise le modèle Whisper d'OpenAI pour une transcription de qualité professionnelle.
                    </div>
                </div>
                <div class="feature-item">
                    <span class="feature-icon">✓</span>
                    <div>
                        <strong>API Simple</strong>: Interface REST facile à utiliser avec un seul endpoint pour la transcription.
                    </div>
                </div>
                <div class="feature-item">
                    <span class="feature-icon">✓</span>
                    <div>
                        <strong>Déploiement facile</strong>: Solution conteneurisée avec Docker pour un déploiement rapide et fiable.
                    </div>
                </div>
            </section>
            
            <div class="btn-container">
                <a href="/docs" class="btn">Essayer l'API de Transcription</a>
            </div>
        </div>
        
        <footer>
            &copy; 2025 API de Transcription Audio - Tous droits réservés
        </footer>
    </body>
    </html>
    """
    return html_content

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