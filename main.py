from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List
import tempfile
import os
import librosa
import uvicorn
from Extraction.emotion_recognizer import EmotionRecognizer  

# Initialize FastAPI app
app = FastAPI(
    title="Speech Emotion Recognition API",
    description="API for recognizing emotions in speech using Wav2Vec2",
    version="1.0.0"
)

# Allow CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load EmotionRecognizer once when the server starts
recognizer = EmotionRecognizer(model_path="E:/Documents/Sem_6/PROJECT/Speech Emotion Recognisation/Backend/emotion_model.pth")

# =================== Response Models ===================

class EmotionResponse(BaseModel):
    emotions: Dict[str, float]
    predominant_emotion: str
    confidence: float

class FormatResponse(BaseModel):
    supported_formats: List[str]

class ModelInfoResponse(BaseModel):
    emotions: List[str]
    is_trained: bool

# =================== Routes ===================

@app.get("/")
async def root():
    return {
        "name": "Speech Emotion Recognition API",
        "version": "1.0.0",
        "description": "API for recognizing emotions in speech",
        "endpoints": [
            {"path": "/", "method": "GET", "description": "API info"},
            {"path": "/formats", "method": "GET", "description": "Supported audio formats"},
            {"path": "/model-info", "method": "GET", "description": "Info about the emotion model"},
            {"path": "/analyze-audio", "method": "POST", "description": "Analyze an uploaded audio file"},
        ]
    }

@app.get("/formats", response_model=FormatResponse)
async def formats():
    return {"supported_formats": ["wav", "mp3", "flac", "ogg", "m4a"]}

@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    return {
        "emotions": recognizer.labels,
        "is_trained": True
    }

@app.post("/analyze-audio", response_model=EmotionResponse)
async def analyze_audio(audio_file: UploadFile = File(...)):
    # Validate file format
    file_extension = os.path.splitext(audio_file.filename)[1][1:].lower()
    if file_extension not in ["wav", "mp3", "flac", "ogg", "m4a"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file_extension}"
        )

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
        content = await audio_file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name

    try:
        # Load audio using librosa
        waveform, sr = librosa.load(temp_file_path, sr=16000)

        # Predict emotion
        predictions = recognizer.predict(waveform, sr)

        # Get top emotion
        top_emotion = max(predictions.items(), key=lambda x: x[1])
        return {
            "emotions": predictions,
            "predominant_emotion": top_emotion[0],
            "confidence": round(top_emotion[1], 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# =================== Run the app ===================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
