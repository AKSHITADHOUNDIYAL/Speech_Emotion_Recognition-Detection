from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import tempfile
import os
import librosa
import torch
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from emotion_detection import model, processor  # Directly import model and processor

# Initialize FastAPI app
app = FastAPI(
    title="Speech Emotion Recognition API",
    description="API for recognizing emotions in speech using Wav2Vec2",
    version="1.0.0"
)

# Add CORS middleware to allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow the frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# =================== Response Models ===================
class EmotionResponse(BaseModel):
    emotions: Dict[str, float]
    predominant_emotion: str
    confidence: float

class FormatResponse(BaseModel):
    supported_formats: List[str]

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
            {"path": "/analyze-audio", "method": "POST", "description": "Analyze an uploaded audio file"},
        ]
    }

@app.get("/formats", response_model=FormatResponse)
async def formats():
    return {"supported_formats": ["wav", "mp3", "flac", "ogg", "m4a"]}

@app.post("/analyze-audio", response_model=EmotionResponse)
async def analyze_audio(audio_file: UploadFile = File(...)):
    file_extension = os.path.splitext(audio_file.filename)[1][1:].lower()
    if file_extension not in ["wav", "mp3", "flac", "ogg", "m4a"]:
        raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_extension}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
        content = await audio_file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name

    try:
        # Load audio using librosa
        waveform, sr = librosa.load(temp_file_path, sr=16000)

        # Preprocess the audio with the processor
        input_values = processor(waveform, return_tensors="pt", sampling_rate=16000).input_values

        # Make the emotion prediction
        with torch.no_grad():
            logits = model(input_values).logits

        # Get predicted emotion class (Assuming you have a predefined list of emotions)
        predicted_class = torch.argmax(logits, dim=-1).item()

        # Define emotions (this should match the classes of your model)
        emotions = ["happy", "sad", "angry", "neutral"]  # Example emotions, replace with your actual classes

        # Get the emotion name
        predicted_emotion = emotions[predicted_class]
        confidence = round(logits[0][predicted_class].item(), 4)

        return EmotionResponse(
            emotions={predicted_emotion: confidence},
            predominant_emotion=predicted_emotion,
            confidence=confidence
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# =================== Run the app ===================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
