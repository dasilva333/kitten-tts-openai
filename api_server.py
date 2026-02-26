from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Optional, Literal
import time
import io
import soundfile as sf
import numpy as np
from kittentts import KittenTTS

# OpenAI Schema
class SpeechRequest(BaseModel):
    model: str
    input: str = Field(..., max_length=4096)
    voice: str
    response_format: Optional[Literal["mp3", "opus", "aac", "flac", "wav", "ogg"]] = "wav"
    speed: Optional[float] = 1.0

# Voice Mapping from OpenAI -> KittenTTS (best effort)
# KittenTTS voices: ['Bella', 'Jasper', 'Luna', 'Bruno', 'Rosie', 'Hugo', 'Kiki', 'Leo']
VOICE_MAP = {
    "alloy": "Jasper",
    "echo": "Bruno",
    "fable": "Hugo",
    "onyx": "Leo",
    "nova": "Bella",
    "shimmer": "Luna",
}

app = FastAPI(title="KittenTTS OpenAI API")

# Global TTS model instance
model = None

@app.on_event("startup")
def load_model():
    global model
    print("Loading KittenTTS 80M model...")
    model = KittenTTS("KittenML/kitten-tts-mini-0.8")
    print("Model loaded successfully.")

def chunk_text(text: str) -> list[str]:
    """Splits text into sentences to avoid ONNX max sequence length limits."""
    # simple split by punctuation
    sentences = []
    current_sentence = ""
    for char in text:
        current_sentence += char
        if char in ['.', '!', '?'] and len(current_sentence.strip()) > 0:
            sentences.append(current_sentence.strip())
            current_sentence = ""
    
    if current_sentence.strip():
        sentences.append(current_sentence.strip())
        
    return sentences if sentences else [text]

import datetime

def log_with_time(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {msg}")

@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest):
    global model
    if not model:
        raise HTTPException(status_code=503, detail="Model is still loading")

    target_voice = "Kiki"
    max_retries = 3
    retry_delay = 0.5 

    last_error = ""
    
    for attempt in range(max_retries):
        try:
            start = time.perf_counter()
            log_with_time(f"Turn Handler: Processing request (Attempt {attempt + 1}/{max_retries})")
            
            # Use the global model instance
            if attempt > 0:
                log_with_time("Attempting self-healing: Re-initializing model session...")
                try:
                    model = KittenTTS("KittenML/kitten-tts-mini-0.8")
                    log_with_time("Model re-initialized successfully.")
                except Exception as reinit_err:
                    log_with_time(f"Self-healing failed: {str(reinit_err)}")
            
            # Determine if we need to chunk
            if len(req.input) > 250:
                sentences = chunk_text(req.input)
                audio_chunks = []
                for s in sentences:
                    audio_chunks.append(model.generate(s, voice=target_voice))
                audio_data = np.concatenate(audio_chunks)
            else:
                audio_data = model.generate(req.input, voice=target_voice)
                
            end = time.perf_counter()
            log_with_time(f"Generated speech in {end - start:.2f}s (Length: {len(req.input)}, Voice: {target_voice})")
            
            # Save to memory buffer
            format_type = req.response_format.upper()
            if format_type not in ["WAV", "OGG", "FLAC"]:
                format_type = "WAV"
                
            subtype = 'VORBIS' if format_type == 'OGG' else None
                
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, audio_data, 24000, format=format_type, subtype=subtype)
            audio_buffer.seek(0)
            
            media_type = f"audio/{format_type.lower()}"
            return Response(content=audio_buffer.read(), media_type=media_type)

        except Exception as e:
            last_error = str(e)
            log_with_time(f"Error during attempt {attempt + 1}: {last_error}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue
            else:
                log_with_time(f"CRITICAL: Failed all {max_retries} attempts.")
                raise HTTPException(status_code=500, detail={
                    "error": "TTS Generation Failed",
                    "cause": last_error,
                    "remedy": "The server attempted self-healing and retries but could not recover. Please check server logs or restart the process if this persists."
                })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)
