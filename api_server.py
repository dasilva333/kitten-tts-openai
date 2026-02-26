from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Optional, Literal
import time
import io
import soundfile as sf
import numpy as np
import datetime
from kittentts import KittenTTS

# OpenAI Schema
class SpeechRequest(BaseModel):
    model: str
    input: str = Field(..., max_length=4096)
    voice: str
    response_format: Optional[Literal["mp3", "opus", "aac", "flac", "wav", "ogg"]] = "wav"
    speed: Optional[float] = 1.0

def log_with_time(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {msg}")

app = FastAPI(title="KittenTTS OpenAI API")

# Global TTS model instance
model = None

@app.on_event("startup")
def load_model():
    global model
    log_with_time("Loading KittenTTS 80M model...")
    model = KittenTTS("KittenML/kitten-tts-mini-0.8")
    log_with_time("Model loaded successfully.")

@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest):
    global model
    if not model:
        raise HTTPException(status_code=503, detail="Model is still loading")

    target_voice = "Kiki"
    # User requested a 20% speed boost for testing.
    speed_factor = req.speed * 1.2
    max_retries = 3
    retry_delay = 0.5 

    last_error = ""
    
    for attempt in range(max_retries):
        try:
            start = time.perf_counter()
            log_with_time(f"Turn Handler: Processing request (Attempt {attempt + 1}/{max_retries}, Speed: {speed_factor:.2f})")
            
            # Use the global model instance
            if attempt > 0:
                log_with_time("Attempting self-healing: Re-initializing model session...")
                try:
                    model = KittenTTS("KittenML/kitten-tts-mini-0.8")
                    log_with_time("Model re-initialized successfully.")
                except Exception as reinit_err:
                    log_with_time(f"Self-healing failed: {str(reinit_err)}")
            
            # Robustness: KittenTTS utility library crashes if input has no speakable characters (like just "...")
            # We filter it here and return a tiny silence instead of crashing the process.
            clean_text = req.input.strip()
            if not any(c.isalnum() for c in clean_text):
                log_with_time(f"Warning: Input '{clean_text}' has no speakable content. Returning silence.")
                audio_data = np.zeros(1000, dtype=np.float32) # ~40ms of silence at 24kHz
            else:
                # The library handles its own chunking at 400 chars, so we don't need manual splitting.
                audio_data = model.generate(clean_text, voice=target_voice, speed=speed_factor)
                
            end = time.perf_counter()
            log_with_time(f"Generated speech in {end - start:.2f}s (Length: {len(clean_text)}, Voice: {target_voice})")
            
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
                    "remedy": "Input sanitization should prevent most crashes. If this error persists, it may be an internal model hang. Please check server console."
                })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)
