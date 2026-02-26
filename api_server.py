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

import re

def phonetic_cleaner(text: str) -> str:
    """Fixes common TTS pronunciation issues with contractions using generalized regex."""
    # Matches any word with 2+ letters followed by 's or ’s
    # Example: "Kiki's" -> "Kikiz", "That's" -> "Thatz"
    # We use a capturing group for the leading letters to preserve them.
    pattern = r"(\b[a-zA-Z]{2,})['’]s\b"
    cleaned = re.sub(pattern, r"\1z", text, flags=re.IGNORECASE)
    return cleaned

def parse_narrative(text: str):
    """
    Splits text into chunks, identifying which ones are 'narrative' (wrapped in asterisks).
    Returns a list of (text, is_narrative) tuples.
    """
    # Regex to find text between asterisks (greedy to avoid splitting multi-word actions)
    # We use a non-capturing group with capturing parens around the asterisk blocks
    # to preserve the split parts.
    parts = re.split(r"(\*[^*]+\*)", text)
    chunks = []
    for p in parts:
        if not p.strip():
            continue
        if p.startswith("*") and p.endswith("*"):
            # It's narrative, strip asterisks
            chunks.append((p.strip("*").strip(), True))
        else:
            # It's normal text
            chunks.append((p.strip(), False))
    return chunks

@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest):
    global model
    if not model:
        raise HTTPException(status_code=503, detail="Model is still loading")

    # Base voices
    VOICE_DIALOGUE = "Kiki"
    VOICE_NARRATIVE = "Luna"
    
    # Base speeds
    base_speed = req.speed * 1.2 # User's requested 20% boost
    
    max_retries = 3
    retry_delay = 0.5 

    last_error = ""
    
    for attempt in range(max_retries):
        try:
            start = time.perf_counter()
            log_with_time(f"Turn Handler: Processing request (Attempt {attempt + 1}/{max_retries}, Base Speed: {base_speed:.2f})")
            
            # Use the global model instance
            if attempt > 0:
                log_with_time("Attempting self-healing: Re-initializing model session...")
                try:
                    model = KittenTTS("KittenML/kitten-tts-mini-0.8")
                    log_with_time("Model re-initialized successfully.")
                except Exception as reinit_err:
                    log_with_time(f"Self-healing failed: {str(reinit_err)}")
            
            # 1. Phonetic Homogenization (run on full text first to maintain context)
            clean_input = phonetic_cleaner(req.input.strip())
            
            # 2. Narrative Parsing
            segments = parse_narrative(clean_input)
            
            if not segments:
                log_with_time("Warning: No speakable content found. Returning silence.")
                audio_data = np.zeros(1000, dtype=np.float32) 
            else:
                audio_chunks = []
                for text, is_narrative in segments:
                    # Robustness check for each chunk
                    if not any(c.isalnum() for c in text):
                        continue
                        
                    voice = VOICE_NARRATIVE if is_narrative else VOICE_DIALOGUE
                    # Narration is slightly slower (90% of base) for "internal monologue" feel
                    speed = base_speed * 0.9 if is_narrative else base_speed
                    
                    log_with_time(f" Generating chunk: '{text[:30]}...' (Voice: {voice}, Speed: {speed:.2f})")
                    audio_chunks.append(model.generate(text, voice=voice, speed=speed))
                
                if not audio_chunks:
                    audio_data = np.zeros(1000, dtype=np.float32)
                else:
                    audio_data = np.concatenate(audio_chunks)
                
            end = time.perf_counter()
            log_with_time(f"Generated total speech in {end - start:.2f}s (Segments: {len(segments)})")
            
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
                    "remedy": "Input segments may have caused a processing error. Check the server console for chunk-level logs."
                })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)
