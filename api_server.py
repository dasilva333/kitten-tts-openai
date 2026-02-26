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

@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest):
    global model
    if not model:
        raise HTTPException(status_code=503, detail="Model is still loading")

    # Determine voice - forcing Kiki based on user request
    target_voice = "Kiki"
    max_retries = 3
    retry_delay = 0.5 # seconds

    for attempt in range(max_retries):
        try:
            start = time.perf_counter()
            
            # Determine if we need to chunk
            if len(req.input) > 250:
                sentences = chunk_text(req.input)
                audio_chunks = []
                for s in sentences:
                    # model.generate might fail due to CUDA/ONNX issues
                    audio_chunks.append(model.generate(s, voice=target_voice))
                audio_data = np.concatenate(audio_chunks)
            else:
                audio_data = model.generate(req.input, voice=target_voice)
                
            end = time.perf_counter()
            print(f"Generated speech in {end - start:.2f}s (Length: {len(req.input)}, Voice: {target_voice})")
            
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
            error_msg = str(e)
            print(f"Attempt {attempt + 1}/{max_retries} failed: {error_msg}")
            
            # If it's a CUDA/ONNX failure, we might want to wait a bit before retrying
            if "CUDA" in error_msg or "ONNXRuntimeError" in error_msg:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    # Final attempt failed
                    print(f"CRITICAL: Failed all {max_retries} attempts. Returning detailed error.")
                    raise HTTPException(status_code=500, detail={
                        "error": "Internal Server Error during TTS generation",
                        "cause": error_msg,
                        "suggestion": "Check GPU memory or consider switching to CPU execution."
                    })
            else:
                # For non-CUDA errors, just raise immediately
                raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)
