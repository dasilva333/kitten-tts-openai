import time
import numpy as np
from kittentts import KittenTTS

def generate_long_text(model, text, voice="Kiki"):
    # Split by sentence to avoid ONNX size limits
    sentences = [s.strip() + "." for s in text.split('.') if s.strip()]
    audio_chunks = []
    for sentence in sentences:
        audio_chunks.append(model.generate(sentence, voice=voice))
    return np.concatenate(audio_chunks)

def run_benchmark(length, text):
    print(f"\n======================================")
    print(f"       BENCHMARK: {length} CHARACTERS")
    print(f"======================================")
    
    print("\n--- Cold Start ---")
    start_cold = time.perf_counter()
    # Cold start includes model initialization
    model_cold = KittenTTS("KittenML/kitten-tts-mini-0.8")
    generate_long_text(model_cold, text)
    end_cold = time.perf_counter()
    print(f"Time Taken (Init + Gen): {end_cold - start_cold:.4f} seconds")
    
    # We will clear the model to simulate a fresh environment, 
    # but strictly speaking Python's GC might hold it. 
    # We'll use the existing model_cold for the warm test.
    print("\n--- Warm Start ---")
    start_warm = time.perf_counter()
    generate_long_text(model_cold, text)
    end_warm = time.perf_counter()
    print(f"Time Taken (Gen only): {end_warm - start_warm:.4f} seconds")

def main():
    # Generate 1500 chars 
    base_1500 = "The AI model is generating audio. "
    text_1500 = (base_1500 * (1500 // len(base_1500))) + "Final sentence."
    
    # Generate 3500 chars 
    base_3500 = "Extended generation test for performance evaluation. "
    text_3500 = (base_3500 * (3500 // len(base_3500))) + "End of test."

    run_benchmark(1500, text_1500)
    run_benchmark(3500, text_3500)

if __name__ == "__main__":
    main()
