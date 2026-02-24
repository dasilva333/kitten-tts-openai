import time
import os
import sys
import soundfile as sf
import torch
from kittentts import KittenTTS

import numpy as np

def generate_and_save(model, text, voice, filename, description):
    print(f"\n--- {description} ---")
    start_time = time.perf_counter()
    
    # Check if text is long, if so, manually chunk by sentence to avoid ONNX limits
    if len(text) > 300:
        sentences = [s.strip() + "." for s in text.split('.') if s.strip()]
        audio_chunks = []
        for sentence in sentences:
            audio_chunks.append(model.generate(sentence, voice=voice))
        audio = np.concatenate(audio_chunks)
    else:
        audio = model.generate(text, voice=voice)
        
    end_time = time.perf_counter()
    
    elapsed_time = end_time - start_time
    print(f"Time Taken (Generation): {elapsed_time:.4f} seconds")
    
    save_start = time.perf_counter()
    sf.write(filename, audio, 24000, format='OGG', subtype='VORBIS')
    save_end = time.perf_counter()
    print(f"Time Taken (Saving File): {save_end - save_start:.4f} seconds")
    print(f"Saved: {filename} ({os.path.getsize(filename)} bytes)")
    
    return elapsed_time

def main():
    print("Initializing Benchmark...")
    voice = "Kiki"
    model_id = "KittenML/kitten-tts-mini-0.8"
    
    init_start = time.perf_counter()
    model = KittenTTS(model_id)
    init_end = time.perf_counter()
    print(f"Model Initialization Time: {init_end - init_start:.4f} seconds")
    
    # Texts
    text_short = "The quick brown fox jumps over the lazy dog"
    
    text_medium = (
        "This is a longer piece of text intended to test the medium length generation capabilities of the model. "
        "It includes multiple sentences and should take a bit longer to process than the short phrase. "
        "We are evaluating how the inference time scales with character count."
    )
    
    # Generate 1500 chars (ensure full sentences)
    base_1500 = "The AI model is generating audio. "
    text_1500 = (base_1500 * (1500 // len(base_1500))) + "Final sentence."
    
    # Generate 3500 chars (ensure full sentences)
    base_3500 = "Extended generation test for performance evaluation. "
    text_3500 = (base_3500 * (3500 // len(base_3500))) + "End of test."

    results = []

    # 1. Short (Cold Start)
    # The first generation will incur any lazy-loading or CUDA initialization penalties.
    desc1_cold = "1. Short Phrase (Cold Start) - 43 chars"
    time1_cold = generate_and_save(model, text_short, voice, "benchmark_1_cold.ogg", desc1_cold)
    results.append((desc1_cold, time1_cold))

    # 1. Short (Warm Start)
    desc1_warm = "1. Short Phrase (Warm) - 43 chars"
    time1_warm = generate_and_save(model, text_short, voice, "benchmark_1_warm.ogg", desc1_warm)
    results.append((desc1_warm, time1_warm))

    # 2. Medium (Warm)
    desc2 = f"2. Medium Paragraph (Warm) - {len(text_medium)} chars"
    time2 = generate_and_save(model, text_medium, voice, "benchmark_2_medium.ogg", desc2)
    results.append((desc2, time2))

    # 3. 1500 Chars (Warm)
    desc3 = "3. 1500 Characters (Warm) - 1500 chars"
    time3 = generate_and_save(model, text_1500, voice, "benchmark_3_1500_chars.ogg", desc3)
    results.append((desc3, time3))

    # 4. 3500 Chars (Warm)
    desc4 = "4. 3500 Characters (Warm) - 3500 chars"
    time4 = generate_and_save(model, text_3500, voice, "benchmark_4_3500_chars.ogg", desc4)
    results.append((desc4, time4))

    print("\n================ BACKGROUND RESULTS ================")
    for desc, t in results:
        print(f"{desc}: {t:.4f} seconds")

if __name__ == "__main__":
    with open('benchmark.log', 'w') as f:
        sys.stdout = f
        sys.stderr = f
        try:
            main()
        except Exception as e:
            import traceback
            traceback.print_exc()

