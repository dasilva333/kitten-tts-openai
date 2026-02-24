import time
from kittentts import KittenTTS

print("Loading model...")
model = KittenTTS("KittenML/kitten-tts-mini-0.8")

text_short = "The quick brown fox jumps over the lazy dog"
text_medium = (
    "This is a longer piece of text intended to test the medium length generation capabilities of the model. "
    "It includes multiple sentences and should take a bit longer to process than the short phrase. "
    "We are evaluating how the inference time scales with character count."
)

print("\n--- 1. Short (Cold Start) ---")
start = time.perf_counter()
model.generate(text_short, voice="Kiki")
end = time.perf_counter()
print(f"Time: {end - start:.4f} seconds (Length: {len(text_short)} chars)")

print("\n--- 2. Short (Warm Start) ---")
start = time.perf_counter()
model.generate(text_short, voice="Kiki")
end = time.perf_counter()
print(f"Time: {end - start:.4f} seconds (Length: {len(text_short)} chars)")

print("\n--- 3. Medium (Warm Start) ---")
start = time.perf_counter()
model.generate(text_medium, voice="Kiki")
end = time.perf_counter()
print(f"Time: {end - start:.4f} seconds (Length: {len(text_medium)} chars)")
