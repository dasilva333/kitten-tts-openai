from kittentts import KittenTTS
import time
import onnxruntime as ort

print("=" * 50)
print("CUDA BENCHMARK TEST")
print("=" * 50)

# Show available providers
print(f"\nONNX Available Providers: {ort.get_available_providers()}")

print('\nLoading model...', flush=True)
model = KittenTTS('KittenML/kitten-tts-mini-0.8')
print('Model loaded!', flush=True)

# Cold start
print('\n--- Test 1: Cold Start (43 chars) ---', flush=True)
start = time.perf_counter()
audio = model.generate('The quick brown fox jumps over the lazy dog', voice='Kiki')
cold_time = time.perf_counter() - start
print(f'Result: {cold_time:.4f} seconds', flush=True)

# Warm start
print('\n--- Test 2: Warm Start (43 chars) ---', flush=True)
start = time.perf_counter()
audio = model.generate('The quick brown fox jumps over the lazy dog', voice='Kiki')
warm_time = time.perf_counter() - start
print(f'Result: {warm_time:.4f} seconds', flush=True)

# Medium text
medium = 'This is a longer piece of text intended to test the medium length generation capabilities of the model. It includes multiple sentences and should take a bit longer to process than the short phrase. We are evaluating how the inference time scales with character count.'
print(f'\n--- Test 3: Medium ({len(medium)} chars) ---', flush=True)
start = time.perf_counter()
audio = model.generate(medium, voice='Kiki')
medium_time = time.perf_counter() - start
print(f'Result: {medium_time:.4f} seconds', flush=True)

print('\n' + '=' * 50)
print('RESULTS SUMMARY')
print('=' * 50)
print(f'Cold (43 chars):  {cold_time:.4f}s')
print(f'Warm (43 chars):  {warm_time:.4f}s')
print(f'Warm ({len(medium)} chars): {medium_time:.4f}s')
print('=' * 50)
