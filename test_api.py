import requests
import json
import time

url = "http://localhost:8090/v1/audio/speech"
headers = {"Content-Type": "application/json"}
data = {
    "model": "tts-1",
    "input": "I have successfully verified the OpenAI-compatible API server using test_api.py. The server is running on port 8090, correctly uses the 80M KittenTTS model, and forces the 'Kiki' voice as requested. I am now finalizing the project documentation. The server generates audio for a short test sentence in ~6 seconds on the GPU.",
    "voice": "alloy"  # The server should ignore this and use Kiki
}

print("Sending request to KittenTTS API...")
start = time.perf_counter()
response = requests.post(url, headers=headers, data=json.dumps(data))
end = time.perf_counter()

if response.status_code == 200:
    print(f"Success! Request took {end - start:.2f} seconds.")
    with open("test_api.wav", "wb") as f:
        f.write(response.content)
    print("Saved to test_api.wav")
else:
    print(f"Failed with status {response.status_code}")
    print(response.text)
