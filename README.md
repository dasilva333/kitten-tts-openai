# KittenTTS OpenAI API

An OpenAI-compatible API server for [KittenTTS](https://github.com/KittenML/KittenTTS), a lightweight and fast text-to-speech library. This project provides a simple way to integrate KittenTTS into applications that already support the OpenAI Audio API.

## Features

- **OpenAI Compatible**: Implements the `/v1/audio/speech` endpoint.
- **Phonetic Cleaning**: Automatically fixes common TTS pronunciation issues (e.g., contractions).
- **Narrative Parsing**: Supports multi-voice generation using Luna for narrative text (wrapped in asterisks) and Kiki for dialogue.
- **CUDA Support**: Optional GPU acceleration for faster inference on long texts.
- **Lightweight**: Uses the 0.8M parameter KittenTTS model for low memory footprint.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/h4rdc/kitten-tts-openai.git
   cd kitten-tts-openai
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install ONNX Runtime (GPU recommended)**:
   ```bash
   pip uninstall onnxruntime
   pip install onnxruntime-gpu
   ```

## Usage

### Start the Server

```bash
python api_server.py
```

The server will start on `http://localhost:8090`.

### Example Request

```bash
curl http://localhost:8090/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "kitten-tts",
    "input": "Hello! *She whispered softly.* How are you today?",
    "voice": "Kiki",
    "response_format": "wav"
  }' \
  --output speech.wav
```

## Performance Benchmarks

Tested on **RTX 4070 Laptop GPU**.

| Text Length | Characters | Latency (Cold) | Latency (Warm) | Throughput |
|-------------|------------|----------------|----------------|------------|
| Short       | 43         | 5.22s          | 4.46s          | 9.6 ch/s   |
| Medium      | 267        | -              | 19.02s         | 14.0 ch/s  |

> [!NOTE]
> For very short texts, CPU inference may be faster than GPU due to data transfer overhead.

## Files

- `api_server.py`: The main FastAPI server.
- `api_benchmark.py`: Comprehensive benchmarking script.
- `CUDA_RESEARCH.md`: Detailed analysis of CPU vs GPU performance for small models.

## License

MIT
