"""
KittenTTS API Benchmark Script
Tests the running API server with hardcoded text of specific lengths.
Measures end-to-end latency including network round-trip.

Usage:
    python api_benchmark.py

Requires:
    - API server running on http://localhost:8090
    - requests library installed
"""

import requests
import json
import time
import os

# Configuration
API_URL = "http://localhost:8090/v1/audio/speech"
OUTPUT_DIR = "benchmark_outputs"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hardcoded test texts with exact character counts
TEST_CASES = [
    {
        "name": "Short",
        "length": 43,
        "text": "The quick brown fox jumps over the lazy dog"
    },
    {
        "name": "Medium", 
        "length": 267,
        "text": "This is a longer piece of text intended to test the medium length generation capabilities of the model. It includes multiple sentences and should take a bit longer to process than the short phrase. We are evaluating how the inference time scales with character count."
    },
    {
        "name": "Long",
        "length": 1500,
        "text": "The AI model is generating audio. " * 46 + "Final sentence to complete the test."
    },
    {
        "name": "Very Long",
        "length": 3500,
        "text": "Extended generation test for performance evaluation. " * 70 + "End of test."
    }
]

def generate_text_of_length(base_text, target_length):
    """Generate text of approximately the target length by repeating base text."""
    if len(base_text) >= target_length:
        return base_text[:target_length]
    
    repetitions = target_length // len(base_text)
    remainder = target_length % len(base_text)
    
    result = (base_text * repetitions) + base_text[:remainder]
    return result.strip()


def benchmark_api_request(text, voice="Kiki", response_format="wav"):
    """
    Make a single API request and measure the latency.
    
    Returns:
        tuple: (latency_seconds, success, error_message)
    """
    payload = {
        "model": "tts-1",
        "input": text,
        "voice": "alloy",  # Server ignores this and uses Kiki
        "response_format": response_format
    }
    
    start = time.perf_counter()
    
    try:
        response = requests.post(API_URL, json=payload, timeout=300)
        end = time.perf_counter()
        
        latency = end - start
        
        if response.status_code == 200:
            return latency, True, None
        else:
            return latency, False, f"HTTP {response.status_code}: {response.text}"
            
    except requests.exceptions.Timeout:
        end = time.perf_counter()
        return end - start, False, "Request timed out (300s)"
    except requests.exceptions.ConnectionError:
        end = time.perf_counter()
        return end - start, False, "Connection refused - is the server running?"
    except Exception as e:
        end = time.perf_counter()
        return end - start, False, str(e)


def run_benchmark():
    """Run the full benchmark suite."""
    print("=" * 70)
    print("KITTENTTS API BENCHMARK")
    print("=" * 70)
    print(f"\nServer: {API_URL}")
    print(f"Voice: Kiki (hardcoded in server)")
    print(f"Output: {OUTPUT_DIR}/")
    print("=" * 70)
    
    # First, check if server is running
    print("\nChecking server connectivity...")
    test_latency, success, error = benchmark_api_request("Test", voice="Kiki")
    
    if not success:
        print(f"❌ Server check failed: {error}")
        print("\nMake sure the API server is running:")
        print("  python api_server.py")
        return
    
    print(f"✓ Server is responsive (test latency: {test_latency:.2f}s)")
    
    # Run benchmarks
    results = []
    
    for test_case in TEST_CASES:
        text = test_case["text"]
        name = test_case["name"]
        length = len(text)
        
        print(f"\n{'─' * 70}")
        print(f"Testing: {name} ({length} characters)")
        print(f"{'─' * 70}")
        
        # Run 3 iterations for each test case
        latencies = []
        successful_runs = 0
        
        for i in range(3):
            print(f"  Run {i+1}/3...", end=" ", flush=True)
            latency, success, error = benchmark_api_request(text)
            
            if success:
                latencies.append(latency)
                successful_runs += 1
                print(f"✓ {latency:.2f}s")
            else:
                print(f"✗ {error}")
        
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            
            # Calculate chars per second
            chars_per_sec = length / avg_latency
            
            results.append({
                "name": name,
                "length": length,
                "avg_latency": avg_latency,
                "min_latency": min_latency,
                "max_latency": max_latency,
                "chars_per_sec": chars_per_sec,
                "successful_runs": successful_runs
            })
            
            print(f"\n  Summary:")
            print(f"    Avg: {avg_latency:.2f}s | Min: {min_latency:.2f}s | Max: {max_latency:.2f}s")
            print(f"    Throughput: {chars_per_sec:.1f} chars/sec")
        else:
            print(f"  ❌ All runs failed for {name}")
            results.append({
                "name": name,
                "length": length,
                "avg_latency": None,
                "error": "All runs failed"
            })
    
    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Test':<15} {'Length':<10} {'Avg Latency':<15} {'Throughput':<15} {'Status'}")
    print("-" * 70)
    
    for r in results:
        if r["avg_latency"] is not None:
            print(f"{r['name']:<15} {r['length']:<10} {r['avg_latency']:<15.2f}s {r['chars_per_sec']:<15.1f} ch/s ✓")
        else:
            print(f"{r['name']:<15} {r['length']:<10} {'N/A':<15} {'N/A':<15} ✗ {r.get('error', '')}")
    
    print("=" * 70)
    
    # Save results to file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(OUTPUT_DIR, f"benchmark_results_{timestamp}.txt")
    
    with open(results_file, 'w') as f:
        f.write("KITTENTTS API BENCHMARK RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Server: {API_URL}\n")
        f.write(f"Voice: Kiki\n\n")
        
        for r in results:
            f.write(f"{r['name']} ({r['length']} chars):\n")
            if r["avg_latency"] is not None:
                f.write(f"  Avg Latency: {r['avg_latency']:.2f}s\n")
                f.write(f"  Min Latency: {r['min_latency']:.2f}s\n")
                f.write(f"  Max Latency: {r['max_latency']:.2f}s\n")
                f.write(f"  Throughput: {r['chars_per_sec']:.1f} chars/sec\n")
            else:
                f.write(f"  Error: {r.get('error', 'Unknown')}\n")
            f.write("\n")
        
        f.write("=" * 70 + "\n")
        f.write("Results also saved to benchmark_results.txt\n")
    
    # Also save to a fixed filename for easy access
    fixed_results_file = os.path.join(OUTPUT_DIR, "benchmark_results.txt")
    with open(fixed_results_file, 'w') as f:
        f.write(f"KITTENTTS API BENCHMARK RESULTS - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        for r in results:
            if r["avg_latency"] is not None:
                f.write(f"{r['name']} ({r['length']} chars): {r['avg_latency']:.2f}s avg ({r['chars_per_sec']:.1f} ch/s)\n")
            else:
                f.write(f"{r['name']} ({r['length']} chars): FAILED - {r.get('error', 'Unknown')}\n")
    
    print(f"\nResults saved to: {results_file}")
    print(f"Quick results saved to: {fixed_results_file}")
    
    return results


if __name__ == "__main__":
    run_benchmark()
