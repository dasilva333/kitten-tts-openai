import torch
import soundfile as sf
from kittentts import KittenTTS
import os

def main():
    print("--- KittenTTS 80M Setup ---")
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available. Falling back to CPU.")

    # Model ID for 80M
    model_id = "KittenML/kitten-tts-mini-0.8"
    print(f"Loading model: {model_id}...")
    
    # Initialize model
    # Note: We stick to basic usage. If CUDA is available, we'll try to move the model.
    try:
        m = KittenTTS(model_id)
        
        # Check if the model has a way to move to device
        # Following basic usage first
        print("Model loaded successfully.")
        
        text = "This is a test of the Kitten T T S eighty million parameter model, saved as an O G G file with C U D A acceleration if available."
        voice = 'Jasper'
        
        print(f"Generating audio for: '{text}' using voice '{voice}'...")
        audio = m.generate(text, voice=voice)
        
        output_file = 'output.ogg'
        print(f"Saving to {output_file}...")
        
        # Save as OGG
        # Sample rate is 24000 according to readme
        sf.write(output_file, audio, 24000, format='OGG', subtype='VORBIS')
        
        if os.path.exists(output_file):
            print(f"Success! {output_file} created. Size: {os.path.getsize(output_file)} bytes")
        else:
            print("Failed to create output file.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
