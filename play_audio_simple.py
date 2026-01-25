#!/usr/bin/env python3
"""
Simplest way to play a WAV file
"""

import sys
import sounddevice as sd
from scipy.io import wavfile
import numpy as np

def play_wav(filename):
    """Play a WAV file using sounddevice"""
    # Load audio file
    sample_rate, audio_data = wavfile.read(filename)
    
    # Convert stereo to mono if needed
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Normalize to float32 for sounddevice
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / 2147483648.0
    else:
        audio_data = audio_data.astype(np.float32)
    
    # Play audio
    print(f"Playing {filename}...")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {len(audio_data)/sample_rate:.2f} seconds")
    
    sd.play(audio_data, sample_rate)
    sd.wait()  # Wait until playback finishes
    
    print("Done!")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python play_audio_simple.py <audio_file.wav>")
        print("\nExample:")
        print("  python play_audio_simple.py miaow_16k.wav")
        sys.exit(1)
    
    play_wav(sys.argv[1])

