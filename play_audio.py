#!/usr/bin/env python3
"""
Simple script to play WAV files
Multiple methods available
"""

import sys
import numpy as np
from scipy.io import wavfile

def play_with_playsound(filename):
    """Method 1: Using playsound (simplest)"""
    try:
        from playsound import playsound
        print(f"Playing {filename}...")
        playsound(filename)
        print("Done!")
    except ImportError:
        print("Install playsound: pip install playsound")
        return False
    return True

def play_with_sounddevice(filename):
    """Method 2: Using sounddevice (you already have this)"""
    try:
        import sounddevice as sd
        from scipy.io import wavfile
        
        print(f"Loading {filename}...")
        sample_rate, audio_data = wavfile.read(filename)
        
        # Convert stereo to mono if needed
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Normalize to float32 for sounddevice
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        
        print(f"Playing... (Sample rate: {sample_rate} Hz, Duration: {len(audio_data)/sample_rate:.2f}s)")
        sd.play(audio_data, sample_rate)
        sd.wait()  # Wait until playback is finished
        print("Done!")
        return True
    except ImportError:
        print("Install sounddevice: pip install sounddevice")
        return False

def play_with_pygame(filename):
    """Method 3: Using pygame (more control)"""
    try:
        import pygame
        pygame.mixer.init()
        print(f"Playing {filename}...")
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        
        # Wait for playback to finish
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
        
        print("Done!")
        return True
    except ImportError:
        print("Install pygame: pip install pygame")
        return False

def play_with_ipython(filename):
    """Method 4: Using IPython (for Jupyter notebooks)"""
    try:
        from IPython.display import Audio
        from scipy.io import wavfile
        
        sample_rate, audio_data = wavfile.read(filename)
        
        # Convert to int16 if needed
        if audio_data.dtype != np.int16:
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            audio_data = (audio_data * 32767).astype(np.int16)
        
        return Audio(audio_data, rate=sample_rate)
    except ImportError:
        print("IPython not available (this is for Jupyter notebooks)")
        return None

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python play_audio.py <audio_file.wav> [method]")
        print("\nMethods:")
        print("  1. playsound (simplest)")
        print("  2. sounddevice (recommended, already installed)")
        print("  3. pygame")
        print("\nExample:")
        print("  python play_audio.py miaow_16k.wav")
        print("  python play_audio.py miaow_16k.wav sounddevice")
        sys.exit(1)
    
    filename = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else 'sounddevice'
    
    print(f"Playing: {filename}")
    print(f"Method: {method}\n")
    
    success = False
    if method == 'playsound':
        success = play_with_playsound(filename)
    elif method == 'sounddevice':
        success = play_with_sounddevice(filename)
    elif method == 'pygame':
        success = play_with_pygame(filename)
    elif method == 'ipython':
        result = play_with_ipython(filename)
        if result:
            print("For Jupyter notebooks, use: display(result)")
        success = result is not None
    else:
        print(f"Unknown method: {method}")
        print("Available methods: playsound, sounddevice, pygame, ipython")
    
    if not success:
        print("\nTrying sounddevice as fallback...")
        play_with_sounddevice(filename)

