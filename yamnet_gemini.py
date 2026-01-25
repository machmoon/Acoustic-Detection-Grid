#!/usr/bin/env python3
"""
YAMNet Audio Classification with Google Gemini API
Simple script - just run it!
"""

import os
import sys
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import scipy.signal
from scipy.io import wavfile
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Google Gemini API
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("ℹ️  Install Gemini: pip install google-generativeai")

# Get Gemini API key from .env file or environment variable
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

if GEMINI_AVAILABLE and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    print("✅ Google Gemini API enabled (using gemini-2.5-flash)")
else:
    gemini_model = None
    if not GEMINI_AVAILABLE:
        print("ℹ️  Gemini not installed (optional)")

# Load YAMNet
print("Loading YAMNet model...")
model = hub.load('https://tfhub.dev/google/yamnet/1')

def class_names_from_csv(class_map_csv_text):
    class_names = []
    with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])
    return class_names

class_map_path = model.class_map_path().numpy()
class_names = class_names_from_csv(class_map_path)
print(f"✅ Loaded {len(class_names)} sound classes\n")

def ensure_sample_rate(original_sample_rate, waveform, desired_sample_rate=16000):
    """Resample waveform if required. Returns float32 in [-1.0, 1.0] range."""
    # Handle stereo audio - convert to mono
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    
    # Convert to float32 and normalize to [-1.0, 1.0] for resampling
    if waveform.dtype == np.int16:
        waveform_float = waveform.astype(np.float32) / 32768.0
    elif waveform.dtype == np.int32:
        waveform_float = waveform.astype(np.float32) / 2147483648.0
    elif waveform.dtype in [np.float32, np.float64]:
        # Already float, ensure in [-1.0, 1.0] range
        max_val = np.abs(waveform).max()
        if max_val > 1.0:
            waveform_float = (waveform / max_val).astype(np.float32)
        else:
            waveform_float = waveform.astype(np.float32)
    else:
        waveform_float = waveform.astype(np.float32)
    
    # Resample if needed
    if original_sample_rate != desired_sample_rate:
        desired_length = int(round(float(len(waveform_float)) /
                                   original_sample_rate * desired_sample_rate))
        print(f"Resampling from {original_sample_rate}Hz to {desired_sample_rate}Hz")
        waveform_float = scipy.signal.resample(waveform_float, desired_length)
    
    # Return as float32 (YAMNet needs this format)
    return desired_sample_rate, waveform_float

def get_gemini_description(sound_class, confidence, predictions):
    """Get AI description from Google Gemini."""
    if not gemini_model:
        return None
    
    try:
        predictions_text = ", ".join([f"{p['class']} ({p['confidence']:.1%})" 
                                     for p in predictions[:3]])
        
        prompt = f"""An acoustic detection system identified the sound: "{sound_class}" with {confidence:.1%} confidence.

Other detected sounds: {predictions_text}

Provide a brief, helpful 1-2 sentence description of what this sound means in a home security/healthcare context. Be practical and actionable. Focus on what the user should do."""
        
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API error: {e}")
        return None

# Main
if __name__ == '__main__':
    wav_file = sys.argv[1] if len(sys.argv) > 1 else 'breakin.wav'
    
    print(f"Processing: {wav_file}")
    sample_rate, wav_data = wavfile.read(wav_file)
    print(f"  Original: {sample_rate} Hz, shape: {wav_data.shape}, dtype: {wav_data.dtype}")
    
    # Process audio: convert stereo to mono and resample if needed
    sample_rate_processed, wav_data_processed = ensure_sample_rate(sample_rate, wav_data)
    print(f"  Processed: {sample_rate_processed} Hz, shape: {wav_data_processed.shape}")
    print(f"  Range: [{wav_data_processed.min():.6f}, {wav_data_processed.max():.6f}]")
    
    # Classify - waveform is already float32 in [-1.0, 1.0] range
    waveform = wav_data_processed
    
    print("\n🔍 Running YAMNet classification...")
    scores, _, _ = model(waveform)
    scores_np = scores.numpy()
    mean_scores = np.mean(scores_np, axis=0)
    
    top_idx = mean_scores.argmax()
    top_class = class_names[top_idx]
    confidence = mean_scores[top_idx]
    
    top_5 = np.argsort(mean_scores)[::-1][:5]
    predictions = [{'class': class_names[i], 'confidence': float(mean_scores[i])} for i in top_5]
    
    # Results
    print(f"\n🎯 Classification:")
    print(f"   Top sound: {top_class}")
    print(f"   Confidence: {confidence:.2%}")
    print(f"\n   Top 5:")
    for i, p in enumerate(predictions, 1):
        print(f"   {i}. {p['class']:30s} {p['confidence']:.2%}")
    
    # AI Description from Gemini
    if gemini_model:
        print(f"\n🤖 Getting AI description from Gemini...")
        description = get_gemini_description(top_class, confidence, predictions)
        if description:
            print(f"\n💡 AI Description:")
            print(f"   {description}")
    else:
        print(f"\nℹ️  Gemini not available (install: pip install google-generativeai)")

