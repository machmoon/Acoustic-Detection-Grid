#!/usr/bin/env python3
"""
YAMNet Audio Classification with Google Gemini API
Classifies audio files and generates AI descriptions of detected sounds.
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

# Load API key from .env file
load_dotenv()

# Initialize Gemini API (optional)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

if GEMINI_AVAILABLE and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    print("✅ Gemini API enabled")
else:
    gemini_model = None

# Load YAMNet model
print("Loading YAMNet model...")
model = hub.load('https://tfhub.dev/google/yamnet/1')

def class_names_from_csv(class_map_csv_text):
    """Load sound class names from YAMNet's class map CSV."""
    class_names = []
    with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])
    return class_names

# Load class names
class_map_path = model.class_map_path().numpy()
class_names = class_names_from_csv(class_map_path)
print(f"✅ Loaded {len(class_names)} sound classes\n")

def ensure_sample_rate(original_sample_rate, waveform, desired_sample_rate=16000):
    """
    Convert audio to YAMNet's required format:
    - Mono (if stereo)
    - 16kHz sample rate
    - Float32 in [-1.0, 1.0] range
    """
    # Convert stereo to mono
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    
    # Normalize to float32 in [-1.0, 1.0] range
    if waveform.dtype == np.int16:
        waveform_float = waveform.astype(np.float32) / 32768.0
    elif waveform.dtype == np.int32:
        waveform_float = waveform.astype(np.float32) / 2147483648.0
    elif waveform.dtype in [np.float32, np.float64]:
        max_val = np.abs(waveform).max()
        if max_val > 1.0:
            waveform_float = (waveform / max_val).astype(np.float32)
        else:
            waveform_float = waveform.astype(np.float32)
    else:
        waveform_float = waveform.astype(np.float32)
    
    # Resample to 16kHz if needed
    if original_sample_rate != desired_sample_rate:
        desired_length = int(round(float(len(waveform_float)) /
                                   original_sample_rate * desired_sample_rate))
        waveform_float = scipy.signal.resample(waveform_float, desired_length)
    
    return desired_sample_rate, waveform_float

def get_gemini_description(sound_class, confidence, predictions):
    """Generate AI description of detected sound using Gemini API."""
    if not gemini_model:
        return None
    
    try:
        # Format top predictions for context
        predictions_text = ", ".join([f"{p['class']} ({p['confidence']:.1%})" 
                                     for p in predictions[:3]])
        
        prompt = f"""An acoustic detection system identified the sound: "{sound_class}" with {confidence:.1%} confidence.

Other detected sounds: {predictions_text}

Provide a brief, helpful 1-2 sentence description of what this sound means in a home security/healthcare context. Be practical and actionable. Focus on what the user should do."""
        
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"⚠️  Gemini API error: {e}")
        return None

if __name__ == '__main__':
    # Get audio file from command line or use default
    wav_file = sys.argv[1] if len(sys.argv) > 1 else 'breakin.wav'
    
    print(f"📁 Processing: {wav_file}")
    sample_rate, wav_data = wavfile.read(wav_file)
    
    # Process audio for YAMNet
    sample_rate_processed, wav_data_processed = ensure_sample_rate(sample_rate, wav_data)
    
    # Classify audio
    print("\n🔍 Running classification...")
    scores, _, _ = model(wav_data_processed)
    scores_np = scores.numpy()
    mean_scores = np.mean(scores_np, axis=0)
    
    # Get top prediction
    top_idx = mean_scores.argmax()
    top_class = class_names[top_idx]
    confidence = mean_scores[top_idx]
    
    # Get top 5 predictions
    top_5 = np.argsort(mean_scores)[::-1][:5]
    predictions = [{'class': class_names[i], 'confidence': float(mean_scores[i])} 
                  for i in top_5]
    
    # Display results
    print(f"\n🎯 Top sound: {top_class} ({confidence:.2%})")
    print(f"\n📊 Top 5 predictions:")
    for i, p in enumerate(predictions, 1):
        print(f"   {i}. {p['class']:30s} {p['confidence']:.2%}")
    
    # Generate AI description if Gemini is available
    if gemini_model:
        print(f"\n🤖 Generating AI description...")
        description = get_gemini_description(top_class, confidence, predictions)
        if description:
            print(f"\n💡 {description}")
    elif not GEMINI_AVAILABLE:
        print(f"\nℹ️  Install Gemini for AI descriptions: pip install google-generativeai")
