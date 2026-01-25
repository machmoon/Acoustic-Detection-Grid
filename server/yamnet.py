#!/usr/bin/env python3
"""
YAMNet Audio Classification (Basic Version)
Classifies audio files using YAMNet without AI descriptions.
"""

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import scipy.signal
from scipy.io import wavfile
import matplotlib.pyplot as plt

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
        print(f"Resampling from {original_sample_rate}Hz to {desired_sample_rate}Hz")
        waveform_float = scipy.signal.resample(waveform_float, desired_length)
    
    return desired_sample_rate, waveform_float

# Main execution
if __name__ == '__main__':
    wav_file_name = 'breakin.wav'
    
    print(f"📁 Processing: {wav_file_name}")
    sample_rate, wav_data = wavfile.read(wav_file_name)
    
    # Process audio for YAMNet
    sample_rate_processed, wav_data_processed = ensure_sample_rate(sample_rate, wav_data)
    
    # Show audio info
    duration = len(wav_data_processed) / sample_rate_processed
    print(f"Duration: {duration:.2f}s")
    print(f"Sample rate: {sample_rate_processed} Hz")
    
    # Classify audio
    print("\n🔍 Running classification...")
    waveform = wav_data_processed
    scores, embeddings, spectrogram = model(waveform)
    
    scores_np = scores.numpy()
    spectrogram_np = spectrogram.numpy()
    mean_scores = np.mean(scores_np, axis=0)
    
    # Get top prediction
    top_class_idx = mean_scores.argmax()
    top_class = class_names[top_class_idx]
    confidence = mean_scores[top_class_idx]
    
    print(f'\n🎯 Top sound: {top_class} ({confidence:.2%})')
    
    # Show top 5 predictions
    top_n = 5
    top_indices = np.argsort(mean_scores)[::-1][:top_n]
    print(f'\n📊 Top {top_n} predictions:')
    for i, idx in enumerate(top_indices, 1):
        print(f'   {i}. {class_names[idx]:30s} {mean_scores[idx]:.2%}')
    
    # Visualization
    plt.figure(figsize=(10, 6))
    
    # Waveform
    plt.subplot(3, 1, 1)
    plt.plot(waveform)
    plt.xlim([0, len(waveform)])
    plt.title('Waveform')
    
    # Spectrogram
    plt.subplot(3, 1, 2)
    plt.imshow(spectrogram_np.T, aspect='auto', interpolation='nearest', origin='lower')
    plt.title('Spectrogram')
    
    # Top class scores over time
    top_class_indices = np.argsort(mean_scores)[::-1][:top_n]
    plt.subplot(3, 1, 3)
    plt.imshow(scores_np[:, top_class_indices].T, aspect='auto', 
               interpolation='nearest', cmap='gray_r')
    plt.title('Top Class Scores Over Time')
    
    # Formatting
    patch_padding = (0.025 / 2) / 0.01
    plt.xlim([-patch_padding-0.5, scores.shape[0] + patch_padding-0.5])
    yticks = range(0, top_n, 1)
    plt.yticks(yticks, [class_names[top_class_indices[x]] for x in yticks])
    plt.ylim(-0.5 + np.array([top_n, 0]))
    
    plt.tight_layout()
    plt.show()
