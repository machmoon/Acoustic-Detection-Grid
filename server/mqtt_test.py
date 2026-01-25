#!/usr/bin/env python3
"""
MQTT Test - Simulates sending and receiving audio over MQTT
Runs both sender and receiver in one script for easy testing.
"""

import os
import sys
import time
import threading
import numpy as np
from scipy.io import wavfile

# MQTT
import paho.mqtt.client as mqtt

# Load YAMNet components
print("Loading YAMNet model...")
import tensorflow as tf
import tensorflow_hub as hub
import csv
import scipy.signal

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

# Config
MQTT_BROKER = os.environ.get('MQTT_BROKER', 'test.mosquitto.org')
MQTT_PORT = int(os.environ.get('MQTT_PORT', 1883))
MQTT_TOPIC = os.environ.get('MQTT_TOPIC', 'audio/test/yamnet')

def wav_to_8bit(wav_file):
    """Convert WAV file to 8-bit unsigned audio data."""
    sample_rate, wav_data = wavfile.read(wav_file)
    
    # Convert stereo to mono
    if len(wav_data.shape) > 1:
        wav_data = np.mean(wav_data, axis=1)
    
    # Normalize to float in [-1.0, 1.0]
    if wav_data.dtype == np.int16:
        wav_float = wav_data.astype(np.float32) / 32768.0
    elif wav_data.dtype == np.int32:
        wav_float = wav_data.astype(np.float32) / 2147483648.0
    else:
        wav_float = wav_data.astype(np.float32)
        max_val = np.abs(wav_float).max()
        if max_val > 1.0:
            wav_float = wav_float / max_val
    
    # Convert to 8-bit unsigned (0-255)
    wav_8bit = ((wav_float + 1.0) * 127.5).astype(np.uint8)
    
    return sample_rate, wav_8bit

def process_8bit_audio(audio_data, sample_rate=16000):
    """Convert 8-bit audio to YAMNet format."""
    # Convert uint8 (0-255) to float32 in [-1.0, 1.0]
    waveform_float = (audio_data.astype(np.float32) - 128.0) / 128.0
    
    # Resample to 16kHz if needed
    if sample_rate != 16000:
        desired_length = int(round(float(len(waveform_float)) / sample_rate * 16000))
        waveform_float = scipy.signal.resample(waveform_float, desired_length)
    
    return waveform_float

def classify_audio(waveform):
    """Classify audio using YAMNet."""
    scores, _, _ = model(waveform)
    scores_np = scores.numpy()
    mean_scores = np.mean(scores_np, axis=0)
    
    top_idx = mean_scores.argmax()
    top_class = class_names[top_idx]
    confidence = mean_scores[top_idx]
    
    top_5 = np.argsort(mean_scores)[::-1][:5]
    predictions = [{'class': class_names[i], 'confidence': float(mean_scores[i])} 
                  for i in top_5]
    
    return top_class, confidence, predictions

# Results storage
received_results = []

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"✅ Connected to MQTT broker: {MQTT_BROKER}")
        client.subscribe(MQTT_TOPIC)
        print(f"📡 Subscribed to: {MQTT_TOPIC}")
    else:
        print(f"❌ Failed to connect (code: {rc})")

def on_message(client, userdata, msg):
    """Receive and classify audio."""
    print(f"\n📥 Received {len(msg.payload)} bytes")
    
    # Convert bytes to numpy array
    audio_data = np.frombuffer(msg.payload, dtype=np.uint8)
    
    # Process 8-bit audio
    waveform = process_8bit_audio(audio_data, 16000)
    print(f"   Processed: {len(waveform)} samples @ 16kHz")
    
    # Classify
    print("🔍 Classifying...")
    top_class, confidence, predictions = classify_audio(waveform)
    
    # Display results
    print(f"\n🎯 TOP SOUND: {top_class} ({confidence:.2%})")
    print(f"\n📊 Top 5 predictions:")
    for i, p in enumerate(predictions, 1):
        print(f"   {i}. {p['class']:30s} {p['confidence']:.2%}")
    
    received_results.append((top_class, confidence))
    
    # Stop the loop after receiving
    client.disconnect()

def main():
    wav_file = sys.argv[1] if len(sys.argv) > 1 else 'breakin16khz.wav'
    
    if not os.path.exists(wav_file):
        print(f"❌ File not found: {wav_file}")
        print("Available WAV files:")
        for f in os.listdir('.'):
            if f.endswith('.wav'):
                print(f"  - {f}")
        return
    
    print(f"📁 Loading: {wav_file}")
    sample_rate, audio_8bit = wav_to_8bit(wav_file)
    print(f"   Sample rate: {sample_rate} Hz")
    print(f"   Duration: {len(audio_8bit)/sample_rate:.2f}s")
    print(f"   8-bit range: [{audio_8bit.min()}, {audio_8bit.max()}]")
    
    # Create MQTT client
    print(f"\n🔌 Connecting to: {MQTT_BROKER}:{MQTT_PORT}")
    client = mqtt.Client(client_id="yamnet_mqtt_test")
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    # Start MQTT loop in background thread
    client.loop_start()
    
    # Wait for subscription
    time.sleep(2)
    
    # Send audio
    print(f"\n📤 Sending {len(audio_8bit)} bytes to: {MQTT_TOPIC}")
    raw_bytes = audio_8bit.tobytes()
    result = client.publish(MQTT_TOPIC, raw_bytes)
    result.wait_for_publish()
    print("✅ Audio sent!")
    
    # Wait for classification result
    print("\n⏳ Waiting for classification result...")
    timeout = 30
    start = time.time()
    while not received_results and (time.time() - start) < timeout:
        time.sleep(0.5)
    
    if received_results:
        print("\n" + "="*50)
        print("✅ MQTT TEST SUCCESSFUL!")
        print(f"   Detected: {received_results[0][0]} ({received_results[0][1]:.2%})")
        print("="*50)
    else:
        print("⚠️  Timeout waiting for result")
    
    client.loop_stop()

if __name__ == '__main__':
    main()

