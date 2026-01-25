#!/usr/bin/env python3
"""
YAMNet Audio Classification via MQTT
Subscribes to MQTT topic, receives 8-bit audio data, and classifies it.
"""

import os
import json
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import scipy.signal
from dotenv import load_dotenv

# MQTT
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("⚠️  Install MQTT: pip install paho-mqtt")

# Gemini API (optional)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

load_dotenv()
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

class_map_path = model.class_map_path().numpy()
class_names = class_names_from_csv(class_map_path)
print(f"✅ Loaded {len(class_names)} sound classes\n")

def process_8bit_audio(audio_data, sample_rate=16000, desired_sample_rate=16000):
    """
    Convert 8-bit audio (uint8, 0-255) to YAMNet format:
    - Convert to float32 in [-1.0, 1.0] range
    - Resample to 16kHz if needed
    """
    # Convert uint8 (0-255) to float32 in [-1.0, 1.0]
    # Center at 0: subtract 128, then normalize
    waveform_float = (audio_data.astype(np.float32) - 128.0) / 128.0
    
    # Ensure mono (in case it's not)
    if len(waveform_float.shape) > 1:
        waveform_float = np.mean(waveform_float, axis=1)
    
    # Resample to 16kHz if needed
    if sample_rate != desired_sample_rate:
        desired_length = int(round(float(len(waveform_float)) /
                                   sample_rate * desired_sample_rate))
        waveform_float = scipy.signal.resample(waveform_float, desired_length)
    
    return desired_sample_rate, waveform_float

def classify_audio(waveform):
    """Classify audio using YAMNet and return results."""
    scores, _, _ = model(waveform)
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
    
    return top_class, confidence, predictions

def get_gemini_description(sound_class, confidence, predictions):
    """Generate AI description of detected sound using Gemini API."""
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
        print(f"⚠️  Gemini API error: {e}")
        return None

# MQTT callbacks
def on_connect(client, userdata, flags, rc):
    """Called when MQTT client connects."""
    if rc == 0:
        print("✅ Connected to MQTT broker")
        topic = os.environ.get('MQTT_TOPIC', 'audio/raw')
        client.subscribe(topic)
        print(f"📡 Subscribed to topic: {topic}")
    else:
        print(f"❌ Failed to connect to MQTT broker (code: {rc})")

def on_message(client, userdata, msg):
    """Called when a message is received on subscribed topic."""
    try:
        # Parse message - can be JSON with metadata or raw audio bytes
        try:
            # Try JSON format first (includes sample_rate, etc.)
            payload = json.loads(msg.payload)
            audio_bytes = bytes(payload.get('audio', payload.get('data', b'')))
            sample_rate = payload.get('sample_rate', 16000)
        except (json.JSONDecodeError, TypeError):
            # Raw audio bytes
            audio_bytes = msg.payload
            sample_rate = 16000  # Default assumption
        
        if len(audio_bytes) == 0:
            print("⚠️  Received empty audio data")
            return
        
        # Convert bytes to numpy array (8-bit unsigned)
        audio_data = np.frombuffer(audio_bytes, dtype=np.uint8)
        
        print(f"\n📥 Received {len(audio_data)} samples ({len(audio_data)/sample_rate:.2f}s @ {sample_rate}Hz)")
        
        # Process 8-bit audio
        sample_rate_processed, waveform = process_8bit_audio(audio_data, sample_rate)
        
        # Classify
        print("🔍 Classifying...")
        top_class, confidence, predictions = classify_audio(waveform)
        
        # Display results
        print(f"\n🎯 Top sound: {top_class} ({confidence:.2%})")
        print(f"📊 Top 5 predictions:")
        for i, p in enumerate(predictions, 1):
            print(f"   {i}. {p['class']:30s} {p['confidence']:.2%}")
        
        # AI description
        if gemini_model:
            print(f"\n🤖 Generating AI description...")
            description = get_gemini_description(top_class, confidence, predictions)
            if description:
                print(f"💡 {description}")
        
        # Publish results back to MQTT (optional)
        result_topic = os.environ.get('MQTT_RESULT_TOPIC', 'audio/classification')
        if result_topic:
            result = {
                'class': top_class,
                'confidence': float(confidence),
                'top_5': predictions,
                'description': description if gemini_model else None
            }
            client.publish(result_topic, json.dumps(result))
            print(f"📤 Published results to: {result_topic}")
        
    except Exception as e:
        print(f"❌ Error processing audio: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main MQTT client setup and loop."""
    if not MQTT_AVAILABLE:
        print("❌ MQTT library not available. Install: pip install paho-mqtt")
        return
    
    # MQTT configuration from environment variables
    mqtt_broker = os.environ.get('MQTT_BROKER', 'localhost')
    mqtt_port = int(os.environ.get('MQTT_PORT', 1883))
    mqtt_username = os.environ.get('MQTT_USERNAME')
    mqtt_password = os.environ.get('MQTT_PASSWORD')
    client_id = os.environ.get('MQTT_CLIENT_ID', 'yamnet_classifier')
    
    print(f"\n🔌 Connecting to MQTT broker: {mqtt_broker}:{mqtt_port}")
    
    # Create MQTT client
    client = mqtt.Client(client_id=client_id)
    
    # Set credentials if provided
    if mqtt_username and mqtt_password:
        client.username_pw_set(mqtt_username, mqtt_password)
    
    # Set callbacks
    client.on_connect = on_connect
    client.on_message = on_message
    
    # Connect and start loop
    try:
        client.connect(mqtt_broker, mqtt_port, 60)
        print("🔄 Starting MQTT loop (press Ctrl+C to stop)...\n")
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
        client.disconnect()
    except Exception as e:
        print(f"❌ MQTT connection error: {e}")

if __name__ == '__main__':
    main()

