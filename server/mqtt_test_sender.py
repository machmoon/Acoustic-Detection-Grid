#!/usr/bin/env python3
"""
MQTT Test Sender - Sends a WAV file as 8-bit audio over MQTT
Usage: python mqtt_test_sender.py [wav_file] [mqtt_broker]
"""

import os
import sys
import json
import numpy as np
from scipy.io import wavfile

try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("❌ Install MQTT: pip install paho-mqtt")
    sys.exit(1)

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
    elif wav_data.dtype in [np.float32, np.float64]:
        wav_float = wav_data.astype(np.float32)
        max_val = np.abs(wav_float).max()
        if max_val > 1.0:
            wav_float = wav_float / max_val
    else:
        wav_float = wav_data.astype(np.float32)
    
    # Convert to 8-bit unsigned (0-255)
    # Map [-1.0, 1.0] to [0, 255]
    wav_8bit = ((wav_float + 1.0) * 127.5).astype(np.uint8)
    
    return sample_rate, wav_8bit

def main():
    # Config
    wav_file = sys.argv[1] if len(sys.argv) > 1 else 'breakin16khz.wav'
    mqtt_broker = sys.argv[2] if len(sys.argv) > 2 else os.environ.get('MQTT_BROKER', 'localhost')
    mqtt_port = int(os.environ.get('MQTT_PORT', 1883))
    mqtt_topic = os.environ.get('MQTT_TOPIC', 'audio/raw')
    
    print(f"📁 Loading: {wav_file}")
    sample_rate, audio_8bit = wav_to_8bit(wav_file)
    
    print(f"   Sample rate: {sample_rate} Hz")
    print(f"   Duration: {len(audio_8bit)/sample_rate:.2f}s")
    print(f"   Samples: {len(audio_8bit)}")
    print(f"   8-bit range: [{audio_8bit.min()}, {audio_8bit.max()}]")
    
    # Connect to MQTT
    print(f"\n🔌 Connecting to MQTT broker: {mqtt_broker}:{mqtt_port}")
    
    client = mqtt.Client(client_id="yamnet_test_sender")
    
    try:
        client.connect(mqtt_broker, mqtt_port, 60)
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        print("\n💡 Start a local MQTT broker:")
        print("   brew install mosquitto")
        print("   mosquitto -v")
        print("\n   Or use a public broker:")
        print("   python mqtt_test_sender.py breakin16khz.wav test.mosquitto.org")
        sys.exit(1)
    
    # Send as JSON with sample rate info
    payload = {
        'audio': audio_8bit.tobytes().hex(),  # Hex-encoded bytes
        'sample_rate': sample_rate,
        'format': '8bit_unsigned'
    }
    
    # Also send raw bytes option
    raw_bytes = audio_8bit.tobytes()
    
    print(f"\n📤 Publishing to topic: {mqtt_topic}")
    print(f"   Payload size: {len(raw_bytes)} bytes")
    
    # Send raw bytes (simpler, yamnet_mqtt.py handles this)
    result = client.publish(mqtt_topic, raw_bytes)
    result.wait_for_publish()
    
    if result.rc == mqtt.MQTT_ERR_SUCCESS:
        print("✅ Audio sent successfully!")
        print(f"\n💡 Run the receiver in another terminal:")
        print(f"   python yamnet_mqtt.py")
    else:
        print(f"❌ Failed to publish (code: {result.rc})")
    
    client.disconnect()

if __name__ == '__main__':
    main()

