#!/usr/bin/env python3
"""
Simple MQTT Receiver - Listen to ESP32 messages
Equivalent to: mosquitto_sub -h test.mosquitto.org -p 1883 -t "goontronics/#"
"""

import paho.mqtt.client as mqtt

MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC = "goontronics/#"

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"✅ Connected to {MQTT_BROKER}")
        client.subscribe(MQTT_TOPIC)
        print(f"📡 Subscribed to: {MQTT_TOPIC}")
        print("Waiting for messages... (Ctrl+C to stop)\n")
    else:
        print(f"❌ Connection failed (code: {rc})")

def on_message(client, userdata, msg):
    print(f"📥 Topic: {msg.topic}")
    print(f"   Payload ({len(msg.payload)} bytes): {msg.payload[:100]}")  # First 100 bytes
    print()

client = mqtt.Client(client_id="yamnet_receiver")
client.on_connect = on_connect
client.on_message = on_message

print(f"🔌 Connecting to {MQTT_BROKER}:{MQTT_PORT}...")
client.connect(MQTT_BROKER, MQTT_PORT, 60)

try:
    client.loop_forever()
except KeyboardInterrupt:
    print("\n👋 Stopped")
    client.disconnect()

