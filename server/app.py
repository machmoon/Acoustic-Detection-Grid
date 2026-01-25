#!/usr/bin/env python3
"""
EchoGuard: Acoustic Radar Dashboard
Flask + WebSocket server with YAMNet classification and MQTT integration
"""

import os
import json
import threading
from datetime import datetime
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import numpy as np
import scipy.signal
from dotenv import load_dotenv

# TensorFlow (load before other imports to suppress warnings)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow_hub as hub
import csv

# MQTT
try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("⚠️  MQTT not available: pip install paho-mqtt")

# Gemini API
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ElevenLabs TTS
try:
    from elevenlabs_tts import speak_alert, ELEVENLABS_API_KEY
    ELEVENLABS_AVAILABLE = bool(ELEVENLABS_API_KEY)
    if ELEVENLABS_AVAILABLE:
        print("✅ ElevenLabs voice alerts enabled")
except ImportError:
    ELEVENLABS_AVAILABLE = False

load_dotenv()

# Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'echoguard-secret-key')
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
events_log = []
system_status = {
    'armed': True,
    'mode': 'away',
    'ambient_db': 32.4,
    'health': 94
}

# Gemini setup
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
gemini_model = None
if GEMINI_AVAILABLE and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    print("✅ Gemini API enabled")

# Load YAMNet
print("Loading YAMNet model...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

def class_names_from_csv(class_map_csv_text):
    class_names = []
    with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])
    return class_names

class_map_path = yamnet_model.class_map_path().numpy()
class_names = class_names_from_csv(class_map_path)
print(f"✅ Loaded {len(class_names)} sound classes")

# Threat classification mapping
THREAT_SOUNDS = {
    'Glass': 'danger',
    'Breaking': 'danger', 
    'Shatter': 'danger',
    'Gunshot': 'danger',
    'Explosion': 'danger',
    'Scream': 'danger',
    'Alarm': 'warning',
    'Siren': 'warning',
    'Door': 'info',
    'Knock': 'info',
    'Doorbell': 'info',
    'Speech': 'info',
    'Dog': 'info',
    'Cat': 'info',
}

def get_threat_level(sound_class):
    """Determine threat level from sound class."""
    for keyword, level in THREAT_SOUNDS.items():
        if keyword.lower() in sound_class.lower():
            return level
    return 'normal'

def process_8bit_audio(audio_data, sample_rate=100000, desired_sample_rate=16000):
    """Convert 8-bit audio to YAMNet format."""
    waveform_float = (audio_data.astype(np.float32) - 128.0) / 128.0
    
    if len(waveform_float.shape) > 1:
        waveform_float = np.mean(waveform_float, axis=1)
    
    if sample_rate != desired_sample_rate:
        desired_length = int(round(float(len(waveform_float)) / sample_rate * desired_sample_rate))
        waveform_float = scipy.signal.resample(waveform_float, desired_length)
    
    return desired_sample_rate, waveform_float

def classify_audio(waveform):
    """Classify audio using YAMNet."""
    scores, _, _ = yamnet_model(waveform)
    scores_np = scores.numpy()
    mean_scores = np.mean(scores_np, axis=0)
    
    top_idx = mean_scores.argmax()
    top_class = class_names[top_idx]
    confidence = float(mean_scores[top_idx])
    
    top_5 = np.argsort(mean_scores)[::-1][:5]
    predictions = [{'class': class_names[i], 'confidence': float(mean_scores[i])} 
                  for i in top_5]
    
    return top_class, confidence, predictions

def get_gemini_description(sound_class, confidence, predictions):
    """Generate AI description using Gemini."""
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

def add_event(sound_class, confidence, threat_level, description=None, db_level=None):
    """Add event to log and broadcast to clients."""
    event = {
        'id': len(events_log) + 1,
        'sound_class': sound_class,
        'confidence': confidence,
        'threat_level': threat_level,
        'description': description,
        'db_level': db_level or np.random.randint(30, 110),
        'timestamp': datetime.now().isoformat(),
        'timestamp_display': datetime.now().strftime('%I:%M %p'),
        'zone': 'Living Room'  # TODO: multi-zone support
    }
    events_log.insert(0, event)
    
    # Keep only last 50 events
    if len(events_log) > 50:
        events_log.pop()
    
    return event

# MQTT callbacks
mqtt_client = None

def on_mqtt_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ Connected to MQTT broker")
        topic = os.environ.get('MQTT_TOPIC', 'goontronics')
        client.subscribe(topic)
        print(f"📡 Subscribed to: {topic}")
        socketio.emit('mqtt_status', {'connected': True})
    else:
        print(f"❌ MQTT connection failed (code: {rc})")
        socketio.emit('mqtt_status', {'connected': False})

def on_mqtt_message(client, userdata, msg):
    """Process incoming MQTT audio data."""
    default_sample_rate = int(os.environ.get('AUDIO_SAMPLE_RATE', 100000))
    
    try:
        # Parse message
        try:
            payload = json.loads(msg.payload)
            audio_bytes = bytes(payload.get('audio', payload.get('data', [])))
            if not audio_bytes and 'audio' in payload:
                audio_bytes = bytes(payload['audio'])
            sample_rate = payload.get('sample_rate', default_sample_rate)
        except (json.JSONDecodeError, TypeError):
            audio_bytes = msg.payload
            sample_rate = default_sample_rate
        
        if len(audio_bytes) == 0:
            return
        
        # Convert to numpy array
        audio_data = np.frombuffer(audio_bytes, dtype=np.uint8)
        
        print(f"📥 Received {len(audio_data)} samples @ {sample_rate}Hz")
        socketio.emit('audio_received', {
            'samples': len(audio_data),
            'sample_rate': sample_rate,
            'duration': len(audio_data) / sample_rate
        })
        
        # Process and classify
        _, waveform = process_8bit_audio(audio_data, sample_rate)
        top_class, confidence, predictions = classify_audio(waveform)
        
        threat_level = get_threat_level(top_class)
        
        # Get AI description
        description = None
        if gemini_model and (threat_level in ['danger', 'warning'] or confidence > 0.5):
            socketio.emit('ai_processing', {'status': 'Analyzing with Gemini AI...'})
            description = get_gemini_description(top_class, confidence, predictions)
        
        # Add to event log
        event = add_event(top_class, confidence, threat_level, description)
        
        # Generate voice alert for threats
        voice_alert = None
        if ELEVENLABS_AVAILABLE and threat_level in ['danger', 'warning']:
            print("🔊 Generating voice alert...")
            voice_alert = speak_alert(top_class, confidence, threat_level, event.get('zone', 'the area'))
        
        # Broadcast to all clients
        result = {
            'event': event,
            'predictions': predictions,
            'ai_description': description,
            'voice_alert': voice_alert  # Base64 audio or None
        }
        
        print(f"🎯 Classified: {top_class} ({confidence:.1%}) - {threat_level}")
        socketio.emit('classification_result', result)
        
    except Exception as e:
        print(f"❌ Error processing audio: {e}")
        import traceback
        traceback.print_exc()

def start_mqtt_client():
    """Start MQTT client in background thread."""
    global mqtt_client
    
    if not MQTT_AVAILABLE:
        print("⚠️  MQTT not available")
        return
    
    mqtt_broker = os.environ.get('MQTT_BROKER', 'test.mosquitto.org')
    mqtt_port = int(os.environ.get('MQTT_PORT', 1883))
    
    print(f"🔌 Connecting to MQTT: {mqtt_broker}:{mqtt_port}")
    
    mqtt_client = mqtt.Client(client_id='echoguard_server')
    mqtt_client.on_connect = on_mqtt_connect
    mqtt_client.on_message = on_mqtt_message
    
    try:
        mqtt_client.connect(mqtt_broker, mqtt_port, 60)
        mqtt_client.loop_start()
    except Exception as e:
        print(f"❌ MQTT connection error: {e}")

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    return jsonify(system_status)

@app.route('/api/events')
def get_events():
    return jsonify(events_log[:20])

# WebSocket events
@socketio.on('connect')
def handle_connect():
    print('🔗 Client connected')
    emit('system_status', system_status)
    emit('events_update', events_log[:10])

@socketio.on('disconnect')
def handle_disconnect():
    print('🔌 Client disconnected')

@socketio.on('arm_system')
def handle_arm(data):
    system_status['armed'] = data.get('armed', True)
    system_status['mode'] = data.get('mode', 'away')
    socketio.emit('system_status', system_status)

@socketio.on('test_classification')
def handle_test():
    """Test classification with a dummy event."""
    event = add_event(
        'Glass (Breaking)', 
        0.94, 
        'danger',
        'A high-frequency impact signature was detected. The sound profile matches tempered glass shattering with 94% confidence.',
        104
    )
    
    # Generate voice alert for test
    voice_alert = None
    if ELEVENLABS_AVAILABLE:
        voice_alert = speak_alert('Glass breaking', 0.94, 'danger', 'the kitchen')
    
    socketio.emit('classification_result', {
        'event': event,
        'predictions': [
            {'class': 'Glass', 'confidence': 0.94},
            {'class': 'Breaking', 'confidence': 0.87},
            {'class': 'Crash', 'confidence': 0.45}
        ],
        'ai_description': event['description'],
        'voice_alert': voice_alert
    })

if __name__ == '__main__':
    # Start MQTT in background
    mqtt_thread = threading.Thread(target=start_mqtt_client, daemon=True)
    mqtt_thread.start()
    
    print("\n🚀 Starting EchoGuard Dashboard...")
    print("   Open http://localhost:8080 in your browser\n")
    
    socketio.run(app, host='0.0.0.0', port=8080, debug=False, allow_unsafe_werkzeug=True)

