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

# Sounddevice for audio playback
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
    print("✅ Sounddevice audio playback enabled")
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("⚠️  Sounddevice not available: pip install sounddevice")

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

# =============================================================================
# TRIANGULATION (TDOA) - 3-Mic Array
# =============================================================================
# Mic positions (mm from center) - isoceles triangle: 50mm, 50mm, 60mm
MIC_POSITIONS = {
    'mic1': (0.0, 50.0),        # 50mm from center, top
    'mic2': (-43.3, -25.0),     # 50mm from center, bottom-left  
    'mic3': (51.96, -30.0),     # 60mm from center, bottom-right
}
SPEED_OF_SOUND = 343000  # mm/s

def gcc_phat(sig1, sig2, fs=100000):
    """
    Generalized Cross-Correlation with Phase Transform.
    More robust to noise and reverberation than basic correlation.
    Returns time delay (tau) in seconds.
    """
    n = len(sig1) + len(sig2)
    SIG1 = np.fft.rfft(sig1, n=n)
    SIG2 = np.fft.rfft(sig2, n=n)
    R = SIG1 * np.conj(SIG2)
    R /= (np.abs(R) + 1e-10)  # Phase transform (whitening)
    cc = np.fft.irfft(R, n=n)
    max_shift = len(sig1) // 2
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    tau = shift / fs
    return tau

def estimate_direction(mic1_audio, mic2_audio, mic3_audio, sample_rate=100000):
    """
    Estimate 2D direction to sound source using TDOA.
    Returns dict with angle (degrees, 0°=up, clockwise) and confidence.
    """
    try:
        # Get time delays between mic pairs
        tau_12 = gcc_phat(mic1_audio, mic2_audio, sample_rate)
        tau_13 = gcc_phat(mic1_audio, mic3_audio, sample_rate)
        tau_23 = gcc_phat(mic2_audio, mic3_audio, sample_rate)
        
        # Convert to distance differences (mm)
        d12 = tau_12 * SPEED_OF_SOUND
        d13 = tau_13 * SPEED_OF_SOUND
        d23 = tau_23 * SPEED_OF_SOUND
        
        # Mic positions
        x1, y1 = MIC_POSITIONS['mic1']
        x2, y2 = MIC_POSITIONS['mic2']
        x3, y3 = MIC_POSITIONS['mic3']
        
        # Least-squares direction estimate
        A = np.array([
            [x1 - x2, y1 - y2],
            [x1 - x3, y1 - y3],
            [x2 - x3, y2 - y3]
        ])
        b = np.array([d12, d13, d23]) * 0.5
        
        # Solve for direction vector
        result, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        # Normalize to get unit direction
        norm = np.linalg.norm(result)
        if norm > 0:
            direction = result / norm
        else:
            direction = np.array([0, 1])
        
        # Convert to angle (degrees, 0° = up/north, clockwise positive)
        angle = np.degrees(np.arctan2(direction[0], direction[1]))
        
        # Confidence based on residuals (lower = better fit)
        confidence = max(0.0, min(1.0, 1.0 - (np.sum(residuals) / 1000) if len(residuals) > 0 else 0.8))
        
        return {
            'angle': float(angle),
            'direction': {'x': float(direction[0]), 'y': float(direction[1])},
            'tdoa_us': {
                'tau_12': float(tau_12 * 1e6),
                'tau_13': float(tau_13 * 1e6),
                'tau_23': float(tau_23 * 1e6)
            },
            'confidence': float(confidence)
        }
    except Exception as e:
        print(f"⚠️  Triangulation error: {e}")
        return None

def process_8bit_audio(audio_data, sample_rate=100000, desired_sample_rate=16000):
    """Convert 8-bit audio to YAMNet format."""
    waveform_float = (audio_data.astype(np.float32) - 128.0) / 128.0
    
    if len(waveform_float.shape) > 1:
        waveform_float = np.mean(waveform_float, axis=1)
    
    if sample_rate != desired_sample_rate:
        desired_length = int(round(float(len(waveform_float)) / sample_rate * desired_sample_rate))
        waveform_float = scipy.signal.resample(waveform_float, desired_length)
    
    return desired_sample_rate, waveform_float

def play_audio(audio_data, sample_rate=100000, playback_rate=44100):
    """Play 8-bit audio through speakers and wait for completion."""
    if not SOUNDDEVICE_AVAILABLE:
        print("⚠️  Cannot play audio - sounddevice not installed")
        return
    
    try:
        # Convert 8-bit unsigned to float [-1, 1]
        audio_float = (audio_data.astype(np.float32) - 128.0) / 128.0
        
        # Resample to playback rate if needed
        if sample_rate != playback_rate:
            num_samples = int(len(audio_float) * playback_rate / sample_rate)
            audio_float = scipy.signal.resample(audio_float, num_samples)
        
        duration = len(audio_float) / playback_rate
        print(f"🔊 Playing audio ({duration:.2f}s @ {playback_rate}Hz)...")
        
        # Play and wait for completion
        sd.play(audio_float, playback_rate)
        sd.wait()  # Block until playback is done
        
        print("✅ Audio playback complete")
    except Exception as e:
        print(f"❌ Audio playback error: {e}")

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

# Frame tracking for ESP32
frame_active = False

def on_mqtt_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ Connected to MQTT broker")
        # Use wildcard to catch all goontronics topics (goontronics, goontronics/esp32, etc.)
        topic = os.environ.get('MQTT_TOPIC', 'goontronics/#')
        client.subscribe(topic)
        print(f"📡 Subscribed to: {topic}")
        socketio.emit('mqtt_status', {'connected': True})
    else:
        print(f"❌ MQTT connection failed (code: {rc})")
        socketio.emit('mqtt_status', {'connected': False})

def on_mqtt_message(client, userdata, msg):
    """Process incoming MQTT audio data with optional triangulation."""
    default_sample_rate = int(os.environ.get('AUDIO_SAMPLE_RATE', 100000))
    
    global frame_active
    
    try:
        # Check for frame start signal
        try:
            text_msg = msg.payload.decode('utf-8').strip()
            if text_msg == "Goontronics Engaged":
                print("🚀 Frame start: Goontronics Engaged - awaiting audio buffer...")
                frame_active = True
                socketio.emit('frame_start', {'status': 'Recording started'})
                return
        except:
            pass  # Not a text message, continue processing as audio
        
        # Print received message info
        print(f"📨 MQTT message received on topic: {msg.topic} ({len(msg.payload)} bytes)")
        
        # Parse message
        direction = None
        payload = None
        
        # Try to parse as JSON, but expect raw bytes most of the time
        try:
            # Only try JSON if it looks like text (starts with { or [)
            if msg.payload and len(msg.payload) > 0:
                first_byte = msg.payload[0]
                if first_byte in (0x7B, 0x5B):  # '{' or '['
                    payload = json.loads(msg.payload.decode('utf-8'))
        except Exception:
            payload = None  # Raw binary data, not JSON
        
        # === MULTI-MIC MODE ===
        # Array format: [[mic1], [mic2], ..., [micN], sample_rate]
        #   - payload[0:-1] = mic arrays (variable number)
        #   - payload[-1] = sample_rate (last element)
        # Object format: {"mic1": [...], "mic2": [...], "mic3": [...], "sample_rate": 100000}
        is_mic_array = payload and isinstance(payload, list) and len(payload) >= 2 and isinstance(payload[0], list)
        is_mic_object = payload and isinstance(payload, dict) and 'mic1' in payload
        
        if is_mic_array or is_mic_object:
            # Extract mic data based on format
            if is_mic_array:
                # payload[0:-1] = mic arrays, payload[-1] = sample_rate
                mic_arrays = payload[0:-1]  # All but last
                sample_rate = payload[-1]    # Last element
                num_mics = len(mic_arrays)
                print(f"📥 Received {num_mics}-mic array: {len(mic_arrays[0])} samples @ {sample_rate}Hz")
            else:
                # Count mic keys (mic1, mic2, mic3, etc.)
                mic_arrays = []
                i = 1
                while f'mic{i}' in payload:
                    mic_arrays.append(payload[f'mic{i}'])
                    i += 1
                num_mics = len(mic_arrays)
                sample_rate = payload.get('sample_rate', 100000)
                print(f"📥 Received {num_mics}-mic object: {len(mic_arrays[0])} samples @ {sample_rate}Hz")
            
            # Convert all mics to float [-1, 1]
            mics_float = []
            for mic_data in mic_arrays:
                mic_np = np.array(mic_data, dtype=np.uint8)
                mic_f = (mic_np.astype(np.float32) - 128.0) / 128.0
                mics_float.append(mic_f)
            
            # Triangulation requires at least 3 mics
            if num_mics >= 3:
                direction = estimate_direction(mics_float[0], mics_float[1], mics_float[2], sample_rate)
                if direction:
                    print(f"📍 Direction: {direction['angle']:.1f}° (confidence: {direction['confidence']:.0%})")
                    socketio.emit('triangulation_result', direction)
            else:
                print(f"⚠️  Only {num_mics} mics - need 3+ for triangulation")
            
            # Average all mics for classification (better SNR)
            audio_avg = np.mean(mics_float, axis=0)
            audio_data = ((audio_avg * 128.0) + 128.0).astype(np.uint8)
        
        # === PRE-COMPUTED DIRECTION FROM NODE ===
        elif payload and isinstance(payload, dict) and 'direction' in payload:
            direction = payload['direction']
            audio_bytes = bytes(payload.get('audio', payload.get('data', [])))
            audio_data = np.frombuffer(audio_bytes, dtype=np.uint8)
            sample_rate = payload.get('sample_rate', default_sample_rate)
            print(f"📥 Received {len(audio_data)} samples with direction: {direction.get('angle', '?')}°")
            
        # === SINGLE MIC MODE (original) ===
        elif payload:
            audio_bytes = bytes(payload.get('audio', payload.get('data', [])))
            if not audio_bytes and 'audio' in payload:
                audio_bytes = bytes(payload['audio'])
            audio_data = np.frombuffer(audio_bytes, dtype=np.uint8)
            sample_rate = payload.get('sample_rate', default_sample_rate)
            print(f"📥 Received {len(audio_data)} samples @ {sample_rate}Hz")
        else:
            # === RAW BYTES FROM ESP32 (no JSON) ===
            # Just raw uint8 ADC samples
            audio_data = np.frombuffer(msg.payload, dtype=np.uint8)
            sample_rate = default_sample_rate  # 100kHz
            print(f"📥 ESP32 raw bytes: {len(audio_data)} samples @ {sample_rate}Hz")
        
        if len(audio_data) == 0:
            return
        
        socketio.emit('audio_received', {
            'samples': len(audio_data),
            'sample_rate': sample_rate,
            'duration': len(audio_data) / sample_rate,
            'has_direction': direction is not None
        })
        
        # Play the received audio through speakers (blocks until done)
        play_audio(audio_data, sample_rate)
        
        # Process and classify
        _, waveform = process_8bit_audio(audio_data, sample_rate)
        top_class, confidence, predictions = classify_audio(waveform)
        
        threat_level = get_threat_level(top_class)
        
        # Get AI description
        description = None
        if gemini_model and (threat_level in ['danger', 'warning'] or confidence > 0.5):
            socketio.emit('ai_processing', {'status': 'Analyzing with Gemini AI...'})
            description = get_gemini_description(top_class, confidence, predictions)
        
        # Add to event log (with direction if available)
        event = add_event(top_class, confidence, threat_level, description)
        if direction:
            event['direction'] = direction
        
        # Generate voice alert for threats
        voice_alert = None
        if ELEVENLABS_AVAILABLE and threat_level in ['danger', 'warning']:
            print("🔊 Generating voice alert...")
            # Include direction in voice alert if available
            if direction:
                zone = f"{direction['angle']:.0f} degrees from the sensor"
            else:
                zone = event.get('zone', 'the area')
            voice_alert = speak_alert(top_class, confidence, threat_level, zone)
        
        # Broadcast to all clients
        result = {
            'event': event,
            'predictions': predictions,
            'ai_description': description,
            'voice_alert': voice_alert,
            'direction': direction  # Include triangulation data
        }
        
        print(f"🎯 Classified: {top_class} ({confidence:.1%}) - {threat_level}")
        if direction:
            print(f"   └─ Direction: {direction['angle']:.1f}° ({direction['confidence']:.0%} confidence)")
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
    """Test classification with a dummy event including triangulation."""
    # Simulated direction from triangulation
    test_direction = {
        'angle': 45.0,
        'direction': {'x': 0.707, 'y': 0.707},
        'tdoa_us': {'tau_12': 87.5, 'tau_13': 102.3, 'tau_23': 14.8},
        'confidence': 0.85
    }
    
    event = add_event(
        'Glass (Breaking)', 
        0.94, 
        'danger',
        'A high-frequency impact signature was detected. The sound profile matches tempered glass shattering with 94% confidence.',
        104
    )
    event['direction'] = test_direction
    
    # Generate voice alert for test
    voice_alert = None
    if ELEVENLABS_AVAILABLE:
        voice_alert = speak_alert('Glass breaking', 0.94, 'danger', '45 degrees from the sensor')
    
    socketio.emit('classification_result', {
        'event': event,
        'predictions': [
            {'class': 'Glass', 'confidence': 0.94},
            {'class': 'Breaking', 'confidence': 0.87},
            {'class': 'Crash', 'confidence': 0.45}
        ],
        'ai_description': event['description'],
        'voice_alert': voice_alert,
        'direction': test_direction
    })
    
    print(f"🧪 Test event with direction: {test_direction['angle']}°")

if __name__ == '__main__':
    # Start MQTT in background
    mqtt_thread = threading.Thread(target=start_mqtt_client, daemon=True)
    mqtt_thread.start()
    
    print("\n🚀 Starting EchoGuard Dashboard...")
    print("   Open http://localhost:8080 in your browser\n")
    
    socketio.run(app, host='0.0.0.0', port=8080, debug=False, allow_unsafe_werkzeug=True)

