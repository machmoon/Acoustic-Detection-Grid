#!/usr/bin/env python3
"""
ElevenLabs Text-to-Speech integration for voice alerts.
"""

import os
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

ELEVENLABS_API_KEY = os.environ.get('ELEVENLABS_API_KEY')
ELEVENLABS_VOICE_ID = os.environ.get('ELEVENLABS_VOICE_ID', 'pNInz6obpgDQGcFmaJgB')  # Default: Adam

# Voice options: https://api.elevenlabs.io/v1/voices
# Adam: pNInz6obpgDQGcFmaJgB (clear, professional)
# Rachel: 21m00Tcm4TlvDq8ikWAM (calm, female)
# Domi: AZnzlk1XvdvUeBnXmlld (strong, confident)

def text_to_speech(text, voice_id=None):
    """
    Convert text to speech using ElevenLabs API.
    Returns base64-encoded audio data for web playback.
    """
    if not ELEVENLABS_API_KEY:
        print("⚠️  ELEVENLABS_API_KEY not set")
        return None
    
    voice_id = voice_id or ELEVENLABS_VOICE_ID
    
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    
    data = {
        "text": text,
        "model_id": "eleven_turbo_v2_5",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.5,
            "use_speaker_boost": True
        }
    }
    
    try:
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 200:
            # Return base64 encoded audio for web playback
            audio_base64 = base64.b64encode(response.content).decode('utf-8')
            return audio_base64
        else:
            print(f"⚠️  ElevenLabs API error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"⚠️  ElevenLabs error: {e}")
        return None


def generate_alert_message(sound_class, confidence, threat_level, zone="detected area"):
    """Generate appropriate alert message based on detection."""
    
    if threat_level == 'danger':
        return f"Alert! {sound_class} detected in {zone} with {int(confidence * 100)} percent confidence. Please investigate immediately."
    
    elif threat_level == 'warning':
        return f"Warning. {sound_class} detected in {zone}. Monitoring situation."
    
    else:
        return f"{sound_class} detected in {zone}."


def speak_alert(sound_class, confidence, threat_level, zone="the monitored area"):
    """
    Generate and return voice alert for a detection.
    Returns base64 audio or None if disabled/error.
    """
    message = generate_alert_message(sound_class, confidence, threat_level, zone)
    return text_to_speech(message)


# Test
if __name__ == '__main__':
    print("Testing ElevenLabs TTS...")
    
    if not ELEVENLABS_API_KEY:
        print("❌ Set ELEVENLABS_API_KEY in .env")
    else:
        audio = speak_alert("Glass breaking", 0.94, "danger", "the living room")
        if audio:
            print(f"✅ Generated audio ({len(audio)} bytes base64)")
            # Save test file
            import base64
            with open("test_alert.mp3", "wb") as f:
                f.write(base64.b64decode(audio))
            print("   Saved to test_alert.mp3")
        else:
            print("❌ Failed to generate audio")

