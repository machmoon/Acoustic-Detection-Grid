# Acoustic Detection Grid

Distributed sound classification system using YAMNet + AI.

## Architecture

```
[ESP32/STM32 Microcontrollers] --MQTT--> [Server] ---> [YAMNet + Gemini AI]
         │                                   │
    Record Audio                      Classify & Alert
```

## Project Structure

```
├── server/          # Python classification server
│   ├── yamnet_mqtt.py      # MQTT receiver + classifier
│   ├── yamnet_gemini.py    # File-based classifier with AI
│   └── requirements.txt    # Python dependencies
│
├── esp32/           # ESP32 Arduino code
│   └── (audio capture + MQTT publish)
│
├── stm32/           # STM32 code
│   └── (audio capture + MQTT publish)
│
└── audio/           # Test audio files
    ├── breakin.wav
    ├── breakin16khz.wav
    └── miaow_16k.wav
```

## Quick Start

### 1. Server Setup

```bash
cd server
pip install -r requirements.txt

# Create .env with your Gemini API key
echo 'GEMINI_API_KEY="your_key_here"' > .env

# Test with a file
python yamnet_gemini.py ../audio/breakin16khz.wav

# Or run MQTT listener
MQTT_BROKER="test.mosquitto.org" python yamnet_mqtt.py
```

### 2. Test MQTT Flow

```bash
cd server
python mqtt_test.py ../audio/breakin16khz.wav
```

### 3. ESP32/STM32 Setup

See `esp32/` and `stm32/` folders for microcontroller code.

## MQTT Protocol

**Topic:** `audio/raw`

**Payload:** Raw 8-bit unsigned audio bytes (0-255)

**Sample Rate:** 16kHz recommended

## Features

- ✅ 521 sound classes (YAMNet)
- ✅ AI descriptions (Gemini)
- ✅ MQTT for IoT devices
- ✅ Real-time classification
