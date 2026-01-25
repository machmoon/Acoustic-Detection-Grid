# Server - YAMNet Classification

Python server that classifies audio using YAMNet and generates AI descriptions.

## Setup

```bash
pip install -r requirements.txt

# Create .env with your Gemini API key (optional)
echo 'GEMINI_API_KEY="your_key_here"' > .env
```

## Scripts

| File | Description |
|------|-------------|
| `yamnet_mqtt.py` | MQTT listener - receives audio, classifies, publishes results |
| `yamnet_gemini.py` | File classifier with Gemini AI descriptions |
| `yamnet.py` | Basic file classifier (no AI) |
| `mqtt_test.py` | Test MQTT send/receive flow |

## Usage

### Classify a File

```bash
python yamnet_gemini.py ../audio/breakin16khz.wav
```

### Run MQTT Listener

```bash
# Use public test broker
MQTT_BROKER="test.mosquitto.org" python yamnet_mqtt.py

# Or set in .env
echo 'MQTT_BROKER="test.mosquitto.org"' >> .env
python yamnet_mqtt.py
```

### Test MQTT Flow

```bash
python mqtt_test.py ../audio/breakin16khz.wav
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | - | Google Gemini API key |
| `MQTT_BROKER` | localhost | MQTT broker address |
| `MQTT_PORT` | 1883 | MQTT broker port |
| `MQTT_TOPIC` | audio/raw | Topic to subscribe to |
| `MQTT_RESULT_TOPIC` | audio/classification | Topic for results |

## Audio Format

- 8-bit unsigned integers (0-255)
- 16kHz sample rate
- Mono

