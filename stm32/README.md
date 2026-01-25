# STM32 - Audio Capture

STM32 code for capturing audio and sending it to the classification server.

## Hardware

- STM32 development board (e.g., STM32F4, STM32H7)
- I2S microphone or analog microphone
- Optional: WiFi module (ESP8266/ESP32) or Ethernet

## Setup

1. Install STM32CubeIDE
2. Configure I2S or ADC for audio capture
3. Configure network (UART to ESP8266 or Ethernet)
4. Build and flash

## Audio Format

- Sample rate: 16kHz
- Bit depth: 8-bit unsigned (0-255)
- Channels: Mono

## Communication Options

### Option 1: UART to ESP8266/ESP32
STM32 captures audio and sends to ESP8266/ESP32 via UART, which then publishes to MQTT.

### Option 2: Ethernet
Use STM32's Ethernet peripheral with MQTT library.

### Option 3: USB
Send audio to PC over USB serial.

## TODO

- [ ] Add I2S microphone code
- [ ] Add ADC audio capture
- [ ] Add UART communication
- [ ] Add network stack

