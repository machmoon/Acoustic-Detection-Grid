#!/usr/bin/env python3
"""Convert WAV to JSON payload (simulates ESP32 format)"""

from scipy.io import wavfile
import numpy as np
import json
import sys
import os

wav_path = sys.argv[1] if len(sys.argv) > 1 else "../audio/breakin.wav"
output_path = sys.argv[2] if len(sys.argv) > 2 else "test_audio.json"

sample_rate, data = wavfile.read(wav_path)

# Stereo → mono
if data.ndim == 2:
    data = data.mean(axis=1)

# Convert to 8-bit unsigned
if data.dtype == np.int16:
    data_8bit = ((data.astype(np.float32) + 32768) / 256).astype(np.uint8)
elif data.dtype in [np.float32, np.float64]:
    data_8bit = ((data + 1.0) * 127.5).astype(np.uint8)
else:
    data_8bit = data.astype(np.uint8)

# Output JSON file
payload = {"sample_rate": int(sample_rate), "audio": data_8bit.tolist()}

with open(output_path, 'w') as f:
    json.dump(payload, f)

print(f"✅ Wrote {output_path}")
print(f"   Samples: {len(data_8bit)}")
print(f"   Sample rate: {sample_rate} Hz")
print(f"   Duration: {len(data_8bit)/sample_rate:.2f}s")
print(f"   File size: {os.path.getsize(output_path) / 1024:.1f} KB")
