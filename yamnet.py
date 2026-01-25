import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv

import matplotlib.pyplot as plt
from IPython.display import Audio
from scipy.io import wavfile
import scipy.signal

# Load the model.
model = hub.load('https://tfhub.dev/google/yamnet/1')

# Find the name of the class with the top score when mean-aggregated across frames.
def class_names_from_csv(class_map_csv_text):
  """Returns list of class names corresponding to score vector."""
  class_names = []
  with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      class_names.append(row['display_name'])

  return class_names

class_map_path = model.class_map_path().numpy()
class_names = class_names_from_csv(class_map_path)

"""Add a method to verify and convert a loaded audio is on the proper sample_rate (16K), otherwise it would affect the model's results."""

def ensure_sample_rate(original_sample_rate, waveform,
                       desired_sample_rate=16000):
  """Resample waveform if required. Returns float32 in [-1.0, 1.0] range."""
  # Handle stereo audio - convert to mono
  if len(waveform.shape) > 1:
    waveform = np.mean(waveform, axis=1)
  
  # Convert to float32 and normalize to [-1.0, 1.0] for resampling
  if waveform.dtype == np.int16:
    waveform_float = waveform.astype(np.float32) / 32768.0
  elif waveform.dtype == np.int32:
    waveform_float = waveform.astype(np.float32) / 2147483648.0
  elif waveform.dtype in [np.float32, np.float64]:
    # Already float, ensure in [-1.0, 1.0] range
    max_val = np.abs(waveform).max()
    if max_val > 1.0:
      waveform_float = (waveform / max_val).astype(np.float32)
    else:
      waveform_float = waveform.astype(np.float32)
  else:
    waveform_float = waveform.astype(np.float32)
  
  # Resample if needed
  if original_sample_rate != desired_sample_rate:
    desired_length = int(round(float(len(waveform_float)) /
                               original_sample_rate * desired_sample_rate))
    print(f"Resampling from {original_sample_rate}Hz to {desired_sample_rate}Hz")
    print(f"  Original length: {len(waveform_float)}, New length: {desired_length}")
    waveform_float = scipy.signal.resample(waveform_float, desired_length)
  
  # Return as float32 (YAMNet needs this format)
  return desired_sample_rate, waveform_float

# wav_file_name = 'speech_whistling2.wav'
# wav_file_name = 'miaow_16k.wav'
wav_file_name = 'breakin.wav'

sample_rate, wav_data = wavfile.read(wav_file_name, 'rb')

import sounddevice as sd

# Play
sd.play(wav_data, sample_rate)
sd.wait()  # Wait until finished
print(sample_rate)
print(wav_data.shape)
print(wav_data.dtype)

# Process audio: convert stereo to mono and resample if needed
sample_rate_processed, wav_data_processed = ensure_sample_rate(sample_rate, wav_data)

# Show some basic information about the audio.
duration = len(wav_data_processed)/sample_rate_processed
print(f'Original: {sample_rate} Hz, Processed: {sample_rate_processed} Hz')
print(f'Total duration: {duration:.2f}s')
print(f'Size of the input: {len(wav_data_processed)}')
print(f'Audio shape: {wav_data_processed.shape}, dtype: {wav_data_processed.dtype}')

# Play processed audio (already float32 from ensure_sample_rate)
print("\nPlaying processed audio...")
sd.play(wav_data_processed, sample_rate_processed)
sd.wait()
print("Done!")


# Listening to the wav file (keep as int16 for Audio display)
# Convert float32 to int16 for IPython Audio
wav_data_int16 = (wav_data_processed * 32767).astype(np.int16)
Audio(wav_data_int16, rate=sample_rate_processed)

"""The `wav_data` needs to be normalized to values in `[-1.0, 1.0]` (as stated in the model's [documentation](https://tfhub.dev/google/yamnet/1))."""

# ensure_sample_rate now returns float32 in [-1.0, 1.0] range, perfect for YAMNet
waveform = wav_data_processed  # Already normalized float32 from ensure_sample_rate

# Verify normalization
print(f"\n✅ Waveform ready for YAMNet:")
print(f"   Shape: {waveform.shape}")
print(f"   Dtype: {waveform.dtype}")
print(f"   Range: [{waveform.min():.6f}, {waveform.max():.6f}]")
print(f"   Sample rate: {sample_rate_processed} Hz")
if sample_rate_processed != 16000:
    print(f"   ⚠️  WARNING: YAMNet requires 16kHz, but got {sample_rate_processed}Hz!")

"""## Executing the Model

Now the easy part: using the data already prepared, you just call the model and get the: scores, embedding and the spectrogram.

The score is the main result you will use.
The spectrogram you will use to do some visualizations later.
"""

# Run the model, check the output.
print("\nRunning YAMNet classification...")
scores, embeddings, spectrogram = model(waveform)

scores_np = scores.numpy()
spectrogram_np = spectrogram.numpy()
mean_scores = np.mean(scores_np, axis=0)
top_class_idx = mean_scores.argmax()
infered_class = class_names[top_class_idx]
confidence = mean_scores[top_class_idx]

print(f'\n🎯 Classification Result:')
print(f'   Top sound: {infered_class}')
print(f'   Confidence: {confidence:.2%}')

# Show top 5 predictions
top_n = 5
top_indices = np.argsort(mean_scores)[::-1][:top_n]
print(f'\n   Top {top_n} predictions:')
for i, idx in enumerate(top_indices, 1):
    print(f'   {i}. {class_names[idx]:30s} {mean_scores[idx]:.2%}')

"""## Visualization

YAMNet also returns some additional information that we can use for visualization.
Let's take a look on the Waveform, spectrogram and the top classes inferred.
"""

plt.figure(figsize=(10, 6))

# Plot the waveform.
plt.subplot(3, 1, 1)
plt.plot(waveform)
plt.xlim([0, len(waveform)])

# Plot the log-mel spectrogram (returned by the model).
plt.subplot(3, 1, 2)
plt.imshow(spectrogram_np.T, aspect='auto', interpolation='nearest', origin='lower')

# Plot and label the model output scores for the top-scoring classes.
mean_scores = np.mean(scores, axis=0)
top_n = 5
top_class_indices = np.argsort(mean_scores)[::-1][:top_n]
plt.subplot(3, 1, 3)
plt.imshow(scores_np[:, top_class_indices].T, aspect='auto', interpolation='nearest', cmap='gray_r')

# patch_padding = (PATCH_WINDOW_SECONDS / 2) / PATCH_HOP_SECONDS
# values from the model documentation
patch_padding = (0.025 / 2) / 0.01
plt.xlim([-patch_padding-0.5, scores.shape[0] + patch_padding-0.5])
# Label the top_N classes.
yticks = range(0, top_n, 1)
plt.yticks(yticks, [class_names[top_class_indices[x]] for x in yticks])
_ = plt.ylim(-0.5 + np.array([top_n, 0]))

plt.show()