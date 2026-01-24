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
  """Resample waveform if required."""
  # Handle stereo audio - convert to mono
  if len(waveform.shape) > 1:
    waveform = np.mean(waveform, axis=1)
  
  if original_sample_rate != desired_sample_rate:
    desired_length = int(round(float(len(waveform)) /
                               original_sample_rate * desired_sample_rate))
    waveform = scipy.signal.resample(waveform, desired_length)

  return desired_sample_rate, waveform

# wav_file_name = 'speech_whistling2.wav'
# wav_file_name = 'miaow_16k.wav'
wav_file_name = 'breakin.wav'

sample_rate, wav_data = wavfile.read(wav_file_name, 'rb')

# Process audio: convert stereo to mono and resample if needed
sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)

# Show some basic information about the audio.
duration = len(wav_data)/sample_rate
print(f'Sample rate: {sample_rate} Hz')
print(f'Total duration: {duration:.2f}s')
print(f'Size of the input: {len(wav_data)}')
print(f'Audio shape: {wav_data.shape}, dtype: {wav_data.dtype}')

# Listening to the wav file (keep as int16 for Audio display)
# Convert to int16 if it's float
if wav_data.dtype != np.int16:
    wav_data_int16 = (wav_data * 32767).astype(np.int16)
else:
    wav_data_int16 = wav_data

Audio(wav_data_int16, rate=sample_rate)

"""The `wav_data` needs to be normalized to values in `[-1.0, 1.0]` (as stated in the model's [documentation](https://tfhub.dev/google/yamnet/1))."""

# Normalize for YAMNet model (convert to float32 in range [-1.0, 1.0])
if wav_data.dtype == np.int16:
    waveform = wav_data.astype(np.float32) / 32768.0
else:
    waveform = wav_data.astype(np.float32)

"""## Executing the Model

Now the easy part: using the data already prepared, you just call the model and get the: scores, embedding and the spectrogram.

The score is the main result you will use.
The spectrogram you will use to do some visualizations later.
"""

# Run the model, check the output.
scores, embeddings, spectrogram = model(waveform)

scores_np = scores.numpy()
spectrogram_np = spectrogram.numpy()
infered_class = class_names[scores_np.mean(axis=0).argmax()]
print(f'The main sound is: {infered_class}')

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