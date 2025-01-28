import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


# Load audio file
def load_audio(audio_file, duration=30, sampling_rate=44100):
   y, sr = librosa.load(audio_file, sr=sampling_rate, duration=duration)
   return y, sr


# Compute the frequency spectrum of the audio
def compute_spectrum(y, sr):
   # Compute the Short-Time Fourier Transform (STFT)
   D = librosa.stft(y)  # Short-Time Fourier Transform (STFT)
   S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)  # Convert amplitude to dB scale
   return D, S_db


# Plot the Spectrum (Spectrogram)
def plot_spectrum(S_db, sr, duration=180):
   # Create a figure to display the spectrum
   plt.figure(figsize=(12, 8))
   librosa.display.specshow(S_db, x_axis='time', y_axis='log', sr=sr)
   plt.colorbar(format='%+2.0f dB')
   plt.title('Amplitude Spectrum (Spectrogram) of the Audio')
   plt.xlabel('Time (s)')
   plt.ylabel('Frequency (Hz)')
   plt.show()


# Generate Harmograph visualization
def generate_harmonograph(y, sr, duration=10, sampling_rate=44100):
   # Create a time array for the harmonograph
   time = np.linspace(0, duration, int(sampling_rate * duration))  # Time array


   # Parameters for harmonic oscillations
   frequency1 = 0.2  # Low frequency oscillation
   frequency2 = 0.5  # High frequency oscillation


   # Create sine waves with different frequencies
   signal1 = np.sin(2 * np.pi * frequency1 * time)
   signal2 = np.sin(2 * np.pi * frequency2 * time)


   # Normalize the audio signal for modulation
   y_normalized = librosa.util.normalize(y)


   # Ensure the length of y matches the duration and resample the audio if necessary
   if len(y_normalized) > len(time):
       y_normalized = y_normalized[:len(time)]  # Trim audio signal
   else:
       # If the audio signal is shorter, resample it
       y_normalized = np.interp(np.linspace(0, len(y_normalized), len(time)), np.arange(len(y_normalized)), y_normalized)


   # Smooth modulation with audio data
   modulation1 = np.interp(y_normalized, (y_normalized.min(), y_normalized.max()), (-0.5, 0.5))
   modulation2 = np.interp(y_normalized, (y_normalized.min(), y_normalized.max()), (-0.5, 0.5))


   # Combine the oscillations with the modulation
   x = signal1 + modulation1
   y = signal2 + modulation2


   # Create line segments for the plot
   points = np.array([x, y]).T.reshape(-1, 1, 2)
   segments = np.concatenate([points[:-1], points[1:]], axis=1)


   # Color based on the amplitude (normalized)
   norm = plt.Normalize(np.min(np.abs(modulation1 + modulation2)), np.max(np.abs(modulation1 + modulation2)))
   lc = LineCollection(segments, cmap='plasma', norm=norm, linewidth=0.6, alpha=0.8)


   # Plotting the Harmograph pattern with LineCollection
   plt.figure(figsize=(8, 8))
   ax = plt.gca()
   ax.add_collection(lc)
   plt.title(f'Harmograph Pattern', fontsize=16)
   plt.xlabel('X', fontsize=12)
   plt.ylabel('Y', fontsize=12)


   # Set axis equal to keep the aspect ratio correct
   plt.axis('equal')
   plt.grid(True, linestyle='--', alpha=0.6)
   plt.show()


# Example usage with audio file
audio_file = r””
y, sr = load_audio(audio_file, duration=180)
D, S_db = compute_spectrum(y, sr)


# Plot the Spectrogram
plot_spectrum(S_db, sr)


# Generate Harmograph visualization
generate_harmonograph(y, sr, duration=180, sampling_rate=30000)


