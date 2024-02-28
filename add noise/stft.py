import librosa

import torch
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
# Step 1: Load an audio file
# Replace 'path_to_your_audio_file.wav' with the path to your actual audio file
y, sr = librosa.load('D:/20241.1-2024.8.31/Micro+photodiode+denoise/LAVSE/dataset/val_data/Pv01/file/noisy_lbay1a.wav', sr=25000)
current_length = len(y)
print(current_length)
target_length = int(3 *sr)
if current_length < target_length:
    padding_length = target_length - current_length
    modified_audio = np.pad(y, (0, padding_length), mode='constant')
else:
    modified_audio = y[:target_length]

print(modified_audio.shape)
#sf.write('D:/20241.1-2024.8.31/Micro+photodiode+denoise/LAVSE/dataset/val_data/Pv01/file/3s_1_lbay1a.wav', modified_audio, 25000)
# Step 2: Apply STFT
# n_fft is the window size, hop_length is the distance between successive windows
n_fft = 2048
hop_length = 512
stft = librosa.stft(modified_audio, n_fft=n_fft, hop_length=hop_length)

# Step 3: Convert to magnitude (and possibly to dB scale for neural network training)
magnitude = np.abs(stft)
magnitude_db = librosa.amplitude_to_db(magnitude)

# Phase Spectrogram
phase_spectrogram = np.angle(stft)
# Optional: Normalize the magnitude_db
magnitude_db_normalized = (magnitude_db - np.min(magnitude_db)) / (np.max(magnitude_db) - np.min(magnitude_db))
magnitude_db_denormalized = (magnitude_db_normalized * (np.max(magnitude_db) - np.min(magnitude_db))) + np.min(magnitude_db)
print(torch.tensor(magnitude_db_normalized).shape)
phase_tensor = torch.from_numpy(phase_spectrogram)
magnitude_tensor = torch.from_numpy(magnitude_db_denormalized)
# Step 4: Save the STFT output as a .pt file
torch.save(magnitude_tensor, 'D:/20241.1-2024.8.31/Micro+photodiode+denoise/LAVSE/dataset/val_data/Pv01/file/vn_P1_1.pt')

torch.save(phase_tensor,'D:/20241.1-2024.8.31/Micro+photodiode+denoise/LAVSE/dataset/val_data/Pv01/file/vn_P1_1_phase.pt')
# Plot the Magnitude Spectrogram
#plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
librosa.display.specshow(librosa.amplitude_to_db(magnitude, ref=np.max), sr=sr, hop_length=hop_length, y_axis='log', x_axis='time')
#librosa.display.specshow(magnitude_db_normalized, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', cmap='magma')
plt.title('Magnitude Spectrogram')
plt.colorbar(format='%+2.0f dB')

# Plot the Phase Spectrogram
plt.subplot(1, 2, 2)
librosa.display.specshow(phase_spectrogram, sr=sr, hop_length=hop_length, y_axis='log', x_axis='time', cmap='twilight')
plt.title('Phase Spectrogram')
plt.colorbar()

plt.tight_layout()
plt.show()

#reconsruction
# magnitude_linear = librosa.db_to_amplitude(magnitude_db_denormalized)
# complex_spectrogram = magnitude_linear * np.exp(1j * phase_spectrogram)
# y_reconstructed = librosa.istft(complex_spectrogram, hop_length=hop_length, win_length=n_fft)
# sf.write('D:/20241.1-2024.8.31/Micro+photodiode+denoise/LAVSE/dataset/train_data/Pt01/file/path_to_output_file.wav', y_reconstructed, sr)

