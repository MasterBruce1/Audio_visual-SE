import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load the audio file
file_path1 = 'D:/20241.1-2024.8.31/Micro+photodiode+denoise/LAVSE/dataset/train_data/Pt02/file/bbaf4a.wav' # Replace with your audio file path
file_path2 = 'D:/20241.1-2024.8.31/Micro+photodiode+denoise/LAVSE/dataset/train_data/Pt02/file/bbal5n.wav'
audio1, sample_rate1 = librosa.load(file_path1, sr=None) # Load with the original sampling rate
audio2, sample_rate2 = librosa.load(file_path2, sr=None) # Load with the original sampling rate
s = librosa.stft(audio1)
print(len(audio2))
print(len(audio1))
# Convert the magnitude spectrogram to dB
S_db = librosa.amplitude_to_db(abs(s))

# Plot the waveform
plt.figure(figsize=(10, 4))
librosa.display.waveshow(audio1, sr=sample_rate1,color="blue")
plt.title('Waveform of loudernoise')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(S_db, sr=sample_rate,x_axis='time', y_axis='log', color="blue")
# plt.colorbar(format='%+2.0f dB')
# plt.title('Spectrogram (dB)')
# plt.xlabel('Time')
# plt.ylabel('Frequency')
# plt.show()