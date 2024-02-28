import torch
import numpy as np
import librosa
import soundfile as sf
stft_clean_phase = torch.load('D:/20241.1-2024.8.31/Micro+photodiode+denoise/LAVSE/dataset/test_data/Ps01/file/c_P1_1_phase.pt')
stft_clean_amp = torch.load('D:/20241.1-2024.8.31/Micro+photodiode+denoise/LAVSE/dataset/test_data/Ps01/clean/sc_P1_1.pt')
#lippt = torch.load(lippt_path)
print(stft_clean_amp.shape)
print(stft_clean_phase.shape)
stft_clean_phase = stft_clean_phase.cpu() 
stft_clean_amp = stft_clean_amp.cpu()
stft_clean_amp = stft_clean_amp.squeeze(0)
amplitude_linear = 10 ** (stft_clean_amp / 20)
amplitude_linear_np = amplitude_linear.numpy()
phase_numpy = stft_clean_phase.numpy()  
complex_phase = np.exp(1j * phase_numpy)
complex_spectrogram = amplitude_linear_np * complex_phase
y_reconstructed = librosa.istft(complex_spectrogram, hop_length=512, win_length=2048)
sf.write('D:/20241.1-2024.8.31/Micro+photodiode+denoise/LAVSE/dataset/test_data/Ps01/file/groundtruth_fulloutput_file.wav', y_reconstructed, samplerate = 25000)