from pesq import pesq
from pystoi import stoi
import librosa

# Load your reference and degraded audio files
ref_audio, sr = librosa.load('D:/20241.1-2024.8.31/Micro+photodiode+denoise/LAVSE/LAVSE-master/LAVSE-master/SP05_clean320.wav', sr=16000)
deg_audio, _ = librosa.load('D:/20241.1-2024.8.31/Micro+photodiode+denoise/LAVSE/LAVSE-master/LAVSE-master/SP05_320.wav', sr=16000)

min_length = min(len(ref_audio), len(deg_audio))
print(len(ref_audio))
ref_audio = ref_audio[:min_length]
deg_audio = deg_audio[:min_length]

# Calculate PESQ score
pesq_score = pesq(sr, ref_audio, deg_audio, 'wb') # 'wb' for wideband, 'nb' for narrowband
print(f'PESQ score: {pesq_score}')
stoi_score = stoi(ref_audio, deg_audio, sr, extended=False) # Use extended=True for the extended STOI
print(f'STOI score: {stoi_score}')