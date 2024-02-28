from pydub import AudioSegment
import random
import librosa.display
import matplotlib.pyplot as plt
import torch

# Load audio files
audio_a = AudioSegment.from_file("D:/20241.1-2024.8.31/Micro+photodiode+denoise/LAVSE/dataset/val_data/Pv01/file/3s_1_lbay1a.wav", sample_rate=25000)
audio_b = AudioSegment.from_file("D:/20241.1-2024.8.31/Micro+photodiode+denoise/testnoisedata/ch08.wav", sample_rate=48000)

# Optional: Resample audio B to match A's sample rate, if necessary
audio_b = audio_b.set_frame_rate(25000)
#increase the volume
audio_b = audio_b + 30 
length_a = len(audio_a)
length_b = len(audio_b)
max_start_point = length_b - length_a
#audio_b.export("D:/20241.1-2024.8.31/Micro+photodiode+denoise/testnoisedata/loudernoise.mp3", format="mp3")
# Select a random 3-second segment from audio file B
#length_of_b = len(audio_b) - 3000  # Subtract 3 seconds in milliseconds
start_point = random.randint(0, max_start_point)
noise_segment = audio_b[start_point:start_point + length_a]

# Overlay noise onto audio file A
mixed_audio = audio_a.overlay(noise_segment)
# Export the new audio file
mixed_audio.export("D:/20241.1-2024.8.31/Micro+photodiode+denoise/LAVSE/dataset/val_data/Pv01/file/noisy_lbay1a.wav", format="wav")

