import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datapath_manage import speaker_list_t,speaker_list_v, speaker_list_s, number_list_t, number_list_v, number_list_s, datapath_manage, dataset
from data_custom import *
 # ********** check cuda **********
print('\n********** check cuda **********\n')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

train_datapath, val_datapath, test_datapath = datapath_manage(dataset)
test_dataset = data_custom(name = 'train', data_path = test_datapath)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ********** define model **********
class LAVSE(nn.Module):
    def __init__(self):
        super(LAVSE, self).__init__()
        # Audio branch//3,1025,147
        self.audio_conv1 = nn.Conv1d(in_channels=1025, out_channels=512, kernel_size=5, stride=1, padding=2)#[3,512,147]
        self.audio_pool = nn.MaxPool1d(kernel_size=4, stride=4, padding=0)#[3,512,36]
        self.audio_conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=5, stride=1, padding=2) #[3,256,36] 

        self.lstm = nn.LSTM(input_size=21888, hidden_size=576, num_layers=1, batch_first=True)
        # Define FC layers for audio and visual outputs
        self.fc_audio = nn.Linear(576, 1025*147)  # audio_output_feature_size = 1025*147
        self.fc_visual = nn.Linear(576, 1*16*36*22)  # visual_output_feature_size = 1*16*36*22

    def forward(self, audio_tensor, visual_tensor):
        # Process audio
        audio_features = F.relu(self.audio_conv1(audio_tensor))
        audio_features = self.audio_pool(audio_features)
        audio_features = F.relu(self.audio_conv2(audio_features))
        audio_features = audio_features.view(audio_features.size(0), -1)  # Flatten audio features [3, 256*36] [3, 9216]

        # Process visual
        visual_features_flattened = visual_tensor.view(visual_tensor.size(0), -1) #[3,1*16*36*22] = [3,12672]

        combined_features = torch.cat((audio_features, visual_features_flattened), dim=1) #[3, 9216+12672]
        combined_features.unsqueeze(1) #[3, 1, 21888]
        # Further processing after concatenation
        combined_features, _ = self.lstm(combined_features)
       
        audio_output = self.fc_audio(combined_features).view(-1, 1025, 147)  # Reshape to [3, 1025, 147]
        visual_output = self.fc_visual(combined_features).view(-1, 1, 16, 36, 22)  # Reshape to [3, 1, 16, 36, 22]
        

        return audio_output, visual_output


def test(model, test_dataset):
    model.to(device)  # Ensure the model is on the correct device
    model.eval()  # Set the model to evaluation mode
    
    #reconstructed_audios = []  # List to store reconstructed audio tensors
    #reconstructed_images = []  # List to store reconstructed image tensors
    
    with torch.no_grad():  # Disable gradient computation
        for i, batch in enumerate(test_dataset):
            batch = {k: v.to(device) for k, v in batch.items()}
            noisy_t, image_t = batch['noisy'], batch['image']
            
            # Run the model to get reconstructed audio and image
            reconstructed_audio, reconstructed_image = model(noisy_t, image_t)
            print( reconstructed_audio.shape)
            print(reconstructed_image.shape)
            # Store the reconstructed tensors
            #reconstructed_audios.append(reconstructed_audio.cpu())
            #reconstructed_images.append(reconstructed_image.cpu())
            
    # Optionally, save the reconstructed tensors to disk
    torch.save(reconstructed_audio, 'D:/20241.1-2024.8.31/Micro+photodiode+denoise/LAVSE/dataset/test_data/Ps01/file/reconstructed_audios.pt')
    torch.save(reconstructed_image, 'D:/20241.1-2024.8.31/Micro+photodiode+denoise/LAVSE/dataset/test_data/Ps01/file/reconstructed_images.pt')

model = LAVSE().to(device)
model.load_state_dict(torch.load('D:/20241.1-2024.8.31/Micro+photodiode+denoise/LAVSE/LAVSE_model.pth', map_location=device))

test(model, test_loader)