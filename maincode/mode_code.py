import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ********** define model **********
class LAVSE(nn.Module):
    def __init__(self):
        super(LAVSE, self).__init__()
        # Audio branch//3,1025,147
        self.audio_conv1 = nn.Conv1d(in_channels=1025, out_channels=512, kernel_size=5, stride=1, padding=2)#[3,512,147]
        self.audio_pool = nn.MaxPool1d(kernel_size=4, stride=4, padding=0)#[3,512,36]
        self.audio_conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=5, stride=1, padding=2) #[3,256,36] 

        # Visual branch//[1, 16, 36, 22]
        # self.visual_conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)#[3,8,16,36,22]
        # self.visual_pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0)#[3, 32, 8, 18, 11]
        # self.visual_conv2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)#[3,16,8,18,11]
        # Define the LSTM layer
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
        # visual_features = F.relu(self.visual_conv1(visual_tensor))
        # visual_features = self.visual_pool(visual_features)
        # visual_features = F.relu(self.visual_conv2(visual_features))
        # visual_features = visual_features.view(visual_features.size(0), -1)  # Flatten visual features[3, ]
        
        # Concatenate audio and visual features
        #print(audio_features.shape)
        #print(visual_features_flattened.shape)
        combined_features = torch.cat((audio_features, visual_features_flattened), dim=1) #[3, 9216+12672]
        combined_features.unsqueeze(1) #[3, 1, 21888]
        # Further processing after concatenation
        combined_features, _ = self.lstm(combined_features)
        #print(combined_features.shape)
        #combined_features = combined_features[:, -1, :] #[3, 576]
        #Audio and visual enhancements
        audio_output = self.fc_audio(combined_features).view(-1, 1025, 147)  # Reshape to [3, 1025, 147]
        visual_output = self.fc_visual(combined_features).view(-1, 1, 16, 36, 22)  # Reshape to [3, 1, 16, 36, 22]
        

        return audio_output, visual_output
    



