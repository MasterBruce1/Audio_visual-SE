import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Define the custom dataset class
class ImageDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)  # PyTorch expects channel first format
        return torch.tensor(image, dtype=torch.float32)

# Define the autoencoder with separate encoder and decoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(5, 5),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=5),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
# Function to test the loaded models
def test_models(encoder, decoder, dataloader,num_images_to_show=4):
    encoder.eval()  # Set encoder to evaluation mode
    decoder.eval()  # Set decoder to evaluation mode
    with torch.no_grad():  # No need to track gradients
        for i, data in enumerate(dataloader):
            data = data.to(device)  # Move data to the same device as models
            encoded = encoder(data)
            torch.save(torch.tensor(encoded), 'D:/20241.1-2024.8.31/Micro+photodiode+denoise/LAVSE/dataset/val_data/Pv01/file/image/vi_P1_1.pt')
            print(encoded.shape)
            #a = torch.load('D:/20241.1-2024.8.31/Micro+photodiode+denoise/LAVSE/dataset/train_data/Pt01/image/i_P1_1.pt')
            decoded = decoder(encoded)
            data, decoded= data.cpu().numpy(), decoded.cpu().numpy()
            encoded_data = encoded[0, 0, :, :].cpu().numpy()
            max_feature_map, _ = torch.max(encoded, dim=1, keepdim=True)


            for j in range(data.shape[0]):
                print(data.shape)
                if j >= num_images_to_show:  # Show only the first num_images_to_show images
                    break
                
                # Display original
                ax = plt.subplot(3, num_images_to_show, j + 1)
                plt.imshow(data[j].reshape(data.shape[2], data.shape[3]), cmap='gray')
                plt.title("Original")
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                # Display original
                ax = plt.subplot(3, num_images_to_show, j + 2)
                plt.imshow(encoded_data, cmap='gray')
                plt.title("encoded feature 1")
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                #display maximum
                ax = plt.subplot(3, num_images_to_show, j + 3)
                plt.imshow(max_feature_map.squeeze().cpu().numpy(), cmap='gray')
                plt.title('Maximum Across Feature Maps')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                # Display reconstruction
                ax = plt.subplot(3, num_images_to_show, j + 4)
                plt.imshow(decoded[j].reshape(decoded.shape[2], decoded.shape[3]), cmap='gray')
                plt.title("Reconstructed")
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

            if i >= num_images_to_show - 1:  # Ensure we don't show more than num_images_to_show images
                break
        plt.show()

# Initialize model instances
encoder = Encoder().to(device)
decoder = Decoder().to(device)
# Load the state dictionaries
encoder.load_state_dict(torch.load('D:/20241.1-2024.8.31/Micro+photodiode+denoise/LAVSE/LAVSE-master/LAVSE-master/autoencoder/saved model/encoder_model.pth', map_location=device))
decoder.load_state_dict(torch.load('D:/20241.1-2024.8.31/Micro+photodiode+denoise/LAVSE/LAVSE-master/LAVSE-master/autoencoder/saved model/decoder_model.pth', map_location=device))
# Define paths, load data, and create DataLoaders
image_folder = 'D:/20241.1-2024.8.31/Micro+photodiode+denoise/LAVSE/dataset/val_data/Pv01/file/image'
test_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, file))]
test_dataset = ImageDataset(test_files)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
# Test the loaded encoder and decoder models
test_models(encoder, decoder, test_loader)