import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#check CUDA
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

# Initialize the autoencoder, optimizer, and loss function
autoencoder = Autoencoder().to(device)
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training function
def train(model, dataloader, val_loader, epochs=1000):
    model.train()
    for epoch in range(epochs):
        train_loss = 0.0
        for data in dataloader:
            # Move data to the same device as model
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
        #print(f'Epoch {epoch+1}, Loss: {loss.item()}')
            train_loss += loss.item() * data.size(0)
        
        train_loss /= len(train_loader.dataset)

        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():  # No gradients needed for validation
            for data in val_loader:
                # Move data to the same device as model
                data = data.to(device)
                output = model(data)
                loss = criterion(output, data)
                val_loss += loss.item() * data.size(0)
        
        val_loss /= len(val_loader.dataset)
        print(f'Epoch {epoch+1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

#validation and visualizaion
def validate_and_visualize(model, dataloader, num_images_to_show=10):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to track gradients
        for i, data in enumerate(dataloader):
            # Forward pass: compute the output of the autoencoder
            # Move data to the same device as model
            data = data.to(device)
            reconstructed = model(data)
            data, reconstructed = data.cpu().numpy(), reconstructed.cpu().numpy()

            # Plot the original and reconstructed images
            for j in range(data.shape[0]):
                if j >= num_images_to_show:  # Show only the first num_images_to_show images
                    break
                
                # Display original
                ax = plt.subplot(2, num_images_to_show, j + 1)
                plt.imshow(data[j].reshape(data.shape[2], data.shape[3]), cmap='gray')
                plt.title("Original")
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                # Display reconstruction
                ax = plt.subplot(2, num_images_to_show, j + 1 + num_images_to_show)
                plt.imshow(reconstructed[j].reshape(reconstructed.shape[2], reconstructed.shape[3]), cmap='gray')
                plt.title("Reconstructed")
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

            if i >= num_images_to_show - 1:  # Ensure we don't show more than num_images_to_show images
                break
        plt.show()

# Define paths, load data, and create DataLoaders
image_folder = 'D:/20241.1-2024.8.31/Micro+photodiode+denoise/GRID cropus dataset#1/visual/croplip/S1/bbaf2n'
image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, file))]
#print("image files:", image_files)
train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)
# Initialize the datasets
train_dataset = ImageDataset(train_files)
print(train_dataset)
val_dataset = ImageDataset(val_files)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=3, shuffle=False)
# Train the model as before
train(autoencoder, train_loader, val_loader)
#validation
validate_and_visualize(autoencoder, val_loader)
# After training, save the encoder and decoder separately
torch.save(autoencoder.encoder.state_dict(), 'D:/20241.1-2024.8.31/Micro+photodiode+denoise/LAVSE/LAVSE-master/LAVSE-master/autoencoder/saved model/encoder_model.pth')
torch.save(autoencoder.decoder.state_dict(), 'D:/20241.1-2024.8.31/Micro+photodiode+denoise/LAVSE/LAVSE-master/LAVSE-master/autoencoder/saved model/decoder_model.pth')
