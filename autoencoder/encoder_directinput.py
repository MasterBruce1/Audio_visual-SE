import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import cv2
from sklearn.model_selection import train_test_split

#input image
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (220, 360)) #width, height
    image = image / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Example usage
image_path = 'D:/20241.1-2024.8.31/Micro+photodiode+denoise/GRID cropus dataset#1/visual/croplip/S1/bbaf2n/swio1s75.jpg'
input_image = preprocess_image(image_path)
print("Shape of Image:", input_image.shape)


# Define the encoder
encoder = models.Sequential([
    layers.Input(shape=(360, 220, 1)),  # Adjust the input shape to match the size of the images  # assuming train_images is (n_samples, height, width, channels)
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((3, 3), padding='same'),
    layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((3, 3), padding='same')
])
encoder.summary()
latent_representation = encoder.predict(input_image)
plt.subplot(2, 1, 1)
plt.imshow(input_image.reshape(input_image.shape[1], input_image.shape[2]), cmap='gray')
plt.subplot(2, 1, 2)
if latent_representation.ndim == 4 and latent_representation.shape[-1] > 0:
    plt.imshow(latent_representation[0, :, :, 0], cmap='gray')  # Visualize the first channel of the first image
    plt.title("Compressed (Encoded) Image")
    plt.axis('off')
    plt.show()
else:
    print("Unexpected shape of the latent representation:", latent_representation.shape)