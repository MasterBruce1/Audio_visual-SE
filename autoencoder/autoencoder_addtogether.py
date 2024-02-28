import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import cv2
from sklearn.model_selection import train_test_split

image_folder = 'D:/20241.1-2024.8.31/Micro+photodiode+denoise/GRID cropus dataset#1/visual/croplip/S1/bbaf2n'
image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder)]
image_files = [file for file in image_files if os.path.isfile(file)]  
train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)  # 20% for validation
def preprocess_image(file):
    # Read the image in grayscale
    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    
    # Resize the image if necessary, here it's redundant since they are already the same size
    #image = cv2.resize(image, (375,225))
    
    # Normalize the pixel values (optional, but often beneficial)
    image = image / 255.0
    
    # Expand dimensions to fit the autoencoder input (batch_size, height, width, channels)
    image = np.expand_dims(image, axis=-1)
    
    return image

# Preprocess and load the images into memory (if the dataset is small enough)
train_images = np.array([preprocess_image(file) for file in train_files])
val_images = np.array([preprocess_image(file) for file in val_files])


# Define the encoder
encoder = models.Sequential([
    layers.Input(shape=(360, 220, 1)),  # Adjust the input shape to match the size of the images  # assuming train_images is (n_samples, height, width, channels)
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2), padding='same')
])

# Define the decoder
decoder = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),
    # Crop the image if necessary to match the original size
    #layers.Cropping2D(cropping=((top_crop, bottom_crop), (left_crop, right_crop))),
    layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')  # Using sigmoid for the last layer because pixel values are normalized between 0 and 1
])

# Connect the encoder and decoder to create the autoencoder
autoencoder = models.Sequential([encoder, decoder])

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder
autoencoder.fit(train_images, train_images,  # autoencoders are trained using input image as both input and output
                epochs=1000,  # number of epochs
                batch_size=4,  # batch size
                shuffle=True,  # shuffle training data before each epoch
                validation_data=(val_images, val_images))  # use validation data to evaluate loss after each epoch

# Save the model
autoencoder.save('D:/20241.1-2024.8.31/Micro+photodiode+denoise/GRID cropus dataset#1/visual/croplip/S1/bbaf2n/autoencoder_model.h5')

# To visualize the reconstruction quality
decoded_imgs = autoencoder.predict(val_images)
n = min(10, len(val_images))
print(n)
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(val_images[i].reshape(train_images.shape[1], train_images.shape[2]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(train_images.shape[1], train_images.shape[2]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()