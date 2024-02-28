import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, Input
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

# Define the input layer
encoder_input = Input(shape=(360, 220, 1), name='encoder_input')
# Add layers to the encoder
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
x = layers.MaxPooling2D((5, 5), padding='same')(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
encoder_output = layers.MaxPooling2D((2, 2), padding='same')(x)
# Create the encoder model
encoder = models.Model(inputs=encoder_input, outputs=encoder_output, name='encoder')

# Define the decoder
decoder_input = Input(shape=(36, 22, 16), name='decoder_input')  # Example shape, adjust based on your encoder output
# Apply the decoder layers as specified
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(decoder_input)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((5, 5))(x)
decoder_output = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
# Create the decoder model
decoder = models.Model(inputs=decoder_input, outputs=decoder_output, name='decoder')
##
autoencoder_input = Input(shape=(360, 220, 1))
encoded = encoder(autoencoder_input)
decoded = decoder(encoded)
autoencoder = models.Model(autoencoder_input, decoded)
# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss= tf.keras.losses.MeanSquaredError() )
#train
autoencoder.fit(train_images, train_images,
                epochs=3000,
                batch_size=4,
                shuffle=True,
                validation_data=(val_images, val_images))



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
print('start saving')
# Save the model
encoder.save('D:/20241.1-2024.8.31/Micro+photodiode+denoise/GRID cropus dataset#1/visual/croplip/S1/encoder_modelnew.h5')
decoder.save('D:/20241.1-2024.8.31/Micro+photodiode+denoise/GRID cropus dataset#1/visual/croplip/S1/decoder_modelnew.h5')
autoencoder.save('D:/20241.1-2024.8.31/Micro+photodiode+denoise/GRID cropus dataset#1/visual/croplip/S1/autoencoder_modelnew.h5')
print('saved')