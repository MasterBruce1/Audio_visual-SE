from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers

encoder = load_model('D:/20241.1-2024.8.31/Micro+photodiode+denoise/GRID cropus dataset#1/visual/croplip/S1/encoder_modelnew.h5')
decoder = load_model('D:/20241.1-2024.8.31/Micro+photodiode+denoise/GRID cropus dataset#1/visual/croplip/S1/decoder_modelnew.h5')
image = cv2.imread('D:/20241.1-2024.8.31/Micro+photodiode+denoise/GRID cropus dataset#1/visual/croplip/S1/lgif1s75.jpg',cv2.IMREAD_GRAYSCALE)
#image = cv2.resize(image, (360, 220))  # Resize to the expected dimensions
image = image / 255.0
image = np.expand_dims(image, axis=-1)
print("Shape of Image:", image.shape)
#image = np.expand_dims(image, axis=0)
#print("Shape of Image:", image.shape)
image= np.array([image])
encoded_data = encoder.predict(image)
#save data
#encoded_tensor = torch.from_numpy(encoded_data)
print("print encodeddata : ", encoded_data)
#save(encoded_tensor, 'D:/20241.1-2024.8.31/Micro+photodiode+denoise/GRID cropus dataset#1/visual/croplip/S1/encoded_data.pt')
np.save('D:/20241.1-2024.8.31/Micro+photodiode+denoise/GRID cropus dataset#1/visual/croplip/S1/encodeddata.npy', encoded_data)
feature_map = encoded_data[:, :, 1]
print("type:", type(encoded_data))
# Average across channels
avg_map = np.mean(encoded_data, axis=-1)

# Or, maximum across channels
max_map = np.max(encoded_data, axis=-1)


decoded_data = decoder.predict(encoded_data)
plt.subplot(3, 1, 1)
plt.imshow(np.squeeze(max_map),cmap='gray')
plt.gray()
plt.subplot(3, 1, 2)
plt.imshow(image.reshape(image.shape[1], image.shape[2]))
plt.subplot(3, 1, 3)
plt.imshow(decoded_data.reshape(image.shape[1], image.shape[2]))
plt.show()
encoder.summary()
# Reconstruct the encoder model from the autoencoder
