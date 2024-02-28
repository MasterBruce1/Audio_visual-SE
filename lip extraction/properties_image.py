import cv2

# read the input image
img = cv2.imread('D:/20241.1-2024.8.31/Micro+photodiode+denoise/GRID cropus dataset#1/visual/croplip/S1/bbaf2n/pbbc7a75.jpg')

# image properties
print("Type:",type(img))
print("Shape of Image:", img.shape)
print('Total Number of pixels:', img.size)
print("Image data type:", img.dtype)

#print("Pixel Values:\n", img)
print("Dimension:", img.ndim)

# if len(img.shape) == 3:
#     gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     print(gray_image.shape)
#     cv2.imwrite('D:/20241.1-2024.8.31/Micro+photodiode+denoise/GRID cropus dataset#1/visual/croplip/S1/bbaf2n/grayscale_image.png', gray_image)