import morlib
import numpy as np
from random import random
from math import ceil
import matplotlib.pyplot as plt
# Create the example image
rawShape = (100, 100)
rawImage = np.zeros(rawShape)
rawX, rawY = rawShape
for x in range(rawX):
    for y in range(rawY):
        rawImage[x, y] = ceil(random() - 0.5)
# Read the example image
img = plt.imread('image.png')
if img.shape[-1] == 1:
    imgGray = np.array(img)
elif img.shape[-1] == 3:
    grayWeight = np.array([0.299, 0.587, 0.114])
    imgGray = np.dot(img, grayWeight)
elif img.shape[-1] == 4:
    grayWeight = np.array([0, 0.299, 0.587, 0.114])
    imgGray = np.dot(img, grayWeight)
else:
    raise Exception('Invaild image.')

# Show the rawImage
'''
plt.imshow(rawImage,cmap='binary')
plt.show()

# Function Test : Padding
plt.subplot(211)
paddedImg=morlib.padding(rawImage,(5,5),border_filled='CONSTANT')
plt.imshow(paddedImg,cmap='binary')
plt.subplot(212)
paddedImg=morlib.padding(rawImage,(5,5),border_filled='NEAREST')
plt.imshow(paddedImg,cmap='binary')
plt.show()

# Function Test : Conv
rev=np.abs(rawImage-1)
print('AND=',morlib.conv(np.array([1,1]),np.array([1,1]),'AND'))
print('OR=',morlib.conv(rawImage,rev,'OR'))
'''

# Function test : imgBinaration
# the image will show with erosion and dilation
imgBW, thershold = morlib.imgBinarization(imgGray)

# Function Test : binErosion & binDilation
kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
imgErosion = morlib.binErosion(imgBW, kernel, border_filled='CONSTANT')
imgDilation = morlib.binDilation(imgBW, kernel, border_filled='CONSTANT')
plt.subplot(221)
plt.imshow(imgGray, cmap='gray')
plt.subplot(222)
plt.imshow(imgBW, cmap='binary')
plt.subplot(223)
plt.imshow(imgErosion, cmap='binary')
plt.subplot(224)
plt.imshow(imgDilation, cmap='binary')
plt.show()