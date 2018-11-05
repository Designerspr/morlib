import morlib
import numpy as np
from random import random
from math import ceil
import matplotlib.pyplot as plt

def quickPlot(position,img,cmap=None,title=None):
    '''Quick Plot Function.'''
    plt.subplot(position)
    if cmap:
        plt.imshow(img,cmap=cmap)
    else:
        plt.imshow(img)
    plt.title(title)

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

# Function test : imgBinaration
# the image will show with padded image.
imgBW, thershold = morlib.imgBinarization(imgGray)

# Function Test : Padding
imgConstantPadded=morlib.padding(imgBW,(5,5),border_filled='CONSTANT')
imgNearestPadded=morlib.padding(imgBW,(5,5),border_filled='NEAREST')

quickPlot(321,img,title='Raw Image')
quickPlot(322,imgGray,cmap='gray',title='GrayScale Image')
quickPlot(323,imgBW,cmap='binary',title='Binary Image')
quickPlot(324,imgConstantPadded,cmap='binary',title='Padding:keyword=Constant')
quickPlot(325,imgNearestPadded,cmap='binary',title='Padding:keyword=Nearest')
plt.show()

# Function Test : binErosion & binDilation & createKernel
kernel = morlib.createKernel((3,3),keyword='X')
imgErosion = morlib.binErosion(imgBW, kernel, border_filled='CONSTANT')
imgDilation = morlib.binDilation(imgBW, kernel, border_filled='CONSTANT')

quickPlot(211,imgErosion,cmap='binary',title='Erosion Image')
quickPlot(212,imgDilation,cmap='binary',title='Dilation Image')
plt.show()

# Function Test: Open/Close/Top-Hat/Black-Hat
imgOpen=morlib.binOpen(imgBW, kernel, border_filled='CONSTANT')
imgClose=morlib.binClose(imgBW, kernel, border_filled='CONSTANT')
imgTopHat=morlib.binTopHat(imgBW, kernel, border_filled='CONSTANT')
imgBlackHat=morlib.binBlackHat(imgBW, kernel, border_filled='CONSTANT')

quickPlot(221,imgOpen,cmap='binary',title='Open')
quickPlot(222,imgClose,cmap='binary',title='Close')
quickPlot(223,imgTopHat,cmap='binary',title='Top Hat')
quickPlot(224,imgBlackHat,cmap='binary',title='Black Hat')
plt.show()