'''
  A morphological function lib based on numpy.  
Morlib provides a series of funtion of morphological, mainly include dilation and erosion.  
To get started with, see demo.py
'''

import numpy as np


def padding(img, kernel_size, border_filled):
    '''Used to add padding around the raw image.  
    Arguments:
        img {nparray} -- the input binary picture.
        kernel_size {tuple/list/ndarray} -- the size of the kernel used. Should be 2n+1,n>=0.
        border_filled {str} --  indicate the way border filled. 'CONSTANT' means filling with 0; 'NEAREST' will filled with nearest value.
    Returns:
        imgPadded {nparray} -- the image with values padded.
    '''
    # Validation check
    try:
        img = np.array(img)
        if len(img.shape) != 2:
            raise Exception('Input shape of image doesn\'t fit.')
        if not (border_filled == 'CONSTANT' or border_filled == 'NEAREST'):
            raise Exception(
                'border_filled parameter dosen\'t fit. Use \'CONSTANT\' or \'NEAREST\' instead.'
            )
    except:
        raise Exception(
            'Invaild input image:expected ndarray or array-like data but get',
            type(img))

    # ZeroPadding
    x, y = img.shape
    dx, dy = kernel_size[0] // 2, kernel_size[1] // 2
    imgPadded = np.zeros((x + 2 * dx, y + 2 * dy))
    fx, fy = imgPadded.shape[0] - 1, imgPadded.shape[1] - 1
    imgPadded[dx:fx + 1 - dx, dy:fy + 1 - dy] = img

    # Nearest if needed
    if border_filled == 'NEAREST':
        for i in range(dy):
            imgPadded[dx:-dx, i] = imgPadded[dx:-dx, dy]
            imgPadded[dx:-dx, fy - i] = imgPadded[dx:-dx, fy - dy]
        for i in range(dx):
            imgPadded[i, dy:-dy] = imgPadded[dx, dy:-dy]
            imgPadded[fx - i, dy:-dy] = imgPadded[fx - dx, dy:-dy]
        imgPadded[:dx, :dy] = imgPadded[dx, dy]
        imgPadded[:dx, fy - dy:] = imgPadded[dx, fy - dy]
        imgPadded[fx - dx:, fy - dy:] = imgPadded[fx - dx, fy - dy]
        imgPadded[fx - dx:, :dy] = imgPadded[fx - dx, dy]
    return imgPadded


def _conv(img, mask, logic):
    '''Convolute the image with the mask of the same size.
    
    Arguments:
        img {ndarray} -- the binary image.
        kernel {ndarray} -- the mask.
        logic {str} -- the logic of the calculation.Use keyword 'AND' or 'OR'.
    Returns:
        {int} -- the binary. 0 or 1.
    '''
    # Validation check
    try:
        if img.shape != mask.shape:
            raise Exception('image size don\'t fit the kernel size')
        if not (logic == 'AND' or logic == 'OR'):
            raise Exception(
                'parameter logic dosen\'t fit. Use \'AND\' or \'OR\' instead.')
    except:
        raise Exception('invaild data type:', type(img), '&', type(mask))

    # Conv
    imask = np.multiply(img, mask)

    # Result
    numOnekernel = len(np.where(mask == 1)[0])
    numOne = len(np.where(imask == 1)[0])
    if logic == 'AND':
        if numOne == numOnekernel:
            return 1
        return 0
    if logic == 'OR':
        if numOne == 0:
            return 0
        return 1


def createKernel(shape, keyword):
    '''A quick kernel creating function.
    
    Arguments:
        shape {array-like} -- the shape of the kernel.
        keyword {string} -- should be 'X' or 'rec'.  
        'X' creates kernel like(shape=(3,3)):[[0, 1, 0],
                                              [1, 1, 1],
                                              [0, 1, 0]]
        while 'rec' does(shape=(3,3)):[[1, 1, 1],
                                       [1, 1, 1],
                                       [1, 1, 1]]
    Returns:
        [ndarray] -- the kernel
    '''

    if len(shape) != 2:
        raise Exception('expect parameters shape with length of 2 but get',
                        len(shape))
    if keyword == 'X':
        kernel = np.zeros(shape)
        kernel[shape[0] // 2, :] = 1
        kernel[:, shape[1] // 2] = 1
    elif keyword == 'rec':
        kernel = np.ones(shape)
    else:
        raise Exception('Undefined keyword:' + keyword)
    return kernel


def imgBinarization(img, thershold=None):
    '''Binarization the given gray-image using given thershold.  
    If thershold isn't given, the function will choose the thershold via Otsu.
    
    Arguments:
        img {ndarray} -- the given gray-image. It should be a 2-d array.
    
    Keyword Arguments:
        thershold {int} -- the thershold chosen.

    Returns:
        binary_img {ndarray} , thershold {int}
    '''
    # Validation check
    try:
        img = np.array(img)
        if len(img.shape) != 2:
            raise Exception('Input shape of image doesn\'t fit.')
        if thershold and (thershold > np.max(img) or thershold < np.min(img)):
            print('Given thershold is invaild. Use Otsu algorithm instead.')
            thershold = None
    except:
        raise Exception('invaild image type:expect ndarray but', type(img))

    # thershold unknown
    if not thershold:
        # Gray scale adjustment
        minV, maxV = np.min(img), np.max(img)
        delta = maxV - minV
        imgAdj = np.array((img - minV) / delta * 255, dtype=np.uint8)
        u, w = np.mean(imgAdj), imgAdj.shape[0] * imgAdj.shape[1]
        gmax = -np.inf

        # Otsu algorithm
        for i in range(1, 255):
            forwardGround = np.where(imgAdj <= i)
            w0 = len(forwardGround[0]) / w
            u0 = np.sum(imgAdj[forwardGround]) / (w0 * w)
            w1 = 1 - w0
            u1 = (u - w0 * u0) / w1
            g = w0 * w1 * (u1 - u0)**2
            if g > gmax:
                gmax, thershold = g, i

        thershold = thershold / 255 * delta + minV
    # Dividing
    imgOut = np.floor(img / thershold)
    imgOut[np.where(imgOut != 0)] = 1
    return imgOut, thershold


def binErosion(img, kernel, border_filled='CONSTANT'):
    '''Execute morphological erosion operation on the given binary picture.  
    We assume that the anchor is at the center of the kernel, so it's better to use odd-length kernel.  
    Arguments:
        img {nparray} -- the input binary picture.
        kernel {nparray} -- the kernel used. Whose length should be 2n+1,n>=0.
        border_filled {str} --  indicate the way border filled. 'CONSTANT' means filling with 0; 'NEAREST' will filled with nearest value.    
    Returns:
        imgOutput {nparray} -- morphological erosion operatated image 
    '''
    # Validation check
    try:
        img, kernel = np.array(img), np.array(kernel)
        if len(img.shape) != 2 or len(kernel.shape) != 2:
            raise Exception('Input shape of image or kernel doesn\'t fit.')
        if not (border_filled == 'CONSTANT' or border_filled == 'NEAREST'):
            raise Exception(
                'border_filled parameter dosen\'t fit. Use \'CONSTANT\' or \'NEAREST\' instead.'
            )
    except:
        raise Exception('invaild image type:expect ndarray but', type(img),
                        'or', type(kernel))

    x, y = img.shape
    dx, dy = kernel.shape
    imgOutput = np.zeros(img.shape)

    # ZeroPadding
    imgPadded = padding(img, kernel.shape, border_filled=border_filled)
    # Erosion Main
    for i in range(x):
        for j in range(y):
            imgOutput[i, j] = _conv(
                imgPadded[i:i + dx, j:j + dy], kernel, logic='AND')
    #Return
    return imgOutput


def binDilation(img, kernel, border_filled='CONSTANT'):
    '''Execute morphological dilation operation on the given binary picture.  
    We assume that the anchor is at the center of the kernel, so it's better to use odd-length kernel.  
    Arguments:
        img {nparray} -- the input binary picture.
        kernel {nparray} -- the kernel used. Whose length should be 2n+1,n>=0.
        border_filled {str} --  indicate the way border filled. 'CONSTANT' means filling with 0; 'NEAREST' will filled with nearest value.
    Returns:
        imgOutput {nparray} -- morphological dilation operatated image 
    '''
    # Validation check
    try:
        img, kernel = np.array(img), np.array(kernel)
        if len(img.shape) != 2 or len(kernel.shape) != 2:
            raise Exception('Input shape of image or kernel doesn\'t fit.')
        if not (border_filled == 'CONSTANT' or border_filled == 'NEAREST'):
            raise Exception(
                'border_filled parameter dosen\'t fit. Use \'CONSTANT\' or \'NEAREST\' instead.'
            )
    except:
        raise Exception('invaild image type:expect ndarray but', type(img),
                        'or', type(kernel))

    x, y = img.shape
    dx, dy = kernel.shape
    imgOutput = np.zeros(img.shape)

    # ZeroPadding
    imgPadded = padding(img, kernel.shape, border_filled=border_filled)
    # Erosion Main
    for i in range(x):
        for j in range(y):
            imgOutput[i, j] = _conv(
                imgPadded[i:i + dx, j:j + dy], kernel, logic='OR')
    #Return
    return imgOutput


def binOpen(img, kernel, border_filled='CONSTANT'):
    '''Execute morphological open operation on the given binary picture, which equals to the dilation operation after erosion operation.  
    We assume that the anchor is at the center of the kernel, so it's better to use odd-length kernel.  
    Arguments:
        img {nparray} -- the input binary picture.
        kernel {nparray} -- the kernel used. Whose length should be 2n+1,n>=0.
        border_filled {str} --  indicate the way border filled. 'CONSTANT' means filling with 0; 'NEAREST' will filled with nearest value.
    Returns:
        imgOpen {nparray} -- morphological open operatated image
    '''
    # Validation check
    try:
        img, kernel = np.array(img), np.array(kernel)
        if len(img.shape) != 2 or len(kernel.shape) != 2:
            raise Exception('Input shape of image or kernel doesn\'t fit.')
        if not (border_filled == 'CONSTANT' or border_filled == 'NEAREST'):
            raise Exception(
                'border_filled parameter dosen\'t fit. Use \'CONSTANT\' or \'NEAREST\' instead.'
            )
    except:
        raise Exception('invaild image type:expect ndarray but', type(img),
                        'or', type(kernel))

    # Erosion + Dilation = Open
    imgErosion = binErosion(img, kernel, border_filled=border_filled)
    imgOpen = binDilation(imgErosion, kernel, border_filled=border_filled)
    return imgOpen


def binClose(img, kernel, border_filled='CONSTANT'):
    '''Execute morphological close operation on the given binary picture, which equals to the dilation operation after erosion operation.  
    We assume that the anchor is at the center of the kernel, so it's better to use odd-length kernel.  
    Arguments:
        img {nparray} -- the input binary picture.
        kernel {nparray} -- the kernel used. Whose length should be 2n+1,n>=0.
        border_filled {str} --  indicate the way border filled. 'CONSTANT' means filling with 0; 'NEAREST' will filled with nearest value.
    Returns:
        imgClose {nparray} -- morphological close operatated image
    '''
    # Validation check
    try:
        img, kernel = np.array(img), np.array(kernel)
        if len(img.shape) != 2 or len(kernel.shape) != 2:
            raise Exception('Input shape of image or kernel doesn\'t fit.')
        if not (border_filled == 'CONSTANT' or border_filled == 'NEAREST'):
            raise Exception(
                'border_filled parameter dosen\'t fit. Use \'CONSTANT\' or \'NEAREST\' instead.'
            )
    except:
        raise Exception('invaild image type:expect ndarray but', type(img),
                        'or', type(kernel))

    # Dilation + Erosion = close
    imgDilation = binDilation(img, kernel, border_filled=border_filled)
    imgClose = binErosion(imgDilation, kernel, border_filled=border_filled)
    return imgClose


def binTopHat(img, kernel, border_filled='CONSTANT'):
    '''Get the Top-Hat image of the given binary picture.  
    Top-Hat image equals to the difference between the raw image and the open operated image.
    We assume that the anchor is at the center of the kernel, so it's better to use odd-length kernel.  
    Arguments:
        img {nparray} -- the input binary picture.
        kernel {nparray} -- the kernel used. Whose length should be 2n+1,n>=0.
        border_filled {str} --  indicate the way border filled. 'CONSTANT' means filling with 0; 'NEAREST' will filled with nearest value.
    Returns:
        imgTopHat {nparray} -- the Top-Hat image
    '''
    # Validation check
    try:
        img, kernel = np.array(img), np.array(kernel)
        if len(img.shape) != 2 or len(kernel.shape) != 2:
            raise Exception('Input shape of image or kernel doesn\'t fit.')
        if not (border_filled == 'CONSTANT' or border_filled == 'NEAREST'):
            raise Exception(
                'border_filled parameter dosen\'t fit. Use \'CONSTANT\' or \'NEAREST\' instead.'
            )
    except:
        raise Exception('invaild image type:expect ndarray but', type(img),
                        'or', type(kernel))

    # Top-Hat = raw - open
    imgOpen = binOpen(img, kernel, border_filled=border_filled)
    imgTopHat = img - imgOpen
    return imgTopHat


def binBlackHat(img, kernel, border_filled='CONSTANT'):
    '''Get the Black-Hat image of the given binary picture.  
    Black-Hat image equals to the difference between the close operated image and the raw image.
    We assume that the anchor is at the center of the kernel, so it's better to use odd-length kernel.  
    Arguments:
        img {nparray} -- the input binary picture.
        kernel {nparray} -- the kernel used. Whose length should be 2n+1,n>=0.
        border_filled {str} --  indicate the way border filled. 'CONSTANT' means filling with 0; 'NEAREST' will filled with nearest value.
    Returns:
        imgBlackHat {nparray} -- the Black-Hat image
    '''
    # Validation check
    try:
        img, kernel = np.array(img), np.array(kernel)
        if len(img.shape) != 2 or len(kernel.shape) != 2:
            raise Exception('Input shape of image or kernel doesn\'t fit.')
        if not (border_filled == 'CONSTANT' or border_filled == 'NEAREST'):
            raise Exception(
                'border_filled parameter dosen\'t fit. Use \'CONSTANT\' or \'NEAREST\' instead.'
            )
    except:
        raise Exception('invaild image type:expect ndarray but', type(img),
                        'or', type(kernel))

    # Black-Hat = raw - open
    imgClose = binClose(img, kernel, border_filled=border_filled)
    imgBlackHat = imgClose - img
    
    imgBlackHat[np.where(imgBlackHat<0)]=0
    return imgBlackHat