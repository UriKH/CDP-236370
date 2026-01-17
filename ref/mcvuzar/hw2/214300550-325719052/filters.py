#
#   @date:  [19/03/2024]
#   @author: [Michal Ozeri, Guy Sudai]
#
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2024
#
from numba import cuda
from numba import njit, cuda, prange
import imageio
import matplotlib.pyplot as plt
import numpy as np


def correlation_gpu(kernel, image):
    '''Correlate using gpu
    Parameters
    ----------
    kernel : numpy array
        A small matrix
    image : numpy array
        A larger matrix of the image pixels
            
    Return
    ------
    An numpy array of same shape as image
    '''
    rows, columns = image.shape
    k_rows, k_columns = kernel.shape
    zero_padded_image = np.zeros((rows + (k_rows - 1), columns + (k_columns - 1)))
    zero_padded_image[(k_rows//2):rows + (k_rows//2), (k_columns//2):columns + (k_columns//2)] = image
    
    dev_image = cuda.to_device(zero_padded_image)
    dev_kernel = cuda.to_device(kernel)
    dev_C = cuda.to_device(np.zeros(image.shape)) # create directly on gpu

    # Set the number of threads in a block
    threadsperblock = columns 
    blockspergrid = rows

    apply_kernel[blockspergrid, threadsperblock](dev_image, dev_kernel, dev_C)
    C = dev_C.copy_to_host()
    return C


@cuda.jit
def apply_kernel(image, kernel, C):
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    k_rows, k_cols = kernel.shape
    for i in range(k_rows):
        for j in range(k_cols):
            C[bx,tx] += image[bx + i, tx + j] * kernel[i, j]


@njit(parallel=True)
def correlation_numba(kernel, image):
    '''Correlate using numba
    Parameters
    ----------
    kernel : numpy array
        A small matrix
    image : numpy array
        A larger matrix of the image pixels
            
    Return
    ------
    An numpy array of same shape as image
    '''
    k_rows, k_cols = kernel.shape
    rows, columns = image.shape
    zero_padded_image = np.zeros((rows + (k_rows - 1), columns + (k_cols- 1)))
    zero_padded_image[(k_rows//2):rows + (k_rows//2), (k_cols//2):columns + (k_cols//2)] = image
    C = np.zeros(image.shape)

    for x in prange(rows):
        for y in prange(columns):
            for i in prange(k_rows):
                for j in prange(k_cols):
                    C[x, y] += zero_padded_image[x + i, y + j] * kernel[i, j]
    return C

def sobel_operator():
    '''Load the image and perform the operator
        ----------
        Return
        ------
        An numpy array of the image
        '''
    pic = load_image()
    # your calculations
    sobel_filter = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    Gx = correlation_numba(sobel_filter, pic)
    Gy = correlation_numba(sobel_filter.transpose(), pic)
    result = np.sqrt(np.power(Gx, 2) + np.power(Gy, 2))
    return result

def load_image(): 
    fname = 'data/image.jpg'
    pic = imageio.imread(fname)
    to_gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114])
    gray_pic = to_gray(pic)
    return gray_pic


def show_image(image):
    """ Plot an image with matplotlib

    Parameters
    ----------
    image: list
        2d list of pixels
    """
    plt.imshow(image, cmap='gray')
    plt.show()
