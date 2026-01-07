#
#   @date:  [TODO: Today's date]
#   @author: [TODO: Student Names]
#
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2025
#
from numba import cuda
from numba import njit, prange
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
    @cuda.jit
    def apply_kernel(d_image, d_kernel, out):
        i, j = d_image.shape

        im_height, im_width = d_image.shape
        k_size_y, k_size_x = d_kernel.shape
        k_size_x = (k_size_x - 1) // 2
        k_size_y = (k_size_y - 1) // 2

        if i < im_height and j < im_width:
            value = 0.0

            for k in range(-k_size_y, k_size_y + 1):
                for l in range(-k_size_x, k_size_x + 1):
                    if i + k >= 0 and i + k < im_height and j + l >= 0 and j + l < im_width:
                        value += d_kernel[k + k_size_y, l + k_size_x] * d_image[i + k, j + l]

            out[i, j] = value

    d_out = cuda.to_device(np.zeros(image.shape))
    d_image = cuda.to_device(image)
    d_kernel = cuda.to_device(kernel)

    threads_per_block = (16, 16)
    blocks_per_grid_x = (image.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (image.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    apply_kernel[blocks_per_grid, threads_per_block](d_image, d_kernel, d_out)
    return d_out.copy_to_host()


@njit
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
    # compute kernel dimensions from kernel center
    k_size_y, k_size_x = kernel.shape
    k_size_x = (k_size_x - 1) // 2
    k_size_y = (k_size_y - 1) // 2

    # prepare the correlation matrix
    im_height, im_width = image.shape
    correlation = np.zeros(shape=(im_height, im_width))

    # compute for each pixel of the image the kernel effect
    for i in prange(im_height):
        for j in prange(im_width):
            # compute for each entry in the kernel
            for k in range(-k_size_y, k_size_y + 1):
                for l in range(-k_size_x, k_size_x + 1):
                    if i + k < 0 or j + l < 0 or i + k >= im_height or j + l >= im_width:
                        continue
                    correlation[i, j] += kernel[k + k_size_y, l + k_size_x] * image[i + k, j + l]
    return correlation


def sobel_operator():
    '''Load the image and perform the operator
        ----------
        Return
        ------
        An numpy array of the image
        '''
    pic = load_image()
    # your calculations

    sobl_kernel = np.array(
        [[1, 0, -1],
         [2, 0, -2],
         [1, 0, -1]]
    )
    gx = correlation_numba(sobl_kernel, pic)
    gy = correlation_numba(sobl_kernel.T, pic)
    return np.sqrt(gx ** 2 + gy ** 2)


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
