#
#   @date:  [2.1.2026]
#   @author: [Ofek Israel, Uri Kasher Hitin]
#
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2025
#
import multiprocessing
import scipy.ndimage as sc
import numpy as np


class Worker(multiprocessing.Process):
    
    def __init__(self, jobs, result, training_data, batch_size):
        ''' Initialize Worker and it's members.

        Parameters
        ----------
        jobs: JoinableQueue
            A jobs Queue for the worker.
        result: Queue
            A results Queue for the worker to put it's results in.
		training_data: 
			A tuple of (training images array, image lables array)
		batch_size:
			workers batch size of images (mini batch size)
        
        You should add parameters if you think you need to.
        '''
        super().__init__()
        self.jobs = jobs
        self.result = result
        self.training_data = training_data
        self.batch_size = batch_size

    @staticmethod
    def rotate(image, angle):
        '''Rotate given image to the given angle

        Parameters
        ----------
        image : numpy array
            An array of size 784 of pixels
        angle : int
            The angle to rotate the image
            
        Return
        ------
        An numpy array of same shape
        '''
        return sc.rotate(image, angle, reshape=False)

    @staticmethod
    def shift(image, dx, dy):
        '''Shift given image

        Parameters
        ----------
        image : numpy array
            An array of shape 784 of pixels
        dx : int
            The number of pixels to move in the x-axis
        dy : int
            The number of pixels to move in the y-axis
            
        Return
        ------
        An numpy array of same shape
        '''
        return sc.shift(image, (dy, dx))
    
    @staticmethod
    def add_noise(image, noise):
        '''Add noise to the image
        for each pixel a value is selected uniformly from the 
        range [-noise, noise] and added to it. 

        Parameters
        ----------
        image : numpy array
            An array of shape 784 of pixels
        noise : float
            The maximum amount of noise that can be added to a pixel

        Return
        ------
        An numpy array of same shape
        '''
        noise_matrix = np.random.uniform(-noise, noise, image.shape)
        return np.minimum(np.maximum(image + noise_matrix, 0), 1)

    @staticmethod
    def skew(image, tilt):
        '''Skew the image

        Parameters
        ----------
        image : numpy array
            An array of size 784 of pixels
        tilt : float
            The skew paramater

        Return
        ------
        An numpy array of same shape
        '''
        def mapping(coord):
            y, x = coord
            return y, x + tilt * y

        return sc.geometric_transform(image, mapping)

    def process_image(self, image):
        '''Apply the image process functions
		Experiment with the random bounds for the functions to see which produces good accuracies.

        Parameters
        ----------
        image: numpy array
            An array of size 784 of pixels

        Return
        ------
        An numpy array of same shape
        '''
        shift = np.random.randint(-7, 7, 2)
        shifted = self.shift(image, *shift)
        rotated = self.rotate(shifted, np.random.randint(-20, 20))
        noise = self.add_noise(rotated, 0.2)
        skewed = self.skew(noise, np.random.uniform(-0.2, 0.2))
        return skewed

    def run(self):
        '''Process images from the jobs queue and add the result to the result queue.
		Hint: you can either generate (i.e sample randomly from the training data)
		the image batches here OR in ip_network.create_batches
        '''
        while not self.jobs.empty():
            image = self.jobs.pop()
            augmented = self.process_image(image)
            self.result.put(augmented)
        # TODO: use the correct methods
