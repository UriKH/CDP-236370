#
#   @date:  [TODO: Today's date]
#   @author: [TODO: Student Names]
#
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2024
#
import multiprocessing
import random
from scipy import ndimage
import numpy as np


class Worker(multiprocessing.Process):
    
    def __init__(self, jobs, result, training_data, batch_size):
        super().__init__()

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
        self.jobs           = jobs
        self.result         = result
        self.training_data  = training_data
        self.batch_size     = batch_size


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
        img_rotated = ndimage.rotate(image.reshape((28,28)), angle, reshape=False, order=0)
        return img_rotated.reshape(784)

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

        return ndimage.shift(image.reshape((28,28)), (-dy, -dx)).reshape(784)
    
    
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
        noise_array = np.random.uniform(low=-noise, high=noise, size=image.shape)
        return np.clip(image+noise_array, a_min=0, a_max=1)

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
        reshaped_image = image.reshape(28,28)
        skewed_image = np.zeros_like(reshaped_image)
        for i in range(0,28):
            for j in range(0,28):
                k = int(np.floor(j+i*tilt))
                if k < 28:
                    skewed_image[i, j] = reshaped_image[i, k]

        return skewed_image.reshape(784)

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
        angle = random.randint(0, 359)
        dx = random.randint(-10, 10)
        dy = random.randint(-10, 10)
        noise = random.uniform(-0.2, 0.2)
        skew_val = random.uniform(-0.5, 0.5)

        return Worker.rotate(Worker.skew(Worker.add_noise(Worker.shift(image, dx, dy),noise), skew_val), angle)

    def run(self):
        '''Process images from the jobs queue and add the result to the result queue.
		Hint: you can either generate (i.e sample randomly from the training data)
		the image batches here OR in ip_network.create_batches
        '''
        while True:
            job = self.jobs.get()
            if job is None: # poison pill
                self.jobs.task_done()
                return
            
            training_images, training_labels = self.training_data[0], self.training_data[1]

            indexes = random.sample(range(0, training_images.shape[0]), self.batch_size)
            images, labels = (training_images[indexes], training_labels[indexes])

            batch_images = np.empty(shape=(self.batch_size * 2, images.shape[1]))
            batch_labels = np.empty(shape=(self.batch_size * 2, labels.shape[1]))

            for i in range(0, self.batch_size):
                batch_images[i*2]  = self.process_image(images[i])
                batch_images[i*2 + 1] = images[i]
                batch_labels[i*2] = labels[i]
                batch_labels[i*2 + 1] = labels[i]
                
            indexes = random.sample(range(0, self.batch_size * 2), self.batch_size)
            self.result.put((batch_images[indexes], batch_labels[indexes]))
            self.jobs.task_done()