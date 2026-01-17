#
#   @date:  [TODO: Today's date]
#   @author: [TODO: Student Names]
#
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2024
#
import numpy as np
from network import *
from preprocessor import Worker
import os
import multiprocessing
import my_queue

class IPNeuralNetwork(NeuralNetwork):
    def __init__(self, sizes=None, learning_rate=1, mini_batch_size=16, number_of_batches=16, epochs=10, matmul=np.matmul):
        super().__init__(sizes, learning_rate, mini_batch_size, number_of_batches, epochs, np.matmul)
        self.result = my_queue.MyQueue() 
        self.jobs = multiprocessing.JoinableQueue()
    

    def fit(self, training_data, validation_data=None):
        '''
        Override this function to create and destroy workers
        '''
        # 1. Create Workers
		# (Call Worker() with self.mini_batch_size as the batch_size)
        num_workers = int(os.environ['SLURM_CPUS_PER_TASK'])
        
        workers = []
        for i in range(num_workers):
            workers.append(Worker(self.jobs, self.result, training_data, batch_size=self.mini_batch_size))
            workers[i].start()
        
        for _ in range(self.epochs * self.number_of_batches):
            self.jobs.put('j')
        

        # Call the parent's fit. Notice how create_batches is called inside super.fit().
        super().fit(training_data, validation_data)
            
        # 3. Stop Workers
        for i in range(num_workers):
            self.jobs.put(None)
        
        self.jobs.join()

        # kill workers?
        for i in range(num_workers):
            workers[i].join()
        
    
    def create_batches(self, data, labels, batch_size):
        '''
        Override this function to return self.number_of_batches batches created by workers
		Hint: you can either generate (i.e sample randomly from the training data) the image batches here OR in Worker.run()
        '''
        new_batches = []
        for _ in range(self.number_of_batches):
            new_batches.append(self.result.get())
        
        return new_batches