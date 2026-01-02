#
#   @date:  [TODO: Today's date]
#   @author: [TODO: Student Names]
#
# This file is for the solutions of the wet part of HW2 in
# "Concurrent and Distributed Programming for Data processing
# and Machine Learning" course (02360370), Winter 2025
#
from hw2_cdp.my_queue import MyQueue
from network import *
from preprocessor import Worker
import os
import multiprocessing as mp

ON_SERVER = False
if ON_SERVER:
    CPUS = int(os.environ['SLURM_CPUS_PER_TASK'])
else:
    CPUS = os.cpu_count()


class IPNeuralNetwork(NeuralNetwork):
    def __init__(self, *args):
        super().__init__(*args)
        self.jobs = mp.JoinableQueue()
        self.results = MyQueue()

    def fit(self, training_data, validation_data=None):
        """
        Override this function to create and destroy workers
        """
        # 1. Create Workers
        # (Call Worker() with self.mini_batch_size as the batch_size)

        workers = [Worker(self.jobs, self.results, training_data, batch_size=self.mini_batch_size) for _ in range(CPUS)]
        for w in workers:
            w.start()

        for _ in range(self.number_of_batches * self.epochs):
            self.jobs.put('do work!')

        # 2. Set jobs
        # Call the parent's fit. Notice how create_batches is called inside super.fit().
        super().fit(training_data, validation_data)

        # 3. Stop Workers
        for _ in workers:
            self.jobs.put(None)
        self.jobs.join()

        for w in workers:
            w.join()
        return

    def create_batches(self, data, labels, batch_size):
        '''
        Override this function to return self.number_of_batches batches created by workers
		Hint: you can either generate (i.e sample randomly from the training data) the image batches here OR in Worker.run()
        '''
        return [self.results.get() for _ in range(self.number_of_batches)]
