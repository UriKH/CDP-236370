from network import *
import itertools
import sys
import numpy as np
import math
import mpi4py
from time import time

mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI


class AsynchronicNeuralNetwork(NeuralNetwork):

    def __init__(self, sizes=list(), learning_rate=1.0, mini_batch_size=16, number_of_batches=16,
                 epochs=10, number_of_masters=1, matmul=np.matmul):
        # calling super constructor
        super().__init__(sizes, learning_rate, mini_batch_size, number_of_batches, epochs, matmul)
        # setting number of workers and masters
        self.num_masters = number_of_masters

    def fit(self, training_data, validation_data=None):
        # MPI setup
        MPI.Init()
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.num_workers = self.size - self.num_masters

        self.layers_per_master = self.num_layers // self.num_masters

        # split up work
        if self.rank < self.num_masters:
            self.do_master(validation_data)
        else:
            self.do_worker(training_data)

        # when all is done
        self.comm.Barrier()
        MPI.Finalize()

    def do_worker(self, training_data):
        """
        worker functionality
        :param training_data: a tuple of data and labels to train the NN with
        """
        # setting up the number of batches the worker should do every epoch
        
        number_of_batches = self.number_of_batches // self.num_workers
        worker_id = self.rank - self.num_masters

        start_idx = worker_id * number_of_batches
        end_idx = (worker_id + 1) * number_of_batches

        data_slice = training_data[0][start_idx:end_idx]
        labels_slice = training_data[1][start_idx:end_idx]

        for epoch in range(self.epochs):
            # creating batches for epoch
            mini_batches = self.create_batches(data_slice, labels_slice, self.mini_batch_size)
            for x, y in mini_batches:
                # do work - don't change this
                self.forward_prop(x)
                nabla_b, nabla_w = self.back_prop(y)

                # send nabla_b, nabla_w to masters 
                requests = []
                for i in range(self.num_layers):
                    requests.append(self.comm.Isend(nabla_w[i], dest = i % self.num_masters, tag = i))
                    requests.append(self.comm.Isend(nabla_b[i], dest = i % self.num_masters, tag = i + self.num_layers))

                MPI.Request.Waitall(requests)
                requests.clear()

                # recieve new self.weight and self.biases values from masters
                for i in range(self.num_layers):
                    requests.append(self.comm.Irecv(nabla_w[i], source = i % self.num_masters, tag = i))
                    requests.append(self.comm.Irecv(nabla_b[i], source = i % self.num_masters, tag = i + self.num_layers))

                MPI.Request.Waitall(requests)

    def do_master(self, validation_data):
        """
        master functionality
        :param validation_data: a tuple of data and labels to train the NN with
        """
        # setting up the layers this master does
        nabla_w = []
        nabla_b = []
        for i in range(self.rank, self.num_layers, self.num_masters):
            nabla_w.append(np.zeros_like(self.weights[i]))
            nabla_b.append(np.zeros_like(self.biases[i]))

        for epoch in range(self.epochs):
            for batch in range(self.number_of_batches):

                # wait for any worker to finish batch and
                # get the nabla_w, nabla_b for the master's layers
                status = MPI.Status()
                self.comm.Recv(nabla_w[0], source=MPI.ANY_SOURCE, tag=self.rank, status=status)
                src = status.Get_source()

                requests = [self.comm.Irecv(nabla_b[0], source=src, tag=self.rank + self.num_layers)]

                global_indices = range(self.rank + self.num_masters, self.num_layers, self.num_masters)
                for local_idx, global_idx in enumerate(global_indices, 1):
                    requests.append(self.comm.Irecv(nabla_w[local_idx], source=src, tag=global_idx))
                    requests.append(self.comm.Irecv(nabla_b[local_idx], source=src, tag=global_idx + self.num_layers))
                
                MPI.Request.Waitall(requests)
                requests.clear()
                
                # calculate new weights and biases (of layers in charge)
                for i, dw, db in zip(range(self.rank, self.num_layers, self.num_masters), nabla_w, nabla_b):
                    self.weights[i] = self.weights[i] - self.eta * dw
                    self.biases[i] = self.biases[i] - self.eta * db

                # send new values (of layers in charge)
                for i in range(self.rank, self.num_layers, self.num_masters):
                    requests.append(self.comm.Isend(self.weights[i], dest=src, tag=i))
                    requests.append(self.comm.Isend(self.biases[i], dest=src, tag=i + self.num_layers))
                
                MPI.Request.Waitall(requests)

            self.print_progress(validation_data, epoch)

        # gather relevant weight and biases to process 0
        reqs = []
        if self.rank != 0:
            for i in range(self.rank, self.num_layers, self.num_masters):
                reqs.append(self.comm.Isend(self.weights[i], dest=0, tag=i))
                reqs.append(self.comm.Isend(self.biases[i], dest=0, tag=i + self.num_layers))
            MPI.Request.Waitall(reqs)
        else:
            for i in range(self.num_layers):
                if i % self.num_masters != 0:
                    src = i % self.num_masters
                    reqs.append(self.comm.Irecv(self.weights[i], source=src, tag=i))
                    reqs.append(self.comm.Irecv(self.biases[i], source=src, tag=i + self.num_layers))
            MPI.Request.Waitall(reqs)
