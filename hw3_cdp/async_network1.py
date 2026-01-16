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
        if not MPI.Is_initialized():
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
        # TODO: add your code
        batches_per_worker = self.number_of_batches // self.num_workers
    
        for epoch in range(self.epochs):
            # creating batches for epoch
            data = training_data[0]
            labels = training_data[1]
            mini_batches = self.create_batches(data, labels, self.mini_batch_size)
            for x, y in mini_batches[:batches_per_worker]:
                # do work - don't change this
                self.forward_prop(x)
                nabla_b, nabla_w = self.back_prop(y)

                # send nabla_b, nabla_w to masters 
                reqs = []
                for layer_idx in range(self.num_layers):
                    master_rank = layer_idx % self.num_masters
                    req_w = self.comm.Isend(nabla_w[layer_idx], dest=master_rank, tag=layer_idx)
                    req_b = self.comm.Isend(nabla_b[layer_idx], dest=master_rank, tag=layer_idx + self.num_layers)
                    reqs.append(req_w)
                    reqs.append(req_b)
                
                # recieve new self.weight and self.biases values from masters
                for layer_idx in range(self.num_layers):
                    master_rank = layer_idx % self.num_masters
                    self.comm.Recv(self.weights[layer_idx], source=master_rank, tag=layer_idx)
                    self.comm.Recv(self.biases[layer_idx], source=master_rank, tag=layer_idx + self.num_layers)

                for req in reqs:
                    req.Wait()

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
            batches_to_process = (self.number_of_batches // self.num_workers) * self.num_workers
            for batch in range(batches_to_process):

                # wait for any worker to finish batch and
                # get the nabla_w, nabla_b for the master's layers
                status = MPI.Status()
                first_layer_idx = self.rank
                self.comm.Recv(nabla_w[0], source=MPI.ANY_SOURCE, tag=first_layer_idx, status=status)
                worker_rank = status.Get_source()
                self.comm.Recv(nabla_b[0], source=worker_rank, tag=first_layer_idx + self.num_layers)

                nabla_idx = 1
                for layer_idx in range(self.rank + self.num_masters, self.num_layers, self.num_masters):
                    self.comm.Recv(nabla_w[nabla_idx], source=worker_rank, tag=layer_idx)
                    self.comm.Recv(nabla_b[nabla_idx], source=worker_rank, tag=layer_idx + self.num_layers)
                    nabla_idx += 1

                # calculate new weights and biases (of layers in charge)
                for i, dw, db in zip(range(self.rank, self.num_layers, self.num_masters), nabla_w, nabla_b):
                    self.weights[i] = self.weights[i] - self.eta * dw
                    self.biases[i] = self.biases[i] - self.eta * db

                # send new values (of layers in charge)
                reqs = []
                for layer_idx in range(self.rank, self.num_layers, self.num_masters):
                    req_w = self.comm.Isend(self.weights[layer_idx], dest=worker_rank, tag=layer_idx)
                    req_b = self.comm.Isend(self.biases[layer_idx], dest=worker_rank, tag=layer_idx + self.num_layers)
                    reqs.append(req_w)
                    reqs.append(req_b)

                for req in reqs:
                    req.Wait()

            self.print_progress(validation_data, epoch)

        # gather relevant weight and biases to process 0
        if self.rank == 0:
            for master_rank in range(1, self.num_masters):
                for layer_idx in range(master_rank, self.num_layers, self.num_masters):
                    self.comm.Recv(self.weights[layer_idx], source=master_rank, tag=layer_idx)
                    self.comm.Recv(self.biases[layer_idx], source=master_rank, tag=layer_idx + self.num_layers)
        else:
            for layer_idx in range(self.rank, self.num_layers, self.num_masters):
                self.comm.Send(self.weights[layer_idx], dest=0, tag=layer_idx)
                self.comm.Send(self.biases[layer_idx], dest=0, tag=layer_idx + self.num_layers)