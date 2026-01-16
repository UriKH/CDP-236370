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
        training_data = training_data[self.rank * number_of_batches: (self.rank + 1)  * number_of_batches]

        for epoch in range(self.epochs):
            # creating batches for epoch
            data = training_data[0]
            labels = training_data[1]
            mini_batches = self.create_batches(data, labels, self.mini_batch_size)
            for x, y in mini_batches:
                # do work - don't change this
                self.forward_prop(x)
                nabla_b, nabla_w = self.back_prop(y)

                # send nabla_b, nabla_w to masters 
                for i in range(self.num_masters):
                    end = (i + 1) * self.layers_per_master
                    if i == self.num_masters - 1:
                        end = self.num_layers
                    MPI.Send(nabla_w[i * self.layers_per_master:end], dest = i, tag = 0)
                    MPI.Send(nabla_b[i * self.layers_per_master:end], dest = i, tag = 1)

                # recieve new self.weight and self.biases values from masters
                for i in range(self.num_masters):
                    end = (i + 1) * self.layers_per_master
                    if i == self.num_masters - 1:
                        end = self.num_layers
                    MPI.Recv(self.weights[i * self.layers_per_master:end], dest = i, tag = 0)
                    MPI.Recv(self.biases[i * self.layers_per_master:end], dest = i, tag = 1)

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
                MPI.Recv(nabla_w, source=MPI.ANY_SOURCE, tag=0)
                MPI.Recv(nabla_b, source=MPI.ANY_SOURCE, tag=1)

                # calculate new weights and biases (of layers in charge)
                for i, dw, db in zip(range(self.rank, self.num_layers, self.num_masters), nabla_w, nabla_b):
                    self.weights[i] = self.weights[i] - self.eta * dw
                    self.biases[i] = self.biases[i] - self.eta * db

                # send new values (of layers in charge)
                for i in range(self.num_workers):
                    MPI.Send(self.weights[self.rank:self.num_layers:self.num_masters], dest=i + self.num_masters, tag=0)
                    MPI.Send(self.biases[self.rank:self.num_layers:self.num_masters], dest=i + self.num_masters, tag=1)

            self.print_progress(validation_data, epoch)

        # gather relevant weight and biases to process 0
        MPI.Send(self.weights[self.rank:self.num_layers:self.num_masters], dest=0, tag=2)
        MPI.Send(self.biases[self.rank:self.num_layers:self.num_masters], dest=0, tag=3)
