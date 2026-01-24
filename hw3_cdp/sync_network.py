from network import *
from my_ring_allreduce import *
import mpi4py

mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI
import random


class SynchronicNeuralNetwork(NeuralNetwork):
    def fit(self, training_data, validation_data=None):
        # initialize MPI if needed
        if not MPI.Is_initialized():
            MPI.Init()
        comm = MPI.COMM_WORLD
        random.seed(42 + comm.Get_rank())
        size = comm.Get_size()

        for epoch in range(self.epochs):
            mini_batches = self.create_batches(training_data[0], training_data[1], self.mini_batch_size // size)

            for x, y in mini_batches:
                # forward and back propagaintion
                self.forward_prop(x)
                my_nabla_b, my_nabla_w = self.back_prop(y)

                # sum gradients using ring allreduce
                sum_op = lambda a, b: a + b
                nabla_w = [np.empty_like(w) for w in my_nabla_w]
                nabla_b = [np.empty_like(b) for b in my_nabla_b]
                for i in range(len(my_nabla_w)):
                    ringallreduce(my_nabla_w[i], nabla_w[i], comm, sum_op)
                    ringallreduce(my_nabla_b[i], nabla_b[i], comm, sum_op)

                # update weights and biases
                self.weights = [w - self.eta * dw for w, dw in zip(self.weights, nabla_w)]
                self.biases = [b - self.eta * db for b, db in zip(self.biases, nabla_b)]

            self.print_progress(validation_data, epoch)

        MPI.Finalize()
