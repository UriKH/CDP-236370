from network import *
import itertools
import sys
import numpy as np
import math
import mpi4py
from time import time

mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI

TAG_GRADS  = 10001
TAG_PARAMS = 10002
TAG_GATHER = 10003
TAG_INIT   = 10004

class AsynchronicNeuralNetwork(NeuralNetwork):

    def __init__(self, sizes=list(), learning_rate=1.0, mini_batch_size=16, number_of_batches=16,
                 epochs=10, number_of_masters=1, matmul=np.matmul):
        super().__init__(sizes, learning_rate, mini_batch_size, number_of_batches, epochs, matmul)
        self.num_masters = number_of_masters

    # Returns the list of layer indices assigned to a given master
    def _get_layer_indices(self, master_rank):
        return list(range(master_rank, self.num_layers, self.num_masters))

    # Calculates the total number of elements needed for a set of layers
    def _calculate_buffer_size(self, layer_indices):
        total_size = 0
        for i in layer_indices:
            total_size += self.weights[i].size + self.biases[i].size
        return total_size

    # Flattens weights and biases into a single contiguous 1D numpy array
    def _flatten_params(self, layer_indices, w_list, b_list):
        # Collect all arrays into a flat list
        flat_arrays = []
        for w, b in zip(w_list, b_list):
            flat_arrays.append(w.ravel())
            flat_arrays.append(b.ravel())
        # Concatenate into one buffer
        return np.concatenate(flat_arrays)

    # Reconstructs the list of weights and biases from a flat buffer
    def _unflatten_params(self, flat_buffer, layer_indices):
        w_list = []
        b_list = []
        offset = 0
        
        for i in layer_indices:
            # Reconstruct Weight
            w_shape = self.weights[i].shape
            w_size = self.weights[i].size
            w_flat = flat_buffer[offset : offset + w_size]
            w_list.append(w_flat.reshape(w_shape))
            offset += w_size
            
            # Reconstruct Bias
            b_shape = self.biases[i].shape
            b_size = self.biases[i].size
            b_flat = flat_buffer[offset : offset + b_size]
            b_list.append(b_flat.reshape(b_shape))
            offset += b_size
            
        return w_list, b_list

    def fit(self, training_data, validation_data=None):
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
        # Splitting the batches among workers
        worker_id = self.rank - self.num_masters

        batches_count = self.number_of_batches // self.num_workers
        remainder = self.number_of_batches % self.num_workers

        worker_batches = batches_count
        # Handle remainder batches - split them among workers
        if worker_id < remainder:
            worker_batches += 1

        # Set up start and end indices for this worker
        start_idx = worker_id * batches_count + min(worker_id, remainder)
        end_idx = start_idx + worker_batches

        # Pre-allocate receive buffers for each master to save time in loop
        recv_buffers = {}
        for m in range(self.num_masters):
            l_ids = self._get_layer_indices(m)
            buf_size = self._calculate_buffer_size(l_ids)
            recv_buffers[m] = np.empty(buf_size, dtype=np.float64)

        for epoch in range(self.epochs):
            # creating batches for epoch
            data = training_data[0]
            labels = training_data[1]
            mini_batches = self.create_batches(data, labels, self.mini_batch_size)
            
            mini_batches = mini_batches[:self.number_of_batches]
            mini_batches = mini_batches[start_idx:end_idx]

            for x, y in mini_batches:
                 # do work - don't change this
                self.forward_prop(x)
                nabla_b, nabla_w = self.back_prop(y)

                # send nabla_b, nabla_w to masters 
                requests = []
                for master in range(self.num_masters):
                    # Figure out which layers this master owns
                    target_layers = self._get_layer_indices(master)
                    
                    # Get the gradients for those specific layers
                    w_subset = [nabla_w[i] for i in target_layers]
                    b_subset = [nabla_b[i] for i in target_layers]
                    
                    # Flatten data into contiguous buffer
                    send_buf = self._flatten_params(target_layers, w_subset, b_subset)
                    
                    # Send data to the master
                    req = self.comm.Isend([send_buf, MPI.DOUBLE], dest=master, tag=TAG_GRADS)
                    req.Wait() # Wait immediately to ensure buffer is safe to reuse if needed

                # Recieve new weightes and biases from masters
                for master in range(self.num_masters):
                    target_layers = self._get_layer_indices(master)
                    
                    # Use pre-allocated buffer
                    buf = recv_buffers[master]
                    
                    # Wait for the updated parameters
                    req = self.comm.Irecv([buf, MPI.DOUBLE], source=master, tag=TAG_PARAMS)
                    req.Wait()
                    
                    # Unflatten back to arrays
                    new_w, new_b = self._unflatten_params(buf, target_layers)

                    # Apply updates to the local worker model
                    for i, w, b in zip(target_layers, new_w, new_b):
                        self.weights[i] = w
                        self.biases[i] = b

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

        # Calculate which layers belong to this master
        master_layers = list(range(self.rank, self.num_layers, self.num_masters))
        
        # Pre-allocate buffer for receiving gradients (Required for Irecv)
        buf_size = self._calculate_buffer_size(master_layers)
        grad_recv_buf = np.empty(buf_size, dtype=np.float64)

        for epoch in range(self.epochs):
            for batch in range(self.number_of_batches):

                # wait for any worker to finish batch and
                # get the nabla_w, nabla_b for the master's layers
                status = MPI.Status()
                # Listen to ANY worker
                req = self.comm.Irecv([grad_recv_buf, MPI.DOUBLE], source=MPI.ANY_SOURCE, tag=TAG_GRADS)
                req.Wait(status) # Pass status object to populate it
                
                worker_rank = status.Get_source()

                # Unflatten the received gradients
                d_w, d_b = self._unflatten_params(grad_recv_buf, master_layers)
                nabla_w = d_w
                nabla_b = d_b

                # calculate new weights and biases (of layers in charge)
                for i, dw, db in zip(master_layers, nabla_w, nabla_b):
                    self.weights[i] = self.weights[i] - self.eta * dw
                    self.biases[i] = self.biases[i] - self.eta * db

                # send new values (of layers in charge)
                curr_w = [self.weights[i] for i in master_layers]
                curr_b = [self.biases[i] for i in master_layers]
                
                # Flatten current weights
                send_buf = self._flatten_params(master_layers, curr_w, curr_b)
                
                req = self.comm.Isend([send_buf, MPI.DOUBLE], dest=worker_rank, tag=TAG_PARAMS)
                req.Wait()

            self.print_progress(validation_data, epoch)

        # Gather weights and biases to send to main master
        final_w = [self.weights[l] for l in master_layers]
        final_b = [self.biases[l] for l in master_layers]
        final_send_buf = self._flatten_params(master_layers, final_w, final_b)

        if self.rank == 0:
            # Main master (rank 0) collects from everyone else to rebuild the full network
            for m in range(1, self.num_masters):
                # Calculate size for master m
                m_indices = self._get_layer_indices(m)
                m_size = self._calculate_buffer_size(m_indices)
                m_buf = np.empty(m_size, dtype=np.float64)
                
                req = self.comm.Irecv([m_buf, MPI.DOUBLE], source=m, tag=TAG_GATHER)
                req.Wait()
                
                recv_w, recv_b = self._unflatten_params(m_buf, m_indices)
                for i, w, b in zip(m_indices, recv_w, recv_b):
                    self.weights[i] = w
                    self.biases[i] = b
        else:
            # Other masters send their parts to main master
            req = self.comm.Isend([final_send_buf, MPI.DOUBLE], dest=0, tag=TAG_GATHER)
            req.Wait()