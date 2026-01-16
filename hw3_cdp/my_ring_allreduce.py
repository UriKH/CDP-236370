import numpy as np
import mpi4py.MPI as MPI


def ringallreduce(send, recv, comm, op):
    """ ring all reduce implementation
    You need to use the algorithm shown in the lecture.

    Parameters
    ----------
    send : numpy array
        the array of the current process
    recv : numpy array
        an array to store the result of the reduction. Of same shape as send
    comm : MPI.Comm
    op : associative commutative binary operator
    """
    size = comm.Get_size()
    rank = comm.Get_rank()
    blockSize = send.shape[0] // size

    for _ in range(2):
        for i in range(size):
            end = (i + 1) * blockSize
            if i == size - 1:
                end = send.shape[0]
            comm.Send(send[i * blockSize: end], dest=(rank + 1) % size)
            temp = np.empty_like(send[i * blockSize: end])
            comm.Recv(temp, source=(rank - 1) % size)
            endR = i * blockSize
            if i == 0:
                endR = send.shape[0]
            recv[((i - 1 + size) % size) * blockSize: endR] = op(recv[((i - 1 + size) % size) * blockSize: endR], temp)
