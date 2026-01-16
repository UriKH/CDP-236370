import numpy as np
import mpi4py.MPI as MPI


def allreduce(send, recv, comm, op):
    """ Naive all reduce implementation

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

    for i in range(size):
        if i == rank:
            pass
        else:
            comm.Send(send, dest=i)

    for i in range(size):
        if i == rank:
            pass
        else:
            temp = np.empty_like(send)
            comm.Recv(temp, source=i)
            recv = op(recv, temp)
