import numpy as np


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

    np.copyto(recv, send)
    temp = np.empty_like(send)
    for i in range(comm.Get_size()):
        if i == comm.Get_rank():
            comm.Bcast(send, i)
        else:
            comm.Bcast(temp, i)
            np.copyto(recv, op(temp, recv))
