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
    
    np.copyto(recv, send)

    reqs = []

    for i in range(size):
        if i != rank:
            req = comm.Isend(send, dest=i)
            reqs.append(req)

    temp = np.empty_like(send)
    
    for i in range(size):
        if i != rank:
            comm.Recv(temp, source=MPI.ANY_SOURCE) 
            recv[:] = op(recv, temp)
            
    MPI.Request.Waitall(reqs)
