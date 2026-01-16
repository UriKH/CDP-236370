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

    np.copyto(recv, send)

    left = (rank - 1 + size) % size
    right = (rank + 1) % size

    def get_chunk_slices(chunk_index, total_len, comm_size):
        base_size = total_len // comm_size
        remainder = total_len % comm_size
        
        if chunk_index < remainder:
            start = chunk_index * (base_size + 1)
            end = start + base_size + 1
        else:
            start = chunk_index * base_size + remainder
            end = start + base_size
        return start, end

    # Reduce-scatter phase
    for i in range(size - 1):
        s_start, s_end = get_chunk_slices(i, send.shape[0], size)
        r_start, r_end = get_chunk_slices((i - 1 + size) % size, send.shape[0], size)
        
        temp = np.empty(r_end - r_start, dtype=send.dtype)

        comm.Sendrecv(
            sendobj = recv[s_start:s_end], dest = right,
            recvobj = temp, source = left
        )

    # All-gather phase
    for i in range(size):
        s_start, s_end = get_chunk_slices(i, send.shape[0], size)
        r_start, r_end = get_chunk_slices((i - 1 + size) % size, send.shape[0], size)

        comm.Sendrecv(
            sendobj = recv[s_start:s_end], dest = right,
            recvobj = recv[s_start:s_end], source = left
        )
