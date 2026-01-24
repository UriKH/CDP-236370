
import numpy as np
from mpi4py import MPI

def ringallreduce(send, recv, comm, op):
    """ 
    Ring all reduce implementation that handles any array shape safely.
    """
    size = comm.Get_size()
    rank = comm.Get_rank()
    np.copyto(recv, send)
    
    recv_flat = recv.reshape(-1)
    total_len = recv_flat.size

    left = (rank - 1 + size) % size
    right = (rank + 1) % size

    # Helper to calculate start/end indices for a flat array
    def get_chunk_slices(chunk_index, total_len, comm_size):
        base_size = total_len // comm_size
        remainder = total_len % comm_size
        
        if chunk_index < remainder:
            start = chunk_index * (base_size + 1)
            end = start + base_size + 1
        else:
            start = (remainder * (base_size + 1)) + ((chunk_index - remainder) * base_size)
            end = start + base_size
        return start, end

    # Reduce-Scatter Phase
    for i in range(size - 1):
        # compute send and recv chunk indices
        send_idx = (rank - i + size) % size
        recv_idx = (rank - i - 1 + size) % size
        s_start, s_end = get_chunk_slices(send_idx, total_len, size)
        r_start, r_end = get_chunk_slices(recv_idx, total_len, size)

        # communicate and reduce
        temp = np.empty(r_end - r_start, dtype=send.dtype)
        comm.Sendrecv(
            sendbuf=recv_flat[s_start:s_end], dest=right,
            recvbuf=temp, source=left
        )
        # perform the operation
        recv_flat[r_start:r_end] = op(recv_flat[r_start:r_end], temp)

    # All-Gather Phase
    for i in range(size - 1):
        # compute send and recv chunk indices
        send_idx = (rank - i + 1 + size) % size
        recv_idx = (rank - i + size) % size
        s_start, s_end = get_chunk_slices(send_idx, total_len, size)
        r_start, r_end = get_chunk_slices(recv_idx, total_len, size)

        # communicate and gather
        comm.Sendrecv(
            sendbuf=recv_flat[s_start:s_end], dest=right,
            recvbuf=recv_flat[r_start:r_end], source=left
        )
