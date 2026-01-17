import numpy as np


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

    rank = comm.Get_rank()
    num_of_processes = comm.Get_size()

    cell_size = send.size // num_of_processes
    
    cells = [(cell_size * r, cell_size * (r + 1)) for r in range(num_of_processes - 1)]
    cells.append((cell_size * (num_of_processes - 1), send.size))
    
    np.copyto(recv, send)
    current = rank
    for _ in range(num_of_processes - 1):
        start, end = cells[current]
        req = comm.Isend(recv[start:end], dest = (rank + 1) % num_of_processes, tag = 0)
        current -= 1
        start, end = cells[current]
        comm.Recv(recv[start:end], source = (rank - 1 + num_of_processes) % num_of_processes, tag = 0)
        np.copyto(recv[start:end], op(recv[start:end], send[start:end]))
        req.Wait()

    current = rank + 1
    for _ in range(num_of_processes - 1):
        start, end = cells[current]
        req = comm.Isend(recv[start:end], dest = (rank + 1) % num_of_processes, tag = 0)
        current -= 1
        start, end = cells[current]
        comm.Recv(recv[start:end], source = (rank - 1 + num_of_processes) % num_of_processes, tag = 0)
        req.Wait()
    
    return recv