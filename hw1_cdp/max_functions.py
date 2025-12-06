import numpy as np
from numba import cuda, njit, prange, float32
import timeit
import math


def max_cpu(A, B):
    """
     Returns
     -------
     np.array
         element-wise maximum between A and B
     """
    C = np.zeros_like(A)
    
    for i in range(A.shape[0]):                 # iterate over rows
        for j in range(A.shape[1]):             # iterate over columns
            C[i, j] = max(A[i, j], B[i, j])     # element-wise maximum
    return C


@njit(parallel=True)
def max_numba(A, B):
    """
     Returns
     -------
     np.array
         element-wise maximum between A and B
     """
    out = np.empty_like(A)          # create output array

    for i in prange(A.shape[0]):        # iterate over rows in parallel
        for j in prange(A.shape[1]):    # iterate over columns in parallel
            # element-wise maximum
            out[i, j] = A[i, j] if A[i, j] > B[i, j] else B[i, j]
    return out


def max_gpu(A, B):
    """
     Returns
     -------
     np.array
         element-wise maximum between A and B
     """
    # Allocate device memory and copy input arrays to device
    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_C = cuda.device_array_like(A)

    # Define dimensions
    blocks, threads = (1000, 1000)
    
    # Launch kernel
    max_kernel[blocks, threads](d_A, d_B, d_C)
    return d_C.copy_to_host() # Copy result back to host


@cuda.jit
def max_kernel(A, B, C):
    """
        Find the maximum value in values and store in result[0]
        """
    # Calculate global thread index
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    # check glocal index is in bounds
    if idx < A.shape[0] * A.shape[1]:
        # Compute row and column from linear index
        row = idx // A.shape[1]
        col = idx % A.shape[1]

        # Element-wise maximum
        C[row, col] = max(A[row, col], B[row, col])


def verify_solution():
    A = np.random.randint(0, 256, (1000, 1000))
    B = np.random.randint(0, 256, (1000, 1000))

    if not np.all(max_cpu(A, B) == np.maximum(A, B)):
        print('[-] max_cpu failed')
        exit(0)
    else:
        print('[+] max_cpu passed')

    if not np.all(max_numba(A, B) == np.maximum(A, B)):
        print('[-] max_numba failed')
        exit(0)
    else:
        print('[+] max_numba passed')

    if not np.all(max_gpu(A, B) == np.maximum(A, B)):
        print('[-] max_gpu failed')
        exit(0)
    else:
        print('[+] max_gpu passed')

    print('[+] All tests passed\n')


# this is the comparison function - keep it as it is.
def max_comparison():
    A = np.random.randint(0, 256, (1000, 1000))
    B = np.random.randint(0, 256, (1000, 1000))

    def timer(f):
        return min(timeit.Timer(lambda: f(A, B)).repeat(3, 20))

    print('[*] CPU:', timer(max_cpu))
    print('[*] Numba:', timer(max_numba))
    print('[*] CUDA:', timer(max_gpu))


if __name__ == '__main__':
    verify_solution()
    max_comparison()
