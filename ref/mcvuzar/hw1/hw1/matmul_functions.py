import numpy as np
from numba import njit, cuda, prange
import timeit


def matmul_transpose_trivial(X):
    rows, columns = np.shape(X)
    res = np.zeros((rows, rows)) # shape of XXT
    for i in range(rows):
        for j in range(rows):
            for k in range(columns):
                res[i,j]+= X[i,k] * X[j,k]
    return res


@njit(parallel=True)
def matmul_transpose_numba(X):
    rows, columns = np.shape(X)
    res = np.zeros((rows, rows)) # shape of XXT
    for i in prange(rows):
        for j in prange(rows):
            for k in prange(columns):
                res[i,j]+= X[i,k] * X[j,k]
    return res


def matmul_transpose_gpu(X):
    rows, _ = np.shape(X)
    dev_X = cuda.to_device(X)
    dev_C = cuda.device_array((rows, rows)) # create directly on gpu
    # Now start the kernel with 1 block and 1024 threads
    matmul_kernel[1, 1024](dev_X, dev_C) 
    C = dev_C.copy_to_host()
    return C


@cuda.jit
def matmul_kernel(A, C):
    rows, columns = A.shape
    pos = cuda.threadIdx.x # each thread starts from its threadIdx- 1..1024
    size = rows**2 # size of result matrix
    while pos < size:
        i = pos // rows # num columns of C = rows
        j = pos % rows # num columns of C = rows
        C[i,j] = 0
        #calc C[i, j]
        for k in range(columns): 
            C[i,j] += A[i,k] * A[j,k] # calc inner product of row i & row j in A
        pos += 1024 # update cells in 1024 jumps


def verify_solution():
    X = np.random.randn(784, 128)
    Xt = X.copy().transpose()

    if not np.allclose(matmul_transpose_trivial(X), np.matmul(X, Xt)):
        print('[-] matmul_transpose_trivial failed')
        exit(0)
    else:
        print('[+] matmul_transpose_trivial passed')

    if not np.allclose(matmul_transpose_numba(X), np.matmul(X, Xt)):
        print('[-] matmul_transpose_numba failed')
        exit(0)
    else:
        print('[+] matmul_transpose_numba passed')

    if not np.allclose(matmul_transpose_gpu(X), np.matmul(X, Xt)):
        print('[-] matmul_transpose_gpu failed')
        exit(0)
    else:
        print('[+] matmul_transpose_gpu passed')

    print('[+] All tests passed\n')


# this is the comparison function - keep it as it is, don't change X or Y.
def matmul_comparison():
    X = np.random.randn(784, 128)
    Xt = X.copy().transpose()

    def timer(f, functionParameters):
        return min(timeit.Timer(lambda: f(X) if functionParameters == 1 else f(X, Xt)).repeat(3, 100))

    # print('Python:', timer(matmul_transpose_trivial, 1)) we will not consider this since it takes infinite time :)
    print('Numpy:', timer(np.matmul, 2))
    print('Numba:', timer(matmul_transpose_numba, 1))
    print('CUDA:', timer(matmul_transpose_gpu, 1))


if __name__ == '__main__':
    verify_solution()
    matmul_comparison()
