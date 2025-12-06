import numpy as np
from numba import njit, cuda, prange
import timeit
import math

def matmul_transpose_trivial(X):
    R = np.zeros((X.shape[0], X.shape[0]))

    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            sum = 0
            for k in range(X.shape[1]):
                sum += X[i, k] * X[j, k]
            R[i, j] = sum

    return R


@njit(parallel=True)
def matmul_transpose_numba(X):
    R = np.zeros((X.shape[0], X.shape[0]))

    for i in prange(X.shape[0]):
        for j in range(X.shape[0]):
            sum = 0
            for k in range(X.shape[1]):
                sum += X[i, k] * X[j, k]
            R[i, j] = sum

    return R


def matmul_transpose_gpu(X):

    n = X.shape[0]

    C = np.zeros((n, n))

    d_X = cuda.to_device(X)
    d_C = cuda.to_device(C)

    threadsperblock = (16, 16)

    blockspergrid_x = math.ceil(n / threadsperblock[0])
    blockspergrid_y = math.ceil(n / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    matmul_kernel[blockspergrid](d_X, d_C)

    R = d_C.copy_to_host()

    return R


@cuda.jit
def matmul_kernel(A, C):
    i, j = cuda.grid(2)

    if i < C.shape[0] and j < C.shape[1]:
        sum = 0
        for k in range(A.shape[1]):
            sum += A[i, k] * A[j, k]
        C[i, j] = sum


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
