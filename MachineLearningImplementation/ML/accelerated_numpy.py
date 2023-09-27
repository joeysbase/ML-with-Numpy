import numpy as np
import numba as nb


@nb.jit(nopython=True)
def acc_dot(a, b):
    return np.dot(a, b)


@nb.jit(nopython=True)
def acc_multiply(a, b):
    return np.multiply(a, b)


@nb.jit(nopython=True)
def acc_log(a, b):
    return np.log(a, b)


@nb.jit(nopython=True)
def acc_exp(a):
    return np.exp(a)


@nb.jit(nopython=True)
def acc_power(a, b):
    return np.power(a, b)


if __name__ == '__main__':
    pass
