import numpy as np


def add_bias_alt(x):
    temp = np.ones((x.shape[0], x.shape[1] + 1))
    temp[:, 1:x.shape[1] + 1] = x
    return temp


def t(itr, a=0.01):
    x = 0.05
    y = 0.1
    r = 0
    for i in range(itr):
        z = 2 * x * y
        u = 2 * y + np.power(x, 2)
        c = x - a * 10 * z
        d = y - a * u
        x = c
        y = d
        r = y * np.power(x, 2) + np.power(y, 2)
    return x, y, r


if __name__ == '__main__':
    pass
