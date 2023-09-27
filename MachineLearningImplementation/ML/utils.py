import numpy as np


def xavier_init(n_in, n_out, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    else:
        np.random.seed(np.random.randint(100))
    return np.random.randn(n_out, n_in + 1) * np.sqrt(2 / (n_in + n_out))


def he_init(n_in, n_out, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    else:
        np.random.seed(np.random.randint(100))
    return np.random.randn(n_out, n_in + 1) * np.sqrt(2 / n_in)


def random_init(n_in, n_out, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    else:
        np.random.seed(np.random.randint(100))
    return np.random.randn(n_out, n_in + 1) * 0.01


def add_bias(x, axis=1):
    """
    adding bias 1 to input data
    :param x: data, two dimensions numpy array
    :param axis: 0 for vertical, 1 for horizontal
    :return: data with bias input
    """
    if axis == 1:
        one_vec = np.ones((x.shape[0], 1))
        return np.hstack([one_vec, x])
    else:
        one_vec = np.ones((1, x.shape[1]))
        return np.vstack([one_vec, x])


# def a():
#     np.random.seed(1)
#     return np.random.randint(100)


INITIALIZER = {'xavier': xavier_init, 'he': he_init, 'random': random_init}

if __name__ == '__main__':
    pass
