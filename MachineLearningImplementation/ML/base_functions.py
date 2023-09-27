import numpy as np


class Logistic:
    def __init__(self):
        self.a = None
        self.z = None

    def calculate(self, z):
        # a = np.copy(x)
        # a[a >= 0] = 1 / (1 + np.exp(-a[a >= 0]))
        # a[a < 0] = np.exp(x[a < 0]) / (1 + np.exp(x[a < 0]))
        # self.a = a
        self.z = z
        self.a = 1.0 / (1 + np.exp(-z))
        return self.a

    def derivative(self):
        return np.multiply(self.a, (1 - self.a))


class Tanh:
    def __init__(self):
        self.a = None
        self.z = None

    def calculate(self, z):
        a = np.exp(z)
        b = np.exp(-z)
        self.a = (a - b) / (a + b)
        self.z = z
        return self.a

    def derivative(self):
        return 1 - np.square(self.a)


class Linear:
    def __init__(self):
        self.a = None
        self.z = None

    def calculate(self, z):
        self.a = z
        self.z = z
        return self.a

    @staticmethod
    def derivative():
        return 1


class Relu:
    def __init__(self):
        self.a = None
        self.z = None

    def calculate(self, z):
        self.z = z
        z = np.maximum(z, 0)
        # z[z <= 0] = 0
        self.a = z
        return self.a

    def derivative(self):
        self.z[self.z > 0] = 1
        return self.z


class Softmax:
    def __init__(self):
        self.a = None
        self.z = None

    def calculate(self, z):
        self.z = z
        # m_example = np.exp(self.z).sum(axis=0)
        corrected_z = self.z - self.z.max(axis=0)
        self.a = np.exp(corrected_z) / np.exp(corrected_z).sum(axis=0)
        return self.a

    def derivative(self):
        return np.multiply(self.a, (1 - self.a))


ACTIVATIONS = {'softmax': Softmax, 'logistic': Logistic, 'linear': Linear, 'relu': Relu,
               'tanh': Tanh}


def softmax_cost(x, y):
    return (y * np.log(x + 1e-9)).sum() / -x.shape[1]


def logistic_cost(x, y):
    return (y * np.log(x + 1e-9) + (1 - y) * np.log(1 - x + 1e-9)).sum() / -x.shape[1]


def linear_cost(x, y):
    return np.square(x - y).sum() / 2 * y.shape[1]


COSTFUNCTIONS = {'softmax': softmax_cost, 'logistic': logistic_cost, 'linear': linear_cost}

if __name__ == '__main__':
    pass
