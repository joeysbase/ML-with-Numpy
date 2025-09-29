import numpy as np


class Kmeans:
    def __init__(self, K):
        self.K = K
        self.mu = None
        self.c = None

    def model(self, x):
        return np.where(np.power((x - self.mu).sum(axis=1), 2).min())

    def cost_fun(self, x):
        pass

    def train(self, x, itr=200):
        np.random.shuffle(x)
        self.mu = x[0:self.K, :]
        self.c = np.zeros((x.shape[0], 1))
        for i in range(itr):
            v = self.mu
            for j in range(self.mu.shape[0]):
                if j == 0:
                    self.c = np.power((x - self.mu).sum(axis=1), 2)[:, np.newaxis]
                else:
                    temp_c = np.power((x - self.mu).sum(axis=1), 2)[:, np.newaxis]
                    ind = temp_c < self.c
                    self.c[ind] = temp_c[ind]
            temp_x = np.hstack((x, self.c))
            temp_x = temp_x[np.argsort(temp_x[:, temp_x.shape[1] - 1])]
            si = np.unique(temp_x, return_index=True)
            t = np.vsplit(temp_x[:, :temp_x.shape[1]], si[1])
            for j in range(self.mu.shape[0]):
                self.mu[j] = t[j].sum(axis=0) / t[j].shape[0]
            if (self.mu - v).sum() == 0:
                break


if __name__ == '__main__':
    pass
