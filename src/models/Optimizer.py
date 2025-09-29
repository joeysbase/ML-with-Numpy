import numpy as np


def sgd_optimizer(self, x, y, yaxis, plot):
    for i in range(self.max_itr):
        if self.shuffle and self.batch_size != self.m_sample:
            z = np.hstack([x, y])
            np.random.shuffle(z)
            x, y = np.hsplit(z, [self.n_sample])
        for j in range(0, self.m_sample, self.batch_size):
            self._cost_fun_derivative(
                x[j : j + self.batch_size, :].T, y[j : j + self.batch_size, :].T
            )
            for k in range(len(self.paras)):
                temp_theta = self.paras[k] - self.learning_rate_init * self.deriva[k]
                self.paras[k] = temp_theta
                # !!! do remember to update the real theta in Affine!!!
                self.affines[k].theta = temp_theta
            if plot:
                z = int(
                    i * np.ceil(self.m_sample / self.batch_size) + j / self.batch_size
                )
                yaxis[z] = self._cost_fun(
                    x[j : j + self.batch_size, :].T, y[j : j + self.batch_size, :].T
                )


def momentum_optimizer(self, x, y, yaxis, plot):
    v = []
    for p in self.paras:
        v.append(np.zeros_like(p))

    for i in range(self.max_itr):
        if self.shuffle and self.batch_size != self.m_sample:
            z = np.hstack([x, y])
            np.random.shuffle(z)
            x, y = np.hsplit(z, [self.n_sample])
        for j in range(0, self.m_sample, self.batch_size):
            self._cost_fun_derivative(
                x[j : j + self.batch_size, :].T, y[j : j + self.batch_size, :].T
            )
            for k in range(len(self.paras)):
                v[k] = self.momentum * v[k] + (1 - self.momentum) * self.deriva[k]
                temp_theta = self.paras[k] - self.learning_rate_init * v[k]
                self.paras[k] = temp_theta
                self.affines[k].theta = temp_theta
            if plot:
                z = int(
                    i * np.ceil(self.m_sample / self.batch_size) + j / self.batch_size
                )
                yaxis[z] = self._cost_fun(
                    x[j : j + self.batch_size, :].T, y[j : j + self.batch_size, :].T
                )


def rmsprop_optimizer(self, x, y, yaxis, plot):
    s = []
    for p in self.paras:
        s.append(np.zeros_like(p))

    for i in range(self.max_itr):
        if self.shuffle and self.batch_size != self.m_sample:
            z = np.hstack([x, y])
            np.random.shuffle(z)
            x, y = np.hsplit(z, [self.n_sample])
        for j in range(0, self.m_sample, self.batch_size):
            self._cost_fun_derivative(
                x[j : j + self.batch_size, :].T, y[j : j + self.batch_size, :].T
            )
            for k in range(len(self.paras)):
                s[k] = self.momentum * s[k] + (1 - self.momentum) * np.square(
                    self.deriva[k]
                )
                temp_theta = self.paras[k] - self.learning_rate_init * self.deriva[
                    k
                ] / (np.sqrt(s[k]) + 1e-8)
                self.paras[k] = temp_theta
                self.affines[k].theta = temp_theta
            if plot:
                z = int(
                    i * np.ceil(self.m_sample / self.batch_size) + j / self.batch_size
                )
                yaxis[z] = self._cost_fun(
                    x[j : j + self.batch_size, :].T, y[j : j + self.batch_size, :].T
                )


def adam_optimizer(self, x, y, yaxis, plot):
    s = [np.zeros_like(p) for p in self.paras]
    v = [np.zeros_like(p) for p in self.paras]

    for i in range(self.max_itr):
        if self.shuffle and self.batch_size != self.m_sample:
            z = np.hstack([x, y])
            np.random.shuffle(z)
            x, y = np.hsplit(z, [self.n_sample])
        for j in range(0, self.m_sample, self.batch_size):
            ep = int(i * np.ceil(self.m_sample / self.batch_size) + j / self.batch_size)
            self._cost_fun_derivative(
                x[j : j + self.batch_size, :].T, y[j : j + self.batch_size, :].T
            )
            for k in range(len(self.paras)):
                v[k] = self.beta_1 * v[k] + (1 - self.beta_1) * self.deriva[k]
                s[k] = self.beta_2 * s[k] + (1 - self.beta_2) * np.square(
                    self.deriva[k]
                )
                v_c = v[k] / (1 - np.power(self.beta_1, ep + 1))
                s_c = s[k] / (1 - np.power(self.beta_2, ep + 1))
                temp_theta = self.paras[k] - self.learning_rate_init * v_c / (
                    np.sqrt(s_c) + 1e-8
                )
                self.paras[k] = temp_theta
                self.affines[k].theta = temp_theta
            if plot:
                yaxis[ep] = self._cost_fun(
                    x[j : j + self.batch_size, :].T, y[j : j + self.batch_size, :].T
                )


OPTIMIZERS = {
    "sgd": sgd_optimizer,
    "momentum": momentum_optimizer,
    "rmsprop": rmsprop_optimizer,
    "adam": adam_optimizer,
}

if __name__ == "__main__":
    pass
