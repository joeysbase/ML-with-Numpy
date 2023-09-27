import numpy as np
import matplotlib.pyplot as plt
import ML.optimizer as om
import tools


class LogisticRegression:
    def __init__(self, feature_num):
        self.theta = np.random.rand(1, feature_num + 1)
        self.deriva = None
        self.x = None
        self.y = None

    def model(self, x):
        # 1 by m_sample matrix
        return 1 / (1 + np.exp(np.dot(self.theta, x.T)))

    def cost_fun(self, x, y):
        return (np.dot(np.log(self.model(x)), y) + np.dot(np.log(1 - self.model(x)), 1 - y)) / -x.shape[0]

    def cost_fun_derivative(self, x, y):
        m = x.shape[0]
        self.deriva = np.dot((y.T - self.model(x)), x) / m

    def train(self, x, y, itr=500, alpha=0.01):
        self.x = tools.add_bias(x)
        self.y = y
        xaixs = np.zeros(itr)
        yaixs = np.zeros(itr)

        for i in range(itr):
            self.cost_fun_derivative(self.x, self.y)
            self.theta = om.gradient_descend(self.theta, self.deriva, alpha=alpha)
            xaixs[i] = i
            yaixs[i] = self.cost_fun(self.x, self.y)

        plt.xlabel('itration')
        plt.ylabel('value')
        plt.plot(xaixs, yaixs)
        plt.show()

    def test(self, x, y):
        y = y.T
        result = np.abs(self.predict(x) - y).sum(axis=0)
        result = result[result == 0]
        return len(result) / y.shape[1]

    def predict(self, x, display_raw=False):
        x = tools.add_bias(x)
        result = self.model(x)
        if not display_raw:
            result[result >= 0.5] = 1
            result[result < 0.5] = 0
        return result


if __name__ == '__main__':
    t = np.loadtxt('../testfile/heart.csv', delimiter=',')

    x_, y_ = np.hsplit(t, [t.shape[1] - 1])
    x_ = tools.z_score(x_)

    x_test, x_train = np.vsplit(x_, [100])
    y_test, y_train = np.vsplit(y_, [100])

    lr = LogisticRegression(x_.shape[1])
    lr.train(x_train, y_train, itr=450, alpha=0.09)
    print(lr.test(x_test, y_test))
