import numpy as np
import matplotlib.pyplot as plt
import Optimizer as om
import pandas as pd
from utils.BaseFunctions import COSTFUNCTIONS
from utils.Preprocessing import ProcessingUnit
from utils.ParamInitializer import add_bias


class LinearRegression:
    """
    let's say we have M exeamples and N features, and therefore theta is a 1 by N+1 matrix, x is an M by N+1 matrix,
    and y is an M by 1 matrix
    """

    def __init__(self, feature_num):
        self.theta = np.random.rand(1, feature_num + 1)
        self.deriva = None
        self.x = None
        self.y = None

    def model(self, x):
        # 1 by m_sample matrix
        return np.dot(self.theta, x.T)

    def cost_fun(self, x, y):
        hx = self.model(x)
        return COSTFUNCTIONS.linear_cost(hx, y.T)

    def cost_fun_derivative(self, x, y):
        # a = self.model(x) - y
        # b = x[:, j:j + 1]
        # c = x.shape[0]
        # d = (a * b).sum() / c
        # return d
        m = x.shape[0]
        self.deriva = np.dot((self.model(x) - y.T), x) / m

    def train(self, x, y, itr=500, alpha=0.01):
        self.x = add_bias(x)
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

    def predict(self, x):
        x = add_bias(x)
        return self.model(x)


if __name__ == '__main__':
    t = pd.read_csv('../testfile/gpa.csv')
    t = t.to_numpy(dtype=float)
    pu = ProcessingUnit(t)
    # pu.z_score()
    # pu.y_to_onehot()
    pu.split_dataset()
    # pu.pca(n_component=0.97)
    xtrain, ytrain = pu.train
    xcv, ycv = pu.cv
    xtest, ytest = pu.test

    # lr = LinearRegression(xtrain.shape[1])
    # lr.train(xtrain, ytrain, itr=200)
    #
    # y_p_train = lr.predict(xtrain)
    # y_p_cv = lr.predict(xcv)
    # y_p_test = lr.predict(xtest)
    #
    # eu1 = EvaluationUnit(y_p_train, ytrain)
    # eu2 = EvaluationUnit(y_p_cv, ycv)
    # eu3 = EvaluationUnit(y_p_test, ytest)
    #
    # print(f'mse -> {eu1.mean_square_error()}')
    # print(f'mse -> {eu2.mean_square_error()}')
    # print(f'mse -> {eu3.mean_square_error()}')
    # print(lr.cost_fun(xtest, ytest))
    plt.scatter(xtrain, ytrain)
    plt.show()
