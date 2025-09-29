import numpy as np
from Preprocessing import one_hot_to_y


class EvaluationUnit:
    def __init__(self, y_p, y_a):
        self.y_predicted = y_p
        self.y_actual = y_a.T
        self.class_num = y_p.shape[0]

    def mean_square_error(self):
        return np.square(self.y_predicted - self.y_actual).sum() / self.y_predicted.shape[1]

    def accuracy(self):
        result = np.abs(self.y_predicted - self.y_actual).sum(axis=0)
        return len(result[result == 0]) / self.y_actual.shape[1]

    def confusing_matrix(self):
        y_p = one_hot_to_y(self.y_predicted)
        y_a = one_hot_to_y(self.y_actual)
        confmat = np.zeros((self.class_num, self.class_num))
        for i in range(len(y_p)):
            confmat[y_p[i], y_a[i]] += 1
        return confmat

    def p_and_r(self):
        confmat = self.confusing_matrix()
        p = confmat.diagonal() / confmat.sum(axis=1)
        r = confmat.diagonal() / confmat.sum(axis=0)
        return p, r


def mean_square_error(x, y):
    return np.power(x - y, 2).sum() / x.shape[1]


def accuracy(x, y):
    result = np.abs(x - y).sum(axis=0)
    return len(result[result == 0]) / y.shape[1]


def confusing_matrix(y_predicted, y_actual, class_num):
    y_p = one_hot_to_y(y_predicted)
    y_a = one_hot_to_y(y_actual)
    confmat = np.zeros((class_num, class_num))
    for i in range(len(y_p)):
        confmat[y_p[i], y_a[i]] += 1
    return confmat


if __name__ == '__main__':
    pass
