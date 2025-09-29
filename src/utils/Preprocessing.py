import numpy as np
import pandas as pd


class ProcessingUnit:
    def __init__(self, d, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
        else:
            np.random.seed(np.random.randint(100))
        np.random.shuffle(d)

        self.d = d
        self.x, self.y = np.hsplit(d, [d.shape[1] - 1])
        if np.issubdtype(self.x.dtype, str):
            self.x = self.x.astype('float64')

        # scaled x will be stored by this, being avaliable only when perfoming feature scaling
        self.scaled_x = self.x
        self.onehot_y = self.y

        # avaliable only whe performing one-hot transformation
        self.y_mapping = {}

        # datasets below are avaliable only when performing data spliting
        self.train = None
        self.cv = None
        self.test = None
        self.splited_data = None
        self.split_flag = 0

        # counting how many labels of y there are
        self.y_label_stats = {}
        for i in range(self.y.shape[0]):
            if self.y_label_stats.get(self.y[i, 0]) is None:
                self.y_label_stats[self.y[i, 0]] = 1
            else:
                self.y_label_stats[self.y[i, 0]] += 1

        # labels of y
        self.y_labels = list(self.y_label_stats.keys())

    def y_to_onehot(self):
        m = self.d.shape[0]
        n = len(self.y_labels)
        onehot = np.zeros((m, n))
        acc = 0
        for i in self.y_labels:
            self.y_mapping[i] = acc
            acc += 1
        for i in range(self.y.shape[0]):
            c = self.y_mapping[self.y[i, 0]]
            onehot[i, c] = 1
        self.onehot_y = onehot
        if np.issubdtype(self.y.dtype, str):
            for i in range(self.y.shape[0]):
                self.y[i, 0] = self.y_mapping[self.y[i, 0]]
            self.y.astype('int8')

    def onehot_to_y(self, onehot_y, y_mapping):
        indices = np.argwhere(onehot_y == 1)
        sorted_indices = np.argsort(indices[:, 1])
        num_y = indices[sorted_indices][:, 0]
        for i in range(num_y.shape[0]):
            num_y[i, 0] = y_mapping[num_y[i, 0]]
        self.y = num_y

    def split_dataset(self, ratio=(0.6, 0.2, 0.2)):
        split_xy = []
        s = 0
        for i in ratio:
            p = int(np.round(self.d.shape[0] * i))
            split_xy.append([self.scaled_x[s:s + p, :], self.onehot_y[s:s + p, :]])
            s = p
        self.train, self.cv, self.test = split_xy
        self.splited_data = [self.train, self.cv, self.test]
        self.split_flag = 1

    def pca(self, n_component):
        if self.split_flag != 1:
            return None
        self.z_score()
        covmat = np.dot(self.train[0].T, self.train[0]) / self.train[0].shape[0]
        u, s, v = np.linalg.svd(covmat)
        lor = np.zeros(len(s))
        for i in range(len(s)):
            lor[i] = s[:i].sum() / s.sum()
        if n_component >= 1:
            for d_set in self.splited_data:
                d_set[0] = np.dot(u[:, :n_component].T, d_set[0].T).T
        elif 0 < n_component <= 1:
            for i in range(len(s)):
                if lor[i] >= n_component:
                    for d_set in self.splited_data:
                        d_set[0] = np.dot(u[:, :i].T, d_set[0].T).T
                    break
        # print(lor)
        # print(s.shape)
        # print(self.train[0].shape)

    # def z_score(self):
    #     self.x = (self.x - self.x.mean(axis=0)) / self.x.std(axis=0)

    def mean_norm(self):
        self.scaled_x = (self.x - self.x.mean(axis=0)) / (self.x.max(axis=0) - self.x.min(axis=0))

    def z_score(self):
        if self.split_flag:
            m = self.train[0].mean(axis=0)
            s = self.train[0].std(axis=0)
            for d_set in self.splited_data:
                d_set[0] = (d_set[0] - m) / s
        else:
            self.scaled_x = (self.x - self.x.mean(axis=0)) / self.x.std(axis=0)

    def get_raw_data(self):
        return self.d

    def get_raw_x(self):
        return self.x

    def get_raw_y(self):
        return self.y


def z_score(x):
    return (x - x.mean(axis=0)) / x.std(axis=0)


def mean_norm(x):
    return (x - x.mean(axis=0)) / (x.max(axis=0) - x.min(axis=0))


def y_to_one_hot(y, y_cate):
    """
    transforming multi-classes target into one-hot version
    :param y: target, two dimensions numpy array, where second dimension equal to 1, y.shape=(m_sample,1)
    :param y_cate: a list, which is listed with all target's category, like ['a','b','c']
    :return: two diemnsions numpy array, one-hot version of input y
    """
    labels = {}
    m = y.shape[0]
    n = len(y_cate)
    onehoty = np.zeros((m, n))
    acc = 0
    for i in y_cate:
        labels[i] = acc
        acc += 1
    for i in range(y.shape[0]):
        c = labels[y[i, 0]]
        onehoty[i, c] = 1
    return onehoty


def one_hot_to_y(onehot_y):
    indices = np.argwhere(onehot_y == 1)
    sorted_indices = np.argsort(indices[:, 1])
    return indices[sorted_indices][:, 0]


def split_dataset(d, y_cate, ratio=(0.6, 0.2, 0.2), zscore=True):
    """
    automatically spliting dataset into train, cv, and test
    :param d: two dimension numpy array, dataset
    :param y_cate: categories of y, one dimension list, like ['apple','orange','grape']
    :param ratio: tuple, spliting ratio of each subset
    :param zscore: applying zscore to dataset
    :return: Xtrain, Ytrain, Xcv, Ycv, Xtest, Ytest
    """
    split_xy = []
    np.random.shuffle(d)
    x, y = np.hsplit(d, [d.shape[1] - 1])
    if zscore:
        x = z_score(x)
    y = y_to_one_hot(y, y_cate)
    s = 0
    for i in ratio:
        p = int(np.round(d.shape[0] * i))
        split_xy.append((x[s:s + p, :], y[s:s + p, :]))
        s = p
    return split_xy


if __name__ == '__main__':
    t = pd.read_csv('../testfile/x.csv')
    t = t.to_numpy(dtype=str)
    # train, cv, test = split_dataset(t, y_cate=[0, 1, 2])
    # print(cv[1])
    pu = ProcessingUnit(t)
    # pu.z_score()
    pu.y_to_onehot()
    pu.split_dataset()
    # pu.pca(n_component=0.98)
    xcv, ycv = pu.cv
    # j = {}
    # for i in range(ycv.shape[0]):
    #     if k.get(ycv[i, 0]) is None:
    #         k[ycv[i, 0]] = 1
    #     else:
    #         k[ycv[i, 0]] += 1
    print(xcv)
    # print(pu.y_label_stats)
    # print(pu.y_labels)
    # print(pu.y_mapping)
