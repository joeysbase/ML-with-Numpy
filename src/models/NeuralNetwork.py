import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from utils.BaseFunctions import COSTFUNCTIONS, ACTIVATIONS, Linear
from utils.ParamInitializer import INITIALIZER, add_bias
from utils.Preprocessing import ProcessingUnit
from utils.Evaluation import EvaluationUnit, accuracy
from sklearn.metrics import accuracy_score


class Layer:
    """
    model of layer in neuralnetwork, storing the number of units, activation function, input, output, bp error,
    and derivative of activation.

    Parameters
    ----------
    activ_fun : {'logistic','linear','softmax','relu','tanh'}
        activation function of this layer

    s : integer
        number of units in this layer

    Attributes
    ----------
    z : ndarray
        input value from previous forward prop

    a : ndarray
        output value of activation function

    delta : ndarray
        back prop error of this layer

    gz : ndarray
        derivative of activation function
    """

    def __init__(self, s, activation_fun):
        self.activ_fun = activation_fun
        self.s = s

        self.z = None
        self.a = None
        self.delta = None
        self.gz = None

    def forward(self, z):
        """calculate the output value of input z using given activation function self.activ_fun.

        Parameters
        ----------
        z : ndarray, self.s by 1 matrix

        Returns
        -------
        output of input z
        """
        self.z = z
        self.a = self.activ_fun.calculate(z)
        return self.a

    def backward(self):
        """calculate the derivative of input z.

        Returns
        -------
        derivative of input z
        """
        self.gz = self.activ_fun.derivative()
        return self.gz


class Affine:
    """
    Model that consist two layers and its corresponding weight.
                            affine
                input layer         output layer
                    O                   O ->unit of layer
                    O       weights     O
                    O                   O

    Parameters
    ----------
    in_layer : instance of class Layer
        The input layer of affine.

    out_layer : instance of class Layer
        The output layer of affine.

    para_init : {'he','xavier','random'}
        The method that is administrated to generate initial weights.

    random_state : integer, default=None
        If given, the generated random weights will be identical at each running time.

    Attributes
    ----------
    theta : ndarray, shape determined by attribute s of input and output layer
        Initial weights.
    """

    def __init__(self, in_layer, out_layer, para_init, random_state=None):
        self.in_layer = in_layer
        self.out_layer = out_layer
        self.para_init = para_init
        self.initializer = INITIALIZER[para_init]

        # randomize parameters
        self.theta = self.initializer(
            self.in_layer.s, self.out_layer.s, random_state=random_state
        )
        self.m = self.theta.shape[1]

    def forward(self):
        z_out = np.dot(self.theta, add_bias(self.in_layer.a, axis=0))
        return self.out_layer.forward(z_out)

    def backward(self):
        self.in_layer.backward()
        self.in_layer.delta = np.multiply(
            np.dot(self.theta[:, 1 : self.m].T, self.out_layer.delta), self.in_layer.gz
        )

    def set_outlayer_delta(self, delta):
        """set back prop errors to the output layer

        Parameters
        ----------
        delta : ndarray, out_layer.s by 1 matrix
            back prop errors of output layer
        """
        self.out_layer.delta = delta

    def set_inlayer_z(self, z_in):
        self.in_layer.forward(z_in)


class NeuralNetwork:
    """
    Model of neural network classifier

    Parameters
    ----------
    hidden_layers : tuple, length=number of hidden layers
        The ith element of the tuple represent the number
        of unit in the ith hidden layer.

    batch_size : integer, default='auto'
        Size of minibatches for stochastic optimizers.
        if set to 'auto', batch_size=min(200,n_sample)

    para_init : {'he','xavier','random'}, default='he'
        The method that is administrated to generate initial weights.

    activation : {'identity', 'logistic', 'tanh', 'relu'}, default='relu'
        Activation function for each hidden layer.

    optimizer : {'sgd','rmsprop','adam'}, default='adam'
        The method that will be administrated to optimize weights.

    learning_rate : {'constant','expdecay'}, default='constant'
        Learning rate schedule

        - 'constant' is a constant learning rate given by
          'learning_rate_init'.

        - 'expdecay' exponentially decreases the learning rate at each
           epoch 't', the formula is shown as following.
           'current learning rate'='learning_rate_init' * pow('decay_rate', 't')

    lambd : float, default=1e-3
        L2 regularization parameter

    learning_rate_init : float, default=1e-3
        The initial learning rate for optimizer

    decay_rate : float, default=0.95
        The base for expdecay learning rate schedule.

    max_itr : integer, default=200
        Maximum number of iterations. The optimizer iterates until convergence
        (determined by 'tol') or this number of iterations.

    momentum : float, default=0.9
        Momentum for gradient descent update. Should be between 0 and 1. Only
        used when optimizer='sgd'.

    beta_1 : float, default=0.9
        Exponential decay rate for estimates of first moment vector in adam,
        should be in [0, 1). Only used when optimizer='adam'.

    beta_2 : float, default=0.999
        Exponential decay rate for estimates of second moment vector in adam,
        should be in [0, 1). Only used when optimizer='adam'.

    n_iter_not_changed : int, default=10
        Maximum number of epochs to not meet ``tol`` improvement.

    tol : float, default=1e-4
        Tolerance for the optimization. When the loss or score is not improving
        by at least ``tol`` for ``n_iter_no_change`` consecutive iterations,
        unless ``learning_rate`` is set to 'adaptive', convergence is
        considered to be reached and training stops.

    verbose : bool, default=False
        Whether to print progress messages to stdout.

    shuffle : bool, default=True
        Whether to shuffle samples in each iteration.
    """

    def __init__(
        self,
        hidden_layers,
        batch_size="auto",
        para_init="he",
        activation="relu",
        output_activation="softmax",
        optimizer="adam",
        learning_rate="constant",
        lambd=1e-3,
        learning_rate_init=1e-3,
        decay_rate=0.95,
        decay_step=1,
        max_itr=200,
        momentum=0.9,
        beta_1=0.9,
        beta_2=0.999,
        tol=1e-4,
        n_iter_not_changed=10,
        verbose=False,
        shuffle=True,
        random_state=None,
    ):

        self.hidden_layers = hidden_layers
        self.para_init = para_init
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.lambd = lambd
        self.decay_rate = decay_rate
        self.decay_step = decay_step
        self.max_itr = max_itr
        self.learning_rate_init = learning_rate_init
        self.momentum = momentum
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.tol = tol
        self.n_iter_not_changed = n_iter_not_changed
        self.verbose = verbose
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.random_state = random_state
        self.activation = activation
        self.output_activation = output_activation
        self.cost_func = output_activation
        self.affine_num = len(hidden_layers)

        self.affines = []
        self.paras = []
        self.deriva = []

        self.m_sample = None
        self.n_sample = None
        self.s_in = None
        self.s_out = None

    def _initialize(self):
        activation = ACTIVATIONS[self.activation]
        output_activation = ACTIVATIONS[self.output_activation]
        l_in = Layer(self.s_in, activation_fun=Linear())

        for i in self.hidden_layers:
            l_out = Layer(i, activation_fun=activation())
            affine = Affine(
                l_in, l_out, para_init=self.para_init, random_state=self.random_state
            )
            l_in = l_out
            self.paras.append(affine.theta)
            self.affines.append(affine)
            self.deriva.append(np.zeros_like(affine.theta))

        l_out = Layer(self.s_out, activation_fun=output_activation())
        affine = Affine(
            l_in, l_out, para_init=self.para_init, random_state=self.random_state
        )
        self.paras.append(affine.theta)
        self.affines.append(affine)
        self.deriva.append(np.zeros_like(affine.theta))

    def _model(self, x):

        self.affines[0].set_inlayer_z(x)
        for affine in self.affines:
            affine.forward()
        return self.affines[self.affine_num].out_layer.a

    def _cost_fun(self, x, y):

        cost_function = COSTFUNCTIONS[self.cost_func]
        hx = self._model(x)
        a = cost_function(hx, y)

        # regulariztion
        b = 0
        if self.lambd != 0:
            theta_num = 0
            for theta in self.paras:
                b += np.square(theta[:, 1 : theta.shape[1]]).sum()
                theta_num += theta.size
            b = self.lambd * b / (2 * theta_num)
        return a + b

    def _cost_fun_derivative(self, x, y):

        # forward propagation, computing a
        last_layer_a = self._model(x)

        # backward propagation, computing delta
        last_layer_delta = last_layer_a - y
        self.affines[self.affine_num].set_outlayer_delta(last_layer_delta)
        for i in range(len(self.affines) - 1, 0, -1):
            self.affines[i].backward()

        # computing derivatives of thetas
        for i in range(len(self.deriva)):
            delta_lplus1 = self.affines[i].out_layer.delta
            a_l_t = add_bias(self.affines[i].in_layer.a, axis=0).T
            self.deriva[i] = np.dot(delta_lplus1, a_l_t) / self.m_sample

        # regularization
        if self.lambd != 0:
            for i in range(len(self.deriva)):
                # m_sample = self.deriva[i].shape[1],[:, 1:m_sample]
                self.deriva[i] += self.lambd * self.paras[i]

    def train(self, x, y, plot=True):
        self.s_in = x.shape[1]
        self.s_out = y.shape[1]

        self.m_sample = x.shape[0]
        self.n_sample = x.shape[1]

        self._initialize()

        if self.batch_size == "auto":
            self.batch_size = min(200, self.n_sample)

        xaxis = np.arange(0, self.max_itr * np.ceil(self.m_sample / self.batch_size))
        yaxis = np.zeros(int(self.max_itr * np.ceil(self.m_sample / self.batch_size)))

        self._fit_optimizer(x, y, yaxis, plot)

        if plot:
            plt.xlabel("itration")
            plt.ylabel("value")
            plt.plot(xaxis, yaxis)
            plt.show()

    def _fit_optimizer(self, x, y, yaxis, plot):
        s = [np.zeros_like(para) for para in self.paras]
        v = [np.zeros_like(para) for para in self.paras]

        n = 0
        previous_acc = accuracy(self.predict(x), y.T)
        alpha = self.learning_rate_init

        for i in range(self.max_itr):

            # shuffle samples at each epoch, when bgd not used
            if self.shuffle and self.batch_size != self.m_sample:
                z = np.hstack([x, y])
                np.random.shuffle(z)
                x, y = np.hsplit(z, [self.n_sample])

            # going through batches
            for j in range(0, self.m_sample, self.batch_size):
                itr_num = int(
                    i * np.ceil(self.m_sample / self.batch_size) + j / self.batch_size
                )
                self._cost_fun_derivative(
                    x[j : j + self.batch_size, :].T, y[j : j + self.batch_size, :].T
                )
                for k in range(len(self.paras)):
                    updated_theta = None
                    if self.optimizer == "adam":
                        v[k] = self.beta_1 * v[k] + (1 - self.beta_1) * self.deriva[k]
                        s[k] = self.beta_2 * s[k] + (1 - self.beta_2) * np.square(
                            self.deriva[k]
                        )
                        corrected_v = v[k] / (1 - np.power(self.beta_1, itr_num + 1))
                        corrected_s = s[k] / (1 - np.power(self.beta_2, itr_num + 1))
                        updated_theta = self.paras[k] - alpha * corrected_v / (
                            np.sqrt(corrected_s) + 1e-8
                        )
                    elif self.optimizer == "rmsprop":
                        s[k] = self.momentum * s[k] + (1 - self.momentum) * np.square(
                            self.deriva[k]
                        )
                        updated_theta = self.paras[k] - alpha * self.deriva[k] / (
                            np.sqrt(s[k]) + 1e-8
                        )
                    elif self.optimizer == "sgdm":
                        v[k] = (
                            self.momentum * v[k] + (1 - self.momentum) * self.deriva[k]
                        )
                        updated_theta = self.paras[k] - alpha * v[k]
                    elif self.optimizer == "sgd":
                        updated_theta = self.paras[k] - alpha * self.deriva[k]
                    else:
                        print(f'unsupported optimizer "{self.optimizer}"')
                        return
                    self.paras[k] = updated_theta
                    self.affines[k].theta = updated_theta

                # loss curve option
                if plot:
                    yaxis[itr_num] = self._cost_fun(
                        x[j : j + self.batch_size, :].T, y[j : j + self.batch_size, :].T
                    )

            if self.verbose:
                print(f"epoch {i + 1} -> loss : {self._cost_fun(x.T, y.T)}")

            # auto stop
            acc = accuracy(self.predict(x), y.T)
            acc_improvement = acc - previous_acc
            previous_acc = acc
            if acc_improvement < self.tol:
                n += 1
                if n > self.n_iter_not_changed:
                    print("converged")
                    return
            else:
                n = 0

                # learning rate decay
            if self.learning_rate == "expdecay":
                if (i + 1) % self.decay_step == 0:
                    alpha = (
                        np.power(self.decay_rate, (i + 1) / self.decay_step)
                        * self.learning_rate_init
                    )

    def predict(self, x, display_prob=False):
        x = x.T
        result = self._model(x)
        if not display_prob:
            result = result - result.max(axis=0)
            result[result == 0] = 1
            result[result < 0] = 0
        return result


# testing below
if __name__ == "__main__":
    t = pd.read_csv("../testfile/dataset.csv")
    t = t.to_numpy(dtype=float)

    pu = ProcessingUnit(t, random_state=0)
    pu.y_to_onehot()
    pu.split_dataset(ratio=(0.8, 0.1, 0.1))
    pu.z_score()
    # pu.pca(n_component=0.97)
    xtrain, ytrain = pu.train
    xcv, ycv = pu.cv
    xtest, ytest = pu.test
    m_train = xtrain.shape[0]

    nn = NeuralNetwork(
        hidden_layers=(172,),
        batch_size="auto",
        optimizer="adam",
        learning_rate="expdecay",
        lambd=0,
        learning_rate_init=0.00585,
        decay_rate=0.98,
        decay_step=2,
        max_itr=500,
        shuffle=True,
        verbose=True,
        random_state=None,
    )
    nn.train(xcv, ycv, plot=False)

    y_p_train = nn.predict(xtrain)
    y_p_cv = nn.predict(xcv)
    y_p_test = nn.predict(xtest)

    eu1 = EvaluationUnit(y_p_train, ytrain)
    eu2 = EvaluationUnit(y_p_cv, ycv)
    eu3 = EvaluationUnit(y_p_test, ytest)
    p, r = eu2.p_and_r()

    print(f"acc_train -> {eu1.accuracy()}")
    print(f"acc_cv -> {eu2.accuracy()}")
    print(f"acc_test -> {eu3.accuracy()}")
    print(accuracy_score(ytrain, y_p_train.T))
    print(f'confusing_matrix -> "\n"{eu2.confusing_matrix()}')
    print("precision -> " + str(list(p)))
    print("recall -> " + str(list(r)))

    # print(nn._cost_fun(xcv.T, ycv.T))
    # print(nn._cost_fun(xtrain.T, ytrain.T))

    # learning curve
    # xaix1 = np.arange(0, xtrain.shape[0]) yaix1 = np.zeros(xtrain.shape[0]) xaix2 = np.arange(0,
    # xtrain.shape[0]) yaix2 = np.zeros(xtrain.shape[0]) for i in range(xtrain.shape[0]): nn = NeuralNetwork(
    # hidden_layers=(302,), activation=af.Relu, para_init='he', reg_lamdba=0.001, output_activation='softmax')
    # nn.train(xtrain[0:i + 1, :], ytrain[0:i + 1, :], max_itr=50, learning_rate_init=0.06, regularize=False,
    # plot=False)
    #
    #     yaix1[i] = nn.cost_fun(xcv.T, ycv.T, regularize=False)
    #     yaix2[i] = nn.cost_fun(xtrain[0:i + 1, :].T, ytrain[0:i + 1, :].T, regularize=False)
    # plt.plot(xaix1, yaix1)
    # plt.plot(xaix2, yaix2)
    # plt.show()
