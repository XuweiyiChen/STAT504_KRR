import scipy as sp
from numpy.linalg import inv
import numpy as np
from scipy import linalg
from numpy import linalg as LA


def kernel_quadratic(x1, x2):
    return np.dot(x1, x2.T) ** 2


def kernel_linear(x1, x2):
    return np.dot(x1, x2.T)


class KernelRidge:
    """
        Simple implementation of a Kernel Ridge Regression using the
        closed form for training.
        https://people.eecs.berkeley.edu/~jrs/189s19/lec/14.pdf
    """

    def __init__(self, kernel_type='linear', regularizer=1.0, gamma=0.5):
        """
        :param kernel_type: Kernel type to use in training.
                        'linear' use linear kernel function.
                        'quadratic' use quadratic kernel function.
                        'gaussian' use gaussian kernel function
        :param regularizer: Value of regularization parameter C
        :param gamma: parameter for gaussian kernel or Polynomial kernel
        """
        self.alphas = None
        self.kernels = {
            'linear': kernel_linear,
            'quadratic': kernel_quadratic,
            'gaussian': self.kernel_gaussian,
            'laplacian': self.laplacian_kernel
        }
        self.kernel_type = kernel_type
        self.kernel = self.kernels[self.kernel_type]
        self.C = regularizer
        self.gamma = gamma
        self.kernel_matrix = None

    # Define kernels
    def laplacian_kernel(self, x, y, sigma):
        gamma = self.gamma
        return np.exp(-linalg.norm(x - y) / sigma)

    def kernel_gaussian(self, x1, x2, gamma=5.0):
        gamma = self.gamma
        return np.exp(-linalg.norm(x1 - x2) ** 2 / (2 * (gamma ** 2)))

    def compute_kernel_matrix(self, X1, X2):
        """
        compute kernel matrix (gram matrix) give two input matrix
        """

        # sample size
        n1 = X1.shape[0]
        n2 = X2.shape[0]

        # Gram matrix
        K = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self.kernel(X1[i], X2[j])

        return K

    def fit(self, X, y):
        """
        training KRR
        :param X: training X
        :param y: training y
        :return: alpha vector
        """
        K = self.compute_kernel_matrix(X, X)

        self.kernel_matrix = K
        self.alphas = np.dot(inv(K + self.C * np.eye(np.shape(K)[0])),
                             y.transpose())

        return self.alphas

    def predict(self, x_train, x_test):
        """
        :param x_train: DxNtr array of Ntr train data points
                        with D features
        :param x_test:  DxNte array of Nte test data points
                        with D features
        :return: y_test, D2xNte array
        """

        k = self.compute_kernel_matrix(x_test, x_train)

        y_test = sp.dot(k, self.alphas)
        return y_test.transpose()
