import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA
from sklearn.metrics.pairwise import rbf_kernel


def gaussian_kernel(x, y, sigma):
    return np.exp(-LA.norm(x - y) ** 2 / (2 * sigma ** 2))


def laplacian_kernel(x, y, sigma):
    return np.exp(-LA.norm(x - y) / sigma)


class KRR:
    """
    main structure for kernel regression
    """

    def __init__(self, x_data, y_label, kernel, sigma):
        self.x_data = x_data
        self.y_label = y_label
        self.kernel = None
        self.sigma = sigma
        if kernel == 'rbf':
            self.kernel = 'rbf'
        elif kernel == 'laplacian':
            self.kernel = 'laplacian'
        else:
            raise NameError('No such kernel.')
        n, d = self.x_data.shape
        self.n = n
        self.d = d
        gram = self.gram()
        self.gram = gram
        self.transpose_y = self.y_label.T

    def gram(self):
        """
        This is the main function for the Kernel ridge regression
        construct the gram matrix
        :param kernel:
        :param x_data:
        :param self: (N * D)
        :param y_label: (N * 1)
        :return: Gram Matrix
        """
        N = self.n

        gram = np.zeros((N, N))
        alpha = 0.1

        for s in range(N):
            for t in range(N):
                if self.kernel == 'rbf':
                    gram[s, t] = gaussian_kernel(self.x_data[s, :], self.y_label[t, :], self.sigma)
                else:
                    gram[s, t] = laplacian_kernel(self.x_data[s, :], self.y_label[t, :], self.sigma)

        return gram

    def predict(self, regularized, x_predict):
        gram_regularized = np.eye(self.n) * regularized + self.gram
        n_predict, d_predict = x_predict.shape
        feature_selection = rbf_kernel(self.x_data, x_predict)
        # print(self.transpose_y.shape)
        # print(self.gram.shape)
        # print(feature_selection.shape)

        return np.matmul(np.matmul(self.transpose_y, self.gram), feature_selection)
