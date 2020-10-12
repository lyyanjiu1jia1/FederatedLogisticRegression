import numpy as np


class LogisticRegression(object):
    def __init__(self, dim, alpha, eta, max_iter, train_method):
        """
        
        :param dim: int, feature dimension
        :param alpha: float, regularizer balancer
        :param eta: float, learning rate
        :param max_iter: int
        :param train_method: str
        """
        self.weight = np.ones((dim + 1, 1))
        self.eta = eta
        self.alpha = alpha
        self.max_iter = max_iter
        self.train_method = train_method
        self.train_method_list = ['gd']

    def _sigmoid(self, x):
        """
        1 / (1 + exp(-w^T x))
        :param x: ndarray
        :return:
        """
        func_val = 1 / (1 + np.exp(-self.weight.transpose().dot(x)))
        return func_val

    @staticmethod
    def data_processing(X, y):
        """
        Remove sample ID and encode label by \pn 1
        :param X:
        :param y:
        :return:
        """
        X = LogisticRegression.remove_id(X)
        X = LogisticRegression.add_homo_column(X)
        y = LogisticRegression.encode_label_by_minus_one(y)
        return X, y

    @staticmethod
    def remove_id(X):
        """

        :param X: ndarray
        :return:
        """
        X = X[1:]
        return X

    @staticmethod
    def add_homo_column(X):
        """

        :param X: ndarray
        :return:
        """
        sample_num = X.shape[0]
        X = np.concatenate((X, np.ones(sample_num)))
        return X

    @staticmethod
    def encode_label_by_minus_one(y):
        """

        :param y: ndarray
        :return:
        """
        for i in y.shape[0]:
            if y[i][0] == 0:
                y[i][0] = -1
        return y


class CentralizedLogisticRegression(LogisticRegression):
    def __init__(self, dim, alpha, eta, max_iter, train_method):
        super(CentralizedLogisticRegression, self).__init__(dim, alpha, eta, max_iter, train_method)

    def train(self, X, y):
        """

        :param X: ndarray, design matrix
        :param y: ndarray, binary label vector, \pn 1 encoded
        :return:
        """
        iter = 0
        while iter < self.max_iter:
            if self.train_method == 'gd':
                self._gradient_descent(X, y)

    def _gradient_descent(self, X, y):
        grad = 0
        for i in X.shape(0):
            xi, yi = X[[i], :], y[i, 0]
            sigmoid_val = self._sigmoid(yi * xi)
            local_grad = (1 - sigmoid_val) * yi * xi - self.alpha * self.weight
            grad += local_grad
        self.weight += 
