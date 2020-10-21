import copy

import numpy as np
from sklearn import metrics, linear_model


class LogisticRegression(object):
    def __init__(self, alpha, eta, max_iter, train_method, balance_mode, oversampling):
        """
        
        :param alpha: float, regularizer balancer
        :param eta: float, learning rate
        :param max_iter: int
        :param train_method: str
                'gd': gradient descent
                'l_gd': line search gradient descent
                ‘sgd’: stochastic gradient descent
                'cd': coordinate descent
                'cgd': conjugate gradient descent
                'single_lbfgs': single-memory BFGS
                'lbfgs': limited-memory BFGS
                'momentum_lbfgs': sandbox version
                'smart_momentum_lbfgs': sandbox version
                'smart_adam_lbfgs': sandbox version
                'trust_region': trust region Newton method
                'liblinear': liblinear
                'dual': liblinear-dual
        :param balance_mode: boolean
        """
        self.weight = None
        self.eta = eta
        self.alpha = alpha
        self.max_iter = max_iter

        if '-' in train_method:
            train_method = train_method.split('-')
            self.train_method = train_method[0]
            self.liblinear_train_method = train_method[1]
        else:
            self.train_method = train_method
            self.liblinear_train_method = None

        self.balance_mode = balance_mode
        self.balance_weight = None

        self.oversampling = oversampling
        self.oversampling_rate = None

        self.loss_trajectory = []
        self.auc_trajectory = []

        self.sample_num = None
        self.dim = None

    def _init_params(self, X, y):
        pass

    def train(self, X, y):
        pass

    def predict(self, X):
        pass

    def validate(self, X, y):
        pass

    def _compute_loss(self, X, y):
        loss = 0
        for i in range(X.shape[0]):
            xi, yi = X[[i], :].transpose(), y[i, 0]
            yi_logit = yi * self.weight.transpose().dot(xi)
            try:
                local_loss = np.log(1 / self._sigmoid(yi_logit))
            except ZeroDivisionError:
                local_loss = -yi_logit

            if np.isinf(local_loss) or np.isnan(local_loss):
                continue
            elif yi == 1:
                local_loss *= self.balance_weight[0]

            loss += local_loss
        loss = loss / self.sample_num + self.alpha / 2 * np.square(np.linalg.norm(self.weight))
        return np.float(loss)

    def _sigmoid(self, x):
        x = np.float(x)
        if x <= 0:
            a = np.exp(x)
            a /= (1. + a)
        else:
            a = 1. / (1. + np.exp(-x))
        return a

    @staticmethod
    def data_processing(X, y):
        """
        Remove sample ID and encode label by \pn 1
        :param X:
        :param y:
        :return:
        """
        X = LogisticRegression.remove_id(X)
        X = LogisticRegression.convert_nan(X)
        X = LogisticRegression.add_homo_column(X)
        y = LogisticRegression.encode_label_by_minus_one(y)
        return X, y

    @staticmethod
    def convert_nan(X):
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if np.isnan(X[i, j]):
                    X[i, j] = 0.0
        return X

    @staticmethod
    def remove_id(X):
        """

        :param X: ndarray
        :return:
        """
        X = X[:, 1:]
        return X

    @staticmethod
    def add_homo_column(X):
        """

        :param X: ndarray
        :return:
        """
        sample_num = X.shape[0]
        X = np.concatenate((X, np.ones((sample_num, 1))), axis=1)
        return X

    @staticmethod
    def encode_label_by_minus_one(y):
        """

        :param y: ndarray
        :return:
        """
        for i in range(y.shape[0]):
            if y[i][0] == 0:
                y[i][0] = -1
        return y


class CentralizedLogisticRegression(LogisticRegression):
    def __init__(self, alpha, eta, max_iter, train_method, balance_mode, oversampling):
        super(CentralizedLogisticRegression, self).__init__(alpha, eta, max_iter, train_method,
                                                            balance_mode, oversampling)

    def train(self, X, y):
        """

        :param X: ndarray, design matrix
        :param y: ndarray, binary label vector, \pn 1 encoded
        :return:
        """
        self._init_params(X, y)
        self._preprocessing(X, y)

        print("init loss = {}, auc = {}".format(self._compute_loss(X, y), self.validate(X, y)))

        use_liblinear = self._liblinear(X, y)
        if use_liblinear:
            return

        for iter in range(self.max_iter):
            print(">>> iter = {}".format(iter))
            if self.train_method == 'gd':
                self._gradient_descent(X, y)
            elif self.train_method == 'l_gd':
                self._line_search_gradient_descent(X, y)
            elif self.train_method == 'sgd':
                self._stochastic_gradient_descent(X, y)
            elif self.train_method == 'cd':
                self._coordinate_descent(X, y, iter % self.dim)
            elif self.train_method == 'cgd':
                self._conjugate_gradient_descent(X, y)
            elif self.train_method == 'single_lbfgs':
                self._single_lbfgs(X, y)
            elif self.train_method == 'lbfgs':
                self._lbfgs(X, y)
            elif self.train_method == 'momentum_lbfgs':
                self._momentum_lbfgs(X, y)
            elif self.train_method == 'smart_momentum_lbfgs':
                self._smart_momentum_lbfgs(X, y)
            elif self.train_method == 'smart_adam_lbfgs':
                self._smart_adam_lbfgs(X, y)
            elif self.train_method == 'trust_region':
                self._trust_region(X, y)
                if not self.search:
                    break
            else:
                raise ValueError("train method {} not supported".format(self.train_method))

            # display loss
            loss = self._compute_loss(X, y)
            self.loss_trajectory.append(loss)

            # display auc
            auc = self.validate(X, y)
            self.auc_trajectory.append(auc)

            # display first 3 weights
            weight_to_display = self.weight[0:3, :]

            # display
            print("loss = {}, auc = {}, weight = {}".format(loss,
                                                            auc,
                                                            weight_to_display.reshape(len(weight_to_display))))

    def _conjugate_gradient_descent(self, X, y):
        grad = self._compute_gradient(X, y)

        if self.prev_grad is None and self.prev_u is None:
            u = grad
        else:
            grad_diff = grad - self.prev_grad
            beta = grad.transpose().dot(grad_diff) / self.prev_u.transpose().dot(grad_diff)
            u = grad - beta * self.prev_u

        self.prev_grad = copy.deepcopy(grad)
        self.prev_u = copy.deepcopy(u)

        denominator = self._conjugate_denominator(X, u)
        numerator = grad.transpose().dot(u)[0, 0]
        weight_diff = numerator / denominator * u
        self.weight = self.weight - weight_diff

    def _preprocessing(self, X, y):
        if self.oversampling_rate:
            for i in range(X.shape[0]):
                if y[i, 0] == 1:
                    X[i] *= self.oversampling_rate
        return X, y
        # # balance positive samples
        # if self.balance_mode:
        #     for i in range(y.shape[0]):
        #         if y[i, 0] == 1:
        #             X[i] = X[i] / self.pos_neg_ratio

    def _trust_region(self, X, y):
        print("trust region Newton begins")

        # check early stop
        g = self._compute_gradient(X, y)
        g_norm = np.linalg.norm(g)
        self.f = self._compute_trust_region_loss(X, y)
        if g_norm <= self.eps * self.g_norm0:
            self.search = False
            return

        # perform update
        M = self._get_preconditioner(X)
        s = self._pcg(X, g, M)
        step_size = self._linear_search_and_update(X, y, s, g)

        if step_size == 0:
            print("line search failed")

    def _pcg(self, X, g, M):
        """

        :param X:
        :param g: gradient
        :param M: preconditioner
        :return:
        """
        # init
        s = np.zeros((self.dim, 1))
        r = -g
        z = self.invert_digonal_matrix(M).dot(r)
        d = copy.deepcopy(z)
        gamma = np.float(r.transpose().dot(z))      # z^\top r
        gamma_sqrt = np.sqrt(gamma)
        Q = 0.0
        cgtol = min((self.eps_cg, gamma_sqrt))
        max_cg_iter = max((self.sample_num, 5))

        # iterate
        for cg_iter in range(max_cg_iter):
            print("cg_iter = {}".format(cg_iter))
            v = self._compute_v(X, d)
            alpha = self._compute_alpha(gamma, d, v)
            s = self._update_s(s, alpha, d)
            r = self._update_r(r, alpha, d)
            new_Q = self._compute_Q(s, r, g)
            Q_diff = self._compute_Q_diff(Q, new_Q)

            if new_Q <= 0 and Q_diff <= 0:
                if cg_iter * Q_diff <= cgtol * new_Q:
                    break
            else:
                print("quadratic approximation > 0 or increasing in CG")
                break

            Q = new_Q
            z = self._update_z(M, r)
            new_gamma = self._update_gamma(z, r)
            beta = self._compute_beta(new_gamma, gamma)
            d = self._update_d(z, beta, d)
            gamma = new_gamma

        print("pcg complete")

        return s

    def _compute_beta(self, new_gamma, gamma):
        beta = new_gamma / gamma
        return beta

    def _update_d(self, z, beta, d):
        d = z + beta * d
        return d

    def _update_gamma(self, z, r):
        gamma = np.float(z.transpose().dot(r))
        return gamma

    def _update_z(self, M, r):
        z = self.invert_digonal_matrix(M).dot(r)
        return z

    def _compute_Q_diff(self, Q, new_Q):
        Q_diff = new_Q - Q
        return Q_diff

    def _compute_Q(self, s, r, g):
        Q = 0.5 * np.float(s.transpose().dot(g - r))
        return Q

    def _update_r(self, r, alpha, d):
        r = r - alpha * d
        return r

    def _update_s(self, s, alpha, d):
        s = s + alpha * d
        return s

    def _compute_alpha(self, gamma, d, v):
        numerator = gamma
        denominator = np.float(d.transpose().dot(v))
        try:
            alpha = numerator / denominator
        except ZeroDivisionError:
            alpha = 1
        return alpha

    def _compute_v(self, X, d):
        """

        :param X:
        :param y:
        :param d:
        :return: v = \sum C \sigma_i (1 - \sigma_i) x_i^\top d x_i
        """
        Hd = np.zeros((self.dim, 1))

        for i in range(self.sample_num):
            xi = X[[i], :].transpose()
            sigma = self._sigmoid(self.weight.transpose().dot(xi))
            scalar = np.float(-self.C * sigma * (1 - sigma) * d.transpose().dot(xi))
            Hd = Hd + scalar * xi

        return Hd

    def _linear_search_and_update(self, X, y, s, g):
        """

        :param X:
        :param y:
        :param s:
        :param g:
        :return: alpha
        """
        # init
        s_s = np.square(np.linalg.norm(s))
        g_s = np.float(g.transpose().dot(s))
        x_s = X.dot(s)
        eta = 1e-2
        loss = 0
        fold = self.f
        alpha = self.init_step_size
        max_num_linesearch = 20
        w_new = copy.deepcopy(self.weight)

        # iterate
        num_linesearch = 0
        for num_linesearch in range(max_num_linesearch):
            w_new = w_new + alpha * s

            w_x = X.dot(w_new)
            w_w = np.square(np.linalg.norm(w_new))
            w_s = np.float(w_new.transpose().dot(s))
            g_temp = copy.deepcopy(self.weight)
            self.weight = w_new
            g = self._compute_gradient(X, y)
            self.weight = g_temp
            g_s = np.float(g.transpose().dot(s))

            for i in range(self.sample_num):
                inner_product = x_s[i, 0] * alpha + w_x[i, 0]
                loss += self._C_times_loss(inner_product, y[i, 0])

            self.f = loss + alpha ** 2 * s_s * w_w / 2 + alpha * w_s

            if self.f - fold <= eta * alpha * g_s:
                break
            else:
                alpha *= 0.5

        if num_linesearch >= max_num_linesearch - 1:
            self.f = fold
            return 0
        else:
            self.weight = w_new

        print("line search and update complete")

        return alpha

    def _C_times_loss(self, wxi, yi):
        ywxi = wxi * yi
        if ywxi >= 0:
            return np.float(self.C * np.log(1 + np.exp(-ywxi)))
        else:
            return np.float(self.C * (-ywxi + np.log(1 + np.exp(ywxi))))

    def _get_preconditioner(self, X):
        diag_elements = np.ones(self.dim)

        for i in range(self.sample_num):
            xi = X[i, :]
            sigma = self._sigmoid(self.weight.reshape(self.dim).dot(xi))
            diag_elements = diag_elements + self.C * sigma * (1 - sigma) * np.square(xi)

        M = np.diag(diag_elements)
        M = (1 - self.alpha_pcg) * np.identity(self.dim) + self.alpha_pcg * M

        print("got preconditioner")

        return M

    def _stochastic_gradient_descent(self, X, y):
        for index in self.shuffle_generator():
            grad = self._compute_local_gradient(X, y, index)
            self.weight = self.weight + self.eta * grad

    def predict(self, X):
        """

        :param X: ndarray
        :return: y_pred
        """
        y_pred = X.dot(self.weight)
        y_pred = 1 / (1 + np.exp(-y_pred))
        return y_pred

    def validate(self, X, y):
        y_pred = self.predict(X)
        fpr, tpr, thresholds = metrics.roc_curve(y, y_pred)
        auc = metrics.auc(fpr, tpr)
        return auc

    def _liblinear(self, X, y):
        if self.train_method == 'liblinear':
            if self.liblinear_train_method == 'liblinear':
                solver = 'liblinear'
                solve_dual = False
            elif self.liblinear_train_method == 'dual':
                solver = 'liblinear'
                solve_dual = True
            else:
                raise ValueError("liblinear train method {} not supported".format(self.liblinear_train_method))
            lr_trainer = linear_model.LogisticRegression(penalty='l2',
                                                         fit_intercept=False,
                                                         solver=solver,
                                                         dual=solve_dual,
                                                         C=1 / self.alpha,
                                                         max_iter=self.max_iter,
                                                         random_state=0)
            lr_trainer.fit(X, y)
            self.weight = lr_trainer.coef_.reshape((len(self.weight), 1))
            return True
        return False

    def _gradient_descent(self, X, y):
        grad = self._compute_gradient(X, y)
        self.weight = self.weight + self.eta * grad

    def _line_search_gradient_descent(self, X, y):
        grad = self._compute_gradient(X, y)

        # line search
        denominator = self._conjugate_denominator(X, grad)
        numerator = grad.transpose().dot(grad)[0, 0]
        eta = numerator / denominator

        self.weight = self.weight - eta * grad

    def _coordinate_descent(self, X, y, target_sample):
        """

        :param X:
        :param y:
        :param target_sample: int, at which coordinate the weight is to be updated
        :return:
        """
        numerator, denominator = 0, 0

        for i in range(self.sample_num):
            xi, yi = X[[i], :].transpose(), y[i, 0]

            # compute numerator
            local_numerator = (1 - self._sigmoid(yi * self.weight.transpose().dot(xi))) * yi * xi[target_sample, 0]
            numerator += local_numerator

            # compute denominator
            sigmoid_val = self._sigmoid(self.weight.transpose().dot(xi))
            aii = sigmoid_val * (1 - sigmoid_val)
            local_denominator = aii * np.square(xi[target_sample, 0])
            denominator += local_denominator

        numerator -= self.alpha * self.weight[target_sample, 0]
        denominator += self.alpha

        grad = numerator / denominator

        self.weight[target_sample, 0] += grad

    def _compute_gradient(self, X, y):
        """
        likelihood gradient, not loss
        :param X:
        :param y:
        :return:
        """
        grad = np.zeros((self.dim, 1))
        for i in range(self.sample_num):
            xi, yi = X[[i], :].transpose(), y[i, 0]
            # sigmoid_val = self._sigmoid(yi * self.weight.transpose().dot(xi))
            # local_grad = (1 - sigmoid_val) * yi * xi
            local_grad = yi / (1 + np.exp(yi * np.float(self.weight.transpose().dot(xi)))) * xi
            if yi == 1:
                local_grad = local_grad * self.balance_weight[0]
            else:
                local_grad = local_grad * self.balance_weight[1]
            grad = grad + local_grad
        grad = grad / self.sample_num - self.alpha * self.weight
        return grad

    def _compute_local_gradient(self, X, y, index):
        xi, yi = X[[index], :].transpose(), y[index, 0]
        sigmoid_val = self._sigmoid(yi * self.weight.transpose().dot(xi))
        grad = (1 - sigmoid_val) * yi * xi - self.alpha * self.weight
        if yi == 1:
            grad = grad * self.balance_weight[0]
        else:
            grad = grad * self.balance_weight[1]
        return grad

    def _lbfgs(self, X, y):
        # compute gradient
        grad = self._compute_gradient(X, y)        # add - to make it a loss gradient

        # direction search
        alpha_array = []
        q = copy.deepcopy(grad)

        # update dg, dw
        if type(self.g_array[0]) == np.ndarray and type(self.g_array[1]) == np.ndarray:
            self.pop_push(self.dg_array, self.g_array[1] - self.g_array[0])
            self.pop_push(self.dw_array, self.w_array[1] - self.w_array[0])

        # update rho, gamma
        if self.dg_array[-1] is not None:
            self.pop_push(self.rho_array, 1 / np.float(self.dg_array[-1].transpose().dot(self.dw_array[-1])))
            gamma = np.float((self.dg_array[-1].transpose().dot(self.dw_array[-1])) /
                             np.square(np.linalg.norm(self.dg_array[-1])))

        # recurse on q
        for rho, dw, dg in zip(reversed(self.rho_array), reversed(self.dw_array), reversed(self.dg_array)):
            if rho is None:
                continue
            alpha = rho * np.float(dw.transpose().dot(q))
            q = q - alpha * dg
            alpha_array.append(alpha)

        # recurse on u
        if self.dg_array[-1] is None:
            u = q
        else:
            u = gamma * q
            for rho, dw, dg, alpha in zip(self.rho_array, self.dw_array, self.dg_array, reversed(alpha_array)):
                if rho is None:
                    continue
                beta = rho * np.float(dg.transpose().dot(u))
                u = u + dw * (alpha - beta)

        # reverse u
        u = -u

        # update g, w array of length two
        self.pop_push(self.g_array, grad)
        self.pop_push(self.w_array, self.weight)

        # line search
        denominator = self._conjugate_denominator(X, u)
        numerator = grad.transpose().dot(u)[0, 0]
        u_coef = numerator / denominator

        weight_diff = u_coef * u
        self.weight = self.weight - weight_diff

    def _single_lbfgs(self, X, y):
        grad = self._compute_gradient(X, y)

        if self.prev_grad is None and self.prev_weight is None:
            u = -grad
        else:
            delta_grad = grad - self.prev_grad
            delta_weight = self.weight - self.prev_weight
            wg = delta_grad.transpose().dot(delta_weight)[0, 0]
            b = 1 + np.square(np.linalg.norm(delta_grad)) / wg
            ag = delta_weight.transpose().dot(grad)[0, 0] / wg
            aw = delta_grad.transpose().dot(grad)[0, 0] / wg - b * ag
            u = -grad + aw * delta_weight + ag * delta_grad

        self.prev_grad = copy.deepcopy(grad)
        self.prev_weight = copy.deepcopy(self.weight)

        denominator = self._conjugate_denominator(X, u)
        numerator = grad.transpose().dot(u)[0, 0]
        u_coef = numerator / denominator

        weight_diff = u_coef * u
        self.weight = self.weight - weight_diff

    def _momentum_lbfgs(self, X, y):
        grad = self._compute_gradient(X, y)

        if self.prev_grad is None and self.prev_weight is None:
            u = -grad
        else:
            delta_grad = grad - self.prev_grad
            delta_weight = self.weight - self.prev_weight
            wg = delta_grad.transpose().dot(delta_weight)[0, 0]
            b = 1 + np.square(np.linalg.norm(delta_grad)) / wg
            ag = delta_weight.transpose().dot(grad)[0, 0] / wg
            aw = delta_grad.transpose().dot(grad)[0, 0] / wg - b * ag
            u = -grad + aw * delta_weight + ag * delta_grad

        self.prev_grad = copy.deepcopy(grad)
        self.prev_weight = copy.deepcopy(self.weight)

        denominator = self._conjugate_denominator(X, u)
        numerator = grad.transpose().dot(u)[0, 0]
        u_coef = numerator / denominator
        weight_diff = u_coef * u

        if self.prev_weight_diff is not None:
            weight_diff += self.gamma * self.prev_weight_diff
        self.prev_weight_diff = copy.deepcopy(weight_diff)

        self.weight = self.weight - weight_diff

    def _smart_momentum_lbfgs(self, X, y):
        grad = self._compute_gradient(X, y)

        if self.prev_grad is None and self.prev_weight is None:
            u = -grad
        else:
            delta_grad = grad - self.prev_grad
            delta_weight = self.weight - self.prev_weight
            wg = delta_grad.transpose().dot(delta_weight)[0, 0]
            b = 1 + np.square(np.linalg.norm(delta_grad)) / wg
            ag = delta_weight.transpose().dot(grad)[0, 0] / wg
            aw = delta_grad.transpose().dot(grad)[0, 0] / wg - b * ag
            u = -grad + aw * delta_weight + ag * delta_grad

        self.prev_grad = copy.deepcopy(grad)
        self.prev_weight = copy.deepcopy(self.weight)

        denominator = self._conjugate_denominator(X, u)
        numerator = grad.transpose().dot(u)[0, 0]
        u_coef = numerator / denominator
        weight_diff = u_coef * u

        loss_reduction = self._check_loss_reduction()

        # if previous loss reduce, add up its momentum
        if self.prev_weight_diff is not None:
            weight_diff += self.gamma * self.prev_weight_diff

        # if current loss reduce, store the momentum
        if loss_reduction:
            self.prev_weight_diff = copy.deepcopy(weight_diff)
        else:
            self.prev_weight_diff = None

        self.weight = self.weight - weight_diff

    def _smart_adam_lbfgs(self, X, y):
        grad = self._compute_gradient(X, y)

        if self.prev_grad is None and self.prev_weight is None:
            u = -grad
        else:
            delta_grad = grad - self.prev_grad
            delta_weight = self.weight - self.prev_weight
            wg = delta_grad.transpose().dot(delta_weight)[0, 0]
            b = 1 + np.square(np.linalg.norm(delta_grad)) / wg
            ag = delta_weight.transpose().dot(grad)[0, 0] / wg
            aw = delta_grad.transpose().dot(grad)[0, 0] / wg - b * ag
            u = -grad + aw * delta_weight + ag * delta_grad

        self.prev_grad = copy.deepcopy(grad)
        self.prev_weight = copy.deepcopy(self.weight)

        denominator = self._conjugate_denominator(X, u)
        numerator = grad.transpose().dot(u)[0, 0]
        u_coef = numerator / denominator

        # if previous loss reduce, add up its momentum
        if self.prev_g is not None:
            self.opt_beta1_decay = self.opt_beta1_decay * self.opt_beta1
            self.opt_beta2_decay = self.opt_beta2_decay * self.opt_beta2
            self.opt_m = self.opt_beta1 * self.opt_m + (1 - self.opt_beta1) * grad
            self.opt_v = self.opt_beta2 * self.opt_v + (1 - self.opt_beta2) * np.square(grad)
            opt_m_hat = self.opt_m / (1 - self.opt_beta1_decay)
            opt_v_hat = self.opt_v / (1 - self.opt_beta2_decay)
            # opt_v_hat = np.array(opt_v_hat, dtype=np.float64)
            weight_diff = u_coef * opt_m_hat / (np.sqrt(opt_v_hat) + 1e-8) * u
        else:
            weight_diff = u_coef * u

        # check loss reduction
        loss_reduction = self._check_loss_reduction()

        # if current loss reduce, store the momentum
        if loss_reduction:
            self.prev_g = copy.deepcopy(grad)
        else:
            self.prev_g = None

        self.weight = self.weight - weight_diff

    def _conjugate_denominator(self, X, u):
        alpha_u_square = self.alpha * np.square(np.linalg.norm(u))

        aii_u_x = 0
        for i in range(self.sample_num):
            xi = X[[i], :].transpose()
            sigmoid_val = self._sigmoid(self.weight.transpose().dot(xi))
            aii = sigmoid_val * (1 - sigmoid_val)
            aii_u_x += aii * np.square(u.transpose().dot(xi)[0, 0])

        return -(alpha_u_square + aii_u_x)

    def _init_params(self, X, y):
        # self.weight = np.random.rand(X.shape[1], 1)
        self.weight = np.zeros((X.shape[1], 1))
        self.sample_num = X.shape[0]
        self.dim = X.shape[1]
        self.pos_num = np.count_nonzero(y == 1)

        if self.balance_mode:
            self.balance_weight = (0.5 / (self.pos_num / self.sample_num),
                                   0.5 / ((self.sample_num - self.pos_num) / self.sample_num))
        else:
            self.balance_weight = (1, 1)

        if self.oversampling:
            self.oversampling_rate = (self.sample_num - self.pos_num) / self.pos_num        # neg / pos
        else:
            self.oversampling_rate = 1

        if self.train_method == 'single_lbfgs':
            self.prev_grad = None
            self.prev_weight = None
        elif self.train_method == 'lbfgs':
            self.memory_length = 10
            self.w_array = [None, None]
            self.g_array = [None, None]
            self.dw_array = [None for _ in range(self.memory_length)]
            self.dg_array = [None for _ in range(self.memory_length)]
            self.rho_array = [None for _ in range(self.memory_length)]
        elif self.train_method == 'momentum_lbfgs' or self.train_method == 'smart_momentum_lbfgs':
            self.prev_grad = None
            self.prev_weight = None
            self.prev_weight_diff = None
            self.gamma = 0.9
        elif self.train_method == 'smart_adam_lbfgs':
            self.prev_grad = None
            self.prev_weight = None
            self.prev_g = None      # for adam
            self.opt_beta1 = 0.9
            self.opt_beta2 = 0.999
            self.opt_beta1_decay = 1.0
            self.opt_beta2_decay = 1.0
            self.opt_m = np.zeros((self.dim, 1))
            self.opt_v = np.zeros((self.dim, 1))
        elif self.train_method == 'cgd':
            self.prev_grad = None
            self.prev_u = None
        elif self.train_method == 'trust_region':
            self.C = 1 / self.alpha         # regularization parameter
            self.alpha_pcg = 1e-2
            self.eps = 1e-1
            self.eps_cg = 5e-1                  # original 5e-1
            self.init_step_size = 1
            self.g0 = self._compute_gradient(X, y)
            self.g_norm0 = np.linalg.norm(self.g0)
            self.f = 0
            self.w_w = np.square(np.linalg.norm(self.weight))
            self.search = True

    def save_metrics(self, file_name):
        """
        Save loss and auc trajectories
        :param file_name: str
        :return:
        """
        dir = r'../data/model_metrics/'
        loss_suffix = r'-loss'
        auc_suffix = r'-auc'

        np.save(dir + file_name + loss_suffix, self.loss_trajectory)
        np.save(dir + file_name + auc_suffix, self.auc_trajectory)

    def shuffle_generator(self):
        shuffled_array = [i for i in range(self.sample_num)]
        np.random.shuffle(shuffled_array)
        for index in shuffled_array:
            yield index

    def _compute_trust_region_loss(self, X, y):
        w_w = np.square(np.linalg.norm(self.weight))
        w_x = X.dot(self.weight)

        f = 0
        for i in range(self.sample_num):
            wxi, yi = w_x[i, 0], y[i, 0]
            f += self._C_times_loss(wxi, yi)
        f += 0.5 * w_w

        return f

    @staticmethod
    def invert_digonal_matrix(diag_matrix):
        new_matrix = np.zeros(diag_matrix.shape)
        for i in range(diag_matrix.shape[0]):
            new_matrix[i, i] = 1 / diag_matrix[i, i]
        return new_matrix

    def _check_loss_reduction(self):
        if len(self.loss_trajectory) < 3:
            return False
        if self.loss_trajectory[-1] - self.loss_trajectory[-2] < 0:
            return True
        else:
            return False

    @staticmethod
    def pop_push(queue, element):
        queue.pop(0)
        queue.append(element)
