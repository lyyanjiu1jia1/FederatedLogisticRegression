from algorithms.lr import CentralizedLogisticRegression
import numpy as np

# params
alpha = 1
eta = 0.15
max_iter = 10000
balance_mode = True
oversampling = False
train_method_list = ('gd', 'l_gd', 'sgd', 'cd',
                     'cgd', 'single_lbfgs', 'lbfgs', 'momentum_lbfgs',
                     'smart_momentum_lbfgs', 'smart_adam_lbfgs',
                     'trust_region',
                     'liblinear-liblinear', 'liblinear-dual')
train_method = train_method_list[6]

# data preprocessing
file_name = 'total_data'        # breast, 3_data, total_data
X = np.load(r'../data/' + file_name + '-x.npy', allow_pickle=True)
y = np.load(r'../data/' + file_name + '-y.npy', allow_pickle=True)
print("data loaded {}-rows-by-{}-columns".format(X.shape[0], X.shape[1]))

# run the algorithm
print("using {} method for training".format(train_method))
centralized_logistic_regression = CentralizedLogisticRegression(alpha, eta, max_iter, train_method,
                                                                balance_mode, oversampling)
centralized_logistic_regression.train(X, y)
auc = centralized_logistic_regression.validate(X, y)
print("final auc = {}".format(auc))
centralized_logistic_regression.save_metrics(file_name)
