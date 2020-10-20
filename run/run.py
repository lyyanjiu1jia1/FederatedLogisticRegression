from algorithms.lr import CentralizedLogisticRegression
import numpy as np
# params
alpha = 1
eta = 0.15
max_iter = 1000
balance_mode = True
train_method_list = ('gd', 'sgd', 'cd', 'cgd', 'lbfgs', 'trust_region', 'liblinear-liblinear', 'liblinear-dual')
train_method = train_method_list[0]

# data preprocessing
file_name = 'breast'        # total_breast, 3_data
X = np.load(r'../data/' + file_name + '-x.npy', allow_pickle=True)
y = np.load(r'../data/' + file_name + '-y.npy', allow_pickle=True)
print("data loaded {}-rows-by-{}-columns".format(X.shape[0], X.shape[1]))

# run the algorithm
print("using {} method for training".format(train_method))
centralized_logistic_regression = CentralizedLogisticRegression(alpha, eta, max_iter, train_method, balance_mode)
centralized_logistic_regression.train(X, y)
auc = centralized_logistic_regression.validate(X, y)
print("final auc = {}".format(auc))
centralized_logistic_regression.save_metrics(file_name)
