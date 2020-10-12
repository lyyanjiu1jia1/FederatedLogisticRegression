from algorithms.lr import CentralizedLogisticRegression
import numpy as np
# params
alpha = 0.1
eta = 0.001
max_iter = 300
train_method_list = ('gd', 'cd', 'lbfgs', 'liblinear-liblinear', 'liblinear-dual')
train_method = train_method_list[2]

# data preprocessing
file_name = 'breast'
X = np.load(r'../data/' + file_name + '-x.npy', allow_pickle=True)
y = np.load(r'../data/' + file_name + '-y.npy', allow_pickle=True)
print("data loaded")

# run the algorithm
centralized_logistic_regression = CentralizedLogisticRegression(alpha, eta, max_iter, train_method)
centralized_logistic_regression.train(X, y)
auc = centralized_logistic_regression.validate(X, y)
print("final auc = {}".format(auc))
centralized_logistic_regression.save_metrics(file_name)
