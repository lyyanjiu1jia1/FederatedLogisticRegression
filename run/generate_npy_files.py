# data preprocessing
from algorithms import utils
from algorithms.lr import CentralizedLogisticRegression
from sklearn import preprocessing
import numpy as np

file_name = 'breast.csv'
df = utils.read_csv_to_dataframe(file_name)
print("data loaded")
X, y = utils.disassemble(df)
X, y = CentralizedLogisticRegression.data_processing(X, y)
X = preprocessing.MinMaxScaler((-1, 1)).fit_transform(X)
print("data preprocessing complete")
np.save(r'../data/' + file_name.split('.')[0] + '-x.npy', X)
np.save(r'../data/' + file_name.split('.')[0] + '-y.npy', y)
