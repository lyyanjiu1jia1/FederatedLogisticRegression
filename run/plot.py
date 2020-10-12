from matplotlib import pyplot as plt
import numpy as np

# settings
epoch_range = (0, 3000)
auc_range = (0.5, 0.75)
loss_range = (0, 3000)

# load data
common_path = r'../data/model_metrics/'
lbfgs_auc = np.load(common_path + 'total_data-lbfgs-auc.npy')[epoch_range[0]:epoch_range[1]]
lbfgs_loss = np.load(common_path + 'total_data-lbfgs-loss.npy')[epoch_range[0]:epoch_range[1]]
gd_auc = np.load(common_path + 'total_data-gd-auc.npy')[epoch_range[0]:epoch_range[1]]
gd_loss = np.load(common_path + 'total_data-gd-loss.npy')[epoch_range[0]:epoch_range[1]]
cd_auc = np.load(common_path + 'total_data-cd-auc.npy')[epoch_range[0]:epoch_range[1]]
cd_loss = np.load(common_path + 'total_data-cd-loss.npy')[epoch_range[0]:epoch_range[1]]
cgd_auc = np.load(common_path + 'total_data-cgd-auc.npy')[epoch_range[0]:epoch_range[1]]
cgd_loss = np.load(common_path + 'total_data-cgd-loss.npy')[epoch_range[0]:epoch_range[1]]

# plot auc
plt.figure(0)
plt.plot(range(len(gd_auc)), gd_auc, linewidth=1.5, color='r', label='Gradient Descent')
plt.plot(range(len(lbfgs_auc)), lbfgs_auc, linewidth=1.5, color='b', label='LBFGS')
plt.plot(range(len(cd_auc)), cd_auc, linewidth=1.5, color='g', label='Coordinate Descent')
plt.plot(range(len(cgd_auc)), cgd_auc, linewidth=1.5, color='y', label='Conjugate Gradient Descent')
plt.legend()
plt.title("AUC v.s. Epoch")
plt.xlabel('epoch')
plt.ylabel('AUC')
plt.xlim(epoch_range)
plt.ylim(auc_range)
plt.grid(True)
plt.show()

# plot loss
plt.figure(1)
plt.plot(range(len(gd_loss)), gd_loss, linewidth=1, color='r', label='Gradient Descent')
plt.plot(range(len(lbfgs_loss)), lbfgs_loss, linewidth=1, color='b', label='LBFGS')
plt.plot(range(len(cd_loss)), cd_loss, linewidth=1, color='g', label='Coordinate Descent')
plt.plot(range(len(cgd_loss)), cgd_loss, linewidth=1, color='y', label='Conjugate Gradient Descent')
plt.legend()
plt.title("Loss v.s. Epoch")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.xlim(epoch_range)
# plt.ylim(loss_range)
plt.grid(True)
plt.show()
