import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets, decomposition
from utils_gpu import load_data
from scipy.io import loadmat

def tsne(X, title_name, index):
    '''t-SNE'''
    tsne = manifold.TSNE(n_components=3, init='pca', random_state=3)
    X_tsne = tsne.fit_transform(X)

    print("Org data dimension is {}. Embedded data dimension is {}".format(X[v].shape[-1], X_tsne.shape[-1]))

    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化

    for i in range(X_norm.shape[0]):
        ax[index].text(X_norm[i, 0], X_norm[i, 1], X_norm[i, 2], str(y[i]), color=plt.cm.Set1(y[i]), 
                  fontdict={'weight': 'bold', 'size': 9})
    
    # ax[index].set_title(title_name)

data_name = '3-sources'
X, Y, input_shape = load_data('data/{}.mat'.format(data_name))
y = Y

data = loadmat(f'save/training_log/lowrank-unsupervised/{data_name}/model_parameters/1.mat')

X = [x.T for x in X]
data = data['H'].T
H_s = []
H_s.append(data[:, 0:250])
H_s.append(data[:, 250:500])
H_s.append(data[:, 500:750])
H_c = data[:, 750:]

fig = plt.figure(num=8, figsize=(15, 8), dpi=600)
#使用add_subplot在窗口加子图，其本质就是添加坐标系
#三个参数分别为：行数，列数，本子图是所有子图中的第几个，最后一个参数设置错了子图可能发生重叠
ax = []
ax.append(fig.add_subplot(4,3,1, projection='3d'))
ax.append(fig.add_subplot(4,3,2, projection='3d'))
ax.append(fig.add_subplot(4,3,3, projection='3d'))
ax.append(fig.add_subplot(4,3,4, projection='3d'))
ax.append(fig.add_subplot(4,3,5, projection='3d'))
ax.append(fig.add_subplot(4,3,6, projection='3d'))
ax.append(fig.add_subplot(4,2,7, projection='3d'))
ax.append(fig.add_subplot(4,2,8, projection='3d'))

# visualize each view's raw data
for v in range(3):
    tsne(X[v], 'View_{}'.format(v+1), v)

# visualize each view's latent representation
for v in range(3, 6):
    tsne(H_s[v-3], 'latent representation of view_{}'.format(v-2), v)

# visualize consistent latent representation
tsne(H_c, 'consistent latent representation', 6)

# visualize joint latent representation
tsne(data, 'joint latent representation', 7)

dirs = 'save/figure/'
if not os.path.exists(dirs):
    os.makedirs(dirs)
plt.savefig(dirs + "t_sne_{}_3d.jpg".format(data_name))
plt.show()