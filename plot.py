import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets, decomposition
from utils_gpu import load_data
from scipy.io import loadmat

data_name = '3-sources'
X, Y, input_shape = load_data('data/{}.mat'.format(data_name))
y = Y

data = loadmat(f'save/training_log/lowrank-unsupervised/{data_name}/model_parameters/1.mat')

X = [x.T for x in X]
lantent = data['H_2nd'][600:,:].T

fig = plt.figure(num=4, figsize=(15, 8),dpi=80)
#使用add_subplot在窗口加子图，其本质就是添加坐标系
#三个参数分别为：行数，列数，本子图是所有子图中的第几个，最后一个参数设置错了子图可能发生重叠
ax = []
# ax.append(fig.add_subplot(2,2,1, projection='3d'))
# ax.append(fig.add_subplot(2,2,2, projection='3d'))
# ax.append(fig.add_subplot(2,2,3, projection='3d'))
# ax.append(fig.add_subplot(2,2,4, projection='3d'))
ax.append(fig.add_subplot(2,2,1))
ax.append(fig.add_subplot(2,2,2))
ax.append(fig.add_subplot(2,2,3))
ax.append(fig.add_subplot(2,2,4))

for v in range(3):
    n_samples, n_features = X[v].shape

    '''t-SNE'''
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=3)
    X_tsne = tsne.fit_transform(X[v])
    # X_pca = decomposition.TruncatedSVD(n_components=3).fit_transform(X[v])
    # X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X[v])
    # X_tsne = X_pca

    print("Org data dimension is {}. \
          Embedded data dimension is {}".format(X[v].shape[-1], X_tsne.shape[-1]))

    '''嵌入空间可视化'''
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化

    for i in range(X_norm.shape[0]):
        ax[v].text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), 
                  fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title('View_{}'.format(v+1))

n_samples, n_features = lantent.shape

'''t-SNE'''
tsne = manifold.TSNE(n_components=2, init='pca', random_state=3)
X_tsne = tsne.fit_transform(lantent)
# X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(lantent)
# X_tsne = X_pca

print("Org data dimension is {}. \
      Embedded data dimension is {}".format(lantent.shape[-1], X_tsne.shape[-1]))

'''嵌入空间可视化'''
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化

for i in range(X_norm.shape[0]):
    ax[3].text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), 
             fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])

plt.title('Joint lantent representation')
dirs = 'save/figure/'
if not os.path.exists(dirs):
    os.makedirs(dirs)
plt.savefig(dirs + "t_sne_{}.jpg".format(data_name))
plt.show()