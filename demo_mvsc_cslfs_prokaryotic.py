import os
import torch as t
import numpy as np
from utils_gpu import load_data
from models.MVSC_CSLFS import Model

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

name = 'prokaryotic'
X, y, input_shape = load_data(f'data/{name}.mat', method=0)
V = len(X)
X = [t.from_numpy(X[v]).float().cuda() for v in range(V)]

config = {
    'name': name,
    'view_num': V,
    'view_shape': [X[v].shape[0] for v in range(V)],
    'instance_num': X[0].shape[1],
    'clusters': np.max(y)+1,
    'K_s': 50,
    'K_c': 30,
    'epochs': 50,
    'seeds': 3,
    'lambda_1': 0.01,
    'lambda_2': 1,
    'lambda_3': 1,
    'initial_method1': 'zeros',
    'initial_method2': 'zeros',
    'mu': 1,
    'pho': 1.4,
    'max_mu': 1e6,
    'thrsh': 1e-5
    }

model = Model(config, isSave=False)
model.fit(X, y)
