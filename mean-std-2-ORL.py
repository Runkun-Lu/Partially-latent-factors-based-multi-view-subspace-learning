import os
import torch as t
import numpy as np
import pandas as pd
from time import strftime, localtime
from utils_gpu import load_data
from models.MVSC_CSLFS import Model

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def metrics_mean_std(X, y, config, rounds=10):
    metrics = {
        'acc': [],
        'acc2': [],
        'acc_last': [],
        'nmi': [],
        'RI': [],
        'precision': [],
        'recall': [],
        'f': []
    }
    for _ in range(rounds):
        model = Model(config, isMultiEpochsTest=True)
        acc, acc2, acc_last, nmi, RI, precision, recall, f = model.fit(X, y)
        metrics['acc'].append(acc)
        metrics['acc2'].append(acc2)
        metrics['acc_last'].append(acc_last)
        metrics['nmi'].append(nmi)
        metrics['RI'].append(RI)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f'].append(f)
    strtime = strftime('%Y-%m-%d-%H-%M-%S', localtime())
    method_dirs = 'save/result-mean-std/lowrank-unsupervised/'

    dirs = method_dirs + config['name'] + '/'
    
    if not os.path.exists(dirs):
        os.makedirs(dirs)
        
    df = pd.DataFrame(metrics)
    df.to_csv(dirs + strtime + '.csv')

name = 'ORL_mtv'
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
    'K_c': 200,
    'epochs': 100,
    'seeds': 3,
    'lambda_1': 10,
    'lambda_2': 10,
    'lambda_3': 0.5,
    'initial_method1': 'zeros',
    'initial_method2': 'zeros',
    'mu': 1e-6,
    'pho': 1.9,
    'max_mu': 1e6,
    'thrsh': 1e-4
    }

metrics_mean_std(X, y, config)
# model = Model(config)
# model.fit(X, y)