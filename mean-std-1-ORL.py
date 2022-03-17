import os
import torch as t
import numpy as np
import pandas as pd
from time import strftime, localtime
from utils_gpu import load_data
from models.MVSC_CSLF import Model

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
    method_dirs = 'save/result-mean-std/standard-unsupervised/'

    dirs = method_dirs + config['name'] + '/'
    
    if not os.path.exists(dirs):
        os.makedirs(dirs)
        
    df = pd.DataFrame(metrics)
    df.to_csv(dirs + strtime + '.csv')

name = 'ORL_mtv'
X, y, input_shape = load_data(f'data/{name}.mat', method=1)
V = len(X)
X = [t.from_numpy(X[v]).float().cuda() for v in range(V)]

config = {
    'name': name,
    'view_num': V,
    'view_shape': [X[v].shape[0] for v in range(V)],
    'instance_num': X[0].shape[1],
    'clusters': np.max(y)+1,
    'K_s': 10,
    'K_c': 50,
    'epochs': 100,
    'seeds': 3,
    'lambda_1': 0.1,
    'lambda_2': 0.1,
    'lambda_3': 1,
    'initial_method1': 'zeros',
    'initial_method2': 'zeros',
    'mu': 1e-2,
    'pho': 1.6,
    'max_mu': 1e6,
    'thrsh': 1e-4
}

metrics_mean_std(X, y, config)
# model = Model(config, isSave=False)
# model.fit(X, y)