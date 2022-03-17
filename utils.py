import sys
import numpy as np
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import normalize, StandardScaler, MaxAbsScaler, MinMaxScaler, Normalizer
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, f1_score, adjusted_rand_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore') 

def load_data(data_path):
    dir = data_path
    data = loadmat(dir)
    X = np.reshape(data['X'], [data['X'].shape[1]])
    gt = np.reshape(data['gt'], [data['gt'].shape[0], ])
    V = len(X)

    if data_path == 'data/reuters':
      X = [StandardScaler().fit_transform(X[v]) for v in range(V)]
    else:
      # X = [X[v]/np.tile(np.sqrt(np.sum(X[v]*X[v], 0)), (X[v].shape[0], 1)) for v in range(V)]
      X = [MinMaxScaler().fit_transform(X[v].T) for v in range(V)]
      # X = [Normalizer().fit_transform(X[v].T) for v in range(V)]
      # X = [StandardScaler().fit_transform(X[v].T) for v in range(V)]
      X = [X[v].T for v in range(V)]
    
    y = data['gt']
    y = np.reshape(y, [-1, ])
    
    if data_path == 'data/prokaryotic.mat' or data_path == 'data/3-sources.mat':
        X, y  = rank(X, y)
    
    input_shape = [X[v].shape[0] for v in range(V)]
    
    return X, y-1, input_shape

def solve_orthonormal(G, Q):
   """ solve the optimal problem with orthonormal constriant
   
   min_R |Q - GR|_F^2 \n
   s.t. R'R = RR' = I

   References
   ----------
   J. Huang, F. Nie, and H. Huang. Spectral rotation versus \n
   k-means in spectral clustering. In AAAI, 2013.
   """

   W = G@Q
   U, S, V = np.linalg.svd(W, full_matrices=False)

   return np.transpose(np.matmul(U, V))



def solve_l21(Y, lambda_):
   """ solve the optimal problem with L21 norm

      min 0.5*|X-Y|_F^2 + lambda|X|_21 
   """

   N = Y.shape[1]
   E = Y
   for n in range(N):
      E[:, n] = solve_l2_pro(Y[:, n], lambda_)

   return E


def solve_l2_pro(y, lambda_):
   """ slove the proximal problrm with L2 norm

      min 0.5*|x-y|_2^2 + lambda*|x|_2
   """

   ny = np.linalg.norm(y)

   if ny > lambda_:
      x = (ny - lambda_)*y/ny
   else:
      x = np.zeros(y.shape)

   return x

def Hungarian(A):
    """ HUNGARIAN Solve the Assignment problem using the Hungarian method.
    """
   
    _, col_ind = linear_sum_assignment(A)
    # Cost can be found as A[row_ind, col_ind].sum()
    return col_ind

def BestMap(L1, L2):
    """ permute labels of L2 to match L1 as good as possible
    """
    L1 = L1.flatten(order='F').astype(float)
    L2 = L2.flatten(order='F').astype(float)
    if L1.size != L2.size:
        sys.exit('size(L1) must == size(L2)')
    Label1 = np.unique(L1)
    nClass1 = Label1.size
    Label2 = np.unique(L2)
    nClass2 = Label2.size
    nClass = max(nClass1, nClass2)

    # For Hungarian - Label2 are Workers, Label1 are Tasks.
    G = np.zeros([nClass, nClass]).astype(float)
    for i in range(0, nClass2):
        for j in range(0, nClass1):
            G[i, j] = np.sum(np.logical_and(L2 == Label2[i], L1 == Label1[j]))

    c = Hungarian(-G)
    newL2 = np.zeros(L2.shape)
    for i in range(0, nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2

def f_p_r_score(gt_s, s):
    N = len(gt_s)
    num_t = 0
    num_h = 0
    num_i = 0
    for n in range(N-1):
        tn = (gt_s[n] == gt_s[n+1:]).astype('int')
        hn = (s[n] == s[n+1:]).astype('int')
        num_t += np.sum(tn)
        num_h += np.sum(hn)
        num_i += np.sum(tn * hn)
    p = r = f = 1
    if num_h > 0:
        p = num_i / num_h
    if num_t > 0:
        r = num_i / num_t
    if p + r == 0:
        f = 0
    else:
        f = 2 * p * r / (p + r)
    return f, p, r

def metrics_score(y_true, y_predict):
   acc = accuracy_score(y_true, y_predict)
   nmi = normalized_mutual_info_score(y_true, y_predict)
   RI = adjusted_rand_score(y_true, y_predict)
   precision, recall, f = f_p_r_score(y_true, y_predict)
   return acc, nmi, RI, precision, recall, f