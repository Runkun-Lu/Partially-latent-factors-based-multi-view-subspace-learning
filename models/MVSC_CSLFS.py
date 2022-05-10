import os
from time import time, strftime, localtime
import cvxpy as cvx
from cvxpy.error import SolverError
from control import lyap 
import numpy as np
import torch as t
import pandas as pd
from scipy.io import savemat
from sklearn.cluster import SpectralClustering

import sys
sys.path.append('..')
from utils_gpu import solve_orthonormal, solve_l21, BestMap, metrics_score, sigma_soft_thresh, soft_thresh

class Model():
    """ Multi-view subspace clustering with consistent and view specific latent factors and subspaces
    """
    def __init__(self, config, isTestPerformancePerEpoch=True, isSave=True, isMultiEpochsTest=False, kernel=None):
        
        self.config = config
        self.name = config['name']
        self.V = config['view_num']
        self.N = config['instance_num']
        self.view_shape = config['view_shape']
        self.clusters = config['clusters']
        self.epochs = config['epochs']
        self.K_s = config['K_s']
        self.K_c = config['K_c']
        self.seeds = config['seeds']
        self.lambda_1 = config['lambda_1']
        self.lambda_2 = config['lambda_2']
        self.lambda_3 = config['lambda_3']
        self.initial_method1 = config['initial_method1']
        self.initial_method2 = config['initial_method2']
        self.mu = config['mu']
        self.pho = config['pho']
        self.max_mu = config['max_mu']
        self.thrsh = config['thrsh']
                
        self.kernel = kernel
        self.isTestPerformancePerEpoch, self.isSave, self.isMultiEpochsTest = isTestPerformancePerEpoch, isSave, isMultiEpochsTest
        if self.isMultiEpochsTest:
            self.isSave = False

    def fit(self, X, y=None):
        
        mu, pho, max_mu, thrsh = self.mu, self.pho, self.max_mu, self.thrsh

        # initialize variables
        print("Initialize Variables:")
        start_time = time()
        pi_1, pi_2, P_s, P_c, H_s, H_c, Z, Z_c, D, D_c, E_r, E_s, E_s_c, Lambda1, Lambda2, Lambda3, Lambda4, Lambda5 = self._initialization()
        end_time = time()
        print(f'Finised the initialization process in {end_time-start_time} seconds')

        # traning is start
        print('Start Training:')
        records1 = {f'cond{i}': [] for i in range(1, 6)}
        records2 = {'acc':[], 'nmi':[], 'RI':[], 'precision':[], 'recall':[], 'f':[], 'time':[]}
        records = dict(records1, **records2)
        best_acc, best_acc2, best_nmi, best_nmi2, best_cond = 0, 0, 0, 0, 1e10
        best_model = {'H': None, 'affinity': None, 'best_epoch': 0}
        
        total_start = time()
        for epoch in range(self.epochs):
            
            epoch_start = time()

            # updating P_s^v and P_c^v
            P_s = self._P_s_solver(X, H_s, H_c, P_s, P_c, mu, Lambda1, E_r)
            P_c = self._P_c_solver(X, H_s, H_c, P_s, P_c, mu, Lambda1, E_r)       

            # updating H_s^v and H_c
            H_s = self._H_s_solver(X, H_s, H_c, P_s, P_c, mu, Lambda1, Lambda2, Z, E_r, E_s)
            H_c = self._H_c_solver(X, H_s, H_c, P_s, P_c, mu, Lambda1, Lambda3, Z, Z_c, E_r, E_s, E_s_c)
            H = [H_s[f'view_{v}'] for v in range(self.V)]
            H.append(H_c)
            H = t.cat(H, 0)

            # updating Z and Z_c
            Z = self._Z_solver(Z, H_s, H_c, Lambda2, Lambda4, E_s, D, mu)
            Z_c = self._Z_c_solver(H_c, Lambda3, Lambda5, D_c, E_s_c, mu)

            # updating D and D_c
            D = self._D_solver(D, Z, Lambda3, Lambda4, mu, pi_2)
            D_c = self._D_c_solver(Z_c, Lambda5, mu, pi_2)
            self.Z, self.D, self.Z_c, self.D_c = Z, D, Z_c, D_c
            
            # updating E_r^v, E_s(or E_s^v), E_s_c
            E_r = self._E_r_solver(E_r, Lambda1, X, P_s, P_c, H_s, H_c, mu, pi_1)
            E_s = self._E_s_solver(Lambda2, E_s, H, H_s, H_c, Z, mu, pi_2)
            E_s_c = self._E_s_c_solver(Lambda3, H_c, Z_c, mu, pi_2)
           
            # updating pi_1 and pi_2
            pi_1 = self._pi_1_solver(E_r)
            pi_2 = self._pi_2_solver(E_s, E_s_c, D, D_c)

            # updating lagrangian multipliers
            Lambda_list = [Lambda1, Lambda2, Lambda3, Lambda4, Lambda5]
            Lambda1, Lambda2, Lambda3, Lambda4, Lambda5 = self._Lambda_solver(Lambda_list, \
                X, P_s, P_c, H, H_s, H_c, E_r, E_s, E_s_c, Z, Z_c, D, D_c, mu)

            # update mu
            if self.name=='3-sources' and mu > 1e-1:
                pho = 2.5          
            mu = min(pho*mu, max_mu)

            epoch_end = time()
            records['time'].append(epoch_end-epoch_start)

            # convergence condition and the comuptation of different loss
            cond1_tmp = [X[v] - P_s[f'view_{v}'] @ H_s[f'view_{v}'] - P_c[f'view_{v}'] @ H_c - E_r[f'view_{v}'] for v in range(self.V)]
            cond1 = sum([t.norm(cond1_tmp[v], float('inf')) for v in range(self.V)])/self.V
            
            records['cond1'].append(cond1.cpu().numpy())

            cond2_tmp = [H_s[f'view_{v}'] - H_s[f'view_{v}'] @ Z[f'view_{v}'] - E_s[f'view_{v}'] for v in range(self.V)]
            cond3_tmp = H_c - H_c @ Z_c - E_s_c
            cond4_tmp = [D[f'view_{v}'] - Z[f'view_{v}'] for v in range(self.V)]
            cond5_tmp = D_c - Z_c

            cond2 = sum([t.norm(cond2_tmp[v], float('inf')) for v in range(self.V)])/self.V
            cond3 = t.norm(cond3_tmp, float('inf'))
            cond4 = sum([t.norm(cond4_tmp[v], float('inf')) for v in range(self.V)])/self.V
            cond5 = t.norm(cond5_tmp, float('inf'))
            
            records['cond2'].append(cond2.cpu().numpy())
            records['cond3'].append(cond3.cpu().numpy())
            records['cond4'].append(cond4.cpu().numpy())
            records['cond5'].append(cond5.cpu().numpy())

            IsConverge = True if (cond1 <= thrsh and cond2 <= thrsh and cond3 <= thrsh and cond4 <= thrsh and cond5 <= thrsh) else False
            del cond1_tmp, cond2_tmp, cond3_tmp, cond4_tmp, cond5_tmp
            cond_sum = cond1 + cond2 + cond3 + cond4 + cond5
            cond = t.tensor([cond1.cpu(), cond2.cpu(), cond3.cpu(), cond4.cpu(), cond5.cpu()])
            
            cond_mean = t.mean(cond)
            
            # process visualization
            if self.isTestPerformancePerEpoch == True:
                y_hat = self.predict()
                y_new = BestMap(y, y_hat+1)
                acc, nmi, RI, precision, recall, f = metrics_score(y, y_new)
                if self.isSave:
                    records['acc'].append(acc)
                    records['nmi'].append(nmi)
                    records['RI'].append(RI)
                    records['precision'].append(precision)
                    records['recall'].append(recall)
                    records['f'].append(f)
                if epoch >= 5 and acc+nmi >= best_acc+best_nmi:
                    best_acc2, best_nmi2 = best_acc, best_nmi
                    best_acc, best_nmi = acc, nmi
                    if self.isMultiEpochsTest:
                        best_RI, best_precision, best_recall, best_f = RI, precision, recall, f
                    if self.isSave:
                        best_model['H_2nd'] = best_model['H'] if epoch != 0 else None
                        best_model['affinity_2nd'] = best_model['affinity'] if epoch != 0 else None
                        best_model['second_epoch'] = best_model['best_epoch'] if epoch != 0 else 0
                        best_model['H'] = H.cpu().numpy()
                        best_model['affinity'] = self._affinity_matrix()
                        best_model['best_epoch'] = epoch
                elif acc+nmi < best_acc+best_nmi and acc+nmi >= best_acc2+best_nmi2:
                    best_acc2, best_nmi2 = acc, nmi
                    if self.isSave:
                        best_model['H_2nd'] = H.cpu().numpy()
                        best_model['affinity_2nd'] = self._affinity_matrix()
                        best_model['second_epoch'] = epoch
                cond_temp = cond.tolist()
                print(f'epoch{epoch}||mu:{mu:.6f}||condions:{cond_temp[0]:.6f},{cond_temp[1]:.6f},{cond_temp[2]:.6f},{cond_temp[3]:.6f},{cond_temp[4]:03f}||acc:{acc:.3f}||nmi:{nmi:.3f}||AR:{RI:.3f}||precision:{precision:.3f}||recall:{recall:.3f}||f-score:{f:.3f}||best acc:{best_acc:.3f}||best nmi:{best_nmi:.3f}')
            else:
                if cond_mean < best_cond:
                    best_cond = cond_mean
                    if self.isSave:
                        best_model['H'] = H.cpu().numpy()
                        best_model['affinity'] = self._affinity_matrix()
                print(f'epoch{epoch}||condions:{cond.tolist()}')

            if self.isMultiEpochsTest and epoch >= 35:
                if self.name == 'reuters' and acc < 0.4:
                    return best_acc, best_acc2, acc, best_nmi, best_RI, best_precision, best_recall, best_f
                elif self.name != 'reuters' and self.name != '3-sources' and acc < 0.65:
                    return best_acc, best_acc2, acc, best_nmi, best_RI, best_precision, best_recall, best_f
                elif self.name == '3-sources' and acc < 0.55:
                    return best_acc, best_acc2, acc, best_nmi, best_RI, best_precision, best_recall, best_f
                
            elif self.isMultiEpochsTest and epoch >= 5:
                if self.name != '3-sources' and acc < 0.25:
                    return best_acc, best_acc2, acc, best_nmi, best_RI, best_precision, best_recall, best_f
                elif self.name == '3-sources' and acc < 0.3:
                    return best_acc, best_acc2, acc, best_nmi, best_RI, best_precision, best_recall, best_f


            if IsConverge:
                total_end = time()
                print(f'Algorithm converges after {epoch+1} epochs, and the total training time is {total_end-total_start}')
                if self.isSave:
                    df = pd.DataFrame(records)
                    self._save_files(df, best_model)
                if self.isMultiEpochsTest:
                    return best_acc, best_acc2, acc, best_nmi, best_RI, best_precision, best_recall, best_f
                else:
                    break
        if self.isSave:
            df = pd.DataFrame(records)
            self._save_files(df, best_model)
        total_end = time()            
        print(f'Algorithm converges after {epoch+1} epochs, and the total training time is {total_end-total_start}')
        if self.isMultiEpochsTest:
            return best_acc, best_acc2, acc, best_nmi, best_RI, best_precision, best_recall, best_f

    def predict(self):
        """ clustering using spectral clustering
        """
        affinity = self._affinity_matrix()
        spectral = SpectralClustering(n_clusters=self.clusters, affinity='precomputed', assign_labels='discretize')
        spectral.fit(affinity)
        clusters = spectral.fit_predict(affinity)
        return clusters
    
    def _initialization(self):
        """ initialize the variables
        """
        
        t.manual_seed(self.seeds)
        
        pi_1 = [1./self.V]*self.V
        pi_2 = [1./(self.V+1)]*(self.V+1)

        P_s, P_c, H_s, E_r, Lambda1 = {}, {}, {}, {}, {} 
        for v in range(self.V):
            P_s[f'view_{v}'] = self._initial(self.view_shape[v], self.K_s, self.initial_method1)
            P_c[f'view_{v}'] = self._initial(self.view_shape[v], self.K_c, self.initial_method1)
            H_s[f'view_{v}'] = self._initial(self.K_s, self.N, self.initial_method1)
            E_r[f'view_{v}'] = t.zeros([self.view_shape[v], self.N]).cuda()
            Lambda1[f'view_{v}'] = t.zeros([self.view_shape[v], self.N]).cuda()
        H_c = self._initial(self.K_c, self.N, self.initial_method1)     
        K = self.K_s*self.V + self.K_c
               
        D, Z, E_s, Lambda2, Lambda4 = {}, {}, {}, {}, {}
        for v in range(self.V):
            Z[f'view_{v}'] = self._initial(self.N, self.N, self.initial_method2)
            if self.initial_method2 != 'zeros':
                Z[f'view_{v}'] = (t.abs(Z[f'view_{v}']) + t.abs(Z[f'view_{v}']))/2.
                Z[f'view_{v}'] = (Z[f'view_{v}'] - t.diag(t.diag(Z[f'view_{v}'])))
            D[f'view_{v}'] = Z[f'view_{v}']
            E_s[f'view_{v}'] = t.zeros([self.K_s, self.N]).cuda()
            Lambda2[f'view_{v}'] = t.zeros([self.K_s, self.N]).cuda()
            Lambda4[f'view_{v}'] = t.zeros([self.N, self.N]).cuda()
        Lambda3 = t.zeros([self.K_c, self.N]).cuda()
        Lambda5 = t.zeros([self.N, self.N]).cuda()
        Z_c = self._initial(self.N, self.N, self.initial_method2)
        if self.initial_method2 != 'zeros':
            Z_c = (t.abs(Z_c) + t.abs(Z_c))/2.
            Z_c = (Z_c - t.diag(t.diag(Z_c)))
        D_c = Z_c
        E_s_c = t.zeros([self.K_c, self.N]).cuda()
        
        return pi_1, pi_2, P_s, P_c, H_s, H_c, Z, Z_c, D, D_c, E_r, E_s, E_s_c, Lambda1, Lambda2, Lambda3, Lambda4, Lambda5

    def _initial(self, shape1, shape2, method):
        if method == 'zeros':
            return t.zeros([shape1, shape2]).cuda()
        elif method == 'ones':
            return t.ones([shape1, shape2]).cuda()
        elif method == 'uniform':
            return t.rand(shape1, shape2).cuda()
        elif method == 'normal':
            return t.randn(shape1, shape2).cuda()

    def _P_s_solver(self, X, H_s, H_c, P_s, P_c, mu, Lambda1, E_r):
        """ updating P_s^v
        """
        
        for v in range(self.V):
            G = H_s[f'view_{v}']            
            Q = Lambda1[f'view_{v}']/mu + X[v] - P_c[f'view_{v}'] @ H_c - E_r[f'view_{v}']
            P_s[f'view_{v}'] = solve_orthonormal(G, Q.T)
        return P_s

    def _P_c_solver(self, X, H_s, H_c, P_s, P_c, mu, Lambda1, E_r):
        """ updating P_c^v
        """
        G = H_c
        for v in range(self.V):          
            Q = Lambda1[f'view_{v}']/mu + X[v] - P_s[f'view_{v}'] @ H_s[f'view_{v}'] - E_r[f'view_{v}']
            P_c[f'view_{v}'] = solve_orthonormal(G, Q.T)
        return P_c

    def _H_s_solver(self, X, H_s, H_c, P_s, P_c, mu, Lambda1, Lambda2, Z, E_r, E_s):
        """ updating H_s^v
        """
      
        epsilon1 = 1e-6*t.eye(self.K_s).cuda()  # to avoid the matrix equals to zero
        epsilon2 = 1e-6*t.eye(self.N).cuda()

        for v in range(self.V):
            A = P_s[f'view_{v}'].T @ P_s[f'view_{v}']
            A += epsilon1

            tmp = t.eye(self.N).cuda() - Z[f'view_{v}']
            B = tmp @ tmp.T + epsilon2

            C = P_s[f'view_{v}'].T @ (Lambda1[f'view_{v}']/mu + X[v] - P_c[f'view_{v}'] @ H_c - E_r[f'view_{v}']) \
                - (Lambda2[f'view_{v}']/mu - E_s[f'view_{v}']) @ tmp.T
            
            H_s[f'view_{v}'] = t.from_numpy(lyap(A.cpu().numpy(), B.cpu().numpy(), C.cpu().numpy())).float().cuda()

        return H_s

    def _H_c_solver(self, X, H_s, H_c, P_s, P_c, mu, Lambda1, Lambda3, Z, Z_c, E_r, E_s, E_s_c):
        """ updating H_c
        """
        sum_P_c = 0
        sum_long = 0
        for v in range(self.V):
            sum_P_c += P_c[f'view_{v}'].T @ P_c[f'view_{v}']
            sum_long += P_c[f'view_{v}'].T @ (Lambda1[f'view_{v}']/mu + X[v] - P_s[f'view_{v}'] @ H_s[f'view_{v}'] - E_r[f'view_{v}'])
        
        A = sum_P_c + 1e-6*t.eye(self.K_c).cuda() 

        tmp = t.eye(self.N).cuda() - Z_c
        B = tmp @ tmp.T + 1e-6*t.eye(self.N).cuda()

        C = sum_long - (Lambda3/mu - E_s_c) @ tmp.T

        return t.from_numpy(lyap(A.cpu().numpy(), B.cpu().numpy(), C.cpu().numpy())).float().cuda()

    def _Z_solver(self, Z, H_s, H_c, Lambda2, Lambda4, E_s, D, mu):
        """updating Z or Z^v
        """
        for v in range(self.V):
            left = t.inverse(H_s[f'view_{v}'].T @ H_s[f'view_{v}'] + t.eye(self.N).cuda())
            right = Lambda4[f'view_{v}']/mu + H_s[f'view_{v}'].T @ Lambda2[f'view_{v}']/mu + D[f'view_{v}'] \
                    - H_s[f'view_{v}'].T @ E_s[f'view_{v}'] + H_s[f'view_{v}'].T @ H_s[f'view_{v}']
            Z[f'view_{v}'] = left @ right
            Z[f'view_{v}'] -= t.diag(t.diag(Z[f'view_{v}']))

        return Z

    def _Z_c_solver(self, H_c, Lambda3, Lambda5, D_c, E_s_c, mu):
        """ updating Z_c
        """
        left = t.inverse(H_c.T @ H_c + t.eye(self.N).cuda())
        right = Lambda5/mu + H_c.T @ Lambda3/mu + D_c - H_c.T @ E_s_c + H_c.T @ H_c
        Z_c = left @ right
        Z_c -= t.diag(t.diag(Z_c))
        return Z_c

    def _D_solver(self, D, Z, Lambda3, Lambda4, mu, pi_2):
        """ updating auxiliary variable D
        """
        for v in range(self.V):
            D[f'view_{v}'] = (mu*Z[f'view_{v}'] - Lambda4[f'view_{v}'])/(self.lambda_2*pi_2[v] + mu)
            # G = Z[f'view_{v}'] - Lambda4[f'view_{v}']/mu
            # mu_hat = self.lambda_2*pi_2[v]/2/mu
            # D[f'view_{v}'] = soft_thresh(G, mu_hat)
        return D

    def _D_c_solver(self, Z_c, Lambda5, mu, pi_2):
        """ updating auxiliary variable D_c
        """
        G = Z_c - Lambda5/mu + 1e-8*t.eye(self.N).cuda()
        mu_hat = (self.lambda_2*pi_2[self.V])/mu
        # mu_hat = mu/(self.lambda_2)

        return sigma_soft_thresh(G, mu_hat)

    def _E_r_solver(self, E_r, Lambda1, X, P_s, P_c, H_s, H_c, mu, pi_1):
        """ updating E_r^v
        """
        for v in range(self.V):
            G = Lambda1[f'view_{v}']/mu + X[v] - P_s[f'view_{v}'] @ H_s[f'view_{v}'] - P_c[f'view_{v}'] @ H_c
            mu_hat = pi_1[v]/mu
            E_r[f'view_{v}'] = solve_l21(G, mu_hat)
        return E_r

    def _E_s_solver(self, Lambda2, E_s, H, H_s, H_c, Z, mu, pi_2):
        """ updating E_s or E_s^v
        """

        for v in range(self.V):
            G = Lambda2[f'view_{v}']/mu + H_s[f'view_{v}'] - H_s[f'view_{v}'] @ Z[f'view_{v}']
            mu_hat = (self.lambda_1*pi_2[v])/mu
            E_s[f'view_{v}'] = solve_l21(G, mu_hat)
        
        return E_s
    
    def _E_s_c_solver(self, Lambda3, H_c, Z_c, mu, pi_2):
        """ updating E_s_c
        """
        G = Lambda3/mu + H_c - H_c @ Z_c
        mu_hat = (self.lambda_1*pi_2[self.V])/mu

        return solve_l21(G, mu_hat)

    def _pi_1_solver(self, E_r):
        """ updating pi_1
        """
        pi_tmp = cvx.Variable(self.V)
        E_tmp = [(self._L_21_norm(E_r[f'view_{v}']).cpu().numpy()) for v in range(self.V)]
        objective = cvx.Minimize(cvx.sum(cvx.multiply(pi_tmp, E_tmp)) + self.lambda_3/2*cvx.power(cvx.norm2(pi_tmp), 2))
        constraints = [cvx.sum(pi_tmp)==1, pi_tmp >= 0.01]

        prob = cvx.Problem(objective, constraints)
        prob.solve(verbose=False)
        
        tmp = pi_tmp.value
        pi_1 = [tmp[v] for v in range(self.V)]
        
        return pi_1

    def _pi_2_solver(self, E_s, E_s_c, D, D_c):
        pi_tmp = cvx.Variable(self.V+1)
        const = [self.lambda_1*(self._L_21_norm(E_s[f'view_{v}'])).cpu().numpy() + \
            self.lambda_2/2*(t.norm(D[f'view_{v}'])**2).cpu().numpy() for v in range(self.V)]
        const.append(self.lambda_1*(self._L_21_norm(E_s_c)).cpu().numpy() + self.lambda_2*(t.norm(D_c, 'nuc')).cpu().numpy())
        # const = [self.lambda_1*(self._L_21_norm(E_s[f'view_{v}'])).cpu().numpy() for v in range(self.V)]
        # const.append(self.lambda_1*(self._L_21_norm(E_s_c)).cpu().numpy())
        objective = cvx.Minimize(cvx.sum(cvx.multiply(pi_tmp, const)) + self.lambda_3/2*cvx.power(cvx.norm2(pi_tmp), 2))
        constraints = [cvx.sum(pi_tmp)==1, pi_tmp >= 0.01]

        prob = cvx.Problem(objective, constraints)
        prob.solve(verbose=False)
        
        tmp = pi_tmp.value
        pi_2 = [tmp[v] for v in range(self.V+1)]

        return pi_2

    def _Lambda_solver(self, Lambda_list, X, P_s, P_c, H, H_s, H_c, E_r, E_s, E_s_c, Z, Z_c, D, D_c, mu):
        """ updating Lambda1, Lambda2, Lambda3, Lambda4, Lambda5, Lambda6
        """
        Lambda1, Lambda2, Lambda3 = Lambda_list[0], Lambda_list[1], Lambda_list[2]

        # updating Lambda1
        for v in range(self.V):
            Lambda1[f'view_{v}'] += mu*(X[v] - P_s[f'view_{v}'] @ H_s[f'view_{v}'] - P_c[f'view_{v}'] @ H_c - E_r[f'view_{v}'])    
        
        # updating Lambda2 and Lambda3

        Lambda4, Lambda5 = Lambda_list[3], Lambda_list[4]
        for v in range(self.V):
            Lambda2[f'view_{v}'] += mu*(H_s[f'view_{v}'] - H_s[f'view_{v}'] @ Z[f'view_{v}'] - E_s[f'view_{v}'])
            Lambda4[f'view_{v}'] += mu*(D[f'view_{v}'] - Z[f'view_{v}'])
        Lambda3 += mu*(H_c - H_c @ Z_c - E_s_c)
        Lambda5 += mu*(D_c - Z_c)
           
        return Lambda1, Lambda2, Lambda3, Lambda4, Lambda5
    
    def _affinity_matrix(self):
        """ compute affinity_matrix
        """
        sum_ = 0
        if self.name == 'reuters' or  self.name == 'uci-digit':
            sum_ = 0
            for v in range(self.V):
                sum_ = sum_ + (t.abs(self.Z[f'view_{v}']) + t.abs(self.Z[f'view_{v}'].T))/2.
            sum_ = sum_/self.V + (t.abs(self.Z_c) + t.abs(self.Z_c.T))/2.
            sum_ /= (self.V+1)
        else:
            sum_ = 0
            for v in range(self.V):
                sum_ = sum_ + (t.abs(self.D[f'view_{v}']) + t.abs(self.D[f'view_{v}'].T))/2.
            sum_ = sum_/self.V + (t.abs(self.D_c) + t.abs(self.D_c.T))/2.
            sum_ /= (self.V+1)
        
        return sum_.cpu().numpy()
    
    def _L_21_norm(self, X):
        return sum([t.norm(X[:, i]) for i in range(X.shape[1])])
    
    def _save_files(self, df, best_model):
        
        strtime = strftime('%Y-%m-%d-%H-%M-%S', localtime())
        method_dirs = 'save/training_log/lowrank-unsupervised/'

        dirs1 = method_dirs + self.name + '/model_parameters/'
        dirs2 = method_dirs + self.name + '/training_log/'
        
        if not os.path.exists(dirs1):
            os.makedirs(dirs1)
        if not os.path.exists(dirs2):
            os.makedirs(dirs2)

        best_model.update(self.config)
        savemat(dirs1 + strtime + ".mat", best_model)
        df.to_csv(dirs2 + strtime + '.csv')


    
    

