from collections import defaultdict
import scipy as sp
import scipy
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import os


def ReLU(x, delta=1e-3):
    return (np.maximum(0,x) + delta*(x<0))


class Trainer():
    def __init__(self):
        pass
        
    def trainer(self, ls_X, ls_modelmat, Y, dates=None,  lr=1e-3, max_iter=200, warm_start=False, save_folder=None, name_model='', normalization_loss=1):
        from collections import defaultdict
        import scipy as sp
        import scipy
        from copy import deepcopy
        from tqdm import tqdm
        EPS = np.finfo(np.float64).eps  # machine epsilon
        """
        Performs stable PIRLS iterations to estimate GAM coefficients
    
        Parameters
        ---------
        X : array-like of shape (n_samples, m_features)
            containing input data
        Y : array-like of shape (n,)
            containing target data
        weights : array-like of shape (n,)
            containing sample weights
    
        Returns
        -------
        losses along iterations
        """
    
        epsilon = 1e-6
        
    
        for k, gam in enumerate(self.gam.pygams):
            if False:
                # validate parameters
                gam._validate_params()
            
                # validate data-dependent parameters
                gam._validate_data_dep_params(ls_X[k])
            
                # set up logging
                if not hasattr(gam, 'logs_'):
                    gam.logs_ = defaultdict(list)
        
                gam.statistics_ = {}
                gam.statistics_['n_samples'] = len(Y)
                gam.statistics_['m_features'] = ls_X[k].shape[1]
    
    
        modelmat = np.concatenate(ls_modelmat, axis=1)
        n, m = modelmat.shape
    
        # initialize GLM coefficients if model is not yet fitted
        if True:
            for k,gam in enumerate(self.gam.pygams):
                if (
                    not gam._is_fitted
                    or len(gam.coef_) != gam.terms.n_coefs
                    or not np.isfinite(gam.coef_).all()
                ):
                    # initialize the model
                    gam.coef_ = gam._initial_estimate(Y, gam._modelmat(ls_X[k]))
    
            assert np.isfinite(
                gam.coef_
            ).all(), "coefficients should be well-behaved, but found: {}".format(gam.coef_)
        ls_P = []
        for gam in self.gam.pygams:
            ls_P.append(gam._P().A)
            #ls_S.append(sp.sparse.diags(np.ones(m) * np.sqrt(EPS)))  # improve condition
        # S += gam._H # add any user-chosen minumum penalty to the diagonal
        S = sp.sparse.diags(np.ones(m) * np.sqrt(EPS))
        P = scipy.linalg.block_diag(*ls_P)
        print(P.shape, S.shape)
        # if we dont have any constraints, then do cholesky now
        self.gam.pygams[0].statistics_['m_features'] = ls_X[0].shape[1] * len(self.gam.pygams)
        E = self.gam.pygams[0]._cholesky(S + P, sparse=False, verbose=self.gam.pygams[0].verbose)
        self.gam.pygams[0].statistics_['m_features'] = ls_X[0].shape[1]
        print(self.gam.pygams[0].terms.get_coef_indices(-1))
        min_n_m = np.min([m, n])
        Dinv = np.zeros((min_n_m + m, m)).T
    
    
        y = deepcopy(Y)  # for simplicity
        for k, gam in enumerate(self.gam.pygams):
            lp = gam._linear_predictor(X=ls_X[k])
            mu = gam.link.mu(lp, gam.distribution)
        # W = np.diag(weights)
    
        # # check for weghts == 0, nan, and update
        # mask = gam._mask(W.diagonal())
        # y = y[mask]  # update
        # lp = lp[mask]  # update
        # mu = mu[mask]  # update
        # W = sp.sparse.diags(W.diagonal()[mask])  # update
    
        # PIRLS Wood pg 183
        #pseudo_data = W.dot(gam._pseudo_data(y, lp, mu))
        pseudo_data = y
    
        # # log on-loop-start stats
        # for k in range(self.L):
        #     self.gam.pygams[k]._on_loop_start(vars(self.gam.pygams[k]).add({'y':y, 'mu':mu, 'gam':self.gam.pygams[k]})
    
        WB = modelmat  # common matrix product
    
        if not(warm_start):
            Q, R = np.linalg.qr(WB)
        
            if not np.isfinite(Q).all() or not np.isfinite(R).all():
                raise ValueError(
                    'QR decomposition produced NaN or Inf. ' 'Check X data.'
                )
        
            # need to recompute the number of singular values
            min_n_m = np.min([m, n])
            Dinv = np.zeros((m, min_n_m))
        
            # SVD
            U, d, Vt = np.linalg.svd(np.vstack([R, E]))
        
            # mask out small singular values
            # svd_mask = d <= (d.max() * np.sqrt(EPS))
        
            np.fill_diagonal(Dinv, d**-1)  # invert the singular values
            U1 = U[:min_n_m, :min_n_m]  # keep only top corner of U
        
            # update coefficients
            B = Vt.T.dot(Dinv).dot(U1.T).dot(Q.T)
    
    
        Q = (WB.T).dot(WB) / normalization_loss + S + P
        q = (WB.T).dot(pseudo_data) / normalization_loss 
        loss = []
        for ite in tqdm(range(max_iter)):   
            if ite!= 0:
                coef_old = deepcopy(coef_new)
            
            if ite==0:
                if not(warm_start):
                    coef_new = B.dot(pseudo_data).flatten()
                    coef_new = np.array(coef_new).ravel()
                    coef_new = ReLU(coef_new, delta=0)
                else:
                    coef_new = self.gam.get_coeffs().reshape(-1)
            else:
                #grad = ( (WB.T).dot(WB)+S+P ).dot(coef_new) - (WB.T).dot(pseudo_data) # - (1/t) /(coef_new+epsilon)
                grad = ( Q ).dot(coef_new) - q # - (1/t) /(coef_new+epsilon)
                coef_new = coef_new - lr * grad / n
                coef_new = np.array(coef_new).ravel()
                coef_new = ReLU(coef_new, delta=0)
                
            diff = 0
            count = 0
            for gam in self.gam.pygams:
                diff += np.linalg.norm(gam.coef_ - coef_new[count:count+len(gam.coef_)]) / np.linalg.norm(coef_new[count:count+len(gam.coef_)])/ len(self.gam.pygams)
                gam.coef_ = coef_new[count:count+len(gam.coef_)]  # update
                count = count+len(gam.coef_) 
    
                # log on-loop-end stats
                gam._on_loop_end(vars())
    
    
            Qhat = np.zeros(len(y))
            for k, gam in enumerate(self.gam.pygams):
                Qhat += ls_modelmat[k] @ gam.coef_
    
            error = np.linalg.norm(Qhat-y)
            if ite!=0 and (error>loss[-1]):
                lr = lr / 4
                print('New lr: ', lr)
                coef_new = deepcopy(coef_old)
            else:
                loss.append(error)
    
            if ite%100==0:
                print('Error: ', error)
                print(diff)
                if not(save_folder is None):
                    gamcoeffs = []
                    for gam in self.gam.pygams:
                        gamcoeffs += list(gam.coef_)
                    self.save_model_parameters(save_folder, name=name_model, add_dic={'dates_training': dates})
                    np.save(os.path.join(save_folder, 'gamcoeffs.npy'), np.array(gamcoeffs))
                    np.save(os.path.join(save_folder, 'losses.npy'), np.array(loss))
    
    
    
        if diff < self.gam.pygams[0].tol:
            return loss
    
        print('did not converge')
        return loss
