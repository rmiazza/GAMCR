import numpy as np
from ..dataset import *
from ..trainer import *
from ..resultsanalysis import *

class GAMCR(Dataset, Trainer, ComputeStatistics):
    def __init__(self, max_lag=24*30*2, features={}, n_splines=10,  lam=10):
        Dataset.__init__(self, max_lag=max_lag, features=features, n_splines=n_splines,  lam=lam)
        Trainer.__init__(self)
        ComputeStatistics.__init__(self)
        
    def train(self, X, matJ, Y, dates = None, lr=1e-3, max_iter=200, warm_start=False, save_folder=None, name_model='', normalization_loss=1, lam_global=0):
        ls_X = [X for l in range(self.L)]
        ls_modelmat = [matJ[:,l,:] for l in range(self.L)]
        loss_curve = self.trainer(ls_X, ls_modelmat, Y, dates=dates, lr=lr, max_iter=max_iter, warm_start=warm_start, save_folder=save_folder, name_model=name_model, normalization_loss=normalization_loss, lam_global=lam_global)


    def predict_transfer_function(self, X):
        nt = X.shape[0]
        H = np.zeros((nt,self.m))
        A = self.gam._modelmat(X)
        for l in range(self.L):            
            H += np.tile( (A @ self.gam.pygams[l].coef_).reshape(-1,1), (1,self.m)) * np.tile( self.basis_splines[l,:].reshape(1,-1), (nt,1))
        return H

    def predict_streamflow(self, matJ):
        nt,L,nft = matJ.shape
        Qhat = np.zeros(nt)
        for k in range(self.L):
            Qhat += matJ[:,k,:] @ self.gam.pygams[k].coef_
        return Qhat


    