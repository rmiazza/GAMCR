import numpy as np
from ..dataset import *
from ..trainer import *
from ..resultsanalysis import *

class GAMCR(Dataset, Trainer, ComputeStatistics):
    """
    Main class of the GAMCR package to learn transfer functions of a given watershed.

    ...

    Attributes
    ----------
    max_lag : int
        maximum lag time consider for the transfer functions
    features : dic
        dictionary of the different features used in the model
    n_splines : int
        number of splines considered for a GAM
    lam : positive float
        regularization parameter related to the smoothing penalty in the GAM

    Methods
    -------
    train(X, matJ, Y, dates = None, lr=1e-3, max_iter=200, warm_start=False, save_folder=None, name_model='', normalization_loss=1, lam_global=0)
        Train the model.
    predict_transfer_function(X)
        Predict the transfer functions from the design matrix X.
    predict_streamflow(matJ)
        Predict the hydrograph from the matrix matJ (obtained from the method 'get_GAMdesign' of the class 'Dataset').
    """
    def __init__(self, max_lag=24*30*2, features={}, n_splines=10,  lam=10):
        Dataset.__init__(self, max_lag=max_lag, features=features, n_splines=n_splines,  lam=lam)
        Trainer.__init__(self)
        ComputeStatistics.__init__(self)
        
    def train(self, X, matJ, Y, dates = None, lr=1e-3, max_iter=200, warm_start=False, save_folder=None, name_model='', normalization_loss=1, lam_global=0):
        """Train the model.

        Parameters
        ----------
        X : array
            Design matrix of the GAM compute from the method 'get_design'. X has dimension: number of timepoints x number of features.
        matJ : array
            Matrix used in the convolution to get the streamflow values (obtained from the method 'get_GAMdesign' of the class 'Dataset').
        dates : array, optional
            Array of dates.
        lr : float, optional
            Initial value of the learning rate. Note that the learning rate will be automatically adjusted to ensure a strict descrease of the training loss.
        max_iter : int, optional
            Maximum number of iterations of the projected gradient descent algorithm.
        warm_start : bool, optional
            If True, the model parameters will be initialized to the parameters saved in the model loaded.
        save_folder : str, optional
            Path of the folder of the studied site where the optimized model will be saved.
        name_model : str, optional
            Custom name of the model that will be saved.
        normalization_loss : positive float, optional
            Normalization factor for the loss (should be kept to 1).
        lam_global : positive float, optional
            Regularization parameter for the smoothing penalty applied on the transfer functions
        """
        ls_X = [X for l in range(self.L)]
        ls_modelmat = [matJ[:,l,:] for l in range(self.L)]
        loss_curve = self.trainer(ls_X, ls_modelmat, Y, dates=dates, lr=lr, max_iter=max_iter, warm_start=warm_start, save_folder=save_folder, name_model=name_model, normalization_loss=normalization_loss, lam_global=lam_global)


    def predict_transfer_function(self, X):
        """Predict the transfer functions from the design matrix X.

        Parameters
        ----------
        X : array
            Design matrix of the GAM compute from the method 'get_design'. X has dimension: number of timepoints x number of features.
        """
        nt = X.shape[0]
        H = np.zeros((nt,self.m))
        A = self.gam._modelmat(X)
        for l in range(self.L):            
            H += np.tile( (A @ self.gam.pygams[l].coef_).reshape(-1,1), (1,self.m)) * np.tile( self.basis_splines[l,:].reshape(1,-1), (nt,1))
        return H

    def predict_streamflow(self, matJ):
        """Predict the hydrograph from the matrix matJ (obtained from the method 'get_GAMdesign' of the class 'Dataset').

        Parameters
        ----------
        matJ : array
            Matrix used in the convolution to get the streamflow values (obtained from the method 'get_GAMdesign' of the class 'Dataset').
        """
        nt,L,nft = matJ.shape
        Qhat = np.zeros(nt)
        for k in range(self.L):
            Qhat += matJ[:,k,:] @ self.gam.pygams[k].coef_
        return Qhat


    