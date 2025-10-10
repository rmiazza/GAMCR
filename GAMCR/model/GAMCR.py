import numpy as np
from ..dataset.dataset import Dataset
from ..trainer.trainer import Trainer
from ..resultsanalysis.compute_statistics import ComputeStatistics


class GAMCR(Dataset, Trainer, ComputeStatistics):
    """
    Main class of the GAMCR package to learn transfer functions of a given
    catchment.

    Attributes
    ----------
    max_lag : int
        Maximum lag time consider for the transfer functions
    features : dic
        Dictionary of the different features used in the model
    n_splines : int
        Number of splines considered for a GAM
    lam : positive float
        Regularization parameter related to the smoothing penalty in the GAM

    Methods
    -------
    train(X, matJ, Y, dates = None, lr=1e-3, max_iter=200, warm_start=False,
          save_folder=None, name_model='', normalization_loss=1, lam_global=0)
        Train the model.
    predict_transfer_function(X)
        Predict the transfer functions from the design matrix X.
    predict_streamflow(matJ)
        Predict the hydrograph from the matrix matJ (obtained from the method
        'get_GAMdesign' of the class 'Dataset').
    """
    def __init__(self, max_lag=24*10, features={}, n_splines=10, lam=10):
        Dataset.__init__(
            self, max_lag=max_lag, features=features,
            n_splines=n_splines, lam=lam
            )
        Trainer.__init__(self)
        ComputeStatistics.__init__(self)

    def train(self, X, matJ, Y, dates=None, lr=1e-3, max_iter=200,
              warm_start=False, save_folder=None, name_model='',
              normalization_loss=1, lam_global=0):
        """Train the model.

        This method prepares the input matrices and calls the internal optimizer
        ('self.trainer') to estimate model parameters through projected gradient
        descent. Each GAM component (one per basis spline) is trained jointly
        using the supplied data.

        Parameters
        ----------
        X : array
            Design matrix of the GAM compute from the method 'get_design'.
            X has dimension: number of timepoints x number of features.
        matJ : array
            Matrix used in the convolution to get the streamflow values
            (obtained from the method 'get_GAMdesign' of the class 'Dataset').
        dates : array, optional
            Array of dates.
        lr : float, optional
            Initial value of the learning rate. Note that the learning rate
            will be automatically adjusted to ensure a strict descrease of the
            training loss.
        max_iter : int, optional
            Maximum number of iterations of the projected gradient descent
            algorithm.
        warm_start : bool, optional
            If True, the model parameters will be initialized to the
            parameters saved in the model loaded.
        save_folder : str, optional
            Path of the folder of the studied site where the optimized model
            will be saved.
        name_model : str, optional
            Custom name of the model that will be saved.
        normalization_loss : positive float, optional
            Normalization factor for the loss (should be kept to 1).
        lam_global : positive float, optional
            Regularization parameter for the smoothing penalty applied on the
            transfer functions
        """
        # Duplicate the design matrix for each basis spline
        ls_X = [X for _ in range(self.L)]

        # Extract the corresponding model matrix slice for each spline
        ls_modelmat = [matJ[:, l, :] for l in range(self.L)]

        self.trainer(
            ls_X, ls_modelmat, Y, dates=dates, lr=lr,
            max_iter=max_iter, warm_start=warm_start, save_folder=save_folder,
            name_model=name_model, normalization_loss=normalization_loss,
            lam_global=lam_global
        )

    def predict_transfer_function(self, X):
        """Compute the time-varying transfer functions from the design matrix X.

        This method reconstructs the transfer functions used in the GAMCR
        convolution model by combining the model design matrix, fitted GAM
        coefficients, and basis splines.

        Parameters
        ----------
        X : array
            Design matrix of the GAM compute from the method 'get_design'.
            X has dimension: number of timepoints x number of features.

        Returns
        -------
        np.ndarray
            Array of shape (n_timepoints, m) containing the predicted transfer
            function values for each time step (n_time) and each lag time (m).
        """
        n_time = X.shape[0]  # number of time steps
        H = np.zeros((n_time, self.m))  # initialize transfer function matrix
        A = self.gam._modelmat(X)  # compute the GAM model matrix

        # Reconstruct transfer functions by combining GAM outputs and basis splines
        for l in range(self.L):
            # Compute contribution from the l-th spline
            gam_output = A @ self.gam.pygams[l].coef_

            # Compute transfer functions
            H += (
                np.tile(gam_output.reshape(-1, 1), (1, self.m))
                * np.tile(self.basis_splines[l, :].reshape(1, -1), (n_time, 1))
            )

        return H

    def predict_streamflow(self, matJ):
        """Predict the streamflow (hydrograph) from the GAM convolution matrix.

        This method reconstructs the predicted discharge time series using the
        model’s fitted GAM coefficients and the convolution matrices produced by
        'Dataset.get_GAMdesign'.

        Parameters
        ----------
        matJ : np.ndarray
            Matrix used in the convolution to get the streamflow values
            (obtained from the method 'get_GAMdesign' of the class 'Dataset').

        Returns
        -------
        np.ndarray
            1D array representing the predicted
            streamflow (hydrograph).
        """
        n_time, L, nft = matJ.shape
        Qhat = np.zeros(n_time)

        # Combine contributions from all GAM components
        for k in range(self.L):
            Qhat += matJ[:, k, :] @ self.gam.pygams[k].coef_

        return Qhat
