import numpy as np
from scipy.interpolate import BSpline
from pygam import LinearGAM
from copy import deepcopy
from collections import defaultdict


class DataGAM():
    def __init__(self, L, n_splines=10, lam=10):
        self.L = L
        self.n_splines = n_splines
        self.lam = lam
        self.m_features = None
        self.edge_knots_ = None

    def get_params(self):
        return {'edge_knots_':self.edge_knots_, 'n_splines':self.n_splines, 'lam': self.lam, 'coeffs': self.get_coeffs(), 'm_features':self.m_features}

    def get_coeffs(self):
        if (
            not self.pygams[0]._is_fitted
            or len(self.pygams[0].coef_) != self.pygams[0].terms.n_coefs
            or not np.isfinite(self.pygams[0].coef_).all()
        ):
            return None
        else:
            coeffs = []
            for l in range(self.L):
                coeffs.append(self.pygams[l].coef_)
            return np.array(coeffs)

    def init_gam_from_design(self, X):
        gam = LinearGAM(terms='auto', n_splines=self.n_splines, lam=self.lam, max_iter=20)
        # validate parameters
        gam._validate_params()
        # validate data-dependent parameters
        gam._validate_data_dep_params(X)
        # set up logging
        if not hasattr(gam, 'logs_'):
            gam.logs_ = defaultdict(list)
        gam.statistics_ = {}
        gam.statistics_['n_samples'] = X.shape[0]
        gam.statistics_['m_features'] = X.shape[1]
        self.edge_knots_ = gam.edge_knots_
        self.m_features =  X.shape[1]
        self.pygams = []
        for l in range(self.L):
            self.pygams.append(deepcopy(gam))


    def init_gam_from_knots(self, edge_knots, m_features, coeffs=None):
        gam = LinearGAM(terms='auto', n_splines=self.n_splines, lam=self.lam, max_iter=20)
        # validate parameters
        gam._validate_params()
        # validate data-dependent parameters
        gam._validate_data_dep_params(np.zeros((1,m_features)))
        # set up logging
        if not hasattr(gam, 'logs_'):
            gam.logs_ = defaultdict(list)
        gam.statistics_ = {}
        gam.statistics_['n_samples'] = 1
        gam.statistics_['m_features'] = m_features
        gam.edge_knots_ = edge_knots
        self.edge_knots_ = edge_knots
        self.m_features =  m_features
        self.pygams = []
        for l in range(self.L):
            self.pygams.append(deepcopy(gam))
        if not(coeffs is None):
            for l in range(self.L):
                self.pygams[l].coef_ = coeffs[l,:]

    def _modelmat(self, X):
        return self.pygams[0]._modelmat(X).A

    