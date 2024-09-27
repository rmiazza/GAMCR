import numpy as np
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
import pickle
from .datagam import DataGAM
from pygam import LinearGAM, s, f
from collections import defaultdict
import scipy as sp
import scipy
import pandas as pd
from copy import deepcopy
import os

class Dataset():
    def __init__(self, max_lag=24*30*2, features={}, n_splines=10,  lam=10):
        self.m = max_lag
        self.compute_spline_basis()
        self.L = self.basis_splines.shape[0]
        if not('p' in features):
            features['p'] = [24*15,24*30,24*30*6,24*30*12,24*30*16]
        if not('pet' in features):
            features['pet'] = [24*15,24*30,24*30*2,24*30*6]
        self.features = features
        self.init = max([np.max(self.features['p']), np.max(self.features['pet'])])
        self.gam = DataGAM(self.L, n_splines=n_splines, lam=lam)

    def load_model(self, path_model, lam=None):
        with open(path_model, 'rb') as handle:
            params = pickle.load(handle)
        if lam is None:
            lam = params['gam']['lam']
        self.m = params['m']
        self.compute_spline_basis()
        self.L = self.basis_splines.shape[0]
        self.init = params['init']
        self.features = params['features']
        self.feature_names = params['feature_names']
        self.gam = DataGAM(self.L, n_splines=params['gam']['n_splines'], lam=lam)
        self.gam.init_gam_from_knots(params['gam']['edge_knots_'], params['gam']['m_features'], coeffs=params['gam'].get('coeffs', None))
        
    def save_model_parameters(self, save_folder, name='', add_dic={}):
        params = add_dic
        params.update({'m':self.m,
                  'init':self.init,
                  'features':self.features,
                  'feature_names':self.feature_names,
                  'gam': self.gam.get_params()
                 })
        if name=='':
            with open('{0}/params.pkl'.format(save_folder), 'wb') as handle:
                pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('{0}/{1}.pkl'.format(save_folder, name), 'wb') as handle:
                pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
    def save_batch_common_GAM(self, allGISID, save_folder, ntest=0, nstart=0, nfiles=40):
        init = self.init
        m = self.m
        common_edge_knots = None
        for GISID in allGISID:
            import os
            path_root = os.path.join(save_folder, str(GISID))
            path = os.path.join(path_root, 'data/')
            if not os.path.exists(path):
                os.makedirs(path)
            pathfile = os.path.join(path_root, 'data_{0}.txt'.format(GISID))
            pet_train, x_train, y_train, dates_train, timeyear_train, pet_test, x_test, y_test, dates_test, timeyear_test = self.get_fluxes(pathfile, ntest=ntest, nstart=nstart)
            
            X = self.get_design(pet_train, x_train, y_train, timeyear_train)
            self.gam.init_gam_from_design(X)
            if common_edge_knots is None:
                common_edge_knots = deepcopy(self.gam.edge_knots_)
            else:
                for i, l in enumerate(common_edge_knots):
                    common_edge_knots[i][0] = min([self.gam.edge_knots_[i][0], l[0]]) 
                    common_edge_knots[i][1] = max([self.gam.edge_knots_[i][1], l[1]])
            self.save_model_parameters(path, name='params_local')
            np.save(os.path.join(path,'X_temp.npy'), X)
            np.save(os.path.join(path,'y_temp.npy'), y_train)
            #np.save(save_folder+'indexes_temp.npy'.format(l), np.array([i for i in range()]))
            np.save(os.path.join(path,'timeyear_temp.npy'), np.array(timeyear_train))
            np.save(os.path.join(path,'dates_temp.npy'), np.array(dates_train))
            np.save(os.path.join(path,'J_temp.npy'), np.array(x_train))
        self.gam.init_gam_from_knots(common_edge_knots, self.gam.m_features)
        
        for GISID in allGISID:
            path_root = os.path.join(save_folder, str(GISID))
            path = os.path.join(path_root, 'data/')
            X = np.load(os.path.join(path,'X_temp.npy'))
            J = np.load(os.path.join(path,'J_temp.npy'))
            timeyear = np.load(os.path.join(path,'timeyear_temp.npy'))
            dates = np.load(os.path.join(path,'dates_temp.npy'), allow_pickle=True)
            y = np.load(os.path.join(path,'y_temp.npy'))
            # os.remove(path+'X_temp.npy')
            # os.remove(path+'timeyear_temp.npy')
            # os.remove(path+'dates_temp.npy')
            # os.remove(path+'y_temp.npy')
            # os.remove(path+'J_temp.npy')

            matJ = self.get_GAMdesign(X, J)
            X = X[m+init:,:]
            Y = y[m+init:]
            dates = dates[m+init:]
            timeyear = timeyear[m+init:]
            nt = matJ.shape[0]
            for l in range(nfiles-1):
                low, up = l*(nt//nfiles), (l+1)*(nt//nfiles)
                np.save(os.path.join(path,'matJ_{0}.npy'.format(l)), matJ[low:up,:,:])
                np.save(os.path.join(path,'X_{0}.npy'.format(l)), X[low:up,:])
                np.save(os.path.join(path,'y_{0}.npy'.format(l)), Y[low:up])
                np.save(os.path.join(path,'indexes_{0}.npy'.format(l)), np.array([i for i in range(low,up)]))
                np.save(os.path.join(path,'timeyear_{0}.npy'.format(l)), np.array(timeyear[low:up]))
                np.save(os.path.join(path,'dates_{0}.npy'.format(l)), np.array(dates[low:up]))
            self.save_model_parameters(path)

    def save_batch(self, save_folder, datafile, nstart=0, nfiles=100):
        init = self.init
        m = self.m
        import os
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        pet_train, x_train, y_train, dates_train, timeyear_train, pet_test, x_test, y_test, dates_test, timeyear_test = self.get_fluxes(datafile, ntest=0, nstart=nstart)
        
        X = self.get_design(pet_train, x_train, y_train, timeyear_train)
        self.gam.init_gam_from_design(X)
        matJ = self.get_GAMdesign(X, x_train)
        X = X[m+init:,:]
        Y = y_train[m+init:]
        dates = dates_train[m+init:]
        timeyear = timeyear_train[m+init:]
        nt = matJ.shape[0]
        for l in range(nfiles-1):
            low, up = l*(nt//nfiles), (l+1)*(nt//nfiles)
            np.save(os.path.join(save_folder,'matJ_{0}.npy'.format(l)), matJ[low:up,:,:])
            np.save(os.path.join(save_folder,'X_{0}.npy'.format(l)), X[low:up,:])
            np.save(os.path.join(save_folder,'y_{0}.npy'.format(l)), Y[low:up])
            np.save(os.path.join(save_folder,'indexes_{0}.npy'.format(l)), np.array([i for i in range(low,up)]))
            np.save(os.path.join(save_folder,'timeyear_{0}.npy'.format(l)), np.array(timeyear[low:up]))
            np.save(os.path.join(save_folder,'dates_{0}.npy'.format(l)), np.array(dates[low:up]))
        self.save_model_parameters(save_folder)
    
    def get_fluxes(self, datafile, nstart=0, ntest=0, size_time_window=None):
        df = pd.read_csv(datafile)
        df = df.fillna(0)
        x = df['p'].to_numpy()
        y = df['q'].to_numpy()
        pet = df['pet'].to_numpy()
        m = self.m
        init = self.init
        if size_time_window is None:
            size_time_window = len(y)
        ntrain = init+size_time_window
        dates = df['date'].to_numpy()
        timeyear = df['timeyear'].to_numpy()
        n = len(dates)
    
        if ntest is None:
            ntest = init+size_time_window
        
        y_test = y[nstart+ntrain:min(nstart+ntrain+ntest,n)]
        x_test = x[nstart+ntrain:min(nstart+ntrain+ntest,n)]
        pet_test = pet[nstart+ntrain:min(nstart+ntrain+ntest,n)]
        dates_test = dates[nstart+ntrain:min(nstart+ntrain+ntest,n)]
        timeyear_test = timeyear[nstart+ntrain:min(nstart+ntrain+ntest,n)]
        y_train = y[nstart:nstart+ntrain]
        x_train = x[nstart:nstart+ntrain]
        pet_train = pet[nstart:nstart+ntrain]
        dates_train = dates[nstart:nstart+ntrain]
        timeyear_train = timeyear[nstart:nstart+ntrain]
        return pet_train, x_train, y_train, dates_train, timeyear_train, pet_test, x_test, y_test, dates_test, timeyear_test

    def get_design(self, pet, x, y, dates):
        m = self.m
        init = self.init
        nfeat = self.basis_splines.shape[0]
        
        nb_features = len(self.features['pet']) + len(self.features['p'])
        ages_max_x = self.features['p'] 
        ages_max_pet = self.features['pet']
        
        nfeatures_out_pet_p = 2
        self.feature_names = ['precipitation intensity', 'last 2 hours precipitation intensity']
        if self.features.get('date', False):
            nfeatures_out_pet_p += 1
            self.feature_names.append('date')
        if self.features.get('timeyear', False):
            nfeatures_out_pet_p += 2
            self.feature_names.append('cos yeartime')
            self.feature_names.append('sin yeartime')
        for el in ages_max_x:
            self.feature_names.append('Weighted avg precipitation over the last {0} hours'.format(el))
        for el in ages_max_pet:
            self.feature_names.append('Weighted avg pet over the last {0} hours'.format(el))
        nb_features += nfeatures_out_pet_p
        Nt = len(x)-1
        X = np.zeros((Nt,nb_features))
        J = x
        for i in range(init,Nt):
            t = int(i)+1
            arg_perio = (dates[i]-int(dates[i]))*2*np.pi
            for j in range(nb_features):
                if j<=1:
                    X[i,j] =  np.sum(J[t-j-1:t])
                elif self.feature_names[j]=='date':
                    X[i,j] =  dates[i]-dates[0]
                elif self.feature_names[j]=='cos yeartime':
                    X[i,j] = np.cos(arg_perio)
                elif self.feature_names[j]=='sin yeartime':
                    X[i,j] = np.sin(arg_perio)
                elif (j>=nfeatures_out_pet_p) and (j-nfeatures_out_pet_p<len(ages_max_x)):
                    ls = np.cumsum(np.ones(ages_max_x[j-nfeatures_out_pet_p]))
                    weights = np.flip( np.exp(-4*ls/ls[-1]) )
                    weights /= np.sum(weights)
                    X[i,j] = np.sum(J[t-ages_max_x[j-nfeatures_out_pet_p]:t]*weights)
                else:
                    jp = j-nfeatures_out_pet_p-len(ages_max_x)
                    ls = np.cumsum(np.ones(ages_max_pet[jp]))
                    weights = np.flip( np.exp(-4*ls/ls[-1]) )
                    weights /= np.sum(weights)
                    X[i,j] =  np.sum(pet[t-ages_max_pet[jp]:t]*weights)
        return X
        

    def get_GAMdesign(self, X, J):
        m = self.m
        init = self.init
        nfeat = self.basis_splines.shape[0]
        Nt = len(J)-1
        A = self.gam._modelmat(X)
        matJ = np.zeros((Nt-m-init,nfeat,A.shape[1]))
        for i in range(m+init,Nt):
            vec = np.flip(J[i-m+1:i+1])
            mat = np.flip(A[i-m+1:i+1,:], axis=0)
            # A: nt x nft     basis: L x 
            matJ[i-m-init,:,:] =  np.sum(vec[None,:,None] * self.basis_splines[:,:,None] * mat[None,:,:], axis=1)
        return matJ

    def load_data(self, save_folder, max_files=100, test_mode=False):
        id_sub_files = np.sort(np.array([name[5:-4] for name in os.listdir(save_folder) if ('matJ' in name)]).astype(int))

        if test_mode:
            id_files_2_load = id_sub_files[-max_files:]
        else:
            id_files_2_load = id_sub_files[:max_files]
        for ite, id_file in enumerate(id_files_2_load):
            if ite==0:
                X = np.load(os.path.join(save_folder,'X_{0}.npy'.format(id_file)))
                matJ = np.load(os.path.join(save_folder,'matJ_{0}.npy'.format(id_file)))
                y = np.load(os.path.join(save_folder,'y_{0}.npy'.format(id_file)))
                timeyear = np.load(os.path.join(save_folder,'timeyear_{0}.npy'.format(id_file)))
                dates = np.load(os.path.join(save_folder,'dates_{0}.npy'.format(id_file)), allow_pickle=True)
            else:
                Xtemp = np.load(os.path.join(save_folder,'X_{0}.npy'.format(id_file)))
                matJtemp = np.load(os.path.join(save_folder,'matJ_{0}.npy'.format(id_file)))
                ytemp = np.load(os.path.join(save_folder,'y_{0}.npy'.format(id_file)))
                timeyeartemp = np.load(os.path.join(save_folder,'timeyear_{0}.npy'.format(id_file)))
                datestemp = np.load(os.path.join(save_folder,'dates_{0}.npy'.format(id_file)),  allow_pickle=True)

                X = np.concatenate((X,Xtemp), axis=0)
                matJ = np.concatenate((matJ,matJtemp), axis=0)
                timeyear = np.concatenate((timeyear,timeyeartemp), axis=0)
                dates = np.concatenate((dates,datestemp), axis=0)

                y = np.concatenate((y,ytemp), axis=0)
        return X, matJ, y, timeyear, dates

    def compute_spline_basis(self, show_splines = False):
        m = self.m
        knots_ref = []
        val = 0
        step = 1
        while val<=m:
            knots_ref.append(val)
            val += step
            step *= 2
        
        nk = len(knots_ref)
        degree = 2
        n = nk+degree+1
        knots = np.concatenate((min(knots_ref)*np.ones(degree),knots_ref,max(knots_ref)*np.ones(degree)))
        c = np.zeros(n)
        
        # Generate B-spline basis functions
        basis_functions = []
        evaluation_points = np.linspace(min(knots), max(knots), m)
        
        basis_values = []
        for i in range(n):
            c[i] = 1
            basis = BSpline(knots, c, degree)
            basis_values.append(basis(evaluation_points))
            c[i] = 0
        basis_values = np.array(basis_values)[1:-3,:]
    
        if show_splines:    
            # Plot the basis functions
            
            
            for i, basis_values_i in enumerate(basis_values):
                plt.plot(evaluation_points, basis_values_i, label=f'Basis {i + 1}')
            
            plt.title('B-spline Basis Functions')
            plt.xlabel('x')
            plt.ylabel('Basis Function Value')
            plt.show()
            
        self.basis_splines = basis_values
        self.knots_splines = knots_ref
    