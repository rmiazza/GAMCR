import pandas as pd
import numpy as np
import sys
from joblib import Parallel, delayed
sys.path.append('/mydata/watres/quentin/code/FLOW/hourly_analysis/')
import GAMCR
data_folder = '/mydata/watres/quentin/code/FLOW/hourly_analysis/RES_GAMCR/real_data/'
save_folder = '/mydata/watres/quentin/code/FLOW/hourly_analysis/RES_GAMCR/real_data_pred/LOO/save/'
import os
import pickle
import matplotlib.pyplot as plt
import copy
import torch

#all_GISID = np.load(os.path.join('/mydata/watres/quentin/code/FLOW/hourly_analysis/RES_GAMCR/real_data_pred/', 'GISID_with_test_NSE_above_30percent.npy'))

mode = '_all'

with open('/mydata/watres/quentin/code/FLOW/hourly_analysis/RES_GAMCR/real_data_pred/GISID2NSE{0}.pkl'.format(mode), 'rb') as handle:
    GISID2NSE = pickle.load(handle)

NSE_min = 0.3
all_GISID = []
for key, val in GISID2NSE.items():
    if val>=NSE_min:
        all_GISID.append(key)
feat_space, all_GISID, dffeat = GAMCR.dataset.get_feat_space(all_GISID=all_GISID, get_df=True, normalize=False)

feat_space = copy.deepcopy(dffeat.to_numpy())


model = GAMCR.model.GAMCR(lam=1)
name_model = '{0}_best_model{1}.pkl'.format(all_GISID[0], mode)
save_folder_file =  os.path.join(data_folder,'{0}'.format(all_GISID[0]))
model.load_model(os.path.join(save_folder_file, name_model))

beta0 = model.gam.get_coeffs()

site2gamcoeffs = torch.zeros((len(all_GISID), beta0.shape[0]*beta0.shape[1]))

for i, GISID in enumerate(all_GISID):
    model = GAMCR.model.GAMCR(lam=1)
    name_model = '{0}_best_model{1}.pkl'.format(GISID, mode)
    save_folder_file =  os.path.join(data_folder,'{0}'.format(GISID))
    model.load_model(os.path.join(save_folder_file, name_model))
    beta0 = model.gam.get_coeffs().reshape(-1)
    site2gamcoeffs[i,:] = torch.tensor(copy.deepcopy(beta0)).float()
    site2gamcoeffs[i,:] = site2gamcoeffs[i,:] * (site2gamcoeffs[i,:]>0)

norms = np.linalg.norm(feat_space, axis=0)
centerings = {}
for j in range(feat_space.shape[1]):
    centerings[j] = np.mean(feat_space[:,j])
    if norms[j]>1e-3:
        feat_space[:,j] = (feat_space[:,j]-(np.mean(feat_space[:,j])))/norms[j]
    else:
        feat_space[:,j] = (feat_space[:,j]-(np.mean(feat_space[:,j])))



    

from sklearn import linear_model


if False:
    ############################## BEGIN CV
    
    from sklearn.model_selection import GridSearchCV, KFold
    from sklearn.linear_model import MultiTaskLasso
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    
    # Initialize the MultiTaskElasticNet model
    model = MultiTaskLasso(random_state=42)
    
    # Define the hyperparameter grid
    param_grid = {
        'alpha': [0.001 + 0.001*i for i in range(100)],  # Range of alpha values
    }
    
    
    # Define a KFold cross-validator
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Set up the GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kf, n_jobs=-1)
    
    # Perform the grid search
    grid_search.fit(feat_space, site2gamcoeffs.numpy())
    
    # Print the best parameters and the corresponding score
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation MSE: {-grid_search.best_score_:.4f}")
    
    alpha = grid_search.best_params_['alpha']
    
    ################################ END CV

    
    for id_test, GISID in enumerate(all_GISID):
        idxs_train = np.array([i for i in range(len(all_GISID)) if not(i==id_test)])
        idxs_test = np.array([id_test])
        clf = linear_model.MultiTaskLasso(alpha=alpha)
        clf.fit(feat_space[idxs_train,:], site2gamcoeffs.numpy()[idxs_train,:])
        coeffs_pred = clf.predict(feat_space)
    
    
        ### Loading model and data
        model = GAMCR.model.GAMCR(lam=1)
        save_folder_file =  os.path.join(data_folder,'{0}'.format(GISID))
        name_model = '{0}_best_model{1}.pkl'.format(GISID, mode)
        model.load_model(os.path.join(save_folder_file, name_model))
        GISIDpath = os.path.join(save_folder_file, 'data')
        X, matJ, y, timeyear, dates = model.load_data(GISIDpath,  max_files=20, test_mode=True)
        coef = 3600 * 1000 / (dffeat.loc[GISID, 'EZG '] * 1000000) 
        y = y * coef
        nt,L,nft = matJ.shape
    
    
        ### Computing prediction using the parameters learned using GAMCR on the given real site
        gamcoeffs = model.gam.get_coeffs()
        H = model.predict_transfer_function(X)
        Qhat =  model.predict_streamflow(matJ)
        
        ### Computing prediction using the parameters given by the MultiTaskLasso
        m = H.shape[1]
        coeffpred_GISID = coeffs_pred[id_test,:].reshape(model.L, coeffs_pred.shape[1]//(model.L))
        for k in range(model.L):
            model.gam.pygams[k].coef_ = coeffpred_GISID[k,:]
        Qhat_pred =  model.predict_streamflow(matJ)
        HpredGISID =  model.predict_transfer_function(X)
    
    
    
        idxs = np.argsort(np.sum(np.abs(clf.coef_), axis=0))
        sum_weights = np.sum(np.abs(clf.coef_), axis=0)
        idxs = np.array([i for i in idxs if sum_weights[i]>1e-8])
        featurenames = dffeat.columns.values
        print(featurenames[np.flip(idxs)])
        
        labels = featurenames[np.flip(idxs)]
        counts = sum_weights[np.flip(idxs)]
    
        # Prepare dictionary for saving
        dico = {
            'alpha': alpha,
            'coeffs': clf.coef_,
            'Qhat': Qhat,
            'Qtrue': y,
            'Qpred': Qhat_pred,
            'Hhat': np.mean(H, axis=0),
            'Hpred': np.mean(HpredGISID, axis=0),
            'selected_features': labels,
            'weights': counts,
            'dates': timeyear
        }
    
        #dico = {'coeffs':clf.coef_ ,'Qhat':Qhat, 'Qtrue':y, 'Qpred':Qhat_pred, 'Hhat':np.mean(H, axis=0), 'Hpred':np.mean(HpredGISID, axis=0), 'selected_features':labels, 'weights':counts, 'dates': timeyear}
        with open(os.path.join(save_folder, 'GISID_{0}.pkl'.format(GISID)), 'wb') as handle:
            pickle.dump(dico, handle, protocol=pickle.HIGHEST_PROTOCOL)

else:
    from joblib import Parallel, delayed
    import numpy as np
    import os
    import pickle
    from sklearn import linear_model
    
    def process_GISID(id_test, GISID, all_GISID, feat_space, site2gamcoeffs, dffeat, data_folder, alpha):
        # Prepare train and test indices
        idxs_train = np.array([i for i in range(len(all_GISID)) if not(i == id_test)])
        idxs_test = np.array([id_test])
    
        # Train MultiTaskLasso model
        if False:
            clf = linear_model.MultiTaskLasso(alpha=alpha, max_iter=10000)
            clf.fit(feat_space[idxs_train, :], np.log(site2gamcoeffs.numpy()[idxs_train, :]+1e-14))
            coeffs_pred = clf.predict(feat_space)
            coeffs_pred = np.exp(coeffs_pred) * (coeffs_pred>np.log(1e-7))
        else:
            clf = linear_model.MultiTaskLasso(alpha=alpha)
            clf.fit(feat_space[idxs_train, :], site2gamcoeffs.numpy()[idxs_train, :])
            coeffs_pred = clf.predict(feat_space)
            coeffs_pred = coeffs_pred * (coeffs_pred>0)

        # Load GAMCR model and data
        model = GAMCR.model.GAMCR(lam=1)
        save_folder_file = os.path.join(data_folder, '{0}'.format(GISID))
        name_model = '{0}_best_model{1}.pkl'.format(GISID, mode)
        model.load_model(os.path.join(save_folder_file, name_model))
        GISIDpath = os.path.join(save_folder_file, 'data')

        
        X, matJ, y, timeyear, dates = model.load_data(GISIDpath, max_files=20)
        
        # Adjust target based on coefficients
        coef = 3600 * 1000 / (dffeat.loc[GISID, 'EZG '] * 1000000)
        y = y * coef
        nt, L, nft = matJ.shape

        # threshold coefficients based on (P-Q)/PET
        pathfile = os.path.join(save_folder_file, 'data_{0}.txt'.format(GISID))
        df = pd.read_csv(pathfile)
        df = df.fillna(0)
        J = df['p'].to_numpy()
        PET = df['pet'].to_numpy()
        Qtrue = np.mean(df['q'].to_numpy() * coef)
        Pmean = np.mean(J)
        ETmean = np.mean(PET)/24

        threshold = (1e-15)
        Qref4mean = np.mean(matJ, axis=0).reshape(-1)
        Qmean = np.sum(Qref4mean * np.abs(coeffs_pred[id_test, :]))
        if Qmean>Pmean:
            coeffs_pred[id_test, :] *= Pmean / Qmean
        betacoeff = ((Pmean-Qmean)/ETmean)
        print(GISID, 'betacoeff init', betacoeff, 'true betacoeff', (Pmean-Qtrue)/ETmean, Pmean*24*365, Qtrue*24*365, ETmean*24*365)

        while ((betacoeff<0.7) or (betacoeff>0.9)) and threshold<1:
            threshold *= 2
            Qmean = np.sum(Qref4mean * coeffs_pred[id_test, :] * (coeffs_pred[id_test, :]>threshold))
            
            betacoeff = ((Pmean-Qmean)/ETmean)
        print('betacoeff', betacoeff, threshold)
        if (  ((betacoeff>=0.7) and (betacoeff<=0.9)) or threshold<1. ):
            print('ok')
            coeffs_pred[id_test, :] = coeffs_pred[id_test, :] * (coeffs_pred[id_test, :]>threshold)
    
        # Prediction using parameters learned from GAMCR
        gamcoeffs = model.gam.get_coeffs()
        H = model.predict_transfer_function(X)
        Qhat = model.predict_streamflow(matJ)
    
        # Prediction using parameters from MultiTaskLasso
        coeffpred_GISID = coeffs_pred[id_test, :].reshape(model.L, coeffs_pred.shape[1] // (model.L))
        for k in range(model.L):
            model.gam.pygams[k].coef_ = coeffpred_GISID[k, :]
        Qhat_pred = model.predict_streamflow(matJ)
        HpredGISID = model.predict_transfer_function(X)
    
        # Feature selection based on learned coefficients
        idxs = np.argsort(np.sum(np.abs(clf.coef_), axis=0))
        sum_weights = np.sum(np.abs(clf.coef_), axis=0)
        idxs = np.array([i for i in idxs if sum_weights[i] > 1e-8])
        featurenames = dffeat.columns.values
        labels = featurenames[np.flip(idxs)]
        counts = sum_weights[np.flip(idxs)]
    
        # Prepare dictionary for saving results
        dico = {
            'alpha': alpha,
            'coeffs': clf.coef_,
            'Qhat': Qhat,
            'Qtrue': y,
            'Qpred': Qhat_pred,
            'Hhat': np.mean(H, axis=0),
            'Hpred': np.mean(HpredGISID, axis=0),
            'selected_features': labels,
            'weights': counts,
            'dates': timeyear
        }
    
        # Save results to a file
        with open(os.path.join(save_folder, 'GISID_{0}.pkl'.format(GISID)), 'wb') as handle:
            pickle.dump(dico, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    alpha = 0.01  # Example alpha value
    
    # Use Parallel to run the process_GISID function for each GISID
    Parallel(n_jobs=-1)(delayed(process_GISID)(
        id_test, GISID, all_GISID, feat_space, site2gamcoeffs, dffeat, data_folder, alpha) 
        for id_test, GISID in enumerate(all_GISID))
