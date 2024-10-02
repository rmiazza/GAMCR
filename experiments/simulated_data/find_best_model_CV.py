import pandas as pd
import numpy as np
import sys
from joblib import Parallel, delayed
sys.path.append('/mydata/watres/quentin/code/FLOW/hourly_analysis/')
import GAMCR
save_folder= '/mydata/watres/quentin/code/FLOW/hourly_analysis/RES_GAMCR/simulated_data/'
import os
import pickle



ls_stations = ['Lugano', 'Pully', 'Basel']

ls_modes = ['flashy', 'notflashy']

for station in ls_stations:
    for mode in ls_modes:
        site = '{0}_{1}'.format(station, mode)
        model_ghost = GAMCR.model.GAMCR(lam=0.1)
        datapath = os.path.join(save_folder, site, 'data')
        X, matJ, y, timeyear, dates = model_ghost.load_data(datapath,  max_files=30, test_mode=True)
        
        ls_lambs = [0.000001 * (10**i) for i in range(2,7)]
        ls_global_lambs = [0.000001 * (10**i) for i in range(3,10)]
        
        # Define the function for model training
        def find_best_model(idx_lam, lam, idx_global_lam, global_lam):
            model = GAMCR.model.GAMCR(lam=lam)
            name_model = '{0}_trained_model_CV_{1}_{2}.pkl'.format(site, idx_lam, idx_global_lam)
            save_folder_file =  os.path.join(save_folder,'{0}'.format(site))
            model.load_model(os.path.join(save_folder_file, name_model),  lam=lam)
            yhat = model.predict_streamflow(matJ)
            NSE = GAMCR.nse(y,yhat)
            coeffs = model.gam.get_coeffs().reshape(-1)
            a = np.mean(model.gam._modelmat(X), axis=0)
            smooth_P = np.kron( GAMCR.build_custom_matrix(model.L), np.dot(a.reshape(-1,1), a.reshape(1,-1)) )
    
            SMOOTHNESS = coeffs.T @ smooth_P @ coeffs
            return [NSE, SMOOTHNESS, idx_lam, idx_global_lam]
    
        if False:
            # Parallelize the nested loops using joblib
            results = Parallel(n_jobs=-1)(delayed(find_best_model)(idx_lam, lam, idx_global_lam, global_lam)
                                          for idx_lam, lam in enumerate(ls_lambs)
                                          for idx_global_lam, global_lam in enumerate(ls_global_lambs))
        else:
            results = []
            for idx_lam, lam in enumerate(ls_lambs):
                for idx_global_lam, global_lam in enumerate(ls_global_lambs):
                    restemp = find_best_model(idx_lam, lam, idx_global_lam, global_lam)
                    results.append(restemp)
        results = np.array(results)
        save_folder_file =  os.path.join(save_folder,'{0}'.format(site))
        np.save(os.path.join(save_folder_file, '{0}_results_CV.npy'.format(site)), results) 
        argmax_NSE = np.argmax(results[:,0])
        print("Best NSE: ", results[argmax_NSE, 0])
        idxs = np.where(results[:,0]> (0.75*results[argmax_NSE,0]))[0]
        print(idxs)
        best_idx = np.argmin(results[idxs,1])
        best_idx_lamb, best_idx_global_lam = results[idxs[best_idx],2], results[idxs[best_idx],3]
        name_best_model = '{0}_trained_model_CV_{1}_{2}.pkl'.format(site, int(best_idx_lamb), int(best_idx_global_lam))
    
        
        with open(os.path.join(save_folder_file, name_best_model), 'rb') as handle:
            saved_best_model = pickle.load(handle)
        with open(os.path.join(save_folder_file, '{0}_best_model.pkl'.format(site)), 'wb') as handle:
            pickle.dump(saved_best_model, handle, protocol=pickle.HIGHEST_PROTOCOL)