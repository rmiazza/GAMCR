import pandas as pd
import numpy as np
import sys
from joblib import Parallel, delayed
sys.path.append('/mydata/watres/quentin/code/FLOW/hourly_analysis/')
import GAMCR
save_folder= '/mydata/watres/quentin/code/FLOW/hourly_analysis/RES_GAMCR/real_data/'
import os
import pickle

all_GISID = [el for el in list(os.walk(save_folder))[0][1] if (not('/' in el) and not('.' in el))]

from get_feat_space import *

feat_space, all_GISID, dffeat = get_feat_space(all_GISID=all_GISID, get_df=True, normalize=False)



if False:
    for GISID in all_GISID:
        save_folder_file =  os.path.join(save_folder,'{0}'.format(GISID))
        if not(os.path.isfile(os.path.join(save_folder_file, '{0}_best_model.pkl'.format(GISID)))):
            try:
                model_ghost = GAMCR.model.GAMCR(lam=0.1)
                GISIDpath = os.path.join(save_folder, str(GISID), 'data')
                X, matJ, y, timeyear, dates = model_ghost.load_data(GISIDpath,  max_files=10, test_mode=False)
                coef = 3600 * 1000 / (dffeat.loc[GISID, 'EZG '] * 1000000) 
                y = y * coef
                
                ls_lambs = [0.000001 * (10**i) for i in range(2,7)]
                ls_global_lambs = [0.000001 * (10**i) for i in range(3,10)]
                
                # Define the function for model training
                def find_best_model(idx_lam, lam, idx_global_lam, global_lam):
                    model = GAMCR.model.GAMCR(lam=lam)
                    name_model = '{0}_trained_model_CV_{1}_{2}.pkl'.format(GISID, idx_lam, idx_global_lam)
                    save_folder_file =  os.path.join(save_folder,'{0}'.format(GISID))
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
                save_folder_file =  os.path.join(save_folder,'{0}'.format(GISID))
                np.save(os.path.join(save_folder_file, '{0}_results_CV.npy'.format(GISID)), results) 
                argmax_NSE = np.argmax(results[:,0])
                print("Best NSE: ", results[argmax_NSE, 0])
                idxs = np.where(results[:,0]> (0.75*results[argmax_NSE,0]))[0]
                print(idxs)
                best_idx = np.argmin(results[idxs,1])
                best_idx_lamb, best_idx_global_lam = results[idxs[best_idx],2], results[idxs[best_idx],3]
                name_best_model = '{0}_trained_model_CV_{1}_{2}.pkl'.format(GISID, int(best_idx_lamb), int(best_idx_global_lam))
            
                
                with open(os.path.join(save_folder_file, name_best_model), 'rb') as handle:
                    saved_best_model = pickle.load(handle)
                with open(os.path.join(save_folder_file, '{0}_best_model.pkl'.format(GISID)), 'wb') as handle:
                    pickle.dump(saved_best_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
            except:
                pass

else:
    for GISID in all_GISID:
        try:
            model_ghost = GAMCR.model.GAMCR(lam=0.1)
            GISIDpath = os.path.join(save_folder, str(GISID), 'data')
            X, matJ, y, timeyear, dates = model_ghost.load_data(GISIDpath,  max_files=38)
            coef = 3600 * 1000 / (dffeat.loc[GISID, 'EZG '] * 1000000) 
            y = y * coef
            
            ls_lambs = [0.000001 * (10**i) for i in range(2,7)]
            ls_global_lambs = [0.000001 * (10**i) for i in range(3,10)]
            
            # Define the function for model training
            def find_best_model(idx_lam, lam, idx_global_lam, global_lam):
                model = GAMCR.model.GAMCR(lam=lam)
                name_model = '{0}_trained_model_CV_{1}_{2}.pkl'.format(GISID, idx_lam, idx_global_lam)
                save_folder_file =  os.path.join(save_folder,'{0}'.format(GISID))
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
            save_folder_file =  os.path.join(save_folder,'{0}'.format(GISID))
            np.save(os.path.join(save_folder_file, '{0}_results_CV_all.npy'.format(GISID)), results) 
            argmax_NSE = np.argmax(results[:,0])
            print("Best NSE: ", results[argmax_NSE, 0])
            idxs = np.where(results[:,0]> (0.9*results[argmax_NSE,0]))[0]
            print(idxs)
            best_idx = np.argmin(results[idxs,1])
            best_idx_lamb, best_idx_global_lam = results[idxs[best_idx],2], results[idxs[best_idx],3]
            name_best_model = '{0}_trained_model_CV_{1}_{2}.pkl'.format(GISID, int(best_idx_lamb), int(best_idx_global_lam))
        
            
            with open(os.path.join(save_folder_file, name_best_model), 'rb') as handle:
                saved_best_model = pickle.load(handle)
            with open(os.path.join(save_folder_file, '{0}_best_model_all.pkl'.format(GISID)), 'wb') as handle:
                pickle.dump(saved_best_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except:
            pass