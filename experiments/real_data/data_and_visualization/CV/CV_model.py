import pandas as pd
import numpy as np
import sys
from joblib import Parallel, delayed
sys.path.append('/mydata/watres/quentin/code/FLOW/hourly_analysis/')
import GAMCR
save_folder= '/mydata/watres/quentin/code/FLOW/hourly_analysis/RES_GAMCR/real_data/'
import os


all_GISID = [el for el in list(os.walk(save_folder))[0][1] if (not('/' in el) and not('.' in el))]

from get_feat_space import *

feat_space, all_GISID, dffeat = get_feat_space(all_GISID=None, get_df=True, normalize=False)


GISID = '152'
model_ghost = GAMCR.model.GAMCR(lam=0.1)
GISIDpath = os.path.join(save_folder, str(GISID), 'data')
X, matJ, y, timeyear, dates = model_ghost.load_data(GISIDpath, max_files=30)

ls_lambs = [0] + [0.000001 * (10**i) for i in range(10)]
ls_global_lambs = [0] + [0.000001 * (10**i) for i in range(10)]

# Define the function for model training
def train_model(idx_lam, lam, idx_global_lam, global_lam):
    model = GAMCR.model.GAMCR(lam=lam)
    model.load_model(os.path.join(GISIDpath, 'params.pkl'),  lam=lam)
    save_folder_file =  os.path.join(save_folder,'{0}'.format(GISID))
    name_model = '{0}_trained_model_CV_{1}_{2}'.format(GISID, idx_lam, idx_global_lam)
    loss = model.train(X, matJ, y, dates=dates, lr=1e-1, max_iter=400, warm_start=False, 
                       save_folder=save_folder_file, name_model=name_model, normalization_loss=1, 
                       lam_global=global_lam)
    return loss

# Parallelize the nested loops using joblib
results = Parallel(n_jobs=-1)(delayed(train_model)(idx_lam, lam, idx_global_lam, global_lam)
                              for idx_lam, lam in enumerate(ls_lambs)
                              for idx_global_lam, global_lam in enumerate(ls_global_lambs))