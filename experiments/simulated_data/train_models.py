import pandas as pd
import numpy as np
import sys
sys.path.append('../../')
import GAMCR
import os

save_folder = os.path.dirname(os.path.abspath(__file__))


all_sites = ['flashy']#,'base','damped']



ls_lambs = [0.000001 * (10**i) for i in range(2,7)]
ls_global_lambs = [0.000001 * (10**i) for i in range(3,10)]


for site in all_sites:
    lam = ls_lambs[1]
    global_lam = ls_global_lambs[3]

    model = GAMCR.model.GAMCR()
    sitepath = os.path.join(save_folder, str(site), 'data')
    X, matJ, y, timeyear, dates = model.load_data(sitepath, max_files=99)
    dates = pd.to_datetime(dates)
    
    # Keeping only the time points with minimum precipitation intensity for training
    idxs = np.where(X[:,0]>=0.05)[0]
    X = X[idxs,:]
    matJ = matJ[idxs,:,:]
    timeyear = timeyear[idxs]
    dates = dates[idxs]
    y = y[idxs]
    
    model.load_model(os.path.join(sitepath, 'params.pkl'),  lam=lam)
    name_model = '{0}_best_model'.format(site)
    save_folder_site = os.path.join(save_folder, str(site))
    loss = model.train(X, matJ, y, dates=dates, lr=1e-1, max_iter=300000, warm_start=False, save_folder=save_folder_site, name_model=name_model, normalization_loss=1, lam_global=global_lam)
