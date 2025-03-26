import pandas as pd
import numpy as np
import sys
sys.path.append('../../')
import GAMCR
import os

save_folder = os.path.dirname(os.path.abspath(__file__))


all_GISID = [el for el in list(os.walk(save_folder))[0][1] if not any(c in el for c in ['/', '.', '_'])]

def filter_dates(GISID, dates):
    # filter the training data based on the dates
    idxsyear = np.where(np.array([date.year for date in dates])<2018)[0]
    if GISID == '46':
        low_month = 7
        up_month = 9
    elif GISID in ['3','44','112']:
        low_month = 6
        up_month = 10
    else:
        low_month = 5
        up_month = 10
    idxsmonth_low = np.where(np.array([date.month for date in dates])>=low_month)[0]
    idxsmonth_up = np.where(np.array([date.month for date in dates])<=up_month)[0]
    idxs = np.intersect1d(idxsyear, idxsmonth_low)
    idxs = np.intersect1d(idxs, idxsmonth_up)
    return idxs



ls_lambs = [0.000001 * (10**i) for i in range(2,7)]
ls_global_lambs = [0.000001 * (10**i) for i in range(3,10)]


for GISID in all_GISID:
    lam = ls_lambs[1]
    global_lam = ls_global_lambs[3]

    model = GAMCR.model.GAMCR()
    GISIDpath = os.path.join(save_folder, str(GISID), 'data')
    X, matJ, y, timeyear, dates = model.load_data(GISIDpath, max_files=99)
    dates = pd.to_datetime(dates)
    
    # Keeping only the snow-free period for training
    idxs = filter_dates(GISID, dates)
    X = X[idxs,:]
    matJ = matJ[idxs,:,:]
    timeyear = timeyear[idxs]
    dates = dates[idxs]
    y = y[idxs]
    
    model.load_model(os.path.join(GISIDpath, 'params.pkl'),  lam=lam)
    name_model = '{0}_best_model'.format(GISID)
    save_folder_GISID = os.path.join(save_folder, str(GISID))
    loss = model.train(X, matJ, y, dates=dates, lr=1e-1, max_iter=6000, warm_start=False, save_folder=save_folder_GISID, name_model=name_model, normalization_loss=1, lam_global=global_lam)
