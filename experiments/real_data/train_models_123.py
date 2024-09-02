import pandas as pd
import numpy as np
import sys
sys.path.append('/mydata/watres/quentin/code/FLOW/hourly_analysis/')
import GAMCR
import os

save_folder= '/mydata/watres/quentin/code/FLOW/hourly_analysis/RES_GAMCR/real_data/'


all_GISID = [el for el in list(os.walk(save_folder))[0][1] if (not('/' in el) and not('.' in el))]



for GISID in all_GISID:
    if GISID[0] in ['1','2','3']:
        model = GAMCR.model.GAMCR()
        GISIDpath = os.path.join(save_folder, str(GISID), 'data')
        X, matJ, y, timeyear, dates = model.load_data(GISIDpath, max_files=35)
        model.load_model(os.path.join(GISIDpath, 'params.pkl'))
        name_model = '{0}_trained_model'.format(GISID)
        loss = model.train(X, matJ, y, dates=dates, lr=1e-1, max_iter=5000, warm_start=False, save_folder=save_folder, name_model=name_model, normalization_loss=1)