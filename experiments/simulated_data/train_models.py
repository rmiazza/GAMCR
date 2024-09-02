import pandas as pd
import numpy as np
import sys
sys.path.append('/mydata/watres/quentin/code/FLOW/hourly_analysis/')
import GAMCR

for station in ['Pully', 'Lugano']:
    for mode in ['flashy', 'notflashy']:
        model = GAMCR.model.GAMCR()
        save_folder = './{0}_{1}/data/'.format(station, mode)
        X, matJ, y, timeyear, dates = model.load_data(save_folder, max_files=70)
        model.load_model('./{0}_{1}/data/params.pkl'.format(station, mode))
        save_folder = './{0}_{1}/'.format(station, mode)
        name_model = '{0}_{1}_trained_model'.format(station, mode)
        loss = model.train(X, matJ, y, dates=dates, lr=1e-1, max_iter=30000, warm_start=False, save_folder=save_folder, name_model=name_model, normalization_loss=1)