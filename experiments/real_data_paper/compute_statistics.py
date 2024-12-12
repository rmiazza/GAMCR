import pandas as pd
import numpy as np
import sys
from joblib import Parallel, delayed
sys.path.append('../../')
import GAMCR
import os
data_folder = os.path.dirname(os.path.abspath(__file__))
import os
import pickle
import matplotlib.pyplot as plt
import copy
import torch

from data_and_visualization.get_data_from_ERRA import *
path_ERRA = os.path.join(data_folder, 'data_and_visualization', 'output_ERRA_forGAMCR')
dicERRA = get_data_from_ERRA(path_ERRA)

#all_GISID = list(dicERRA.keys())
all_GISID = [el for el in list(os.walk(data_folder))[0][1] if not any(c in el for c in ['/', '.', '_'])]

all_data = True

for site in all_GISID: #['184','47','112','150', '27']: 
    name_model = '{0}_best_model.pkl'.format(site)
    model = GAMCR.model.GAMCR(lam=0.1)
    datapath = os.path.join(data_folder, site, name_model)
    model.load_model(datapath)
    site_folder =  os.path.join(data_folder,site)

    if True:
        nblocks = len(dicERRA[site]['groups_precip'])
        groups_precip = dicERRA[site]['groups_precip']
        groups_precip[0] = (1,groups_precip[0][1])

        def filter_dates(dates, all_data=False):
            dates = pd.to_datetime(dates)
            if not(all_data):
                idxsyear = np.where(np.array([date.year for date in dates])<2018)[0]
            if site == '46':
                low_month = 7
                up_month = 9
            elif site in ['3','44','112']:
                low_month = 6
                up_month = 10
            else:
                low_month = 5
                up_month = 10
            idxsmonth_low = np.where(np.array([date.month for date in dates])>=low_month)[0]
            #np.where(dates.apply(lambda x: x.month>=low_month))[0]
            #idxsmonth_up = np.where(dates.apply(lambda x: x.month<=up_month))[0]
            idxsmonth_up = np.where(np.array([date.month for date in dates])<=up_month)[0]
            if not(all_data):
                idxs = np.intersect1d(idxsyear, idxsmonth_low)
            else:
                idxs = idxsmonth_low
            idxs = np.intersect1d(idxs, idxsmonth_up)
            return idxs
            
        save_folder=None
        if all_data:
            save_folder = os.path.join(site_folder, 'results')
        model.compute_statistics(site_folder, site, nblocks = nblocks, min_precip = groups_precip[0][0], groups_precip=groups_precip, max_files=99, filtering_time_points=lambda x: filter_dates(x, all_data=all_data), save_folder=save_folder)
    else:
        model.compute_statistics(site_folder, site)