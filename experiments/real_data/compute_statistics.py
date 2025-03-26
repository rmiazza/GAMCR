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
if dicERRA is None:
    print('No ERRA results found. Ensembles for precipitation and streamflow will be set using quantiles.')
    
all_GISID = [el for el in list(os.walk(data_folder))[0][1] if not any(c in el for c in ['/', '.', '_'])]

# Set to True to get results on the complete dataset (and not only the training set)
all_data = True

for site in all_GISID:
    name_model = '{0}_best_model.pkl'.format(site)
    model = GAMCR.model.GAMCR(lam=0.1)
    datapath = os.path.join(data_folder, site, name_model)
    model.load_model(datapath)
    site_folder =  os.path.join(data_folder,site)

    if not(dicERRA is None):
        nblocks = len(dicERRA[site]['groups_precip'])
        groups_precip = dicERRA[site]['groups_precip']

        if groups_precip[0][0]<0.1:
            # note that we consider ERRA has been run by removing all precipitation events below 0.5mm.h^{-1}
            # Since for some sites, aggregation was used in ERRA, ERRA ensembles for those sites could still provide a minimum for the first bin smaller than 0.5.
            groups_precip[0] = (0.5,groups_precip[0][1])
        min_precip = groups_precip[0][0]
    else:
        nblocks = 4
        min_precip = 0.5
        groups_precip = "auto"

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
    model.compute_statistics(site_folder, site, nblocks = nblocks, min_precip = min_precip, groups_precip=groups_precip, groups_wetness="auto", max_files=99, filtering_time_points=lambda x: filter_dates(x, all_data=all_data), save_folder=save_folder)