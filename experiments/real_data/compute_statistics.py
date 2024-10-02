import pandas as pd
import numpy as np
import sys
from joblib import Parallel, delayed
sys.path.append('/mydata/watres/quentin/code/FLOW/hourly_analysis/')
import GAMCR
data_folder = '/mydata/watres/quentin/code/FLOW/hourly_analysis/RES_GAMCR/real_data/'
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


for site in all_GISID:
    name_model = '{0}_best_model{1}.pkl'.format(site, mode)
    model = GAMCR.model.GAMCR(lam=0.1)
    datapath = os.path.join(data_folder, site, name_model)
    model.load_model(datapath)
    site_folder =  os.path.join(data_folder,site)

    if False:
        dicERRA = get_data_from_ERRA()
        nblocks = len(dicERRA[site]['groups_precip'])
        groups_precip = dicERRA[site]['groups_precip']
        groups_precip[0] = (1,groups_precip[0][1])
        model.compute_statistics(site_folder, site, nblocks = nblocks, min_precip = groups_precip[0][0], groups_precip=groups_precip, max_files=39)
    else:
        model.compute_statistics(site_folder, site)
    
    