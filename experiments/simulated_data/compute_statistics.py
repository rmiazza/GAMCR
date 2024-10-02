import pandas as pd
import numpy as np
import sys
from joblib import Parallel, delayed
sys.path.append('/mydata/watres/quentin/code/FLOW/hourly_analysis/')
import GAMCR
save_folder= '/mydata/watres/quentin/code/FLOW/hourly_analysis/RES_GAMCR/simulated_data/'
import os
import pickle
from get_data_from_ERRA import *


ls_stations = ['Lugano', 'Pully', 'Basel']

ls_modes = ['flashy', 'notflashy']

for station in ls_stations:
    for mode in ls_modes:
        site = '{0}_{1}'.format(station, mode)
        model = GAMCR.model.GAMCR(lam=0.1)
        datapath = os.path.join(save_folder, site, '{0}_best_model.pkl'.format(site))
        model.load_model(datapath)
        site_folder = os.path.join(save_folder, site)

        if True:
            dicERRA = get_data_from_ERRA()
            nblocks = len(dicERRA[site]['groups_precip'])
            groups_precip = dicERRA[site]['groups_precip']
            groups_precip[0] = (1,groups_precip[0][1])
            model.compute_statistics(site_folder, site, nblocks = nblocks, min_precip = groups_precip[0][0], groups_precip=groups_precip, max_files=96)
        else:
            model.compute_statistics(site_folder, site)
        
        