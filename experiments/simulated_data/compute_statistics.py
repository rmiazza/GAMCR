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

all_sites = ['flashy']#,'base','damped']

all_data = True

for site in all_sites:
    name_model = '{0}_best_model.pkl'.format(site)
    model = GAMCR.model.GAMCR(lam=0.1)
    datapath = os.path.join(data_folder, site, name_model)
    model.load_model(datapath)
    site_folder =  os.path.join(data_folder,site)

    nblocks = len(dicERRA[site]['groups_precip'])
    groups_precip = dicERRA[site]['groups_precip']
    groups_precip[0] = (1,groups_precip[0][1])


    save_folder=None
    if all_data:
        save_folder = os.path.join(site_folder, 'results')
    model.compute_statistics(site_folder, site, nblocks = nblocks, min_precip = groups_precip[0][0], groups_precip=groups_precip, max_files=99, save_folder=save_folder)
