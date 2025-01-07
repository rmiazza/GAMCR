import pandas as pd
import numpy as np
import sys
sys.path.append('../../')

import GAMCR
import os

save_folder_ref = os.path.dirname(os.path.abspath(__file__))

all_sites = ['flashy']#,'base','damped']

model = GAMCR.model.GAMCR(max_lag=24*10, features = {'timeyear':True})   

if True:
    for site in all_sites:
        save_folder = os.path.join(save_folder_ref, '{0}/data/'.format(site))
        datafile = os.path.join(save_folder_ref, '{0}/data_{0}.txt'.format(site))
        model.save_batch(save_folder, datafile, nfiles=40)
else:
    model.save_batch_common_GAM(all_sites, save_folder_ref, nfiles=40)
