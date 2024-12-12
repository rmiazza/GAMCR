import pandas as pd
import numpy as np
import sys
sys.path.append('../../')

import GAMCR
import os

save_folder_ref = os.path.dirname(os.path.abspath(__file__))

all_GISID = [el for el in list(os.walk(save_folder_ref))[0][1] if not any(c in el for c in ['/', '.', '_'])]

model = GAMCR.model.GAMCR(features = {'date':True})   
if True:
    for site in all_GISID:
        save_folder = os.path.join(save_folder_ref, '{0}/data/'.format(site))
        datafile = os.path.join(save_folder_ref, '{0}/data_{0}.txt'.format(site))
        model.save_batch(save_folder, datafile, nfiles=40)
else:
    model.save_batch_common_GAM(all_GISID, save_folder_ref, nfiles=40)