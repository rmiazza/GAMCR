import pandas as pd
import numpy as np
import sys
sys.path.append('/mydata/watres/quentin/code/FLOW/hourly_analysis/')
import GAMCR
import os

save_folder= '/mydata/watres/quentin/code/FLOW/hourly_analysis/RES_GAMCR/real_data/'

all_GISID = [el for el in list(os.walk(save_folder))[0][1] if (not('/' in el) and not('.' in el))]

model = GAMCR.model.GAMCR(features = {'date':True})    
model.save_batch_common_GAM(all_GISID, save_folder)