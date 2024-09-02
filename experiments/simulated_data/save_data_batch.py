import pandas as pd
import numpy as np
import sys
sys.path.append('/mydata/watres/quentin/code/FLOW/hourly_analysis/')
import GAMCR

for site in ['Pully_flashy','Pully_notflashy', 'Lugano_flashy', 'Lugano_notflashy']:
    model = GAMCR.model.GAMCR(features = {'date':True})    
    save_folder = './{0}/data/'.format(site)
    datafile = './{0}/data_{0}.txt'.format(site)
    model.save_batch(save_folder, datafile)