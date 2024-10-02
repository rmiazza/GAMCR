import sys
sys.path.append('/mydata/watres/quentin/code/FLOW/hourly_analysis/')
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import os
import numpy as np

def filter_col_name(colname):
    if '<' in colname:
        i = 0
        while  colname[i]!='<':
            i += 1
        istart = int(i+1)
        while  colname[i]!='|':
            i += 1
        return 0, float(colname[istart:i])
    elif '>' in colname:
        i = 0
        while  colname[i]!='>':
            i += 1
        istart = int(i+1)
        while  colname[i]!='|':
            i += 1
        return float(colname[istart:i]), 1000
    else:
        i = 0
        while  colname[i]!='=':
            i += 1
        istart = int(i+1)
        while  colname[i]!='-':
            i += 1
        low = float(colname[istart:i])
        i += 1 
        istart = int(i)
        while  colname[i]!='|':
            i += 1
        return low, float(colname[istart:i])

def get_data_from_ERRA():
    dico = {}
    site2params = {'Pully_flashy': {'m': 25, 'agg':2, 'h':2},
                   'Pully_notflashy': {'m': 25, 'agg':2, 'h':1},
                   'Lugano_flashy': {'m': 25, 'agg':2, 'h':2},
                   'Lugano_notflashy': {'m': 25, 'agg':2, 'h':2},
                   'Basel_flashy': {'m': 25, 'agg':2, 'h':2},
                   'Basel_notflashy': {'m': 20, 'agg':4, 'h':2}
                  }
    
    ls_stations = ['Pully', 'Basel', 'Lugano']
    ls_modes = ['flashy', 'notflashy']
    for station in ls_stations:
        for mode in ls_modes:
            site = '{0}_{1}'.format(station, mode)
            dico[site] = {}
            m, agg, h = site2params[site]['m'], site2params[site]['agg'], site2params[site]['h']
            df = pd.read_csv('/mydata/watres/quentin/code/FLOW/hourly_analysis/RES_GAMCR/simulated_data/outputs_fromERRA_p_bins/{0}/{0}_data_NRF_m={1}_agg={2}_h={3}_nlin.txt'.format(site, m, agg, h), sep='\t')
            colnames_nrf = [col for col in df.columns if 'NRF' in col]
            groups_precip = []
            for colname in colnames_nrf:
                groups_precip.append(filter_col_name(colname))
            dico[site]['groups_precip'] = groups_precip
            
            dico[site]['lagtime'] =  df.loc[:,'lagtime'].values
            NRF = np.zeros((len(colnames_nrf), len(dico[site]['lagtime'])))
            for k, col in enumerate(colnames_nrf):
                NRF[k,:] = df.loc[:,col].values
            dico[site]['group2NRF'] = NRF
    return dico