import sys
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

def get_data_from_ERRA(pathERRA):
    dico = {}
    site2params = {'damped': {'m': 40, 'agg':6, 'h':2},
                   'base': {'m': 40, 'agg':3, 'h':2},
                   'flashy': {'m': 40, 'agg':2, 'h':2}
                  }
    ls_stations = ['damped', 'base', 'flashy']
    for site in ls_stations:
        dico[site] = {}
        m, agg, h = site2params[site]['m'], site2params[site]['agg'], site2params[site]['h']
        df = pd.read_csv(os.path.join(pathERRA, '{0}/{0}_NRF_m={1}_agg={2}_h={3}_nlin.txt'.format(site, m, agg, h)), sep='\t')
        colnames_nrf = [col for col in df.columns if 'NRF' in col]
        groups_precip = []
        for colname in colnames_nrf:
            groups_precip.append(filter_col_name(colname))
        # removing the first group with precipitation events with less than 1mm
        dico[site]['groups_precip'] = groups_precip[1:]
        dico[site]['lagtime'] =  df.loc[:,'lagtime'].values
        NRF = np.zeros((len(colnames_nrf)-1, len(dico[site]['lagtime'])))
        for k, col in enumerate(colnames_nrf[1:]):
            NRF[k,:] = df.loc[:,col].values
        dico[site]['group2NRF'] = NRF
    return dico
