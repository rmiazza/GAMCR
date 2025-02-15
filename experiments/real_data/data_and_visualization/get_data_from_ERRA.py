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
    
    
def get_data_from_ERRA(path_ERRA, mode="training"):
    dico = {}
    sites = os.listdir(path_ERRA)
    allGISID = []
    for site in sites:
        nameGISID = ''
        count = 6
        nameGISID = site[count:]
        allGISID.append(nameGISID)
    for GISID in allGISID:
        dico[GISID] = {}
        site = [f for f in sites if f'GISID-{GISID}' in f][0]
        
        path_site = os.path.join(path_ERRA, site)
        files = os.listdir(os.path.join(path_ERRA, site))
        file = [f for f in files if '{0}_NRF'.format(mode) in f][0]
        path_NRF = os.path.join(path_site, file)
        df = pd.read_csv(path_NRF, sep='\t')
        colnames_nrf = [col for col in df.columns if 'NRF' in col]
        groups_precip = []
        for colname in colnames_nrf:
            groups_precip.append(filter_col_name(colname))
        # removing the first group with precipitation events with less than 1mm
        dico[GISID]['groups_precip'] = groups_precip[1:]
        dico[GISID]['lagtime'] =  df.loc[:,'lagtime'].values
        NRF = np.zeros((len(colnames_nrf)-1, len(dico[GISID]['lagtime'])))
        for k, col in enumerate(colnames_nrf[1:]):
            NRF[k,:] = df.loc[:,col].values
        dico[GISID]['group2NRF'] = NRF

        # Error bars
        colnames_se = [col for col in df.columns if 'se_p' in col]
        SE = np.zeros((len(colnames_se)-1, len(dico[GISID]['lagtime'])))
        for k, col in enumerate(colnames_se[1:]):
            SE[k,:] = df.loc[:,col].values
        dico[GISID]['group2NRF_SE'] = SE

        files = os.listdir(os.path.join(path_ERRA, site))
        file = [f for f in files if '{0}_avgRRD'.format(mode) in f][0]
        path_avgRRD = os.path.join(path_site, file)
        df = pd.read_csv(path_avgRRD, sep='\t')
        dico[GISID]['lagtime_RRD'] =  df.loc[:,'lagtime'].values
        dico[GISID]['wtd_avg_RRD_p'] =  df.loc[:,'wtd_avg_RRD_p|all'].values
    return dico

def get_data_from_ERRA_old(path_ERRA):
    dico = {}
    sites = os.listdir(path_ERRA)
    allGISID = []
    for site in sites:
        nameGISID = ''
        count = 6
        while site[count]!='_':
            nameGISID += site[count]
            count += 1
        allGISID.append(nameGISID)
    for GISID in allGISID:
        dico[GISID] = {}
        site = [f for f in sites if f'GISID-{GISID}_' in f][0]
        
        path_site = os.path.join(path_ERRA, site)
        files = os.listdir(os.path.join(path_ERRA, site))
        file = [f for f in files if 'training_NRF' in f][0]
        path_NRF = os.path.join(path_site, file)
        df = pd.read_csv(path_NRF, sep='\t')
        colnames_nrf = [col for col in df.columns if 'NRF' in col]
        groups_precip = []
        for colname in colnames_nrf:
            groups_precip.append(filter_col_name(colname))
        # removing the first group with precipitation events with less than 1mm
        dico[GISID]['groups_precip'] = groups_precip[1:]
        dico[GISID]['lagtime'] =  df.loc[:,'lagtime'].values
        NRF = np.zeros((len(colnames_nrf)-1, len(dico[GISID]['lagtime'])))
        for k, col in enumerate(colnames_nrf[1:]):
            NRF[k,:] = df.loc[:,col].values
        dico[GISID]['group2NRF'] = NRF
        
        files = os.listdir(os.path.join(path_ERRA, site))
        file = [f for f in files if 'training_avgRRD' in f][0]
        path_avgRRD = os.path.join(path_site, file)
        df = pd.read_csv(path_avgRRD, sep='\t')
        dico[GISID]['lagtime_RRD'] =  df.loc[:,'lagtime'].values
        dico[GISID]['wtd_avg_RRD_p'] =  df.loc[:,'wtd_avg_RRD_p|all'].values
    return dico