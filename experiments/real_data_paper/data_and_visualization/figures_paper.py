import pandas as pd
import numpy as np
import sys
import os
path = os.path.abspath("")
print(path)
sys.path.append(path)
save_folder = os.path.join(path, 'figures/')
folder_ERRA = os.path.join(path, 'ERRA_data_plot/')
import pickle
import matplotlib.pyplot as plt
import copy
import torch
from datetime import timedelta, datetime
import matplotlib.dates as mdates

def nse(observed, simulated):
    """
    Calculate the Nash-Sutcliffe Efficiency (NSE).
    
    Parameters:
    observed (array-like): Array of observed values.
    simulated (array-like): Array of simulated values.
    
    Returns:
    float: NSE value.
    """
    observed = np.array(observed)
    simulated = np.array(simulated)
    
    # Calculate the mean of the observed data
    mean_observed = np.mean(observed)
    
    # Compute the numerator and denominator of the NSE formula
    numerator = np.sum((observed - simulated) ** 2)
    denominator = np.sum((observed - mean_observed) ** 2)
    
    # Compute NSE
    nse_value = 1 - (numerator / denominator)
    
    return nse_value


all_GISID = ['50','14']
all_sitecode = ['BAFU-2312','BAFU-2132']

    
from datetime import datetime
for ii in range(len(all_GISID)):

    GISID = all_GISID[ii]
    sc = all_sitecode[ii]
    m = m_list[ii]
    agg = agg_list[ii]
    h = h_list[ii]
    with open(os.path.join(path, 'save', 'GISID_{0}.pkl'.format(GISID)), 'rb') as handle:
        dico = pickle.load(handle)
    
    NRF = pd.read_csv(folder_ERRA+r'/GISID-{}_{}/GISID-{}_{}_fordraft_NRF_m={}{}_h={}_nlin.txt'.format(GISID,sc,GISID,sc,m,agg,h), sep='\t')
    RRD = pd.read_csv(folder_ERRA+r'/GISID-{}_{}/GISID-{}_{}_fordraft_avgRRD_m={}{}_h={}_nlin.txt'.format(GISID,sc,GISID,sc,m,agg,h), sep='\t')

    fig, axs = plt.subplots(2, 2, figsize=(15,6), gridspec_kw={'width_ratios': [4, 1]})
    dates = [datetime(int(year), 1, 1) + (datetime(int(year)+1, 1, 1) - datetime(int(year), 1, 1)) * (year - int(year)) for year in dico['dates']]

    ########## Streamflow time series
    # Estimated streamflow using the model trained on the site considered
    maxQ = np.max([np.max(dico['Qhat'][-24*365:]),np.max(dico['Qtrue'][-24*365:]),np.max(dico['Qpred'][-24*365:])])
    axs[0,0].plot(dates[-24*365:], dico['Qtrue'][-24*365:], label='Measurements', lw=0.9, color='grey')
    axs[0,0].plot(dates[-24*365:], dico['Qhat'][-24*365:], label='GAMCR training estimation', lw=0.9, color='mediumseagreen', linestyle='-.')
    axs[0,0].legend(loc='upper left')    
    axs[0,0].text(.01, .70, 'NSE: {:.3f}'.format(nse( dico['Qhat'], dico['Qtrue'] )), size=10,color='grey',ha='left', va='top', transform=axs[0,0].transAxes)
    axs[0,0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=np.arange(1, 13, 3)))
    axs[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    axs[0,0].set_ylim(ymax=maxQ*1.05)

    # Predicted streamflow when predicting the transfer functions on this site that was held out when training our model from catchment features to model parameters 
    axs[1,0].plot(dates[-24*365:], dico['Qtrue'][-24*365:], label='Measurements', lw=0.9, color='grey')
    axs[1,0].plot(dates[-24*365:], dico['Qpred'][-24*365:], label='GAMCR LOO estimation', lw=0.9, color='darkorange',  linestyle='-.')
    axs[1,0].legend(loc='upper left') 
    axs[1,0].text(.01, .70, 'NSE: {:.3f}'.format(nse( dico['Qpred'], dico['Qtrue'] )), size=10,color='grey',ha='left', va='top', transform=axs[1,0].transAxes)
    axs[1,0].xaxis.set_major_locator(mdates.MonthLocator(bymonth=np.arange(1, 13, 3)))
    axs[1,0].xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    axs[1,0].set_ylim(ymax=maxQ*1.05)


    ########## Transfer functions
    maxRRD = np.max([np.max(dico['Hhat'][:24*4]),np.max(dico['Hpred'][:24*4]),np.max(RRD['wtd_avg_RRD_p|all'])])
    # Estimated global transfer function using the model trained on the site considered
    axs[0,1].plot(RRD['lagtime']/24, RRD['wtd_avg_RRD_p|all'], label='ERRA', color='grey', lw=0.9)
    axs[0,1].fill_between(RRD['lagtime']/24, RRD['wtd_avg_RRD_p|all']-RRD['se_p|all'], RRD['wtd_avg_RRD_p|all']+RRD['se_p|all'],color='lightgrey', alpha=0.6)
    axs[0,1].plot([i/24 for i in range(24*4)], dico['weighted_avg_RRD_hat'][:24*4], label='GAMCR on test data', color='mediumseagreen',  linestyle='-.', lw=0.9)
    axs[0,1].legend(loc='upper right', fontsize=7)
    axs[0,1].set_ylim(ymax=maxRRD*1.05)
    # Predicted global transfer on this site that was held out wh3n training our model from catchment features to model parameters 
    axs[1,1].plot(RRD['lagtime']/24, RRD['wtd_avg_RRD_p|all'], label='ERRA', color='grey', lw=0.9)
    axs[1,1].fill_between(RRD['lagtime']/24, RRD['wtd_avg_RRD_p|all']-RRD['se_p|all'], RRD['wtd_avg_RRD_p|all']+RRD['se_p|all'],color='lightgrey', alpha=0.6)
    axs[1,1].plot([i/24 for i in range(24*4)], dico['weighted_avg_RRD_pred'][:24*4], label='GAMCR LOO estimation', color='darkorange',  linestyle='-.', lw=0.9)
    axs[1,1].legend(loc='upper right', fontsize=7)
    axs[1,1].set_xlabel('Lag time (days)', fontsize=13)
    axs[1,1].set_ylim(ymax=maxRRD*1.05)

    fig.text(0.09, 0.5, r'Streamflow $(mm\, h^{-1})$', va='center', ha='center', rotation='vertical', fontsize=12)
    fig.text(0.71, 0.5, r'Weighted average RRD $(h^{-1})$', va='center', ha='center', rotation='vertical', fontsize=12)

    # fig.suptitle('GISID: {0}'.format(all_GISID[ii]))

    fig.savefig(folder_ERRA+r'/GISID-{}_{}/Estimate_GISID_{}.pdf'.format(GISID,sc,GISID), bbox_inches='tight') 
    # plt.show()
    # quit()
