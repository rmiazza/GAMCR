import matplotlib.pyplot as plt
import pickle
import pandas as pd
import os
import numpy as np


def show_tf_p_q(site_folder, site, stratif_wetness=True, show_CI=True, weighted=True, alpha=0.1, maxT=None):
    folder = os.path.join(site_folder, "results")
    H_weighted_avg = np.load(os.path.join(folder, 'H_weighted_avg.npy'))
    H_avg = np.load(os.path.join(folder, 'H_avg.npy'))
    m = H_avg.shape[1]

    with open(os.path.join(folder, 'groups_precip.pkl'), 'rb') as handle:
        groups_precip = pickle.load(handle)
        nJ = len(groups_precip)
    with open(os.path.join(folder, 'groups_wetness.pkl'), 'rb') as handle:
        groups_wetness = pickle.load(handle)
        nQ = len(groups_wetness)
    
    with open(os.path.join(folder, 'group2p_range.pkl'), 'rb') as handle:
        group2p_range = pickle.load(handle)
    with open(os.path.join(folder, 'group2q_range.pkl'), 'rb') as handle:
        group2q_range = pickle.load(handle)
    with open(os.path.join(folder, 'group2nbpoints.pkl'), 'rb') as handle:
        group2nbpoints = pickle.load(handle)

    try:
        H_weighted_avg_true = np.load(os.path.join(folder, 'H_weighted_avg_true.npy'))
        H_avg_true = np.load(os.path.join(folder, 'H_avg_true.npy'))
        with open(os.path.join(folder, 'group2nbpoints_true.pkl'), 'rb') as handle:
            group2nbpoints_true = pickle.load(handle)
        true_tfs = True
    except:
        true_tfs = False
        pass
        

    if maxT is None:
        maxT = m
    x = np.arange(0,m,1)/24
    colors = ['red', 'orange', 'green', 'cyan', 'blue']
    if stratif_wetness:
        K= nQ
    else:
        K = nJ
    fig,a =  plt.subplots(int(np.ceil(K/2)),2)
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make space for the super title
    idx2legend = {}
    for j in range(nJ):
        for k in range(nQ):
            if stratif_wetness:
                low, up = np.round(group2q_range[k], 3)
                upleg = up if k!=(nQ-1) else 'max'
                id_x, id_y = k//2, k%2
                a[id_x][id_y].set_title('Wetness range: {0}-{1}'.format(low,upleg))
                id_col = j
                low, up = np.round(group2p_range[nQ*j+k], 1)
                upleg = up if j!=(nJ-1) else 'max'
                title_leg = 'Precipitation range'
                idx2legend[j] = '{0}-{1}'.format(low,upleg)
            else:
                low, up = np.round(group2p_range[nQ*j+k], 1)
                upleg = up if j!=(nJ-1) else 'max'
                id_x, id_y = j//2, j%2
                a[id_x][id_y].set_title('Precipitation range: {0}-{1}'.format(low,upleg))
                id_col = k
                low, up = np.round(group2q_range[k], 3)
                upleg = up if k!=(nQ-1) else 'max'
                title_leg = 'Ant. wetness range'
                idx2legend[k] = '{0}-{1}'.format(low,upleg)
                
            if weighted:
                a[id_x][id_y].plot(x[:maxT],H_weighted_avg[nQ*j+k,:maxT], color=colors[id_col])
                if true_tfs:
                    a[id_x][id_y].plot(x[:maxT],H_weighted_avg_true[nQ*j+k,:maxT], linestyle='--', color=colors[id_col])
                a[id_x][id_y].set_ylabel('NRF', fontsize=14)
            else:
                a[id_x][id_y].plot(x[:maxT],H_avg[nQ*j+k,:maxT], color=colors[id_col])
                if true_tfs:
                    a[id_x][id_y].plot(x[:maxT],H_avg[nQ*j+k,:maxT],linestyle='--', color=colors[id_col])
                a[id_x][id_y].set_ylabel('RRD', fontsize=14)
                    
    for idx in range(K):
        try:
            plt.plot([],[], color=colors[idx], label=idx2legend[idx])
        except:
            pass
    
    fig.suptitle('{0}'.format(site))
    fig.subplots_adjust(top=0.88, hspace=0.4)
    plt.legend(title=title_leg)
    plt.show()


def show_tf_p_or_q(site_folder, site, stratif_wetness=True, weighted=True, show_CI=True, alpha=0.1, maxT=None, dataERRA=None, figname=None):
    folder = os.path.join(site_folder, "results")
    H_weighted_avg = np.load(os.path.join(folder, 'H_weighted_avg.npy'))
    H_avg = np.load(os.path.join(folder, 'H_avg.npy'))
    m = H_avg.shape[1]
    
    with open(os.path.join(folder, 'groups_precip.pkl'), 'rb') as handle:
        groups_precip = pickle.load(handle)
        nJ = len(groups_precip)
    with open(os.path.join(folder, 'groups_wetness.pkl'), 'rb') as handle:
        groups_wetness = pickle.load(handle)
        nQ = len(groups_wetness)
        
    with open(os.path.join(folder, 'group2p_range.pkl'), 'rb') as handle:
        group2p_range = pickle.load(handle)
    with open(os.path.join(folder, 'group2q_range.pkl'), 'rb') as handle:
        group2q_range = pickle.load(handle)
    with open(os.path.join(folder, 'group2nbpoints.pkl'), 'rb') as handle:
        group2nbpoints = pickle.load(handle)

    if stratif_wetness:
        K = nQ
    else:
        K = nJ
    try:
        H_weighted_avg_true = np.load(os.path.join(folder, 'H_weighted_avg_true.npy'))
        H_avg_true = np.load(os.path.join(folder, 'H_avg_true.npy'))
        with open(os.path.join(folder, 'group2nbpoints_true.pkl'), 'rb') as handle:
            group2nbpoints_true = pickle.load(handle)
        true_tfs = True
        tf_true = np.zeros((K,m))
        norm_true = np.zeros(K)
    except:
        true_tfs = False
        pass

    if maxT is None:
        maxT = m
    if not(dataERRA is None):
        maxT = int(np.max(dataERRA['lagtime']))
    x = np.arange(0,maxT,1)/24
    colors = ['red', 'orange', 'green', 'cyan', 'blue']
    tf = np.zeros((K,m))
    norm = np.zeros(K)
    idx2legends = ['' for k in range(K)]

    for j in range(nJ):
        for k in range(nQ):
            if stratif_wetness:
                idx = k
                low,up = np.round(group2q_range[nQ*j+k], 3)
                upleg = up if idx!=(nQ-1) else 'max'
                idx2legends[idx] = '{0}-{1}'.format(low,upleg)
                tit_legend = 'Antecedent wetness'
            else:
                idx = j
                low,up = np.round(group2p_range[nJ*j+k], 1)
                upleg = up if idx!=(nJ-1) else 'max'
                idx2legends[idx] = '{0}-{1}'.format(low,upleg)
                tit_legend = 'Precipitation intensity'


            if group2nbpoints[nQ*j+k]>1:
                norm[idx] += group2nbpoints[nQ*j+k]
                tf[idx,:] += H_weighted_avg[nQ*j+k,:] * group2nbpoints[nQ*j+k]
            if true_tfs:
                if group2nbpoints_true[nQ*j+k]>1:
                    norm_true[idx] += group2nbpoints_true[nQ*j+k]
                    tf_true[idx,:] += H_weighted_avg_true[nQ*j+k,:] * group2nbpoints_true[nQ*j+k]
    for idx in range(K):
        tf[idx,:] /= norm[idx]
    for idx in range(K):
        plt.plot(x[:maxT],tf[idx,:maxT], color=colors[idx])
        plt.plot([],[], color=colors[idx], label=str(idx2legends[idx]))

    plt.plot([],[], color='black',  label='GAMCR')
    if true_tfs:
        for idx in range(K):
            tf_true[idx,:] /= norm_true[idx]
        for idx in range(K):
            plt.plot(x[:maxT],tf_true[idx,:maxT], color=colors[idx], linestyle='--')
        plt.plot([],[], color='black', linestyle='--', label='Ground truth')
    
    if weighted:
        if not(dataERRA is None):
            for idx in range(K):
                plt.plot(dataERRA['lagtime']/24,dataERRA['group2NRF'][idx,:], color=colors[idx], linestyle=':')
            plt.plot([],[], color='black', linestyle=':', label='ERRA')

        plt.ylabel('NRF', fontsize=14)
    else:
        plt.ylabel('RRD', fontsize=14)
    plt.xlabel('Lag (in days)', fontsize=14)
    plt.legend(title=tit_legend)
    if not(figname is None):
        plt.savefig(figname+'.png', dpi=250, bbox_inches='tight')
    else:
        plt.title("{0}".format(site))
    plt.show()


def get_colors(n):
    cmap = plt.cm.get_cmap('tab20', n)  # You can use other colormaps like 'viridis', 'plasma', etc.
    return [cmap(i) for i in range(n)]

def show_tf_global(global_path, all_sites, log_abs=False, weighted=True, alpha=0.1, maxT=24*20, log_ordo=False, show_sites_labels=True, figsave=True):
    site2tf = {}
    colors = get_colors(len(all_sites))
    
    #colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', '#FFA500', '#800080', '#A52A2A', '#FFC0CB', '#00FF00', '#808000', '#000080', '#008080']
    id_site = 0
    for site in all_sites:
        folder = os.path.join(global_path, site, 'results')

        H_weighted_avg = np.load(os.path.join(folder, 'H_weighted_avg.npy'))
        H_avg = np.load(os.path.join(folder, 'H_avg.npy'))
        m = H_avg.shape[1]
    
            
        with open(os.path.join(folder, 'groups_precip.pkl'), 'rb') as handle:
            groups_precip = pickle.load(handle)
            nJ = len(groups_precip)
        with open(os.path.join(folder, 'groups_wetness.pkl'), 'rb') as handle:
            groups_wetness = pickle.load(handle)
            nQ = len(groups_wetness)
        
        with open(os.path.join(folder, 'group2p_range.pkl'), 'rb') as handle:
            group2p_range = pickle.load(handle)
        with open(os.path.join(folder, 'group2q_range.pkl'), 'rb') as handle:
            group2q_range = pickle.load(handle)
        with open(os.path.join(folder, 'group2nbpoints.pkl'), 'rb') as handle:
            group2nbpoints = pickle.load(handle)

        try:
            H_weighted_avg_true = np.load(os.path.join(folder, 'H_weighted_avg_true.npy'))
            H_avg_true = np.load(os.path.join(folder, 'H_avg_true.npy'))
            with open(os.path.join(folder, 'group2nbpoints_true.pkl'), 'rb') as handle:
                group2nbpoints_true = pickle.load(handle)
            true_tfs = True
            tf_true = np.zeros(m)
            norm_true = 0
        except:
            true_tfs = False
            pass
        
        x = np.arange(0,m,1)/24
        tf = np.zeros(m)
        norm = 0
    
        for j in range(1,nJ):
            for k in range(nQ):
                if weighted:
                    if group2nbpoints[nQ*j+k]>1:
                        norm += group2nbpoints[nQ*j+k]
                        tf += H_weighted_avg[nQ*j+k,:] * group2nbpoints[nQ*j+k]

                    if true_tfs:
                        if group2nbpoints_true[nQ*j+k]>1:
                            norm_true += group2nbpoints_true[nQ*j+k]
                            tf_true += H_weighted_avg_true[nQ*j+k,:] * group2nbpoints_true[nQ*j+k]
                else:
                    if group2nbpoints[nQ*j+k]>1:
                        norm += group2nbpoints[nQ*j+k]
                        tf += H_avg[nQ*j+k,:] * group2nbpoints[nQ*j+k]
                    if true_tfs:
                        if group2nbpoints_true[nQ*j+k]>1:
                            norm_true += group2nbpoints_true[nQ*j+k]
                            tf_true += H_avg_true[nQ*j+k,:] * group2nbpoints_true[nQ*j+k]

        tf /= norm
        if true_tfs:
            tf_true /= norm_true
        if log_abs:
            abs = np.log10(x[:maxT])
        else:
            abs = x[:maxT]
        if log_ordo:
            if show_sites_labels:
                plt.plot(abs,np.log10(tf[:maxT]), color=colors[id_site], label=site)
            else:
                plt.plot(abs,np.log10(tf[:maxT]), color=colors[id_site])
            if true_tfs:
                plt.plot(abs,np.log10(tf_true[:maxT]), color=colors[id_site], linestyle='--')            
        else:
            if show_sites_labels:
                plt.plot(abs,tf[:maxT], color=colors[id_site], label=site)
            else:
                plt.plot(abs,tf[:maxT], color=colors[id_site])
            if true_tfs:
                plt.plot(abs,tf_true[:maxT], color=colors[id_site], linestyle='--')

        id_site += 1
    
    if weighted:
        if log_ordo:
            title = 'Log(NRF)'
        else:
            title = 'NRF'
    else:
        if log_ordo:
            title = 'Log(RRD)'
        else:
            title = 'RRD'
    plt.ylabel(title, fontsize=14)
    if not(log_abs):
        title += '_days'
        plt.xlabel('Lag in days', fontsize=14)
    else:
        title += '_logdays'
        plt.xlabel('Log( Lag in days )', fontsize=14)
            
    plt.plot([],[], color='black', label='GAMCR')
    if true_tfs:
        plt.plot([],[], color='black', linestyle='--', label='Ground truth')
    plt.legend()
    title += 'global.png'
    if figsave:
        plt.savefig(title, dpi=250, bbox_inches='tight')
    plt.show()

def show_vs_precip_intensity(global_path, all_sites, weighted=True, log_ordo=False, dataERRA=None, show_GAMCR=True, show_sites_labels=True, figsave=None):
    site2tf = {}
    import math
    colors = get_colors(len(all_sites))
    #colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', '#FFA500', '#800080', '#A52A2A', '#FFC0CB', '#00FF00', '#808000', '#000080', '#008080']
    id_site = 0

    site2area_esti = {}
    site2peak_esti = {}
    site2mean_esti = {}
    site2peaklag_esti = {}

    site2quantiles = {}
    
    site2area_esti_noweight = {}
    site2peak_esti_noweight = {}
    site2mean_esti_noweight = {}
    site2peaklag_esti_noweight = {}

    site2area_true = {}
    site2peak_true = {}
    site2mean_true = {}
    site2peaklag_true = {}

    site2quantiles_true = {}
    
    site2area_true_noweight = {}
    site2peak_true_noweight = {}
    site2mean_true_noweight = {}
    site2peaklag_true_noweight = {}

    for site in all_sites:
        folder = os.path.join(global_path, site, 'results')

        H_weighted_avgbis = np.load(os.path.join(folder, 'H_weighted_avg.npy'))
        H_avgbis = np.load(os.path.join(folder, 'H_avg.npy'))
        m = H_avgbis.shape[1]
        
        with open(os.path.join(folder, 'groups_precip.pkl'), 'rb') as handle:
            groups_precip = pickle.load(handle)
            nJ = len(groups_precip)
        with open(os.path.join(folder, 'groups_wetness.pkl'), 'rb') as handle:
            groups_wetness = pickle.load(handle)
            nQ = len(groups_wetness)
        with open(os.path.join(folder, 'group2p_range.pkl'), 'rb') as handle:
            group2p_range = pickle.load(handle)
        with open(os.path.join(folder, 'group2q_range.pkl'), 'rb') as handle:
            group2q_range = pickle.load(handle)
        with open(os.path.join(folder, 'group2nbpoints.pkl'), 'rb') as handle:
            group2nbpoints = pickle.load(handle)

        group2means_precip = np.load(os.path.join(folder, 'group2means_precip.npy'))
        group2means_wetness = np.load(os.path.join(folder, 'group2means_wetness.npy'))


        K = nJ
        try:
            H_weighted_avgbis_true = np.load(os.path.join(folder, 'H_weighted_avg_true.npy'))
            H_avgbis_true = np.load(os.path.join(folder, 'H_avg_true.npy'))
            H_weighted_avg_true = np.zeros((K,m))
            H_avg_true = np.zeros((K,m))
            group2means_precip_true = np.load(os.path.join(folder, 'group2means_precip_true.npy'))
            group2means_wetness_true = np.load(os.path.join(folder, 'group2means_wetness_true.npy'))

            quantiles_precip_true = np.zeros(K)
            with open(os.path.join(folder, 'group2nbpoints_true.pkl'), 'rb') as handle:
                group2nbpoints_true = pickle.load(handle)
            true_tfs = True
            tf_true = np.zeros(m)
            norm_true = np.zeros(K)
        except:
            true_tfs = False
            pass

        H_weighted_avg = np.zeros((K,m))
        H_avg = np.zeros((K,m))
        quantiles_precip = np.zeros(K)
        norm = np.zeros(K)
        for j in range(K):
            for k in range(nQ):
                idx = j
                if group2nbpoints[nQ*j+k]>1:
                    norm[idx] += group2nbpoints[nQ*j+k]
                    H_weighted_avg[idx,:] += H_weighted_avgbis[nQ*j+k,:] * group2nbpoints[nQ*j+k]
                    H_avg[idx,:] += H_avgbis[nQ*j+k,:] * group2nbpoints[nQ*j+k]
                    quantiles_precip[idx] += group2means_precip[nQ*j+k] * group2nbpoints[nQ*j+k]

                if true_tfs:
                    if group2nbpoints_true[nQ*j+k]>1:
                        norm_true[idx] += group2nbpoints_true[nQ*j+k]
                        H_weighted_avg_true[idx,:] += H_weighted_avgbis_true[nQ*j+k,:] * group2nbpoints_true[nQ*j+k]
                        H_avg_true[idx,:] += H_avgbis_true[nQ*j+k,:] * group2nbpoints_true[nQ*j+k]
                        quantiles_precip_true[idx] += group2means_precip_true[nQ*j+k] * group2nbpoints_true[nQ*j+k]
        for idx in range(K):
            H_weighted_avg[idx,:] /= norm[idx]
            quantiles_precip[idx] /= norm[idx]
            if true_tfs:
                H_weighted_avg_true[idx,:] /= norm_true[idx]
                quantiles_precip_true[idx] /= norm_true[idx]

        site2area_esti[site] = np.zeros(K)
        site2peak_esti[site] = np.zeros(K)
        site2mean_esti[site] = np.zeros(K)
        site2peaklag_esti[site] = np.zeros(K)

        site2area_esti_noweight[site] = np.zeros(K)
        site2peak_esti_noweight[site] = np.zeros(K)
        site2mean_esti_noweight[site] = np.zeros(K)
        site2quantiles[site] = quantiles_precip
        site2peaklag_esti_noweight[site] = np.zeros(K)

        for k in range(K):
            site2area_esti[site][k] = np.sum(H_weighted_avg[k,:])
            site2peak_esti[site][k] = np.max(H_weighted_avg[k,:])
            site2mean_esti[site][k] = np.mean(H_weighted_avg[k,:])
            site2peaklag_esti[site][k] = np.argmax(H_weighted_avg[k,:])

            site2area_esti_noweight[site][k] = np.sum(H_avg[k,:])
            site2peak_esti_noweight[site][k] = np.max(H_avg[k,:])
            site2mean_esti_noweight[site][k] = np.mean(H_avg[k,:])
            site2peaklag_esti_noweight[site][k] = np.argmax(H_avg[k,:])

        if true_tfs:
            
            site2area_true[site] = np.zeros(K)
            site2peak_true[site] = np.zeros(K)
            site2mean_true[site] = np.zeros(K)
            site2peaklag_true[site] = np.zeros(K)
    
            site2area_true_noweight[site] = np.zeros(K)
            site2peak_true_noweight[site] = np.zeros(K)
            site2mean_true_noweight[site] = np.zeros(K)
            site2quantiles_true[site] = quantiles_precip_true
            site2peaklag_true_noweight[site] = np.zeros(K)
    
            for k in range(K):
                site2area_true[site][k] = np.sum(H_weighted_avg_true[k,:])
                site2peak_true[site][k] = np.max(H_weighted_avg_true[k,:])
                site2mean_true[site][k] = np.mean(H_weighted_avg_true[k,:])
                site2peaklag_true[site][k] = np.argmax(H_weighted_avg_true[k,:])
    
                site2area_true_noweight[site][k] = np.sum(H_avg_true[k,:])
                site2peak_true_noweight[site][k] = np.max(H_avg_true[k,:])
                site2mean_true_noweight[site][k] = np.mean(H_avg_true[k,:])
                site2peaklag_true_noweight[site][k] = np.argmax(H_avg_true[k,:])
    
    
    if true_tfs:
        if weighted:
            stats_true = {'area': site2area_true, 'peak': site2peak_true, 'peaklag': site2peaklag_true, 'mean':site2mean_true, 'quantiles':site2quantiles_true}
        else:
            stats_true = {'area': site2area_true_noweight, 'peak': site2peak_true_noweight, 'peaklag': site2peaklag_true_noweight, 'mean':site2mean_true_noweight, 'quantiles':site2quantiles_true}

    if weighted:
        stats_esti = {'area': site2area_esti, 'peak': site2peak_esti, 'peaklag': site2peaklag_esti, 'mean':site2mean_esti, 'quantiles':site2quantiles}    
        TF = 'NRF'
    else:
        stats_esti = {'area': site2area_esti_noweight, 'peak': site2peak_esti_noweight, 'peaklag': site2peaklag_esti_noweight, 'mean':site2mean_esti_noweight, 'quantiles':site2quantiles}
        TF = 'RRD'

    markers = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*', 'h', 'H']

    stat2label = {'area': '{0} runoff volume'.format(TF),
                  'peak': '{0} peak height'.format(TF),
                  'mean': '{0} mean'.format(TF),
                  'peaklag': '{0} peak lag'.format(TF)
                 }

    def get_stat_ERRA(group2nrf, stat):
        if stat == "area":
            return np.sum(group2nrf, axis=1)
        elif stat == "peak":
            return np.max(group2nrf, axis=1)
        elif stat == "mean":
            return np.mean(group2nrf, axis=1)
        elif stat == "peaklag":
            return np.argmax(group2nrf, axis=1)
    for stat in ['area', 'peak', 'mean', 'peaklag']:
        tickslabel = []
        count_fig = -1
        max_val = -float('inf')
        min_val = float('inf')
        for id_station,site in enumerate(all_sites):
                count_fig += 1
                K = len(stats_esti[stat][site])
                if log_ordo:
                    if show_GAMCR:
                        plt.scatter(stats_esti['quantiles'][site], np.log(stats_esti[stat][site]), marker='x', c=[colors[id_station] for gh in range(K)])
                        plt.plot(stats_esti['quantiles'][site], np.log(stats_esti[stat][site]), c=colors[id_station])
                    if true_tfs:
                        plt.scatter(stats_true['quantiles'][site], np.log(stats_true[stat][site]), marker='+', c=[colors[id_station] for gh in range(K)])
                        plt.plot(stats_true['quantiles'][site], np.log(stats_true[stat][site]), linestyle='--', c=colors[id_station])
                    if dataERRA != None:
                        erra_stats = get_stat_ERRA(dataERRA[site]['group2NRF'], stat)
                        plt.scatter(stats_true['quantiles'][site], np.log(erra_stats), marker='|', c=[colors[id_station] for gh in range(K)])
                        plt.plot(stats_true['quantiles'][site], np.log(erra_stats), linestyle=':', c=colors[id_station])
                else:
                    if show_GAMCR:
                        plt.scatter(stats_esti['quantiles'][site], stats_esti[stat][site], marker='+', c=[colors[id_station] for gh in range(K)])
                        plt.plot(stats_esti['quantiles'][site], stats_esti[stat][site],  c=colors[id_station])
                    if true_tfs:
                        plt.scatter(stats_true['quantiles'][site], stats_true[stat][site], marker='x', c=[colors[id_station] for gh in range(K)])
                        plt.plot(stats_true['quantiles'][site], stats_true[stat][site],  linestyle='--', c=colors[id_station])
                    if dataERRA != None:
                        erra_stats = get_stat_ERRA(dataERRA[site]['group2NRF'], stat)
                        plt.scatter(stats_true['quantiles'][site], erra_stats, marker='|', c=[colors[id_station] for gh in range(K)])
                        plt.plot(stats_true['quantiles'][site], erra_stats, linestyle=':', c=colors[id_station])

                if show_sites_labels:
                    plt.plot([], [], label=site, c=colors[id_station])

        title = stat
        if show_GAMCR:
            title += '_GAMCR'
            plt.plot([],[], color='black', label='GAMCR')
        plt.plot([],[], color='black', linestyle='--', label='Ground truth')
        if dataERRA != None:
            title += '_ERRA'
            plt.plot([],[], color='black', linestyle=':', label='ERRA')

#        plt.legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize='small', title='Site')
        plt.legend(loc=4)

        plt.xlabel('Precipitation ($mm.h^{-1}$)', fontsize=14)
        if log_ordo:
            plt.ylabel('Log( {0} )'.format(stat2label[stat]), fontsize=14)
        else:
            plt.ylabel(stat2label[stat], fontsize=14)
        if figsave:
            plt.savefig(f'{title}.png', dpi=250, bbox_inches='tight')
        plt.title(stat, fontsize=14)
        plt.show()
