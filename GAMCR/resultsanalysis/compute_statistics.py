import pandas as pd
import os
import numpy as np
import pickle


class ComputeStatistics():
    """
    Class allowing to compute different statistics from the trained model.

    ...

    Methods
    -------
    compute_statistics(site_folder, site, nblocks = 4, min_precip = 1,
                       groups_wetness=None, groups_precip=None, max_files=20,
                       normalization_streamflow=1)
         Compute some statistics on the predicted transfer functions
    """
    def __init__(self):
        pass

    def compute_statistics(
            self, site_folder, site, nblocks=4, min_precip=1,
            groups_wetness=None, groups_precip=None, max_files=20,
            normalization_streamflow=1, save_folder=None,
            filtering_time_points=None):
        """
        Compute information on the learned transfer functions,
        such as:
            - the global average NRD/RDD
            - the average NRF/RRD over some ensembles (you can stratify either
            by precipitation intensity, antecendent wetness or by both)
            - the area, mean, peak and peak lag of the transfer function over
            different ensembles for the precipitation intensity.

        Parameters
        ----------
        site_folder : str
            Path of the folder corresponding to the site
        site : str
            Name of the site
        nblocks : int
            If groups_wetness or groups_precip are None, then we use 'nblocks'
            ensembles to averaged the learned transfer functions (stratifting
            by both antecent wetness and precipitation intensity)
        min_precip : int
            Minimum precipitation intensity considered to define an event
        groups_wetness : dic
            Define the lower and upper values of the ensembles considered to
            stratify with respect to the antecedent wetness
        groups_precip : dic
            Define the lower and upper values of the ensembles considered to
            stratify with respect to precipitation intensity
        max_files : int
            Maximum number of files loaded to compute the statistics (among
            the ones saved when preprocessing the data using one of the "save_batch"
            type method
        normalization_streamflow : positive float
            Normalization vector to apply on the loaded streamflow time series
            (typically to go from cubic meter per second to mm per hour).
        filtering_time_points: function
            Takes as input a list of dates and return the position indexes
            that we should keep to compute the statistics. WARNING: currently,
            this functionality is only supported without the ground truth data
            (I should code a similar filtering procedure to make sure statistics
            from predicted and ground truth data are computed on the same dataset).
        """
        if save_folder is None:
            save_folder = os.path.join(site_folder, 'results')

        save_folder_details = os.path.join(site_folder, 'results', 'detailedresults')

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        if not os.path.exists(save_folder_details):
            os.makedirs(save_folder_details)

        m = self.m

        data_folder = os.path.join(site_folder, 'data')
        X, matJ, y, timeyear, dates = self.load_data(data_folder, max_files=max_files)

        if filtering_time_points is not None:
            idxs = filtering_time_points(dates)
            X = X[idxs, :]
            matJ = matJ[idxs, :, :]
            timeyear = timeyear[idxs]
            dates = dates[idxs]
            y = y[idxs]

        y /= normalization_streamflow

        idx_precip_intensity = int(0)
        H = self.predict_transfer_function(X)

        yhat = self.predict_streamflow(matJ)
        np.save(os.path.join(save_folder_details, 'predicted_streamflow.npy'), yhat)
        np.save(os.path.join(save_folder_details, 'timeyear.npy'), timeyear)

        df_streamflow = pd.DataFrame({'date': timeyear, 'predicted_streamflow': yhat, 'true_streamflow': y})
        df_streamflow.to_csv(os.path.join(save_folder, 'streamflow.csv'), index=False)

        tpos = np.where(X[:, idx_precip_intensity] > min_precip)[0]
        p_sorted = np.sort(X[:, idx_precip_intensity][tpos])
        n = len(p_sorted)

        if groups_precip is None:
            groups_precip = [[0, 10000]]
        elif groups_precip == "auto":
            groups_precip = []
            low = min_precip

            for k in range(nblocks):
                up = p_sorted[int((n-1)//(nblocks))*(k+1)]
                if k == nblocks-1:
                    up += 10000
                groups_precip.append((low, up))
                low = up

        with open(os.path.join(save_folder_details, 'groups_precip.pkl'), 'wb') as handle:
            pickle.dump(groups_precip, handle, protocol=pickle.HIGHEST_PROTOCOL)

        q_sorted = np.sort(y)
        n = len(q_sorted)

        if groups_wetness is None:
            groups_wetness = [[0, 10000]]
        elif groups_wetness == "auto":
            groups_wetness = []
            low = 0

            for k in range(nblocks):
                up = q_sorted[int((n-1)//(nblocks))*(k+1)]

                if k == nblocks-1:
                    up += 10000

                groups_wetness.append((low, up))
                low = up

        with open(os.path.join(save_folder_details, 'groups_wetness.pkl'), 'wb') as handle:
            pickle.dump(groups_wetness, handle, protocol=pickle.HIGHEST_PROTOCOL)

        T = {}
        g = 0
        group2p_range = {}
        group2q_range = {}

        for low, up in groups_precip:
            for lowq, upq in groups_wetness:
                a = np.where(
                    (X[1:, idx_precip_intensity] < up)
                    * (low <= X[1:, idx_precip_intensity])*(y[:-1] < upq)
                    * (lowq <= y[:-1]))[0]
                T[g] = np.array([int(1+el) for el in a]).astype(int)
                group2p_range[g] = [low, up]
                group2q_range[g] = [lowq, upq]
                g += 1

        with open(os.path.join(save_folder_details, 'group2p_range.pkl'), 'wb') as handle:
            pickle.dump(group2p_range, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(save_folder_details, 'group2q_range.pkl'), 'wb') as handle:
            pickle.dump(group2q_range, handle, protocol=pickle.HIGHEST_PROTOCOL)

        H_weighted_avg = np.zeros((len(T), m))
        H_avg = np.zeros((len(T), m))
        weighted_avg_RRD = np.zeros((len(T), m))
        group2nbpoints = {}
        group2means_precip = np.zeros(len(T))
        group2means_wetness = np.zeros(len(T))

        for k in range(len(T)):
            group2nbpoints[k] = len(T[k])
            block = T[k]
            Jt = np.tile((X[:, idx_precip_intensity][block]).reshape(-1, 1), (1, m))
            group2means_precip[k] = np.mean(Jt)
            group2means_wetness[k] = np.mean(y[np.array([el-1 for el in block]).astype(int)])
            H_weighted_avg[k, :] = np.mean(H[block, :]*Jt, axis=0)
            H_weighted_avg[k, :] = H_weighted_avg[k, :] * (H_weighted_avg[k, :] > 0)
            weighted_avg_RRD[k, :] = np.mean(H[block, :]*Jt, axis=0) / np.mean(Jt[:, 0])
            H_avg[k, :] = np.mean(H[block, :], axis=0)
            H_avg[k, :] = H_avg[k, :] * (H_avg[k, :] >= 0)

        try:
            lst_transfer = np.load(os.path.join(data_folder, 'lst_transfer.npy'))
            transfer = np.load(os.path.join(data_folder, 'transfer.npy'))
            true_tfs = True
        except:
            true_tfs = False

        if true_tfs:
            normalization = np.zeros(len(T))
            H_weighted_avg_true = np.zeros((len(T), m))
            H_avg_true = np.zeros((len(T), m))
            weighted_avg_RRD_true = np.zeros((len(T), m))
            lsJ_events = []

            group2nbpoints_true = {k: 0 for k in range(len(T))}
            group2means_precip_true = np.zeros(len(T))
            group2means_wetness_true = np.zeros(len(T))
            df = pd.read_csv(os.path.join(site_folder, 'data_'+site+'.txt'))

            for count, t in enumerate(lst_transfer[:-1]):
                lsJ_events.append(df['p'].iloc[t-1])
                # we use "t-1" since files have been saved using R where
                # indexing starts at 1 (and not at 0 like in Python)
                # assert (df['p'].iloc[t-1]>0.9)

                for k in range(len(T)):
                    count_weights = 0

                    if ((group2p_range[k][0] <= df['p'][t-1])
                        and (df['p'][t-1] <= group2p_range[k][1])
                        and (group2q_range[k][0] <= df['q'][t-2])
                        and (df['q'][t-2] <= group2q_range[k][1])):

                        group2nbpoints_true[k] += 1
                        normalization[k] += 1
                        H_weighted_avg_true[k, :] += transfer[count, :m]
                        H_avg_true[k, :] += transfer[count, :m] / df['p'].iloc[t-1]
                        weighted_avg_RRD_true[k, :] += transfer[count, :m]
                        count_weights += df['p'].iloc[t-1]
                        group2means_precip_true[k] += df['p'].iloc[t-1]
                        group2means_wetness_true[k] += df['q'][t-2]

                    weighted_avg_RRD_true[k, :] /= count_weights

            for k, norm in enumerate(normalization):
                H_weighted_avg_true[k, :] /= norm
                group2means_precip_true[k] /= norm
                group2means_wetness_true[k] /= norm
                H_weighted_avg_true[k, :] = H_weighted_avg_true[k, :] * (H_weighted_avg_true[k, :] >= 0)
                H_avg_true[k, :] /= norm
                H_avg_true[k, :] = H_avg_true[k, :] * (H_avg_true[k, :] >= 0)

            np.save(os.path.join(save_folder_details, 'group2means_wetness_true.npy'), group2means_wetness_true)
            np.save(os.path.join(save_folder_details, 'group2means_precip_true.npy'), group2means_precip_true)

            np.save(os.path.join(save_folder_details, 'NRF_avg_true.npy'), H_weighted_avg_true)
            np.save(os.path.join(save_folder_details, 'RRD_avg_true.npy'), H_avg_true)   
            np.save(os.path.join(save_folder_details, 'weighted_avg_RRD_true.npy'), weighted_avg_RRD_true)

            with open(os.path.join(save_folder_details, 'group2nbpoints_true.pkl'), 'wb') as handle:
                pickle.dump(group2nbpoints_true, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(save_folder_details, 'group2nbpoints.pkl'), 'wb') as handle:
            pickle.dump(group2nbpoints, handle, protocol=pickle.HIGHEST_PROTOCOL)

        K = len(T)
        site2area = np.zeros(K)
        site2peak = np.zeros(K)
        site2mean = np.zeros(K)
        site2area_esti = np.zeros(K)
        site2peak_esti = np.zeros(K)
        site2mean_esti = np.zeros(K)

        site2area_noweight = np.zeros(K)
        site2peak_noweight = np.zeros(K)
        site2mean_noweight = np.zeros(K)
        site2area_esti_noweight = np.zeros(K)
        site2peak_esti_noweight = np.zeros(K)
        site2mean_esti_noweight = np.zeros(K)

        for k in range(K):
            if true_tfs:
                site2area[k] = np.sum(H_weighted_avg_true[k, :])
                site2peak[k] = np.max(H_weighted_avg_true[k, :])
                site2mean[k] = np.mean(H_weighted_avg_true[k, :])
                site2area_noweight[k] = np.sum(H_avg_true[k, :])
                site2peak_noweight[k] = np.max(H_avg_true[k, :])
                site2mean_noweight[k] = np.mean(H_avg_true[k, :])

            site2area_esti[k] = np.sum(H_weighted_avg[k, :])
            site2peak_esti[k] = np.max(H_weighted_avg[k, :])
            site2mean_esti[k] = np.mean(H_weighted_avg[k, :])
            site2area_esti_noweight[k] = np.sum(H_avg[k, :])
            site2peak_esti_noweight[k] = np.max(H_avg[k, :])
            site2mean_esti_noweight[k] = np.mean(H_avg[k, :])

        np.save(os.path.join(save_folder_details, 'NRF_avg.npy'), H_weighted_avg)
        np.save(os.path.join(save_folder_details, 'RRD_avg.npy'), H_avg)
        np.save(os.path.join(save_folder_details, 'weighted_avg_RRD.npy'), weighted_avg_RRD)   

        np.save(os.path.join(save_folder_details, 'group2means_wetness.npy'), group2means_wetness)
        np.save(os.path.join(save_folder_details, 'group2means_precip.npy'), group2means_precip)
        np.save(os.path.join(save_folder_details, 'dates.npy'), timeyear)

        weights_events = np.tile(X[:, 0].reshape(-1, 1), (1, m))
        weighted_avg_RRD_global = np.mean(H * weights_events, axis=0)/np.mean(X[:, 0])
        np.save(os.path.join(save_folder_details, 'weighted_avg_RRD.npy'), weighted_avg_RRD_global)

        df_transfer = {}
        df_transfer['weighted_avg_RRD'] = weighted_avg_RRD_global.reshape(-1)
        df_transfer['RRD'] = np.mean(H, axis=0)
        df_transfer['NRF'] = np.mean(H * weights_events, axis=0)

        for k in range(K):
            df_transfer[f'predicted_NRF_group_{k}'] = H_weighted_avg[k, :]
            df_transfer[f'predicted_RRD_group_{k}'] = H_avg[k, :]
            df_transfer[f'predicted_weighted_avg_RRD_group_{k}'] = weighted_avg_RRD[k, :]

        if true_tfs:
            for k in range(K):
                df_transfer[f'true_NRF_group_{k}'] = H_weighted_avg_true[k, :]
                df_transfer[f'true_RRD_group_{k}'] = H_avg_true[k, :]
                df_transfer[f'true_weighted_avg_RRD_group_{k}'] = weighted_avg_RRD_true[k, :]

            max_length = min([H_avg.shape[1], H_avg_true.shape[1]])
            for key, val in df_transfer.items():
                df_transfer[key] = val[:max_length]

        for key, val in df_transfer.items():
            print(key, len(df_transfer[key]))

        df_transfer = pd.DataFrame(df_transfer)
        df_transfer.to_csv(os.path.join(save_folder, 'NRF_RRD.csv'), index=False)

        df_groups = {'Precipitation range': [], 'Streamflow range': []}
        for g in range(len(T)):
            [low, up] = group2p_range[g]
            low = np.round(low, 4)
            up = np.round(up, 4)
            df_groups['Precipitation range'].append(str(low) + '-' + str(up))

            [lowq, upq] = group2q_range[g]
            lowq = np.round(lowq, 4)
            upq = np.round(upq, 4)
            df_groups['Streamflow range'].append(str(lowq) + '-' + str(upq))

        df_groups = pd.DataFrame(df_groups)
        df_groups.to_csv(os.path.join(save_folder, 'groups.csv'), index=False)

        stats_esti = {
            'area': site2area_esti,
            'peak': site2peak_esti,
            'mean': site2mean_esti
        }
        stats_esti_noweight = {
            'area': site2area_esti_noweight,
            'peak': site2peak_esti_noweight,
            'mean': site2mean_esti_noweight
        }

        with open(os.path.join(save_folder_details, 'stats_NRF.pkl'), 'wb') as handle:
            pickle.dump(stats_esti, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(save_folder_details, 'stats_RRD.pkl'), 'wb') as handle:
            pickle.dump(stats_esti_noweight, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if true_tfs:
            stats_true = {'area': site2area, 'peak': site2peak, 'mean': site2mean}
            stats_true_noweight = {'area': site2area_noweight, 'peak': site2peak_noweight, 'mean': site2mean_noweight}

            with open(os.path.join(save_folder_details, 'stats_NRF_true.pkl'), 'wb') as handle:
                pickle.dump(stats_true, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(save_folder_details, 'stats_RRD_true.pkl'), 'wb') as handle:
                pickle.dump(stats_true_noweight, handle, protocol=pickle.HIGHEST_PROTOCOL)
