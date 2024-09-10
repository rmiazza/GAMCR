

import pyreadr
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt


for station in ['Pully', 'Lugano']:
    for mode in ['flashy','notflashy']:
        site = '{0}_{1}'.format(station, mode)
        folder = '/mydata/watres/quentin/code/FLOW/data/simulated_data_events_updatesummer2024/{0}'.format(site)

        idx_evts = pyreadr.read_r(folder+'/idx_evts.rda') # also works for Rds, rda
        idx_evts = np.array(list(idx_evts.values()))[0,:,0].astype(int)
        transfer = []
        lst_transfer = []
        dates = []
        for i, idx in tqdm(enumerate(idx_evts[1:])):
            try:
                dfp = pd.read_csv(folder+'/{0}_flow_{1}_fluxes.txt'.format(str(idx), mode), sep=',')
                transfer.append(list(dfp['tf'].to_numpy()))
                lst_transfer.append(idx)
                dates.append(dfp.loc[0,'t'])
            except:
                pass
        transfer = np.array(transfer[:-1])
        dates = np.array(dates)
        np.save(folder+'/lst_transfer.npy', np.array(lst_transfer))
        np.save(folder+'/transfer.npy', np.array(transfer))
        np.save(folder+'/dates.npy', np.array(dates))