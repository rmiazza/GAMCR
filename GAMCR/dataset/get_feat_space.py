import numpy as np
import pandas as pd
import os

import sys
import os

# Get the directory of the current file
current_file_directory = os.path.dirname(os.path.abspath(__file__))

# Add this directory to sys.path
# sys.path.append(current_file_directory)
# GISID_GOOD_SITES = pd.read_csv('sites_damseffect_WBgood_coords.csv', encoding = "ISO-8859-1")['ID_GIS_gag'].to_numpy()

# GISID_GOOD_SITES_str = [str(el) for el in GISID_GOOD_SITES]

def get_feat_space(all_GISID=None, get_df=False, normalize=False):
    path_daily = '/mydata/watres/quentin/code/FLOW/data/Daily_Data/'
    path = '/mydata/watres/quentin/code/FLOW/data/'
    file_catchprop = 'CH_Catchments_Geodata_MF_20221209.csv'
    os.chdir(path)
    
    pathdata = '/mydata/watres/quentin/code/FLOW/data/GISID2data/'
    
    
    df = pd.read_csv(path+file_catchprop, header=None,  engine='python')
    
    data_type = df.iloc[0,:]
    description = df.iloc[1,:]
    IDs = df.iloc[2,:]
    df = df.drop([0,1,2])
    rows_idx = list(df.iloc[:,0])
    
    df_info = df.iloc[:,[0,9]]
    df_info.columns = ['GIS_ID', 'INFO']
    
    categories = ['Area', 'Quality', 'Response', 'Climate', 'Altitude', 'Slope', 'runoff accumulation', 
                  'storage capacity', 'permeability', 'waterlogging', 'thoroughness', 'land use',
                 'ground cover' ] + 5*['Geology'] + ['Quaternary Deposits']
    category = None
    category2feature = {'Area': []}
    count = 0
    
    columns = []
    
    for i in range(len(data_type)):
        if type(data_type[i])!=str:
            category2feature[categories[count]].append(IDs[i])
        else:
            count += 1
            try:
                category2feature[categories[count]].append(IDs[i])
            except:
                category2feature[categories[count]] = [IDs[i]]
        columns.append(categories[count]+ ' '+IDs[i])
            
    features2idxcategory = {}
    for i,cat in enumerate(categories):
        for feature in category2feature[cat]:
            features2idxcategory[feature] = i
    df.columns = IDs
    
    
    gis_id = df['GIS_ID']
    df = df.drop(columns=['GIS_ID'])
    df.index = gis_id
    
    df_data = df.iloc[:,4:]
    df_data = df_data.drop(df_data.columns[[3, 4]],axis = 1)
    
    
    cols = list(df_data.columns)
    idx = 0
    
    feature2remove = []
    
    # Removing the categorical quality feature
    while idx<len(cols):
        if 'Alpine' in cols[idx]:
            feature2remove.append(cols[idx])
            idx = len(cols)
        idx += 1
    
    # Removing the features with some NaNs
    for el, row in df_data.isna().sum().items():
        if row !=0:
            #print(el, row)
            feature2remove.append(el)
            
    # Converting to float
    df_data = df_data.replace(',','.', regex=True).astype(float)
    
    # Removing irrelevant features based on stds
    col2std = df_data.std()
    for feat, std in col2std.items():
        if std<0.01:
            feature2remove.append(feat)
    df_data = df_data.drop(feature2remove, axis=1)
    features = df_data.columns
    
    n_features = len(features)
    df_data.head()

    df_data = df_data[df_data.columns]
    if all_GISID is None:
        df_final = df_data.loc[[index in GISID_GOOD_SITES_str for index in df_data.index]]
    else:
        df_final = df_data.loc[[index in all_GISID for index in df_data.index]]
    all_GISID = df_final.index.to_numpy()

    import copy
    feat_space = copy.deepcopy(df_final.to_numpy())
    if normalize:
    #    print(feat_space.shape, all_GISID)
        centerings = {}
        if feat_space.shape[0] != 1:
            norms = np.linalg.norm(feat_space, axis=0)
            for j in range(feat_space.shape[1]):
                centerings[j] = np.mean(feat_space[:,j])
                if norms[j]>1e-3:
                    feat_space[:,j] = (feat_space[:,j]-(np.mean(feat_space[:,j])))/norms[j]
                else:
                    feat_space[:,j] = (feat_space[:,j]-(np.mean(feat_space[:,j])))
        if get_df:
            return feat_space, all_GISID, df_final, centerings, norms
        else:
            return feat_space, all_GISID
    else:
        if get_df:
            return feat_space, all_GISID, df_final
        else:
            return feat_space, all_GISID
