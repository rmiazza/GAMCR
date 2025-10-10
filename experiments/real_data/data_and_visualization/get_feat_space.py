import numpy as np
import pandas as pd
import copy


def get_feat_space(path_Catchments_Geodata, all_GISID=None, get_df=False, normalize=False):
    """Extract and prepare the catchment feature space from a catchment attributes file.

    This function reads a geospatial catchment-property table and constructs
    a numerical feature matrix. It performs column filtering, cleaning, conversion, and optional
    normalization. The resulting "feature space" represents quantitative
    descriptors (e.g., area, slope, climate, geology) for the specified
    catchments.

    Steps:
        1. Load the catchment geodata file (CSV) and parse header metadata.
        2. Group features by their physical or geological category.
        3. Remove categorical, missing, or non-informative features.
        4. Convert numerical strings to floats.
        5. Optionally normalize each feature column to zero mean and unit norm.
        6. Return the cleaned and optionally normalized feature space along
           with associated catchment IDs (and optionally the processed DataFrame).

    Args:
        path_Catchments_Geodata (str):
            Path to the CSV file containing catchment attributes.
        all_GISID (list or np.ndarray, optional):
            List or array of GIS IDs identifying which catchments to include.
            If None, all catchments are processed.
        get_df (bool, optional):
            If True, also return the cleaned pandas DataFrame (and normalization
            statistics if applicable). Defaults to False.
        normalize (bool, optional):
            If True, normalize each feature to have zero mean and unit Euclidean
            norm. Defaults to False.

    Returns:
        tuple:
            Depending on the chosen options:

            - If `normalize` is False:
                * (np.ndarray, np.ndarray): (feature_space, all_GISID)
                * or (np.ndarray, np.ndarray, pd.DataFrame): adds df_final if `get_df=True`.

            - If `normalize` is True:
                * (np.ndarray, np.ndarray): (normalized_feature_space, all_GISID)
                * or (np.ndarray, np.ndarray, pd.DataFrame, dict, np.ndarray):
                  adds df_final, centering means, and feature norms if `get_df=True`.

    Notes:
        - The CSV is expected to have categorical headers in the first three rows.
        - Features with missing values or very low variability (std < 0.01)
          are automatically excluded.
        - The function assumes that numeric entries may use commas as decimal
          separators and converts them appropriately.
        - Normalization uses Euclidean norm scaling per feature column.
    """
    df_catchment_properties = pd.read_csv(path_Catchments_Geodata, header=None, engine='python')

    # Save catchment properties headers
    data_type = df_catchment_properties.iloc[0, :]  # category
    IDs = df_catchment_properties.iloc[2, :]  # feature ID/label
    df_catchment_properties = df_catchment_properties.drop([0, 1, 2])  # drop headers

    df_info = df_catchment_properties.iloc[:, [0, 9]]  # collect catchment ID and information on data quality
    df_info.columns = ['GIS_ID', 'INFO']

    # Translate categories from german to english and associate
    # each category with its corresponding features
    categories = ['Area', 'Quality', 'Response', 'Climate', 'Altitude', 'Slope', 'runoff accumulation',
                  'storage capacity', 'permeability', 'waterlogging', 'thoroughness', 'land use',
                  'ground cover'] + 5*['Geology'] + ['Quaternary Deposits']
    category2feature = {'Area': []}

    columns = []
    count = 0
    for i in range(len(data_type)):
        if type(data_type[i]) is not str:
            category2feature[categories[count]].append(IDs[i])
        else:
            count += 1
            try:
                category2feature[categories[count]].append(IDs[i])
            except:
                category2feature[categories[count]] = [IDs[i]]
        columns.append(categories[count] + ' ' + IDs[i])

    # Associate each feature to its category (defined with an ID)
    features2idxcategory = {}
    for i, cat in enumerate(categories):
        for feature in category2feature[cat]:
            features2idxcategory[feature] = i
    df_catchment_properties.columns = IDs

    gis_id = df_catchment_properties['GIS_ID']
    df_catchment_properties = df_catchment_properties.drop(columns=['GIS_ID'])
    df_catchment_properties.index = gis_id

    # Split main catchment information and attributes
    df_data = df_catchment_properties.iloc[:, 4:]
    df_data = df_data.drop(df_data.columns[[3, 4]], axis=1)

    cols = list(df_data.columns)
    idx = 0

    feature2remove = []

    # Removing the categorical quality feature
    while idx < len(cols):
        if 'Alpine' in cols[idx]:
            feature2remove.append(cols[idx])
            idx = len(cols)
        idx += 1

    # Removing the features with some NaNs
    for el, row in df_data.isna().sum().items():
        if row != 0:
            feature2remove.append(el)

    # Converting to float
    df_data = df_data.replace(',', '.', regex=True).astype(float)

    # Removing irrelevant features based on low standard deviation
    # (features with nearly constant values provide no useful information)
    col2std = df_data.std()
    for feat, std in col2std.items():
        if std < 0.01:  # Threshold for variability
            feature2remove.append(feat)
    df_data = df_data.drop(feature2remove, axis=1)

    # Ensure consistent column order and subset data to selected catchments (all_GISID)
    df_data = df_data[df_data.columns]
    df_final = df_data.loc[[index in all_GISID for index in df_data.index]]
    all_GISID = df_final.index.to_numpy()

    feat_space = copy.deepcopy(df_final.to_numpy())

    # Optional normalization of features
    if normalize:
        centerings = {}  # Store mean values used for centering each feature

        # Only normalize if there’s more than one catchment
        if feat_space.shape[0] != 1:
            # Compute the Euclidean norm of each feature column
            norms = np.linalg.norm(feat_space, axis=0)

            # Center and scale each feature column
            for j in range(feat_space.shape[1]):
                centerings[j] = np.mean(feat_space[:, j])

                if norms[j] > 1e-3:
                    # Normalize to zero mean and unit norm
                    feat_space[:, j] = (feat_space[:, j] - (np.mean(feat_space[:, j]))) / norms[j]
                else:
                    # If feature has negligible norm, only mean-center it
                    feat_space[:, j] = (feat_space[:, j] - (np.mean(feat_space[:, j])))

        if get_df:
            return feat_space, all_GISID, df_final, centerings, norms
        else:
            return feat_space, all_GISID

    else:
        # Return raw (non-normalized) feature space and optional DataFrame
        if get_df:
            return feat_space, all_GISID, df_final
        else:
            return feat_space, all_GISID
