"""
Train GAMCR models for multiple catchments.

This script loops over all catchment directories, loads preprocessed data,
filters it to retain only snow-free periods, and trains a GAMCR model for each
catchment using specified regularization parameters.

Usage
-----
Run directly from the command line or VS Code:

    python train_models.py

Requirements
------------
- Each catchment folder should be located in the same directory as this script.
- Each folder must include:
    - A 'data/' subdirectory with '.npy' data files.
    - A 'params.pkl' file containing model parameters.
- The 'GAMCR' package must be importable (either installed or in the project path).

Outputs
-------
- Trained model weights and metadata are saved in each catchment’s folder.
"""

import pandas as pd
import numpy as np
import os
import sys

# Add project root to sys.path (two levels up from this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.append(PROJECT_ROOT)

# Import after adjusting sys.path
import GAMCR  # noqa: E402

# List all subdirectories and keep only clean catchment/site IDs
# (exclude hidden folders or folder names containing '/', '.', or '_')
all_gis_id = [
    folder for folder in next(os.walk(SCRIPT_DIR))[1]
    if not any(char in folder for char in ['/', '.', '_'])
]


def filter_dates(gis_id, dates):
    """Select training data indices within specified years and months.

    This function filters a list or array of datetime objects, returning the
    indices of dates that meet both of the following criteria:
      1. The year is earlier than 2018.
      2. The month falls within a catchment-specific snow-free period.

    The month range depends on the catchment ID ('gis_id'):
        - '46' → July to September
        - '3', '44', '112' → June to October
        - all others → May to October

    Parameters
    ----------
    gis_id : str or int
        Unique identifier of the catchment or site.
    dates : list or array-like of datetime.datetime
        List of datetime objects representing the time points to be filtered.

    Returns
    -------
    np.ndarray
        Array of integer indices corresponding to dates that satisfy both
        the year (< 2018) and month range conditions.
    """
    # Select all dates before 2018
    cutoff_year = 2018
    idxsyear = np.where(np.array([date.year for date in dates]) < cutoff_year)[0]

    # Define month ranges based on catchment ID
    if gis_id == '46':
        low_month = 7
        up_month = 9
    elif gis_id in ['3', '44', '112']:
        low_month = 6
        up_month = 10
    else:
        low_month = 5
        up_month = 10

    # Select dates within the month range
    idxs_month_low = np.where(np.array([date.month for date in dates]) >= low_month)[0]
    idxs_month_up = np.where(np.array([date.month for date in dates]) <= up_month)[0]

    # Combine year and month filters
    idxs = np.intersect1d(idxsyear, idxs_month_low)
    idxs = np.intersect1d(idxs, idxs_month_up)

    return idxs


# Regularization settings for model training
ls_lambs = [1e-6 * (10**i) for i in range(2, 7)]
ls_global_lambs = [1e-6 * (10**i) for i in range(3, 10)]

# Main training loop
for gis_id in all_gis_id:
    lam = ls_lambs[1]
    global_lam = ls_global_lambs[3]

    # Initialize model
    model = GAMCR.model.GAMCR()

    # Define paths
    gis_id_path = os.path.join(SCRIPT_DIR, str(gis_id), 'data')
    save_folder_gis_id = os.path.join(SCRIPT_DIR, str(gis_id))
    data_file = os.path.join(gis_id_path, 'params.pkl')

    # Load data
    X, matJ, y, timeyear, dates = model.load_data(gis_id_path, max_files=99)
    dates = pd.to_datetime(dates)

    # Filter to snow-free period for training
    idxs = filter_dates(gis_id, dates)
    X = X[idxs, :]
    matJ = matJ[idxs, :, :]
    timeyear = timeyear[idxs]
    dates = dates[idxs]
    y = y[idxs]

    # Load model parameters and train
    model.load_model(data_file, lam=lam)
    name_model = f'{gis_id}_best_model'

    model.train(
        X, matJ, y,
        dates=dates, lr=1e-1, max_iter=6000,
        warm_start=False, save_folder=save_folder_gis_id,
        name_model=name_model, normalization_loss=1,
        lam_global=global_lam
    )
