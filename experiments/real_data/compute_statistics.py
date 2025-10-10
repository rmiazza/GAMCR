"""
Postprocess GAMCR model results and compute runoff-response statistics.

This script loads pre-trained GAMCR models for each catchment folder, retrieves
precipitation ensemble settings from ERRA (if available), applies date filters,
and computes diagnostic statistics on runoff responses.

Usage
-----
Run directly from the command line or VS Code:

    python compute_statistics.py

Requirements
------------
- Each catchment folder must contain a `{gis_id}_best_model.pkl` file.
- Optional: ERRA output stored in 'data_and_visualization/output_ERRA_forGAMCR/'.
- The `GAMCR` package must be available in the project path.
"""

import pandas as pd
import numpy as np
import os
import sys

# Add project root to sys.path (two levels up from this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.append(PROJECT_ROOT)

import GAMCR  # noqa: E402
from data_and_visualization.get_data_from_ERRA import get_data_from_ERRA

# Load ERRA results
path_ERRA = os.path.join(SCRIPT_DIR, 'data_and_visualization', 'output_ERRA_forGAMCR')
dic_ERRA = get_data_from_ERRA(path_ERRA)

if dic_ERRA is None:
    print('No ERRA results found. Ensembles for precipitation and streamflow will be set using quantiles.')

# List all subdirectories and keep only clean catchment/site IDs
# (exclude hidden folders or folder names containing '/', '.', or '_')
all_gis_id = [
    folder for folder in next(os.walk(SCRIPT_DIR))[1]
    if not any(char in folder for char in ['/', '.', '_'])
]

# Set to True to process all available data (not only the training set)
all_data = True


# Helper function for date filtering
def filter_dates(dates, all_data=False):
    """Filter dates by year and by catchment-specific snow-free period."""
    dates = pd.to_datetime(dates)

    if not (all_data):
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
    idxsmonth_low = np.where(np.array([date.month for date in dates]) >= low_month)[0]
    idxsmonth_up = np.where(np.array([date.month for date in dates]) <= up_month)[0]

    # Combine year and month filters
    if not (all_data):
        idxs = np.intersect1d(idxsyear, idxsmonth_low)
    else:
        idxs = idxsmonth_low

    idxs = np.intersect1d(idxs, idxsmonth_up)
    return idxs


# Main analysis loop
for gis_id in all_gis_id:
    name_model = f'{gis_id}_best_model.pkl'
    model_path = os.path.join(SCRIPT_DIR, gis_id, name_model)
    site_folder = os.path.join(SCRIPT_DIR, gis_id)

    # Initialize and load trained model
    model = GAMCR.model.GAMCR(lam=0.1)
    model.load_model(model_path)

    # Define ERRA or default ensemble settings
    if dic_ERRA is not None and gis_id in dic_ERRA:
        groups_precip = dic_ERRA[gis_id]['groups_precip']
        nblocks = len(groups_precip)

        if groups_precip[0][0] < 0.1:
            # note that we consider ERRA has been run by removing all
            # precipitation events below 0.5mm.h^{-1}
            # Since for some sites, aggregation was used in ERRA, ERRA
            # ensembles for those sites could still provide a minimum for the
            # first bin smaller than 0.5.
            groups_precip[0] = (0.5, groups_precip[0][1])
        min_precip = groups_precip[0][0]
    else:
        # Default settings when ERRA results are unavailable
        nblocks = 4
        min_precip = 0.5
        groups_precip = "auto"

    # Define result save folder
    save_folder = os.path.join(site_folder, 'results') if all_data else None

    # Compute and save statistics
    model.compute_statistics(
        site_folder, gis_id, nblocks=nblocks, min_precip=min_precip,
        groups_precip=groups_precip, groups_wetness="auto", max_files=99,
        filtering_time_points=lambda x: filter_dates(x, all_data=all_data),
        save_folder=save_folder
    )
