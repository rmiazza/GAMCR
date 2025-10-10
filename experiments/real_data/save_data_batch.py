"""
Batch preprocessing of hydrological data for multiple catchments.

This script loops over all catchment directories in the current folder,
creates a GAMCR model instance, and calls its 'save_batch' method to
preprocess and save the model input data in multiple '.npy' batches.

Usage:
    python save_data_batch.py

Requirements:
    - The current directory must contain one subfolder per catchment (e.g. "101", "102").
    - Each subfolder must include a file named 'data_<gis_id>.txt'.
    - The 'GAMCR' package must be accessible (via relative import or 'sys.path').

Outputs:
    Preprocessed '.npy' files are saved inside:
        {site}/data/
"""


import os
import sys

# Add project root to sys.path (two levels up from this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.append(PROJECT_ROOT)

# Import after adjusting sys.path
import GAMCR  # noqa: E402

# List all subdirectories and keep only clean catchment/site IDs
# (exclude hidden folders or names containing '/', '.', or '_')
all_gis_id = [
    folder for folder in next(os.walk(SCRIPT_DIR))[1]
    if not any(char in folder for char in ['/', '.', '_'])
]

# Create a GAMCR model instance
model = GAMCR.model.GAMCR(max_lag=24*10, features={'timeyear': True})

# Loop through each catchment/site folder
for site in all_gis_id:
    save_folder = os.path.join(SCRIPT_DIR, f'{site}', 'data')
    datafile = os.path.join(SCRIPT_DIR, f'{site}', f'data_{site}.txt')

    # Preprocess and save data batches for this site
    model.save_batch(save_folder, datafile, nfiles=40)
