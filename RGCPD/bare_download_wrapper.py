#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 16:31:58 2019

@author: semvijverberg
"""

#%%
import os
import numpy as np
import pandas as pd


# =============================================================================
# Data wil downloaded to path_raw
# =============================================================================
base_path = "/Users/semvijverberg/surfdrive/RGCPD_jetlat/"
path_raw = os.path.join("/Users/semvijverberg/surfdrive/Data_ERAint/", 
                        'input_raw')
path_pp  = os.path.join("/Users/semvijverberg/surfdrive/Data_ERAint/", 
                        'input_pp')
if os.path.isdir(path_raw) == False : os.makedirs(path_raw)
if os.path.isdir(path_pp) == False: os.makedirs(path_pp)

# *****************************************************************************
# Step 1 Create dictionary and variable class (and optionally download ncdfs)
# *****************************************************************************
# The dictionary is used as a container with all information for the experiment
# The dic is saved after the post-processes step, so you can continue the experiment
# from this point onward with different configurations. It also stored as a log
# in the final output.

ex = dict(
     {'dataset'     :       'interim',
     'grid_res'     :       2.5,
     'startyear'    :       1979, # download startyear
     'endyear'      :       2018, # download endyear
     'months'       :       list(range(1,12+1)), #downoad months
     'time'         :       pd.DatetimeIndex(start='00:00', end='23:00', 
                                freq=(pd.Timedelta(6, unit='h'))),
     'base_path'    :       base_path,
     'path_raw'     :       path_raw,
     'path_pp'      :        path_pp}
     )

if ex['dataset'] == 'interim':
    import download_ERA_interim_API as ECMWF
elif ex['dataset'] == 'era5':
    import download_ERA5_API as ECMWF
    

# Option 1111111111111111111111111111111111111111111111111111111111111111111111
# Download ncdf fields (in ex['vars']) through cds?
# 11111111111111111111111111111111111111111111111111111111111111111111111111111
# only reanalysis fields

# Info to download ncdf from ECMWF, atm only analytical fields (no forecasts)
# You need the cds-api-client package for this option.

# See https://confluence.ecmwf.int/display/CKB/How+to+download+ERA5

ex['vars']     =   [
                    ['t2m', 'u'],              # ['name_var1','name_var2', ...]
                    ['167.128', '131.128'],    # ECMWF param ids
                    ['sfc', 'pl'],             # Levtypes
                    [0, 200],                  # Vertical levels
                    ]

for idx in range(len(ex['vars'][0]))[:]:
    # class for ECMWF downloads
    var_class = ECMWF.Var_ECMWF_download(ex, idx)
    ex[ex['vars'][0][idx]] = var_class
    ECMWF.retrieve_field(var_class)
