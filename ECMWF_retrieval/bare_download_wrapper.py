#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 16:31:58 2019

@author: semvijverberg
"""

#%%
import os, inspect, sys
import numpy as np
import pandas as pd
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
local_base_path = "/Users/semvijverberg/surfdrive/"
local_script_dir = os.path.join(local_base_path, "Scripts/RGCPD/RGCPD" )

# cluster_base_path = "/p/projects/climber3/atm_data/"
cluster_base_path = "/scistor/ivm/data_catalogue/reanalysis/ERA5/"

cluster_script_dir = "/scistor/ivm/svg460/Scripts/RGCPD/RGCPD"



try:
    os.chdir(local_script_dir)
    sys.path.append(local_script_dir)
    base_path = local_base_path
except:
    os.chdir(cluster_script_dir)
    sys.path.append(cluster_script_dir)
    base_path = cluster_base_path
# =============================================================================
# Data wil downloaded to path_raw
# =============================================================================


dataset   = 'ERA5' # choose 'era5' or 'ERAint' or era20c
path_raw = os.path.join(base_path,f'{dataset}/input_raw/')
                        
if os.path.isdir(path_raw) == False : os.makedirs(path_raw)


# *****************************************************************************
# Step 1 Create dictionary and variable class (and optionally download ncdfs)
# *****************************************************************************
# The dictionary is used as a container with all information.

ex = dict(
     {'dataset'     :       dataset,
     'grid_res'     :       2.5,
     'startyear'    :       1979, # download startyear
     'endyear'      :       2018, # download endyear
     'months'       :       list(range(1,12+1)), #downoad months
     # for monthly means of daily means, choose 'moda' or 'mnth'
     # for daily means choose 'oper' or 'enda' (for accumulations)
     'stream'       :       'oper', 
     'time'         :       pd.date_range(start='00:00', end='23:00',
                                freq=(pd.Timedelta(6, unit='h'))),
     'area'         :       'global', # [North, West, South, East]. Default: global
     'CDO_command'  :       'daymean',
     'base_path'    :       base_path,
     'path_raw'     :       path_raw}
     )

if ex['dataset'] == 'ERAint' or ex['dataset'] == 'era20c':
    import download_ERA_interim_API as ECMWF
elif ex['dataset'] == 'ERA5':
    import download_ERA5_API as ECMWF


# Option 1111111111111111111111111111111111111111111111111111111111111111111111
# Download ncdf fields (in ex['vars']) through cds?
# 11111111111111111111111111111111111111111111111111111111111111111111111111111
# only reanalysis fields

# Info to download ncdf from ECMWF, atm only analytical fields (no forecasts)
# You need the cds-api-client package for this option.

# See https://confluence.ecmwf.int/display/CKB/How+to+download+ERA5

ex['vars']     =    [
                    ['v500hpa'],              # ['name_var1','name_var2', ...]
                    ['v_component_of_wind'],    # ECMWF param ids
                    ['pl'],             # Levtypes ('sfc' or 'pl')
                    [['500']],                  # Vertical levels
                    ]

for idx in range(len(ex['vars'][0]))[:]:
    # class for ECMWF downloads
    var_class = ECMWF.Var_ECMWF_download(ex, idx)
    ex[ex['vars'][0][idx]] = var_class
    ECMWF.retrieve_field(var_class)
