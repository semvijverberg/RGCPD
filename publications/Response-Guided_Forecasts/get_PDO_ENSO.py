#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: semvijverberg
'''
import os, inspect, sys
if sys.platform == 'linux':
    import matplotlib as mpl
    mpl.use('Agg')

user_dir = os.path.expanduser('~')
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-2])
RGCPD_func = os.path.join(main_dir, 'RGCPD')
cluster_func = os.path.join(main_dir, 'clustering/')
if cluster_func not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(RGCPD_func)
    sys.path.append(cluster_func)

import numpy as np
import matplotlib.pyplot as plt
import plot_maps
import cartopy.crs as ccrs
from RGCPD import RGCPD
from RGCPD import BivariateMI

TVpath = '/Users/semvijverberg/surfdrive/output_RGCPD/Response-Guided/tf15_nc5_dendo_5e87d.nc'
path_out_main = '/Users/semvijverberg/surfdrive/output_RGCPD/Response-Guided/'
name_or_cluster_label = 1
name_ds = 'ts'
start_end_TVdate = ('06-01', '08-31')
start_end_date = ('1-1', '12-31')

tfreq = 1
# In[5]:


list_of_name_path = [(name_or_cluster_label, TVpath),
                     ('sst', '/Users/semvijverberg/surfdrive/ERA5/input_raw/sst_1979-2018_1_12_daily_1.0deg.nc')]


rg = RGCPD(list_of_name_path=list_of_name_path,
               list_for_MI=None,
               start_end_TVdate=start_end_TVdate,
               start_end_date=start_end_date,
               tfreq=tfreq, lags_i=np.array([0]),
               path_outmain=path_out_main)

rg.plot_df_clust()

# ### Post-processing Target Variable
rg.pp_TV(name_ds=name_ds, detrend=False, anomaly=False)


rg.traintest(method='random10')

rg.pp_precursors()




#%% Get PDO & ENSO

import climate_indices

df_PDO, PDO_pattern = climate_indices.PDO(rg.list_precur_pp[-1][1], rg.df_splits)
df_ENSO = climate_indices.ENSO_34(rg.list_precur_pp[-1][1], rg.df_splits)
df = df_ENSO.merge(df_PDO,
                    left_index=True,
                    right_index=True).merge(rg.df_splits,
                                            left_index=True,
                                            right_index=True)
file_path = os.path.join(rg.path_outsub1, 'PDO_ENSO34_ERA5_1979_2018.h5')
import functions_pp
functions_pp.store_hdf_df({'df_data':df}, file_path)
