#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 09:59:06 2021

@author: semvijverberg
"""
#%%
import os, sys
import matplotlib as mpl
import pandas as pd

user_dir = os.path.expanduser('~')
os.chdir(os.path.join(user_dir,
                      'surfdrive/Scripts/RGCPD/publications/paper_Raed/'))
curr_dir = os.path.join(user_dir, 'surfdrive/Scripts/RGCPD/RGCPD/')
main_dir = '/'.join(curr_dir.split('/')[:-2])
RGCPD_func = os.path.join(main_dir, 'RGCPD')
assert main_dir.split('/')[-1] == 'RGCPD', 'main dir is not RGCPD dir'
cluster_func = os.path.join(main_dir, 'clustering/')
fc_dir = os.path.join(main_dir, 'forecasting')

if cluster_func not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(RGCPD_func)
    sys.path.append(cluster_func)
    sys.path.append(fc_dir)


filepath_RGCPD_preds = '/Users/semvijverberg/surfdrive/output_paper3/USDA_Soy/TimeSeriesSplit_10/s1/predictions_s1_continuous.h5'
filepath_target_RGCPD = os.path.join(main_dir, 'publications/paper_Raed/data/usda_soy_spatial_mean_ts.nc')
filepath_USDA_frcst = os.path.join(main_dir, 'publications/paper_Raed/data/yield_spatial_avg_midwest.csv')

filepath_orig_midwest = '/Users/semvijverberg/Dropbox/VIDI_Coumou/Paper3_Sem/GDHY_MIRCA2000_Soy/USDA/ts_spatial_avg_midwest.csv'
filepath_orig_all = '/Users/semvijverberg/Dropbox/VIDI_Coumou/Paper3_Sem/GDHY_MIRCA2000_Soy/USDA/ts_spatial_avg.csv'


def read_csv_Raed(path):
    orig = pd.read_csv(path)
    orig = orig.drop('Unnamed: 0', axis=1)
    orig .index = pd.to_datetime([f'{y}-01-01' for y in orig .Year])
    return orig.drop('Year', 1)

from RGCPD import RGCPD
import func_models as fc_utils
import functions_pp


list_of_name_path = [('', read_csv_Raed(filepath_orig_all))]


rg = RGCPD(list_of_name_path,
           tfreq=None)
rg.pp_TV(name_ds='Soy_Yield', ext_annual_to_mon=False)

# df_preds = functions_pp.load_hdf5(filepath_RGCPD_preds)
df_USDA_midwest = read_csv_Raed(filepath_USDA_frcst)
df_orig_midwest = read_csv_Raed(filepath_orig_midwest)
df_orig_all     = read_csv_Raed(filepath_orig_all)

# ax = df_USDA.plot()
# rg.df_fullts.plot(ax=ax)
# orig.plot(ax=ax)
# df_orig_all.plot(ax=ax)

