#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 09:59:06 2021

@author: semvijverberg
"""
#%%
import os, sys
import matplotlib.pyplot as plt
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

# Sem
filepath_RGCPD_preds = '/Users/semvijverberg/surfdrive/output_paper3/USDA_Soy/random_5_always_data_mask/s1/predictions_s1_continuous.h5'
filepath_target_RGCPD = os.path.join(main_dir, 'publications/paper_Raed/data/usda_soy_spatial_mean_ts.nc')
filepath_nc_mean_allways_data = os.path.join(main_dir, 'publications/paper_Raed/data/usda_soy_spatial_mean_ts_allways_data.nc')

# Raed csv
filepath_orig_midwest = '/Users/semvijverberg/Dropbox/VIDI_Coumou/Paper3_Sem/GDHY_MIRCA2000_Soy/USDA/ts_spatial_avg_midwest.csv'
filepath_orig_all = '/Users/semvijverberg/Dropbox/VIDI_Coumou/Paper3_Sem/GDHY_MIRCA2000_Soy/USDA/ts_spatial_avg.csv'
# Raed (from Beguería et al. 2020)
filepath_USDA_frcst = os.path.join(main_dir, 'publications/paper_Raed/data/yield_spatial_avg_midwest.csv')


# new csv per state
filepath_Raed_state = os.path.join(main_dir, 'publications/paper_Raed/data/masked_rf_gs_state_USDA.csv')
def read_csv_Raed(path):
    orig = pd.read_csv(path)
    orig = orig.drop('Unnamed: 0', axis=1)
    orig.index = pd.to_datetime([f'{y}-01-01' for y in orig.Year])
    orig.index.name = 'time'
    return orig.drop('Year', 1)

def read_csv_State(path, State: str=None, col='obs_yield'):
    orig = read_csv_Raed(path)
    orig = orig.set_index('State', append=True)
    orig = orig.pivot_table(index='time', columns='State')[col]
    if State is None:
        State = orig.columns
    return orig[State]

from RGCPD import RGCPD
import func_models as fc_utils
import functions_pp, core_pp

rg_mistake = RGCPD([('', filepath_target_RGCPD)],tfreq=None)
rg_mistake.pp_TV(name_ds='Soy_Yield', ext_annual_to_mon=False)
rg_mistake.df_fullts = rg_mistake.df_fullts.rename({'Soy_Yield':'Sem spatial mean all data'},axis=1)


rg_always = RGCPD([('', filepath_nc_mean_allways_data)],tfreq=None)
rg_always.pp_TV(name_ds='Soy_Yield', ext_annual_to_mon=False)
rg_always.df_fullts = rg_always.df_fullts.rename({'Soy_Yield':'Sem spatial mean all data that was present every timestep'},axis=1)



df_USDA_midwest = read_csv_Raed(filepath_USDA_frcst)
df_USDA_midwest = df_USDA_midwest.rename({'Soy_Yield':'New data with USDA forecast (Mid-West)'}, axis=1)
df_orig_midwest = read_csv_Raed(filepath_orig_midwest).rename({'Soy_Yield':'Orig csv Raed (Mid-West)'}, axis=1)
df_orig_all     = read_csv_Raed(filepath_orig_all).rename({'Soy_Yield':'Orig csv Raed all spatial data'}, axis=1)

#%% comparison between target pre-processing
f, ax = plt.subplots()
ax = df_orig_midwest.plot(ax=ax, c='red', title='Red is orig csv mid-west spatial data mean')
rg_mistake.df_fullts.plot(ax=ax, c='blue')

f, ax = plt.subplots()
ax = df_orig_midwest.plot(ax=ax, c='red', title='Red is orig csv mid-west spatial data mean')
rg_always.df_fullts.plot(ax=ax, c='blue')

f, ax = plt.subplots()
ax = df_orig_all.plot(ax=ax, c='red', title='Red is orig csv all spatial data mean')
rg_always.df_fullts.plot(ax=ax, c='blue')

f, ax = plt.subplots()
ax = df_USDA_midwest[['obs_yield']].plot(ax=ax, c='red', title='Red is USDA obs Beguería et al. 2020')
rg_always.df_fullts.plot(ax=ax, c='blue')
df_orig_midwest.plot(ax=ax)

#%%
filepath_RGCPD_hindcast = '/Users/semvijverberg/surfdrive/output_paper3/USDA_Soy_csv_midwest_bimonthly_random_10_s1_1950_2019/predictions_s1_continuous.h5'
df_preds = functions_pp.load_hdf5(filepath_RGCPD_hindcast)['df_predictions']
df_preds = functions_pp.get_df_test(df_preds) ; df_preds.index.name='time'
xr_obs = df_preds[['raw_target']].to_xarray().to_array().squeeze()


trend = xr_obs - core_pp.detrend_lin_longterm(xr_obs)
recon = df_preds.iloc[:,[0]] + trend.values[None,:].T #.values[1:][None,:].T + float(rg_always.df_fullts.mean())
ax = recon.plot()
df_preds[['raw_target']].plot(ax=ax)
#%%
pred = df_preds[[0]] + trend.values[None,:].T
ax = pred.plot()
df_USDA_midwest[['frcst_aug_yield']].plot(ax=ax)
df_preds[['raw_target']].plot(ax=ax)

#%%
read_csv_State(filepath_Raed_state).mean(axis=1).plot()
