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
import seaborn as sns

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

path_raw = user_dir + '/surfdrive/ERA5/input_raw'


# Sem
filepath_RGCPD_preds = '/Users/semvijverberg/surfdrive/output_paper3/USDA_Soy/random_5_always_data_mask/s1/predictions_s1_continuous.h5'
filepath_target_RGCPD = os.path.join(main_dir, 'publications/paper_Raed/data/init_usda_soy_spatial_mean_ts.nc')
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
import functions_pp, core_pp, climate_indices

detrend = True
rg_mistake = RGCPD([('', filepath_target_RGCPD),
                    ('sst', os.path.join(path_raw, 'sst_1950-2019_1_12_monthly_1.0deg.nc'))],
                   tfreq=None)
rg_mistake.pp_TV(name_ds='Soy_Yield', ext_annual_to_mon=False,
                 detrend=detrend)
rg_mistake.df_fullts = rg_mistake.df_fullts.rename({'Soy_Yield':'Sem spatial mean all data'},
                                                   axis=1)


rg_always = RGCPD([('', filepath_nc_mean_allways_data)],tfreq=None)
rg_always.pp_TV(name_ds='Soy_Yield', ext_annual_to_mon=False, detrend=detrend)
rg_always.df_fullts = rg_always.df_fullts.rename({'Soy_Yield':'Sem spatial mean all data that was present every timestep'},axis=1)

rg_always.df_fullts.merge(rg_mistake.df_fullts, left_index=True, right_index=True).corr()


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
f, ax = plt.subplots()
df_States = read_csv_State(filepath_Raed_state)
df_States_mean = df_States.mean(axis=1) ; df_States_mean.name = 'mean over States'
plt.plot(df_States_mean, label='mean over State', c='blue')
plt.plot(df_orig_all, color='red', label='original csv Raed mean over all non-NaN spatial data')
ax.legend()
rg_always.df_fullts.plot(ax=ax, c='black')
#%% Test different detrending for State timeseries
import core_pp
df_States = read_csv_State(filepath_Raed_state)
ds, lintrend = core_pp.detrend_lin_longterm(df_States.to_xarray().to_array().T, return_trend=True)
df_States_lin = ds.to_dataframe('Yield').pivot_table(index='time',
                                                      columns='variable')
df_lintrend = lintrend.to_dataframe('Yield').pivot_table(index='time',
                                                      columns='variable')

df_States_lin.columns = df_States_lin.columns.droplevel(0)
df_lintrend.columns = df_lintrend.columns.droplevel(0)
cols = df_States_lin.corr()[['MINNESOTA']].sort_values(by='MINNESOTA', ascending=False).index

df_States = read_csv_State(filepath_Raed_state)
df_States_linnew, df_lin_new = core_pp.detrend(df_States.copy(), method='loess',
                                               return_trend=True, kwrgs_detrend={'order':2})
# df_States_loess, loess_fit = core_pp.detrend_loess(df_States.copy(), return_trend=True)
# df_loess_fit = pd.DataFrame(loess_fit, columns=df_States.columns, index=df_States.index)
# df_States_lin = (df_States_lin - df_States_lin.mean(0)) / df_States_lin.std(0)



FG = sns.FacetGrid(df_States_lin.stack().reset_index(), col='variable', col_wrap=4)
for ia, ax in enumerate(FG.fig.axes):
    ax.plot(df_States[cols[ia]], label=cols[ia])
    ax.plot(df_lintrend[cols[ia]])
    ax.plot(df_lin_new[cols[ia]])
    ax.set_title(cols[ia])
    std = df_States_lin.std().mean()
    #ax.axhline(y=0) ; ax.set_ylim(-3*std,3*std)
#%%

# df_States_lin = (df_States_lin - df_States_lin.mean(0)) / df_States_lin.std(0)

FG = sns.FacetGrid(df_States_loess.stack().reset_index(), col='variable', col_wrap=4)
for ia, ax in enumerate(FG.fig.axes):
    ax.plot(df_States[cols[ia]], label=cols[ia])

    ax.set_title(cols[ia])
    # ax.axhline(y=0) ; ax.set_ylim(-3,3)

#%%
rg_mistake.pp_TV(name_ds='Soy_Yield', ext_annual_to_mon=False,
                 detrend=True)
rg_mistake.pp_precursors()
df_PDO, patterns = climate_indices.PDO(rg_mistake.list_precur_pp[0][1])
df_PDO = df_PDO.groupby(df_PDO.loc[0].index.year).mean()
df_PDO.index = rg_mistake.df_fullts.index

df_Pac = functions_pp.load_hdf5('/Users/semvijverberg/surfdrive/output_paper3/hindcast/USDA_Soy_hindcast/bimonthly_random_20_s1/pandas_dfs_04-06-21_10hr.h5')
df_Pac  = df_Pac['df_data']
df_Pac = df_Pac[['Soy_Yield'] + [k for k in df_Pac.columns if '..1..sst' in k]]
df_Pac = df_Pac.mean(axis=0, level=1).mean(axis=1) # mean over folds, mean over months March-October
df_Pac = pd.DataFrame(df_Pac, columns=['eastern Pacific'])

df_merge = rg_mistake.df_fullts.merge(df_PDO[['PDO']], left_index=True, right_index=True)
df_merge = df_merge.merge(df_Pac, left_index=True, right_index=True)
df_merge = df_merge.merge(rg_always.df_fullts.rename({rg_always.df_fullts.columns[0]:'new'}, axis=1),
                          left_index=True, right_index=True)
df_merge = (df_merge - df_merge.mean(axis=0)) / df_merge.std(0)

#%%
f, ax = plt.subplots(figsize=(15,8))
df_merge.rolling(10, center=True, min_periods=10).mean().plot(ax=ax,
                                                              style={'Soy_Yield':'black',
                                                                     'PDO':'--',
                                                                     'new':':g'})
ax.lines[0].set_linewidth(4)
ax.lines[2].set_linewidth(4)
ax.legend(['init Soy_Yield', 'PDO (corr={:.2f})'.format(df_merge.corr().iloc[0,1]),
           'eastern Pacific (corr={:.2f})'.format(df_merge.corr().iloc[0,2]),
          'New (always data) (corr={:.2f})'.format(df_merge.corr().iloc[0,3])],
          title="10 yr rolling mean applied")
plt.axhline(0, color='grey')
plt.axvline(5)
plt.show()
#%%
import itertools, os, re
import make_country_mask, enums
var_filename = '/Users/semvijverberg/Dropbox/VIDI_Coumou/Paper3_Sem/GDHY_MIRCA2000_Soy/USDA/masked_rf_gs_county_grids_pp.nc'
xarray, States = make_country_mask.create_mask(var_filename, kwrgs_load={}, level='US_States')
xarray = xarray.where(xarray.values != -1)
#%%
All_states = ['ALABAMA', 'DELAWARE', 'ILLINOIS', 'INDIANA', 'IOWA', 'KENTUCKY',
              'MARYLAND', 'MINNESOTA', 'MISSOURI', 'NEW JERSEY', 'NEW YORK',
              'NORTH CAROLINA', 'NORTH DAKOTA', 'OHIO', 'PENNSYLVANIA',
              'SOUTH CAROLINA', 'TENNESSEE', 'VIRGINIA', 'WISCONSIN']

method = 'random_20' ; s =1
path_search = '/Users/semvijverberg/Desktop/cluster/surfdrive/output_paper3/forecast'
df_score_States = []
for STATE in All_states:

    path_search_State = os.path.join(path_search, STATE, method)
    hash_str = f'scores_s{s}_continuous.h5'
    f_name = None
    for root, dirs, files in os.walk(path_search_State):
        for file in files:
            if re.findall(f'{hash_str}', file):
                print(f'Found file {file}')
                f_name = file
    if f_name is not None:
        d_dfs = functions_pp.load_hdf5(os.path.join(path_search_State,
                                                    f's{s}',
                                                    f_name))
        df_scores = d_dfs['df_scores'] ; df_scores.index = [STATE]
        df_score_States.append(df_scores)
df_score_States = pd.concat(df_score_States)

#%%
import xarray as xr
import find_precursors
SKIP_STATES = ['NEW YORK']
months = ['August', 'July', 'June'];
metrics = ['corrcoef', 'r2_score'] # 'MAE', 'RMSE',
xr_score = xarray.copy() ; xr_score.attrs = {}
list_xr = [xr_score.copy().expand_dims('metric', axis=0) for m in metrics]
xr_score = xr.concat(list_xr, dim = 'metric')
xr_score['metric'] = ('metric', metrics)
list_xr = [xr_score.expand_dims('month', axis=0) for m in months]
xr_score = xr.concat(list_xr, dim = 'month')
xr_score['month'] = ('month', months)

for im, month in enumerate(xr_score.month.values):
    for s, metric in enumerate(xr_score.metric.values):
        int_metric = {}
        for STATE in All_states:
            if STATE not in df_score_States.index or STATE in SKIP_STATES:
                continue
            US_States_format = ' '.join([s.lower().capitalize() for s in STATE.split(' ')])
            abbrev = enums.us_state_abbrev[US_States_format]
            integer = States.__dict__[abbrev].real
            score = df_score_States.loc[STATE].loc[month, metric]
            int_metric[integer] = score
        xr_score[im, s] = find_precursors.view_or_replace_labels(xarray.copy(),
                                                          list(int_metric.keys()),
                                                          list(int_metric.values()))
#%%
fig = plot_maps.plot_corr_maps(xr_score, col_dim='month', row_dim='metric',
                               size=4, clevels=np.arange(-.5,0.51,.1),
                               cbar_vert=0, hspace=.1, add_cfeature='LAKES')
