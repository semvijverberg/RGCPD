#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 09:59:06 2020

@author: semvijverberg
"""

from __future__ import division
import os, inspect, sys
import matplotlib as mpl
if sys.platform == 'linux':
    mpl.use('Agg')
else:
    # Optionally set font to Computer Modern to avoid common missing font errors
    mpl.rc('font', family='serif', serif='cm10')

    mpl.rc('text', usetex=True)
    mpl.rcParams['text.latex.preamble'] = r'\boldmath'
import numpy as np
from time import time
import cartopy.crs as ccrs ; import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import matplotlib
from sklearn import metrics
import pandas as pd
import xarray as xr
import csv
# import sklearn.linear_model as scikitlinear
import argparse

user_dir = os.path.expanduser('~')
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-2])
RGCPD_func = os.path.join(main_dir, 'RGCPD')
assert main_dir.split('/')[-1] == 'RGCPD', 'main dir is not RGCPD dir'
cluster_func = os.path.join(main_dir, 'clustering/')
fc_dir = os.path.join(main_dir, 'forecasting')
data_dir = os.path.join(main_dir,'publications/paper2/data')
if cluster_func not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(RGCPD_func)
    sys.path.append(cluster_func)
    sys.path.append(fc_dir)

path_raw = user_dir + '/surfdrive/ERA5/input_raw'


from RGCPD import RGCPD
from RGCPD import BivariateMI
import class_BivariateMI
import func_models as fc_utils
import functions_pp; import df_ana
import plot_maps; import core_pp
import wrapper_PCMCI as wPCMCI


# targets = ['temp', 'RW']


# if region == 'eastern':
targets = ['easterntemp', 'westerntemp']

periods = ['JA_center', 'JA_shiftright', 'JA_shiftleft', 'JJA_center']
seeds = np.array([1,2,3])
combinations = np.array(np.meshgrid(targets, periods, seeds)).T.reshape(-1,3)

i_default = 0 #8

def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Optional arguments
    parser.add_argument("-i", "--intexper", help="intexper", type=int,
                        default=i_default)
    # Parse arguments
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parseArguments()
    out = combinations[args.intexper]
    target = out[0]
    period = out[1]
    seed = int(out[2])
    # remove_PDO = bool(int(out[3]))
    if target[-4:]=='temp':
        tfreq = 15
    else:
        tfreq = 60
    print(f'arg {args.intexper} f{out}')


calc_ts = 'region mean'

if target[-4:] == 'temp':
    TVpath = user_dir + '/surfdrive/output_RGCPD/circulation_US_HW/tf15_nc3_dendo_0ff31.nc'
    alpha_corr = .05
    cluster_label = 2
    name_ds='ts'
    if target == 'westerntemp':
        cluster_label = 1
    elif target == 'easterntemp':
        cluster_label = 2
elif target[-2:] == 'RW':
    cluster_label = 'z500'
    name_ds = f'0..0..{cluster_label}_sp'
    alpha_corr = .05
    if target == 'easternRW':
        TVpath = os.path.join(data_dir, '2020-10-29_13hr_45min_east_RW.h5')
    elif target == 'westernRW':
        TVpath = os.path.join(data_dir, '2020-10-29_10hr_58min_west_RW.h5')

# select target period
if period == 'JA_center':
    start_end_TVdate = ('07-01', '08-31')
elif period == 'JA_shiftleft':
    start_end_TVdate = ('06-25', '08-24')
elif period == 'JA_shiftright':
    start_end_TVdate = ('07-08', '09-06')
elif period == 'JJA_center':
    start_end_TVdate = ('07-01', '08-31')

precur_aggr = tfreq
method     = 'ranstrat_10' ;
n_boot = 5000
min_detect_gc = 0.9
append_main = ''
# name_csv = f'skill_scores_tf{tfreq}.csv'


#%% run RGPD
# start_end_TVdate = ('06-01', '08-31')
start_end_date = ('3-1', start_end_TVdate[-1]) # focus on spring/summer. Important for regressing out influence of PDO (might vary seasonally)
# =============================================================================
# change SED
# =============================================================================
start_end_date = ('01-01', '12-31')

list_of_name_path = [(cluster_label, TVpath),
                     ('sst', os.path.join(path_raw, 'sst_1979-2018_1_12_daily_1.0deg.nc'))]


list_import_ts = None #[('PDO', os.path.join(data_dir, f'PDO_2y_rm_25-09-20_15hr.h5')),
#                   ('PDO1y', os.path.join(data_dir, 'PDO_1y_rm_25-11-20_17hr.h5'))]

list_for_MI   = [BivariateMI(name='sst', func=class_BivariateMI.corr_map,
                            alpha=alpha_corr, FDR_control=True,
                            kwrgs_func={}, group_split='together',
                            distance_eps=500, min_area_in_degrees2=10,
                            calc_ts=calc_ts, selbox=(130,260,-10,60),
                            lags=np.array([0,1,2,3,4,5]))]

path_out_main = os.path.join(main_dir, f'publications/paper2/output/{target}{append_main}/')

rg = RGCPD(list_of_name_path=list_of_name_path,
           list_for_MI=list_for_MI,
           list_import_ts=list_import_ts,
           start_end_TVdate=start_end_TVdate,
           start_end_date=start_end_date,
           start_end_year=None,
           tfreq=tfreq,
           path_outmain=path_out_main)
if rg.list_for_MI[0]._name == 'sst_corr_map':
    title = r'$corr(SST_{t_{gap}}$' + f'$, T^{target[0].capitalize()}_t)$'
else:
    title = r'$parcorr(SST_{t-lag}, mx2t_t\ |\ SST_{t-1},mx2t_{t-1})$'
subtitles = np.array([[f'gap = {(l-1)*tfreq} days' for l in rg.list_for_MI[0].lags]]) #, f'lag 2 (15 day lead)']] )
subtitles[0][0] = 'No gap'
kwrgs_plotcorr = {'row_dim':'split', 'col_dim':'lag','aspect':2, 'hspace':-.47,
              'wspace':-.15, 'size':3, 'cbar_vert':-.1,
              'units':'Corr. Coeff. [-]', 'zoomregion':(130,260,-10,60),
              'clim':(-.60,.60), 'map_proj':ccrs.PlateCarree(central_longitude=220),
              'y_ticks':np.arange(-10,61,20), 'x_ticks':np.arange(140, 280, 25),
              'subtitles':subtitles, 'title':title,
              'title_fontdict':{'fontsize':20, 'fontweight':'bold', 'y':1.07}}
precur = rg.list_for_MI[0]

append_str='{}d_{}'.format(tfreq, calc_ts.split(' ')[0])
kwrgs_MI = [str(i)+str(k) for i,k in precur.kwrgs_func.items()]
if len(kwrgs_MI) != 0:
    append_str += '_' + '_'.join(kwrgs_MI)
#%%



rg.pp_TV(name_ds=name_ds, detrend=False)

subfoldername = '_'.join(['vs_lags',rg.hash, period,
                      str(precur_aggr), str(alpha_corr), method,
                      str(seed)])

rg.pp_precursors()
rg.traintest(method=method, seed=seed, subfoldername=subfoldername)
rg.calc_corr_maps()
rg.cluster_list_MI()
kwrgs_plotlabels = kwrgs_plotcorr.copy()
kwrgs_plotlabels.pop('title') ; kwrgs_plotlabels.pop('units')
rg.quick_view_labels(save=True, append_str=f'{tfreq}d',
                     min_detect_gc=min_detect_gc,
                     kwrgs_plot=kwrgs_plotlabels)
# plotting corr_map
rg.plot_maps_corr(var='sst', save=True,
                  kwrgs_plot=kwrgs_plotcorr,
                  min_detect_gc=min_detect_gc,
                  append_str=f'{tfreq}d')


#%% Get PDO and apply low-pass filter
import climate_indices, filters
from func_models import standardize_on_train




if 'df_PDOsplit' not in globals():
    df_PDO, PDO_patterns = climate_indices.PDO(rg.list_precur_pp[0][1],
                                               None) #rg.df_splits)
    # summerdates = core_pp.get_subdates(dates, start_end_TVdate)
    df_PDOsplit = df_PDO.loc[0].copy()#.loc[summerdates]
    # standardize = preprocessing.StandardScaler()
    # standardize.fit(df_PDOsplit[df_PDOsplit['TrainIsTrue'].values].values.reshape(-1,1))
    # df_PDOsplit = pd.DataFrame(standardize.transform(df_PDOsplit['PDO'].values.reshape(-1,1)),
    #                 index=df_PDOsplit.index, columns=['PDO'])
df_PDOsplit = df_PDOsplit[['PDO']].apply(standardize_on_train,
                     args=[pd.Series(np.ones(df_PDOsplit[['PDO']].size), dtype=bool,
                                     index=df_PDOsplit[['PDO']].index)],
                     result_type='broadcast')

# Butter Lowpass
dates = df_PDOsplit.index
freqraw = (dates[1] - dates[0]).days
ls = ['solid', 'dotted', 'dashdot']
fig, ax = plt.subplots(1,1, figsize=(12,3))
lowpasses = [1,2]
list_dfPDO = []
for i, yr in enumerate(lowpasses):
    window = int(yr*functions_pp.get_oneyr(dates).size) # 2 year
    if i ==0:
        ax.plot_date(dates, df_PDOsplit.values, label=f'Raw ({freqraw} day means)',
                  alpha=.3, linestyle='solid', marker=None)
    df_PDObw = pd.Series(filters.lowpass(df_PDOsplit, period=window).squeeze(),
                         index=dates, name=f'PDO{yr}bw')
    ax.plot_date(dates, df_PDObw, label=f'Butterworth {yr}-year low-pass',
            color='red',linestyle=ls[i], linewidth=1, marker=None)
    df_PDOrm = df_PDOsplit.rolling(window=window, center=True, min_periods=1).mean()
    df_PDOrm = df_PDOrm.rename({'PDO':f'PDO{yr}rm'}, axis=1)
    ax.plot_date(dates, df_PDOrm,
                 label=f'Rolling mean {yr}-year low-pass', color='green',linestyle=ls[i],
                 linewidth=1, marker=None)
    list_dfPDO.append(df_PDObw) ; list_dfPDO.append(df_PDOrm)
    ax.legend()
filepath = os.path.join(rg.path_outsub1, 'Low-pass filter {}.pdf'.format(lowpasses))
plt.savefig(filepath, bbox_inches='tight')
#%%
df_PDOs = pd.concat(list_dfPDO,axis=1)
functions_pp.store_hdf_df({'df_data':df_PDOs},
                          file_path=os.path.join(data_dir, 'df_PDOs.h5'))
#%%
rg.list_import_ts = [('PDO', os.path.join(data_dir, 'df_PDOs.h5'))]
rg.get_ts_prec()
#%% forecasting



def prediction_wrapper(df_data, keys: list=None, match_lag: bool=False,
                       n_boot: int=1):

    alphas = np.append(np.logspace(.1, 1.5, num=25), [250])
    kwrgs_model = {'scoring':'neg_mean_squared_error',
                   'alphas':alphas, # large a, strong regul.
                   'normalize':False}

    fc_mask = df_data.iloc[:,-1].loc[0]#.shift(lag, fill_value=False)
    target_ts = df_data.iloc[:,[0]].loc[0][fc_mask]
    target_ts = (target_ts - target_ts.mean()) / target_ts.std()

    out = rg.fit_df_data_ridge(df_data=df_data,
                               target=target_ts,
                               keys=keys,
                               tau_min=min(lags), tau_max=max(lags),
                               kwrgs_model=kwrgs_model,
                               match_lag_region_to_lag_fc=match_lag,
                               transformer=fc_utils.standardize_on_train)

    prediction, weights, models_lags = out
    # get skill scores
    clim_mean_temp = float(target_ts.mean())
    RMSE_SS = fc_utils.ErrorSkillScore(constant_bench=clim_mean_temp).RMSE
    MAE_SS = fc_utils.ErrorSkillScore(constant_bench=clim_mean_temp).MAE
    CRPSS = fc_utils.CRPSS_vs_constant_bench(constant_bench=clim_mean_temp).CRPSS
    score_func_list = [RMSE_SS, fc_utils.corrcoef, CRPSS, MAE_SS]

    df_train_m, df_test_s_m, df_test_m, df_boot = fc_utils.get_scores(prediction,
                                                             df_data.iloc[:,-2:],
                                                             score_func_list,
                                                             n_boot = n_boot,
                                                             blocksize=blocksize,
                                                             rng_seed=1)
    n_splits = df_data.index.levels[0].size # test for high alpha
    for col in df_test_m.columns.levels[0]:
        cvfitalpha = [models_lags[f'lag_{col}'][f'split_{s}'].alpha_ for s in range(n_splits)]
        print('lag {} mean alpha {:.0f}'.format(col, np.mean(cvfitalpha)))
        maxalpha_c = list(cvfitalpha).count(alphas[-1])
        if maxalpha_c > n_splits/3:
            print(f'\nlag {col} alpha {int(np.mean(cvfitalpha))}')
            print(f'{maxalpha_c} splits are max alpha\n')
            # maximum regularization selected. No information in timeseries
            # df_test_m.loc[:,pd.IndexSlice[col, 'corrcoef']][:] = 0
            # df_boot.loc[:,pd.IndexSlice[col, 'corrcoef']][:] = 0
            no_info_fc.append(col)
    df_test = functions_pp.get_df_test(prediction.merge(df_data.iloc[:,-2:],
                                                    left_index=True,
                                                    right_index=True)).iloc[:,:-2]
    return prediction, df_test, df_test_m, df_boot, models_lags, weights, df_test_s_m, df_train_m


no_info_fc = []
if precur_aggr <= 15:
    blocksize=2
else:
    blocksize=1

lowpass = 1 # which low-pass filtered PDO timeseries to remove from normal SST
lwp_method = 'bw'
region_labels = [1, 3]
rename_labels_d = {'1..1..sst': 'mid-Pacific (label 1)',
                   '1..3..sst': 'east-Pacific (label 2)'} # for plot only



for match_lag in [True]:#[False, True]: #
    print(f'match lag {match_lag}')
    lags = np.array([0,1,2,3,4,5]) ;

    y_keys = [k for k in rg.df_data.columns[:-2] if k not in df_PDOs.columns]
    y_keys = [k for k in y_keys if k not in [rg.TV.name]] # also remove LFV from target?
    if match_lag==False: # only keep regions of lag=0
        y_keys = [k for k in y_keys if k.split('..')[0] == str(0)]
    keys = [k for k in y_keys if k not in [rg.TV.name]] # not use target as precursor
    keys = [k for k in y_keys if int(k.split('..')[1]) in region_labels]

    # remove PDO low-pass
    df_data_r2PDO = rg.df_data.copy()
    # df_data_r2PDO = df_data_r2PDO.rename(rename_labels_d, axis=1) ; y_keys = list(rename_labels_d.values())
    df_data_r2PDO[y_keys], fig = wPCMCI.df_data_remove_z(df_data_r2PDO,
                                                       z=[f'PDO{lowpass}{lwp_method}'],
                                                       keys=y_keys,
                                                       standardize=False)
    fig_path = os.path.join(rg.path_outsub1,
                            f'{precur._name}_r{lowpass}PDO_{period}_match{match_lag}_{append_str}')
    fig.savefig(fig_path+rg.figext, bbox_inches='tight')
    # df_data_r2PDO.loc[0][keys].corrwith(rg.df_data.loc[0][keys]).plot(kind='bar')


    # =============================================================================
    # Predictions
    # =============================================================================
    out_regr2PDO = prediction_wrapper(df_data_r2PDO.copy(), keys=keys,
                                     match_lag=match_lag, n_boot=n_boot)
    out = prediction_wrapper(rg.df_data.copy(), keys=keys,
                             match_lag=match_lag, n_boot=n_boot)
    # out_PDO = prediction_wrapper(rg.df_data.copy(), keys=['PDO2bw'],
    #                              match_lag=False, n_boot=n_boot)
    df_lwp = rg.df_data.copy()
    df_lwp[keys] = df_ana.loop_df_ana(df_lwp, filters.lowpass, keys=keys,
                                kwrgs={'period':round(.25*functions_pp.get_oneyr(df_lwp.loc[0]).size)})
    out_lwp = prediction_wrapper(df_lwp, keys=keys,
                                 match_lag=True, n_boot=0)

    df_comb = pd.merge(df_lwp.iloc[:,0],
                       pd.merge(df_ana.loop_df_ana(df_lwp, filters.lowpass, keys=keys,
                        kwrgs={'period':2*functions_pp.get_oneyr(df_lwp.loc[0]).size}),
                       df_lwp.iloc[:,1:], left_index=True, right_index=True),
                       left_index=True, right_index=True)
    y_keys = [k for k in df_comb.columns[:-2] if k not in df_PDOs.columns]
    if match_lag==False: # only keep regions of lag=0
        y_keys = [k for k in y_keys if k.split('..')[0] == str(0)]
    keys = [k for k in y_keys if k not in [rg.TV.name]] # not use target as precursor
    keys = [k for k in keys if int(k.split('..')[1]) in region_labels]
    out_comb = prediction_wrapper(df_comb, keys=keys,
                             match_lag=True, n_boot=0)
    # df_lwpboth = rg.df_data.copy()
    # keys_incl_target = [k for k in rg.df_data.columns[:-2] if k not in df_PDOs.columns]
    # df_lwpboth[keys_incl_target] = df_ana.loop_df_ana(df_lwpboth, filters.lowpass, keys=keys_incl_target,
    #                             kwrgs={'period':2*functions_pp.get_oneyr(df_lwp.loc[0]).size})
    # out_lwpboth = prediction_wrapper(df_lwpboth, keys=keys,
    #                          match_lag=True, n_boot=500)

    # out_regr1PDO = prediction_wrapper(df_data_r1PDO, keys=keys,
    #                             match_lag=match_lag, n_boot=n_boot)

    # [rg.df_data.copy().loc[s].loc[:,['0..1..sst', '1..1..sst', '2..1..sst', '3..1..sst']].corr() for s in range(10)]

    # =============================================================================
    # Plot
    # =============================================================================
    #%%
    orientation = 'horizontal'
    alpha = .05
    # scores_rPDO1 = out_regr1PDO[2]
    scores_n = out[2]
    scores_rPDO2 = out_regr2PDO[2]
    metrics_cols = ['corrcoef', 'RMSE']
    rename_m = {'corrcoef': 'Corr. coeff.', 'RMSE':'RMSE-SS',
                'MAE':'MAE-SS', 'CRPSS':'CRPSS'}
    if orientation=='vertical':
        f, ax = plt.subplots(len(metrics_cols),1, figsize=(6, 5*len(metrics_cols)),
                         sharex=True) ;
    else:
        f, ax = plt.subplots(1,len(metrics_cols), figsize=(6.5*len(metrics_cols), 5),
                         sharey=False) ;
    c1, c2 = '#3388BB', '#EE6666'
    for i, m in enumerate(metrics_cols):
        labels = [str((l-1)*tfreq) if l != 0 else 'No gap' for l in lags]
        # normal SST
        ax[i].plot(labels, scores_n.reorder_levels((1,0), axis=1).loc[0][m].T,
                label=f'${target[7].capitalize()}^{target[0].capitalize()}: SST$',
                color=c2,
                linestyle='solid')
        ax[i].fill_between(labels,
                        out[3].reorder_levels((1,0), axis=1)[m].quantile(1-alpha/2.),
                        out[3].reorder_levels((1,0), axis=1)[m].quantile(alpha/2.),
                        edgecolor=c2, facecolor=c2, alpha=0.3,
                        linestyle='solid', linewidth=2)
        # Regress out PDO 2yr
        ax[i].plot(labels, scores_rPDO2.reorder_levels((1,0), axis=1).loc[0][m].T,
                label=f'${target[7].capitalize()}^{target[0].capitalize()}:$'+r' $SST$ $\vert$ $\overline{PDO}$',#'\ \left(2yr \right)$',
                color=c1,
                linestyle='-.') ;
        ax[i].fill_between(labels,
                        out_regr2PDO[3].reorder_levels((1,0), axis=1)[m].quantile(1-alpha/2.),
                        out_regr2PDO[3].reorder_levels((1,0), axis=1)[m].quantile(alpha/2.),
                        edgecolor=c1, facecolor=c1, alpha=0.3,
                        linestyle='dashed', linewidth=2)
        # low-pass filtered SST timeseries
        # ax[i].plot(labels, out_lwp[2].reorder_levels((1,0), axis=1).loc[0][m].T,
        #         label=f'${target[7].capitalize()}^{target[0].capitalize()}:$'+r' $\overline{SST}_{lowpass\ seasonal}$',
        #         color='green',
        #         linestyle='dashed') ;
        # ax[i].fill_between(labels,
        #                 out_lwp[3].reorder_levels((1,0), axis=1)[m].quantile(1-alpha/2.),
        #                 out_lwp[3].reorder_levels((1,0), axis=1)[m].quantile(alpha/2.),
        #                 edgecolor='grey', facecolor='grey', alpha=0.3,
        #                 linestyle='solid', linewidth=2)


        if m == 'corrcoef':
            ax[i].set_ylim(-.3,.6)
        else:
            ax[i].set_ylim(-.15,.35)
        ax[i].axhline(y=0, color='black', linewidth=1)
        ax[i].tick_params(labelsize=16)
        if i == len(metrics_cols)-1 and orientation=='vertical':
            ax[i].set_xlabel('Lead time defined as gap [days]', fontsize=18)
        else:
            ax[i].set_xlabel('Lead time defined as gap [days]', fontsize=18)
        if i == 0:
            ax[i].legend(loc='lower right', fontsize=14)
        if target == 'westerntemp' or orientation == 'horizontal':
            ax[i].set_ylabel(rename_m[m], fontsize=18, labelpad=-4)


    f.subplots_adjust(hspace=.1)
    f.subplots_adjust(wspace=.2)
    title = f'{precur_aggr}-day mean ${target[7].capitalize()}^{target[0].capitalize()}$ prediction'
    if orientation == 'vertical':
        f.suptitle(title, y=.92, fontsize=18)
    else:
        f.suptitle(title, y=.95, fontsize=18)
    f_name = 'fc_{}_a{}'.format(precur._name,
                                  precur.alpha) + '_' + \
                                 f'matchlag{match_lag}_nb{n_boot}' + \
                                 f'{method}_{append_str}_{lwp_method}_{lowpass}'
    fig_path = os.path.join(rg.path_outsub1, f_name)+rg.figext

    f.savefig(fig_path, bbox_inches='tight')
#%%
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib import gridspec
# import datetime


def plot_date_years(ax, dates, years, fontsize=12):
    minor = np.arange(len(y_true.dropna()))
    step = int(dates.year.size /np.unique(dates.year).size)
    major = [i for i in minor[::step] if dates[i].year in years]
    ax.set_xticks(np.arange(len(dates)), minor=True)
    ax.set_xticks(major)
    ax.set_xticklabels(years, fontsize=fontsize);


lag = 3
normal = out[1][lag]
fc_lwp = out_lwp[1][lag]
y_true = out_lwp[1]['2ts']
df_PDOraw = df_PDOsplit[['PDO']].copy()
df_PDOraw['year'] = df_PDOraw.index.year
_d = y_true.index
# nonsummer_d = rg.df_data.loc[0]['RV_mask'][~rg.df_data.loc[0]['RV_mask']]
lowpass_cond_fc = 1


df_PDOplotlw = df_PDOs[f'PDO{lowpass_cond_fc}bw'].loc[_d - pd.Timedelta('0d')]
# df_PDOnonsummer = df_PDOs[f'PDO{lowpass_cond_fc}bw'].loc[nonsummer_d.index]
df_PDOnonsummer = df_PDOs[f'PDO{lowpass_cond_fc}bw']
# yrmeanPDO = df_PDOnonsummer.groupby(df_PDOnonsummer.index.year).mean()
yrmeanPDO = df_PDOraw.shift(-184-30).dropna().groupby('year').mean()
yrmeanPDO.index = yrmeanPDO.index.astype(int)
years = yrmeanPDO.abs() > yrmeanPDO.std()
dates_strPDO = pd.to_datetime([d for d in _d if bool(years.loc[d.year].values)])
mask = pd.Series(np.zeros(_d.size), index=_d, dtype=bool)
mask.loc[dates_strPDO] = True


c1, c2 = '#3388BB', '#EE6666'
# f, ax = plt.subplots(2,1, figsize=(10, 8), sharex=False,
#                      width_ratios=[3,5],
#                      height_ratios=[1,1])
fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 5])
facecolor='grey'
ax0 = plt.subplot(gs[1], facecolor=facecolor)
ax0.patch.set_alpha(0.2)
ax1 = plt.subplot(gs[0], facecolor=facecolor)
ax1.patch.set_alpha(0.2)
ax0.plot(normal.values, color=c2, linestyle='--', lw=2.5,
         label=f'${target[7].capitalize()}^{target[0].capitalize()}:\ SST$')
ax0.plot(y_true.values, color='black', linestyle='solid', lw=1,
         label=f'${target[7].capitalize()}^{target[0].capitalize()}$ observed', alpha=.8)
# ax0.plot(fc_lwp.values, color='green',linestyle='dashed',
#          label=f'${target[0].capitalize()}$-${target[7].capitalize()}:$'+r' $\overline{SST}_{lowpass\ seasonal}$')
ax0.fill_between(range(y_true.index.size), normal.values,
                   where=mask, color='lightblue', alpha=1)
legend1 = ax0.legend(loc='lower left', fontsize=14, frameon=True)
ax0.add_artist(legend1)
ax0.set_ylim(-3,3)
ax0.margins(x=.05) ; ax1.margins(x=.05)

# skill score 1
SS_normal = fc_utils.ErrorSkillScore(constant_bench=float(y_true.mean())).MAE(y_true, normal)
SS_strong = fc_utils.ErrorSkillScore(constant_bench=float(y_true.mean())).MAE(y_true[mask], normal[mask])
patch = mpatches.Patch(edgecolor=c2, facecolor='lightblue',linewidth=2, linestyle='dashed',
                           label='MAE-SS {:.2f}'.format(SS_strong))
line = Line2D([0], [0], color=c2, lw=2,ls='dashed',
                  label='MAE-SS {:.2f}'.format(SS_normal))
legend2 = ax0.legend(loc='upper center', handles=[patch, line], fontsize=14,
                     frameon=True)
# ax0.add_artist(legend2)
# skill score 2
SS_normal = fc_utils.ErrorSkillScore(constant_bench=float(y_true.mean())).RMSE(y_true, normal)
SS_strong = fc_utils.ErrorSkillScore(constant_bench=float(y_true.mean())).RMSE(y_true[mask], normal[mask])
patch = mpatches.Patch(edgecolor=c2, facecolor='lightblue',linewidth=2, linestyle='dashed',
                           label='RMSE-SS {:.2f}'.format(SS_strong))
line = Line2D([0], [0], color=c2, lw=2,ls='dashed',
                  label='RMSE-SS {:.2f}'.format(SS_normal))
legend2 = ax0.legend(loc='upper right', handles=[patch, line], fontsize=14,
                     frameon=True)
ax0.add_artist(legend1)
ax0.add_artist(legend2)


ax1.plot(df_PDOraw.loc[y_true.index].values, color='blue', ls='dashed')
ax1.plot(df_PDOplotlw.values, color='blue', alpha=.5)
ax1.fill_between(range(y_true.index.size), df_PDOplotlw.values,
                   where=mask, color='lightblue', alpha=1)
plt.subplots_adjust(right=1)
ax1.axhline(y=0, color='black', lw=1)
ax1.set_ylim(-3.5,3.5)
plot_date_years(ax0, dates=y_true.index, years = np.arange(1980, 2021, 5))
plot_date_years(ax1, dates=y_true.index, years = np.arange(1980, 2021, 5))
ax0.tick_params(labelsize=14)
ax1.tick_params(labelsize=14)

patch = mpatches.Patch(edgecolor=c2, color='lightblue',
                           label='Anomalous PDO: strong boundary condition forcing',
                           alpha=1)
line = Line2D([0], [0], color='blue', lw=2, ls='dashed',
                  label='$PDO$')
linelwp = Line2D([0], [0], color='blue', lw=2, alpha=.5,
                  label=r'$\overline{PDO}$')
legend1 = ax1.legend(loc='upper left', handles=[line, linelwp], fontsize=13,
                     frameon=True)
legend2 = ax1.legend(loc='lower right', handles=[patch], fontsize=13)
ax1.add_artist(legend1)
ax1.add_artist(legend2)
f_name = f'conditional_forecast_lag{lag}_lwp{lowpass_cond_fc}'
fig_path = os.path.join(rg.path_outsub1, f_name)+rg.figext
fig.savefig(fig_path, bbox_inches='tight')

#%%
plt.scatter(df_PDOraw.loc[y_true.index][~mask].abs(),
            fc_utils.CRPSS_vs_constant_bench(constant_bench=float(y_true.mean()),
                                             return_mean=False).CRPSS(y_true[~mask], fc_lwp[~mask]),
            color=c2);
plt.scatter(df_PDOraw.loc[y_true.index][mask].abs(),
            fc_utils.CRPSS_vs_constant_bench(constant_bench=float(y_true.mean()),
                                             return_mean=False).CRPSS(y_true[mask], fc_lwp[mask]),
            color='lightblue');
plt.ylim(-.5,1)

#%%
# remove PDO df
os.remove(os.path.join(data_dir, 'df_PDOs.h5'))

#%%
# if experiment == 'adapt_corr':

#     corr = dm[monthkeys[0]].mean(dim='split').drop('time')
#     list_xr = [corr.expand_dims('months', axis=0) for i in range(len(monthkeys))]
#     corr = xr.concat(list_xr, dim = 'months')
#     corr['months'] = ('months', monthkeys)

#     np_data = np.zeros_like(corr.values)
#     np_mask = np.zeros_like(corr.values)
#     for i, f in enumerate(monthkeys):
#         corr_xr = dm[f]
#         vals = corr_xr.mean(dim='split').values
#         np_data[i] = vals
#         mask = corr_xr.mask.mean(dim='split')
#         np_mask[i] = mask

#     corr.values = np_data
#     mask = (('months', 'lag', 'latitude', 'longitude'), np_mask )
#     corr.coords['mask'] = mask

#     precur = rg.list_for_MI[0]
#     f_name = 'corr_{}_a{}'.format(precur.name,
#                                 precur.alpha) + '_' + \
#                                 f'{experiment}_lag{lag}_' + \
#                                 f'tf{precur_aggr}_{method}'

#     corr.to_netcdf(os.path.join(rg.path_outsub1, f_name+'.nc'), mode='w')
#     import_ds = core_pp.import_ds_lazy
#     corr = import_ds(os.path.join(rg.path_outsub1, f_name+'.nc'))[precur.name]
#     subtitles = np.array([monthkeys])
#     kwrgs_plot = {'aspect':2, 'hspace':.3,
#                   'wspace':-.4, 'size':1.25, 'cbar_vert':-0.1,
#                   'units':'Corr. Coeff. [-]',
#                   'clim':(-.60,.60), 'map_proj':ccrs.PlateCarree(central_longitude=220),
#                   'y_ticks':False,
#                   'x_ticks':False,
#                   'subtitles':subtitles}

#     fig = plot_maps.plot_corr_maps(corr, mask_xr=corr.mask, col_dim='months',
#                                    row_dim=corr.dims[1],
#                                    **kwrgs_plot)

#     fig_path = os.path.join(rg.path_outsub1, f_name)+rg.figext
# #%%
#     plt.savefig(fig_path, bbox_inches='tight')


#%%

# df = df_test_b.stack().reset_index(level=1)
# dfx = df.groupby(['level_1'])
# axes = dfx.boxplot()
# axes[0].set_ylim(-0.5, 1)
#%%
# import seaborn as sns
# df_ap = pd.concat(list_test_b, axis=0, ignore_index=True)
# df_ap['months'] = np.repeat(monthkeys, list_test_b[0].index.size)
# # df_ap.boxplot(by='months')
# ax = sns.boxplot(x=df_ap['months'], y=df_ap['mean_squared_error'])
# ax.set_ylim(-0.5, 1)
# plt.figure()
# ax = sns.boxplot(x=df_ap['months'], y=df_ap['corrcoef'])
# ax.set_ylim(-0.5, 1)

# #%%
# columns_my_order = monthkeys
# fig, ax = plt.subplots()
# for position, column in enumerate(columns_my_order):
#     ax.boxplot(df_test_b.loc[column], positions=[position,position+.25])

# ax.set_xticks(range(position+1))
# ax.set_xticklabels(columns_my_order)
# ax.set_xlim(xmin=-0.5)
# plt.show()


#%%

