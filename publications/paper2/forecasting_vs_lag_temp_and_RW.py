#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 09:59:06 2020

@author: semvijverberg
"""


import os, inspect, sys
if sys.platform == 'linux':
    import matplotlib as mpl
    mpl.use('Agg')
import numpy as np
from time import time
import cartopy.crs as ccrs ; import matplotlib.pyplot as plt
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

# Optionally set font to Computer Modern to avoid common missing font errors
# matplotlib.rc('font', family='serif', serif='cm10')

# matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']

# targets = ['temp', 'RW']


# if region == 'eastern':
targets = ['easterntemp', 'westerntemp']

periods = ['JA_center', 'JA_shiftright', 'JA_shiftleft']
seeds = np.array([1,2,3])
combinations = np.array(np.meshgrid(targets, periods, seeds)).T.reshape(-1,3)

i_default =9

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
    alpha_corr = .01
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

precur_aggr = tfreq
method     = 'ran_strat10' ;
n_boot = 5000

append_main = ''
# name_csv = f'skill_scores_tf{tfreq}.csv'


#%% run RGPD
# start_end_TVdate = ('06-01', '08-31')
start_end_date = ('3-1', '08-31') # focus on spring/summer. Important for regressing out influence of PDO (might vary seasonally)
list_of_name_path = [(cluster_label, TVpath),
                     ('sst', os.path.join(path_raw, 'sst_1979-2018_1_12_daily_1.0deg.nc'))]

lowpass = '2y'
list_import_ts = [('PDO', os.path.join(data_dir, f'PDO_{lowpass}_rm_25-09-20_15hr.h5'))]

list_for_MI   = [BivariateMI(name='sst', func=class_BivariateMI.corr_map,
                            alpha=alpha_corr, FDR_control=True,
                            kwrgs_func={}, group_split='together',
                            distance_eps=1200, min_area_in_degrees2=10,
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
    title = r'$corr(SST_t, mx2t_t)$'
else:
    title = r'$parcorr(SST_t, mx2t_t\ |\ SST_{t-1},mx2t_{t-1})$'
subtitles = np.array([[f'gap = {(l-1)*tfreq} days' for l in rg.list_for_MI[0].lags]]) #, f'lag 2 (15 day lead)']] )
subtitles[0][0] = 'No gap'
kwrgs_plotcorr = {'row_dim':'split', 'col_dim':'lag','aspect':2, 'hspace':-.47,
              'wspace':-.15, 'size':3, 'cbar_vert':-.1,
              'units':'Corr. Coeff. [-]', 'zoomregion':(130,260,-10,60),
              'clim':(-.60,.60), 'map_proj':ccrs.PlateCarree(central_longitude=220),
              'y_ticks':np.arange(-10,61,20), 'x_ticks':np.arange(140, 280, 25),
              'subtitles':subtitles, 'title':title,
              'title_fontdict':{'fontsize':16, 'fontweight':'bold', 'y':1.07}}
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
rg.quick_view_labels(save=True, append_str=f'{tfreq}d',
                     min_detect_gc=.5)
# plotting corr_map
rg.plot_maps_corr(var='sst', save=True,
                  kwrgs_plot=kwrgs_plotcorr,
                  min_detect_gc=1.0,
                  append_str=f'{tfreq}d')



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
    RMSE_SS = fc_utils.RMSE_vs_constant_bench(RMSE_vs_constant_bench=clim_mean_temp).RMSE
    score_func_list = [RMSE_SS, fc_utils.corrcoef]

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



for match_lag in [False, True]:
    print(f'match lag {match_lag}')
    lags = np.array([0,1,2,3,4,5]) ;


    keys = [k for k in rg.df_data.columns[:-2] if k not in [rg.TV.name, 'PDO']]
    if match_lag==False: # only keep regions of lag=0
        keys = [k for k in keys if k.split('..')[0] == str(0)]

    y_keys = [k for k in keys if 'sst' in k]
    df_data_rPDO = rg.df_data.copy()
    df_data_rPDO[y_keys], fig = wPCMCI.df_data_remove_z(df_data_rPDO, z=['PDO'], keys=y_keys,
                                                standardize=False)

    fig_path = os.path.join(rg.path_outsub1,
                            f'{precur._name}_rPDO_{period}_match{match_lag}_{append_str}')
    fig.savefig(fig_path+rg.figext, bbox_inches='tight')
    plt.figure()
    # df_data_rPDO.loc[0][keys].corrwith(rg.df_data.loc[0][keys]).plot(kind='bar')
    out_regrPDO = prediction_wrapper(df_data_rPDO.copy(), keys=keys,
                                     match_lag=match_lag, n_boot=n_boot)
    out = prediction_wrapper(rg.df_data.copy(), keys=keys,
                             match_lag=match_lag, n_boot=n_boot)
    outPDO = prediction_wrapper(rg.df_data.copy(), keys=['PDO'],
                                match_lag=False, n_boot=0)

    # [rg.df_data.copy().loc[s].loc[:,['0..1..sst', '1..1..sst', '2..1..sst', '3..1..sst']].corr() for s in range(10)]

    # =============================================================================
    # Plot
    # =============================================================================
    alpha = .05
    scores_rPDO = out_regrPDO[2]
    scores_n = out[2]
    score_PDO = outPDO[2]
    metrics_cols = ['corrcoef', 'RMSE']
    rename_m = {'corrcoef': 'Corr. coeff', 'RMSE':'RMSE-SS'}
    f, ax = plt.subplots(len(metrics_cols),1, figsize=(6, 5*len(metrics_cols)),
                         sharex=True) ;
    c1, c2 = '#3388BB', '#EE6666'
    for i, m in enumerate(metrics_cols):
        labels = [str((l-1)*tfreq) if l != 0 else 'No gap' for l in lags]
        # normal SST
        ax[i].plot(labels, scores_n.reorder_levels((1,0), axis=1).loc[0][m].T,
                label='SST',
                color=c2,
                linestyle='solid')
        ax[i].fill_between(labels,
                        out[3].reorder_levels((1,0), axis=1)[m].quantile(1-alpha/2.),
                        out[3].reorder_levels((1,0), axis=1)[m].quantile(alpha/2.),
                        edgecolor=c2, facecolor=c2, alpha=0.3,
                        linestyle='solid', linewidth=2)
        # regressed out PDO
        ax[i].plot(labels, scores_rPDO.reorder_levels((1,0), axis=1).loc[0][m].T,
                label=r'SST | regr. out $PDO_{lwp}$',
                color=c1,
                linestyle='dashed') ;
        ax[i].fill_between(labels,
                        out_regrPDO[3].reorder_levels((1,0), axis=1)[m].quantile(1-alpha/2.),
                        out_regrPDO[3].reorder_levels((1,0), axis=1)[m].quantile(alpha/2.),
                        edgecolor=c1, facecolor=c1, alpha=0.3,
                        linestyle='dashed', linewidth=2)
        # Only PDO
        ax[i].plot(labels, score_PDO.reorder_levels((1,0), axis=1).loc[0][m].T,
                label=r'$PDO_{lwp}$',
                color='grey',
                linestyle='-.') ;
        if m == 'corrcoef':
            ax[i].set_ylim(-.3,.6)
        else:
            ax[i].set_ylim(-.1,.5)
        ax[i].axhline(y=0, color='black', linewidth=1)
        ax[i].tick_params(labelsize=16)
        if i == len(metrics_cols)-1:
            ax[i].set_xlabel('Lead time defined as gap [days]', fontsize=18)
        if i == 0:
            ax[i].legend()
        if target == 'westerntemp':
            ax[i].set_ylabel(rename_m[m], fontsize=18)

    f.subplots_adjust(hspace=.1)
    title = f'{precur_aggr}-day mean {target[:7]} U.S. {target[7:]}. pred.'
    f.suptitle(title, y=.92, fontsize=18)
    f_name = 'fc_{}_a{}'.format(precur._name,
                                  precur.alpha) + '_' + \
                                 f'matchlag{match_lag}_' + \
                                 f'{method}_{append_str}'
    fig_path = os.path.join(rg.path_outsub1, f_name)+rg.figext
    f.savefig(fig_path, bbox_inches='tight')

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

