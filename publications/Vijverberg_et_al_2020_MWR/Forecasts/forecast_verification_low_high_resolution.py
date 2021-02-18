#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 11:59:19 2020

@author: semvijverberg
"""
import os, inspect, sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
user_dir = os.path.expanduser('~')
curr_dir = '/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/Sub-seasonal_statistical_forecasts_of_eastern_United_States_extreme_temperature_events/Forecasts' # script directory
main_dir = '/'.join(curr_dir.split('/')[:-3])
data_dir = '/'.join(curr_dir.split('/')[:-1]) + '/data'
RGCPD_dir = os.path.join(main_dir, 'RGCPD')
fc_dir = os.path.join(main_dir, 'forecasting')
df_ana_dir = os.path.join(main_dir, 'df_analysis/df_analysis/')
if fc_dir not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(RGCPD_dir)
    sys.path.append(df_ana_dir)
    sys.path.append(fc_dir)



import plot_maps
import cartopy.crs as ccrs
from RGCPD import RGCPD
from RGCPD import BivariateMI
import func_models as fc_utils
import functions_pp

ERA_data = data_dir + '/CPPA_ERA5_14-05-20_08hr_lag_0_c378f.h5'
RV = user_dir + '/surfdrive/output_RGCPD/easternUS/tf1_n_clusters5_q95_dendo_c378f.nc'
list_of_name_path = [(1, RV)]
list_import_ts = [('ts', ERA_data)]

rg = RGCPD(list_of_name_path=list_of_name_path,
           list_import_ts=list_import_ts,
            start_end_TVdate=('06-24', '08-22'),
            start_end_date=('1-1', '12-31'),
            tfreq=15,
            path_outmain=user_dir+'/surfdrive/output_RGCPD')

rg.pp_TV(name_ds='q95')

rg.traintest()
rg.get_ts_prec()
#%%
# from stat_models import logit_skl
from stat_models_cont import ScikitModel
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics

def prediction_wrapper(q):
    fcmodel = ScikitModel(scikitmodel=LogisticRegressionCV).fit
    kwrgs_model = { 'class_weight':{ 0:1, 1:1},
                    'scoring':'neg_brier_score',
                    'penalty':'l2',
                    'solver':'lbfgs'}

    lag = 4
    keys = ['0..PEPsv'] #rg.df_data.columns[2:-2]
    keys = [k for k in rg.df_data.columns[2:-2] if 'sst' in k]
    target_ts = rg.TV_ts# - rg.TV_ts.mean()) / rg.TV_ts.std()
    # target_ts = rg.df_data_ext.loc[0][['mx2t']][rg.df_data.loc[0]['RV_mask']]
    target_ts  = target_ts.to_dataframe('target')[['target']]
    target_ts.index.name = None
    target_ts = (target_ts > target_ts.quantile(q=q)).astype(int)
    out = rg.fit_df_data_ridge(target=target_ts,
                               fcmodel=fcmodel,
                               keys=keys,
                               tau_min=0, tau_max=lag,
                               kwrgs_model=kwrgs_model)

    prediction, weights, models_lags = out

    df_test = functions_pp.get_df_test(prediction.merge(rg.df_data.iloc[:,-2:].copy(),
                                                    left_index=True,
                                                    right_index=True)).iloc[:,:-2]

    # get skill scores
    clim_mean_temp = float(target_ts.mean())
    SS = fc_utils.ErrorSkillScore(constant_bench=clim_mean_temp)
    BSS = SS.BSS
    score_func_list = [metrics.roc_auc_score, BSS, fc_utils.ErrorSkillScore().AUC_SS]

    df_train_m, df_test_s_m, df_test_m, df_boot = fc_utils.get_scores(prediction,
                                                             rg.df_data.iloc[:,-2:],
                                                             score_func_list,
                                                             score_per_test=False,
                                                             n_boot = 0,
                                                             blocksize=2,
                                                             rng_seed=1)
    return df_train_m, df_test_m, df_boot, df_test, models_lags, SS

q1df_train_m, q1df_test_m, q1df_boot, q1df_test, q1models_lags, SS = prediction_wrapper(.90)
q2df_train_m, q2df_test_m, q2df_boot, q2df_test, q2models_lags, SS = prediction_wrapper(.66)

print(q1df_test_m.loc[0])

# m = models_lags[f'lag_{lag}']['split_2']

#%%
qrange = [.50, .66, .75, .85, .90, .95] ; d = {}
for q in qrange:
    d[q] = prediction_wrapper(q)[1]

#%%
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib import gridspec
# import datetime





orientation = 'horizontal'
alpha = .05
# scores_rPDO1 = out_regr1PDO[2]
scores_n = q1df_test_m
metrics_cols = ['roc_auc_score', 'BSS', 'AUC_SS']
rename_m = {'roc_auc_score': 'AUC', 'BSS':'BSS', 'AUC_SS':'AUC-SS'}
if orientation=='vertical':
    f, ax = plt.subplots(len(metrics_cols),1, figsize=(6, 5*len(metrics_cols)),
                     sharex=True) ;
else:
    f, ax = plt.subplots(1,len(metrics_cols), figsize=(6.5*len(metrics_cols), 5),
                     sharey=False) ;
c1, c2 = '#3388BB', '#EE6666'
for i, m in enumerate(metrics_cols):
    labels = [str((l-1)*rg.tfreq) if l != 0 else 'No gap' for l in q1df_test.columns[1:]]
    # normal SST
    ax[i].plot(labels, q1df_test_m.reorder_levels((1,0), axis=1).loc[0][m].T,
            label=f'forecast low resolution',
            color=c2,
            linestyle='solid')
    ax[i].plot(labels, q2df_test_m.reorder_levels((1,0), axis=1).loc[0][m].T,
            label=f'forecast high resolution',
            color=c1,
            linestyle='solid')
    if m == 'roc_auc_score':
        ax[i].set_ylim(0,1)
    else:
        ax[i].set_ylim(-.6,.6)


#%%
def plot_date_years(ax, y, years, fontsize=12):
    dates = y.index
    minor = np.arange(len(y.dropna()))
    step = int(dates.year.size /np.unique(dates.year).size)
    major = [i for i in minor[::step] if dates[i].year in years]
    ax.set_xticks(np.arange(len(dates)), minor=True)
    ax.set_xticks(major)
    ax.set_xticklabels(years, fontsize=fontsize);

c1, c2 = '#EE6666', '#7b2cbf'
plotlag = 3
q1y_true = q1df_test[['target']]
q1y_pred = q1df_test[[plotlag]]
q2y_true = q2df_test[['target']]
q2y_pred = q2df_test[[plotlag]]


fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(2, 3)
facecolor='grey'
# ax0 = plt.subplot(gs[1,0], facecolor=facecolor)
# ax0.patch.set_alpha(0.2)
# ax1 = plt.subplot(gs[1, 1], facecolor=facecolor)
# ax1.patch.set_alpha(0.2)
# ax2 = plt.subplot(gs[0, :], facecolor=facecolor)
# ax2.patch.set_alpha(0.2)


ax1 = fig.add_subplot(gs[1, :], facecolor=facecolor)
ax1.patch.set_alpha(0.2)
ax0 = fig.add_subplot(gs[0,:], facecolor=facecolor, sharex=ax1)
ax0.patch.set_alpha(0.2)
plt.setp(ax0.get_xticklabels(), visible=False)
# ax2 = plt.subplot(gs[2:, :], facecolor=facecolor)
# ax2.patch.set_alpha(0.2)

# event threshold 1
ax0.plot(q1y_pred.values, color=c1, linestyle='solid', lw=2.5)
ax0.axhline(y=float(q1y_true.mean()), color='black', linewidth=1)
plot_date_years(ax0, q1y_true, years = np.arange(1980, 2021, 5))
ax0.fill_between(range(q1y_true.size), 0, q1y_true.values.squeeze(),
                 alpha=.5, color='black')

ax0.set_ylim(0,1.2)
ax0.set_yticks(np.arange(0,1.21,.25))
ax0.set_yticklabels(np.array(np.arange(0,1.21,.25)*100,dtype=int), fontsize=12)
ax0.set_ylabel('Forecast probability', fontsize=16)

# skill score event threshold 1
SS_normal = q1df_test_m.reorder_levels((1,0), axis=1).loc[0]['BSS'][plotlag]
AUC = q1df_test_m.reorder_levels((1,0), axis=1).loc[0]['roc_auc_score'][plotlag]

line1 = Line2D([0], [0], color=c1, lw=2,ls='solid',
                  label=f'Forecast {15*(plotlag-1)} day lead-time')
patch = mpatches.Patch(facecolor='black',linewidth=0, alpha=.5,
                           label='Events ($90^{th}$ perc.)')
# legend2 = ax0.legend(loc='lower left', handles=[line1, patch], fontsize=14,
#                      frameon=True)
# ax0.add_artist(legend2)
ax0.text(1, 0.09, 'clim. probability',
        transform=ax0.transAxes, fontsize=16,
        verticalalignment='bottom', horizontalalignment='right')
props = dict(boxstyle='round', facecolor=c1, alpha=0.5)
ax0.text(0.25, 0.95, 'BSS {:.2f}'.format(SS_normal),
        transform=ax0.transAxes, fontsize=16,
        verticalalignment='top', bbox=props)
ax0.text(0.55, 0.95, 'AUC-ROC {:.2f}'.format(AUC),
        transform=ax0.transAxes, fontsize=16,
        verticalalignment='top', bbox=props)
ax0.set_title('Low forecast quality (BSS<0) for extreme heat events, but high AUC-ROC',
              fontsize=16)

# event threshold 2
ax1.plot(q2y_pred.values, color=c2, linestyle='solid', lw=2.5,
         label=f'forecast lag {plotlag}')
ax1.axhline(y=float(q2y_true.mean()), color='black', linewidth=1)
plot_date_years(ax1, q1y_true, years = np.arange(1980, 2021, 5))

ax1.set_ylim(0,1.2)
ax1.set_yticks(np.arange(0,1.21,.25))
ax1.set_yticklabels(np.array(np.arange(0,1.21,.25)*100,dtype=int), fontsize=12)
ax1.set_ylabel('Forecast probability', fontsize=16)
ax1.margins(x=.05) ; ax1.margins(x=.05)
ax1.fill_between(range(q2y_pred.size), 0, q2y_true.values.squeeze(),
                 alpha=.5, color='black')
ax1.set_title('Better forecast quality for anomalous heat events, AUC-ROC not increased',
              fontsize=16)




# skill score event threshold 2
SS_normal = q2df_test_m.reorder_levels((1,0), axis=1).loc[0]['BSS'][plotlag]
AUC = q2df_test_m.reorder_levels((1,0), axis=1).loc[0]['roc_auc_score'][plotlag]

line1 = Line2D([0], [0], color=c2, lw=2,ls='solid',
                  label=f'Forecast {15*(plotlag-1)} day lead-time')
patch = mpatches.Patch(facecolor='black',linewidth=0, alpha=.5,
                           label='Events ($66^{th}$ perc.)')
# legend2 = ax1.legend(loc='lower left', handles=[line1, patch], fontsize=14,
#                      frameon=True)
# ax1.add_artist(legend2)
ax1.text(1, 0.29, 'clim. probability',
        transform=ax1.transAxes, fontsize=16,
        verticalalignment='bottom', horizontalalignment='right')
props = dict(boxstyle='round', facecolor=c2, alpha=0.5)
ax1.text(0.25, 0.95, 'BSS {:.2f}'.format(SS_normal),
        transform=ax1.transAxes, fontsize=16,
        verticalalignment='top', bbox=props)
ax1.text(0.55, 0.95, 'AUC-ROC {:.2f}'.format(AUC),
        transform=ax1.transAxes, fontsize=16,
        verticalalignment='top', bbox=props)

ax1.tick_params(which='y', labelsize=18)
fig.suptitle('Probabilistic forecast of temperature events', y=0.96, fontsize=20, fontweight='bold')

png1 = BytesIO()
dpi = 300
fig.savefig(png1, format='png', dpi=dpi ,
            bbox_inches='tight')
fig.savefig(png1, format='png', dpi=dpi ,
            bbox_inches='tight')
png2 = Image.open(png1)
png2.save(os.path.join(functions_pp.get_download_path(), f'BSS_vs_AUC_{dpi}dpi.tif'))
png1.close()


#%%
# # skill versus threshold
# metrics_cols = ['BSS', 'AUC_SS']
# colors = ['blue', 'green'] ; lstyles = ['solid', 'dashed']
# rename_m = {'roc_auc_score': 'AUC', 'BSS':'BSS', 'AUC_SS':'AUC-ROC-SS'}
# for i, m in enumerate(metrics_cols):
#     ss = [d[q].reorder_levels((1,0), axis=1).loc[0][m].loc[plotlag] for q in qrange]
#     ax2.plot(qrange, ss, label=rename_m[m], c=colors[i], ls=lstyles[i])
# ax2.legend(loc='upper center')


# for i, m in enumerate(metrics_cols):
#     c_sc = [c1 if q in [.90] else colors[i] for q in qrange]
#     c_sc[np.argwhere(np.array(qrange)==.66)[0][0]] = c2
#     ss = [d[q].reorder_levels((1,0), axis=1).loc[0][m].loc[plotlag] for q in qrange]
#     ax2.scatter(qrange, ss, label=rename_m[m],
#                 s=[120 if q in [.66,.90] else 40 for q in qrange],
#                 c=c_sc)
# ax2.axhline(y=0, color='black', linewidth=1)
# ax2.set_ylim(-.1,1)
# ax2.set_ylabel('Skill Score', fontsize=12)
# ax2.set_xlabel('Percentile threshold for event timeseries', fontsize=12)
# png1 = BytesIO()
# fig.savefig(png1, format='png', dpi=600,
#             bbox_inches='tight')
# png2 = Image.open(png1)
# png2.save(os.path.join(functions_pp.get_download_path(), 'BSS_vs_AUC.tiff'))
# png1.close()
# #%%
# import pandas as pd
# from validation import get_metrics_sklearn
# clim = np.zeros(q1y_true.size) ; clim[:] = q1y_true.mean()
# _q1df_ = get_metrics_sklearn(q1df_test.iloc[:,0], q1df_test.iloc[:,1:],
#                            pd.Series(clim,index=q1df_test.index))[0]
# clim = np.zeros(q2y_true.size) ; clim[:] = q2y_true.mean()
# _q2df_ = get_metrics_sklearn(q2df_test.iloc[:,0], q2df_test.iloc[:,1:],
#                            pd.Series(clim,index=q2df_test.index))[0]


