#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 09:59:06 2020

@author: semvijverberg
"""


import os, inspect, sys
import numpy as np
from time import time
import cartopy.crs as ccrs ; import matplotlib.pyplot as plt

user_dir = os.path.expanduser('~')
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-2])
RGCPD_func = os.path.join(main_dir, 'RGCPD')
cluster_func = os.path.join(main_dir, 'clustering/')
fc_dir = os.path.join(main_dir, 'forecasting')
if cluster_func not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(RGCPD_func)
    sys.path.append(cluster_func)
    sys.path.append(fc_dir)

path_raw = user_dir + '/surfdrive/ERA5/input_raw'


from RGCPD import RGCPD
from RGCPD import BivariateMI
import class_BivariateMI
import find_precursors

TVpath = '/Users/semvijverberg/surfdrive/output_RGCPD/circulation_US_HW/tf15_nc3_dendo_0ff31.nc'
path_out_main = os.path.join(main_dir, 'publications/paper2/output/east_forecast/')
path_data = os.path.join(main_dir, 'publications/paper2/data/')
cluster_label = 2
name_ds='ts'



start_end_date = ('1-1', '12-31')
tfreq = 15

#%% run RGPD
start_end_TVdate = ('06-01', '08-31')
list_of_name_path = [(cluster_label, TVpath),
                     ('sst', os.path.join(path_raw, 'sst_1979-2018_1_12_daily_1.0deg.nc'))]


list_for_MI   = [BivariateMI(name='sst', func=class_BivariateMI.parcorr_map_time,
                            alpha=.05, FDR_control=True,
                            kwrgs_func={'precursor':True},
                            distance_eps=800, min_area_in_degrees2=1,
                            calc_ts='region mean', selbox=(120,260,-10,90),
                            lags=np.array([0]))]



rg = RGCPD(list_of_name_path=list_of_name_path,
           list_for_MI=list_for_MI,
           list_import_ts=None,
            start_end_TVdate=start_end_TVdate,
            start_end_date=start_end_date,
            start_end_year=None,
            tfreq=tfreq,
            path_outmain=path_out_main,
            append_pathsub='_' + name_ds)


rg.pp_TV(name_ds=name_ds, detrend=False)
rg.pp_precursors()

rg.traintest(method='random10')
rg.calc_corr_maps()

lags = rg.list_for_MI[0].lags

#%%
import matplotlib
# Optionally set font to Computer Modern to avoid common missing font errors
matplotlib.rc('font', family='serif', serif='cm10')

matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']

save = True
SST_green_bb = (140,235,20,59)#(170,255,11,60)
# subtitles = np.array([[f'lag {l}: SST vs E-U.S. mx2t' for l in rg.lags]])
title = r'$parcorr(SST_t, mx2t_t\ |\ SST_{t-1},mx2t_{t-1})$'
subtitles = np.array([['']]) #, f'lag 2 (15 day lead)']] )
kwrgs_plot = {'row_dim':'split', 'col_dim':'lag','aspect':2, 'hspace':-.47,
              'wspace':-.15, 'size':3, 'cbar_vert':-.08,
              'units':'Corr. Coeff. [-]', 'zoomregion':(130,260,-10,60),
              'clim':(-.60,.60), 'map_proj':ccrs.PlateCarree(central_longitude=220),
              'n_yticks':6, 'x_ticks':np.arange(130, 280, 25),
              'subtitles':subtitles, 'title':title,
              'title_fontdict':{'fontsize':16, 'fontweight':'bold'}}
rg.plot_maps_corr(var='sst', save=save,
                  kwrgs_plot=kwrgs_plot,
                  min_detect_gc=1.0)

#%%


rg.cluster_list_MI()
rg.quick_view_labels(save=False)
#%%
precur = rg.list_for_MI[0]
new_labels, label_num = find_precursors.split_region_by_lonlat(precur.prec_labels.copy(),
                                                    label=3,
                                                    kwrgs_mask_latlon={'latmax':10},
                                                    trialplot=False)
new_labels, label_num = find_precursors.split_region_by_lonlat(new_labels,
                                                    label=2,
                                                    plot_l=0,
                                                    kwrgs_mask_latlon={'lonmin':150},
                                                    trialplot=False)
precur.prec_labels = new_labels
rg.quick_view_labels(save=True)
# rg.get_ts_prec()
#%%
from sklearn import metrics
import pandas as pd
import func_models as fc_utils
import functions_pp; import df_ana

list_of_name_path = [(cluster_label, TVpath)]
# list_import_ts = [('RW-sst 60d', '/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/paper2/output/east_forecast/z5000..0..z500_sp_140-300-20-73_3jun-2aug_lag0-60_0..0..z500_sp_random10s1/2020-07-27_11hr_27min_df_data_sst_dt1_tf60_140-300-20-73.h5')]

# list_import_ts = [('sstpattern', '/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/paper2/output/east/z5000..0..z500_sp_0ff31_10jun-24aug_lag0-0_0..0..z500_sp_random10s1/2020-07-02_11hr_52min_df_data_NorthPacAtl_dt1_0ff31.h5'),
                   # ('sstregions', '/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/paper2/output/east/z5000..0..z500_sp_0ff31_10jun-24aug_lag0-0_0..0..z500_sp_random10s1/2020-07-02_12hr_10min_df_data_NorthPacAtl_dt1_0ff31.h5')]

list_import_ts = [('mx2t-sst 15d', '/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/paper2/output/east_forecast/2ts_0ff31_10jun-24aug_sst_ts_random10s1/2020-09-21_13hr_16min_df_data_sst_dt1_tf15_0ff31.h5')]
# list_import_ts = [('mx2t-sst 15d', '/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/paper2/output/east_forecast/2ts_0ff31_10jun-24aug_sst_ts_random10s1/2020-09-21_15hr_34min_df_data_sst_dt1_tf15_0ff31_a1E-3.h5')]
list_import_ts = [('mx2t-sst 15d', '/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/paper2/output/east_forecast/2ts_0ff31_10jun-24aug_sst_ts_random10s1/2020-09-21_17hr_48min_df_data_sst_dt1_tf15_0ff31_parcorr.h5')]
# list_import_ts = [('mx2t-sst 60d', '/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/paper2/output/east_forecast/2ts_0ff31_3jun-2aug_sst_ts_random10s1/2020-09-22_09hr_58min_df_data_sst_dt1_tf60_0ff31_parcorrtime.h5')]
list_import_ts = [('RW-sst 60d', '/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/paper2/output/east_forecast/z5000..0..z500_sp_140-300-20-73_3jun-2aug_60tf_sst_parcorrtime_random10s1/2020-09-28_14hr_13min_df_data_sst_dt1_tf60_140-300-20-73.h5')]



months = {'May'         : ('05-01', '05-30'),
          'June'        : ('06-01', '06-30'),
          'July'        : ('07-01', '07-30'),
          'August'      : ('08-01', '08-30'),
          'september'   : ('09-01', '09-30')}

months = {'May-June'    : ('05-01', '06-30'),
          'June-July'   : ('06-01', '07-30'),
           'July-Aug'    : ('07-01', '08-31'),
           'Aug-Sept'    : ('08-01', '09-30'),
           'Sept-Okt'    : ('09-01', '10-31')}

monthkeys= list(months.keys()) ; oneyrsize = 0

n_boot = 1000
blocksize=1
lag = 1
precur_aggr = 60
list_test = []
list_test_b = []
# keys_ext = [f'{lag}..1..sst', f'{lag}..2..sst']
keys_ext = [f'0..3..sst', f'0..7..sst']
# keys_ext = [f'0..3..sst', f'0..8..sst']

rg = RGCPD(list_of_name_path=list_of_name_path,
            list_import_ts=list_import_ts,
            start_end_TVdate=('06-01', '08-31'),
            start_end_date=start_end_date,
            start_end_year=None,
            tfreq=tfreq,
            path_outmain=path_out_main,
            append_pathsub='_' + name_ds)

for month, start_end_TVdate in months.items():

    rg.start_end_TVdate = start_end_TVdate
    rg.pp_TV(name_ds=name_ds, detrend=False)
    rg.traintest(method='random10')
    rg.get_ts_prec(precur_aggr=precur_aggr, keys_ext=keys_ext)

    if monthkeys.index(month) >= 1:
        nextyr = functions_pp.get_oneyr(rg.df_data['RV_mask'].loc[0][rg.df_data['RV_mask'].loc[0]])
        if nextyr.size != oneyrsize:
            raise ValueError

    oneyr = functions_pp.get_oneyr(rg.df_data['RV_mask'].loc[0][rg.df_data['RV_mask'].loc[0]])
    oneyrsize = oneyr.size


    kwrgs_model = {'scoring':'neg_mean_squared_error',
                   'alphas':np.logspace(.1, 2, num=25),
                   'normalize':False}

    keys = keys_ext
    target_ts = rg.df_data.iloc[:,[0]].loc[0][rg.df_data.iloc[:,-1].loc[0]]
    target_ts = (target_ts - target_ts.mean()) / target_ts.std()

    out = rg.fit_df_data_ridge(target=target_ts,
                               keys=keys,
                               tau_min=lag, tau_max=lag,
                               kwrgs_model=kwrgs_model,
                               transformer=fc_utils.standardize_on_train)

    predict, weights, models_lags = out
    prediction = predict.rename({predict.columns[0]:'temp',lag:'Prediction'},axis=1)
    score_func_list = [metrics.mean_squared_error, fc_utils.corrcoef]
    df_train_m, df_test_s_m, df_test_m, df_boot = fc_utils.get_scores(prediction,
                                                             rg.df_data.iloc[:,-2:],
                                                             score_func_list,
                                                             n_boot = n_boot,
                                                             blocksize=blocksize,
                                                             rng_seed=2)
    print(df_test_m)
    df_boot['mean_squared_error'] = 1-df_boot['mean_squared_error']
    list_test_b.append(df_boot)
    list_test.append(df_test_m)
    m = models_lags[f'lag_{lag}']['split_0']
    print(m.alpha_)
    idx_alpha = np.argwhere(kwrgs_model['alphas']==m.alpha_)[0][0]
    if idx_alpha in [0,24]:
        print(f'\nadapt alphas, idx is {idx_alpha}\n')


    # df_ana.loop_df(df=rg.df_data[keys], colwrap=1, sharex=False,
    #                      function=df_ana.plot_timeseries,
    #                      kwrgs={'timesteps':rg.dates_TV.size,
    #                                  'nth_xyear':5})

df_test = functions_pp.get_df_test(prediction.merge(rg.df_data.iloc[:,-2:],
                                                        left_index=True,
                                                        right_index=True)).iloc[:,:2]

corrvals = [test.values[0,1] for test in list_test]
MSE_SS_vals = [1-test.values[0,0] for test in list_test]

df_scores = pd.DataFrame({'RMSE-SS':MSE_SS_vals, 'Corr. Coef.':corrvals},
                         index=monthkeys)
df_test_b = pd.concat(list_test_b, keys = monthkeys,axis=1)

yerr = [] ; alpha = .05
for i in range(df_test_b.columns.size):
    Eh = 1 - alpha/2 ; El = alpha/2
    tup = [df_test_b.iloc[:,i].quantile(El), df_test_b.iloc[:,i].quantile(Eh)]
    mean = df_scores.values.flatten()[i]
    tup = abs(mean-tup)
    yerr.append(tup)
ax = df_scores.plot.bar(rot=0, yerr=np.array(yerr).reshape(2,2,5),
                        capsize=8, error_kw=dict(capthick=1))
ax.set_ylabel('Skill Score', fontsize=16)
ax.set_xlabel('Months to predict temperature', fontsize=16)
ax.tick_params(axis='both', labelsize=14)
ax.legend(fontsize=16, frameon=True, facecolor='grey',
              framealpha=.5)
ax.set_ylim(-0.5, 1)
plt.savefig(os.path.join(rg.path_outsub1,
             f'skill_score_vs_months_{precur_aggr}tf_lag{lag}_nb{n_boot}.pdf'))

#%%

df = df_test_b.stack().reset_index(level=1)
dfx = df.groupby(['level_1'])
axes = dfx.boxplot()
axes[0].set_ylim(-0.5, 1)
#%%
import seaborn as sns
df_ap = pd.concat(list_test_b, axis=0, ignore_index=True)
df_ap['months'] = np.repeat(monthkeys, list_test_b[0].index.size)
# df_ap.boxplot(by='months')
ax = sns.boxplot(x=df_ap['months'], y=df_ap['mean_squared_error'])
ax.set_ylim(-0.5, 1)
plt.figure()
ax = sns.boxplot(x=df_ap['months'], y=df_ap['corrcoef'])
ax.set_ylim(-0.5, 1)

#%%
columns_my_order = monthkeys
fig, ax = plt.subplots()
for position, column in enumerate(columns_my_order):
    ax.boxplot(df_test_b.loc[column], positions=[position,position+.25])

ax.set_xticks(range(position+1))
ax.set_xticklabels(columns_my_order)
ax.set_xlim(xmin=-0.5)
plt.show()

#%% Store data
# rg.cluster_list_MI()
# rg.quick_view_labels()
# rg.get_ts_prec(precur_aggr=1)
# rg.store_df()


#%% Ridge part 1 - load data
precur_aggr = None
lags = rg.list_for_MI[0].lags
rg.cluster_list_MI()
rg.list_for_MI[0].calc_ts = 'pattern cov'
rg.quick_view_labels()
rg.get_ts_prec(precur_aggr=precur_aggr)
rename = {'0..0..sst_sp': 'mx2t-SST lag 0',
          f'{lags[1]}..0..sst_sp': 'mx2t-SST lag 1',
          f'{lags[1]}..1..sst': 'mx2t-SST r1 lag 1',
          f'{lags[1]}..2..sst': 'mx2t-SST r2 lag 1',
          '0..1..sst': 'mx2t-SST r1 lag 0',
          '0..2..sst': 'mx2t-SST r2 lag 0'}
rg.df_data = rg.df_data.rename(rename, axis=1)

#%% Ridge part 2 - fit

import df_ana; import sklearn, functions_pp, func_models
keys = ['mx2t-SST lag 1']
# keys = ['mx2t-SST r1 lag 1', 'mx2t-SST r2 lag 1']
# keys = ['SST lag 1']
kwrgs_model = {'scoring':'neg_mean_squared_error',
               'alphas':np.logspace(-3, 1, num=10)}
s = 0
rename_target = {'2ts':'E-U.S. mx2t'}
name_target = list(rename_target.values())[0]
rg.df_data = rg.df_data.rename(rename_target, axis=1)
target_ts = rg.df_data.loc[s].iloc[:,[0]][rg.df_data.loc[s]['RV_mask']].copy()
target_mean = target_ts.mean().squeeze()
target_std = target_ts.std().squeeze()
# standardize :
target_ts = (target_ts - target_mean) / target_std
predict, coef, model = rg.fit_df_data_ridge(keys=keys, target=target_ts, tau_min =1,
                                            tau_max=1,
                                            transformer=func_models.standardize_on_train,
                                            kwrgs_model=kwrgs_model)
prediction = predict.rename({1:'Prediction'}, axis=1)

# AR1
AR1, c, m = rg.fit_df_data_ridge(keys=[name_target], target=target_ts,
                                 tau_min =1,
                                 tau_max=1,
                                 transformer=func_models.standardize_on_train,
                                 kwrgs_model=kwrgs_model)
AR1 = AR1.rename({1:'AR1 fit'}, axis=1)

#%%
df_splits = rg.df_data[['TrainIsTrue', 'RV_mask']]
df_AR1test = functions_pp.get_df_test(AR1.merge(df_splits,
                                                    left_index=True,
                                                    right_index=True)).iloc[:,:2]
df_test = functions_pp.get_df_test(prediction.merge(df_splits,
                                                    left_index=True,
                                                    right_index=True)).iloc[:,:2]


# df_AR1test = (df_AR1test - target_mean) / target_std

fig, ax = df_ana.plot_df(df = df_test[['Prediction']], function=df_ana.plot_timeseries,
               figsize=(15,5), kwrgs={'selyears':range(1979,2019)})
ax.lines[0].set_label('SST pattern lag 1')
ax.scatter(df_test[['Prediction']].index,
           df_test[['Prediction']].values, label=None,
           color=ax.lines[0].get_color(),
           s=15)
# ax.plot(df_test[['Rossby wave (z500)']], label='Truth', color='black',
#         linewidth=1)
ax.plot(df_test[[name_target]], label='Truth', color='black',
        linewidth=1)

ax.plot(df_AR1test[['AR1 fit']], label='AR1', color='grey',
        linewidth=1, linestyle='--')
ax.set_title(f'Out of sample (10-fold) {rg.precur_aggr}-day aggr. RW prediction from lagged SST pattern',
             fontsize=16)
ax.hlines(y=0,xmin=pd.to_datetime('1979-01-01'),
          xmax=pd.to_datetime('2018-12-31'), color='black')

ax.set_xlim(pd.to_datetime('1979-01-01'), xmax=pd.to_datetime('2020-12-31'))
ax.legend()
ax.tick_params(axis='both', labelsize=14)
ax.set_ylabel('Standardized E-U.S. RW timeseries', fontsize=16)
ax.set_ylim(-3,3)
MSE_func = sklearn.metrics.mean_squared_error
fullts = rg.df_data.loc[0].iloc[:,0]
Persistence = fullts.shift(1)[rg.df_data.loc[0]['RV_mask']]
Persistence = (Persistence - target_mean) / target_std
MSE_model = MSE_func(df_test.iloc[:,0],df_test.iloc[:,1], squared=False)
MSE_pers  = MSE_func(df_test.iloc[:,0],Persistence, squared=False)
MSE_AR1  = MSE_func(df_test.iloc[:,0],df_AR1test.iloc[:,1], squared=False)
Corr_pers = np.corrcoef(Persistence.values.squeeze(), df_test.iloc[:,0].values)
Corr_AR1 = np.corrcoef(df_test.iloc[:,0].values, df_AR1test.iloc[:,1].values)

text1 =  'Corr. coeff. model          : {:.2f}\n'.format(df_test.corr().iloc[0,1])
text1 +=  'Corr. coeff. persistence : {:.2f}\n'.format(Corr_pers[0][1])
text1 +=  'Corr. coeff. AR1             : {:.2f}'.format(Corr_AR1[0][1])
text2 = r'RMSE model          : {:.2f} $\sigma$'.format(MSE_model) + '\n'
text2 += r'RMSE persistence : {:.2f} $\sigma$'.format(MSE_pers) + '\n'
text2 += r'RMSE AR1             : {:.2f} $\sigma$'.format(MSE_AR1)

ax.text(.029, .035, text1,
        transform=ax.transAxes, horizontalalignment='left',
        fontdict={'fontsize':14},
        bbox = dict(boxstyle='round', facecolor='white', alpha=1,
                    edgecolor='black'))
ax.text(.385, .035, text2,
        transform=ax.transAxes, horizontalalignment='left',
        fontdict={'fontsize':14},
        bbox = dict(boxstyle='round', facecolor='white', alpha=1,
                    edgecolor='black'))

figname = rg.path_outsub1 +f'/forecast_lagged_SST_tf{rg.precur_aggr}.pdf'
plt.savefig(figname, bbox_inches='tight')


#%%
path_data = '/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/paper2/output/east/2ts_0ff31_10jun-24aug_lag0-0_ts_from_imports/2020-07-06_11hr_58min_df_data_sstpattern_sstregionssst_dt1_0ff31.h5'
path_allNH = '/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/paper2/output/east/2ts_0ff31_10jun-24aug_lag0-0_ts_from_imports/2020-07-07_14hr_06min_df_data_sstpattern_sstregionssst_dt1_0ff31.h5'

from class_fc import fcev

start_time = time()

kwrgs_events = {'event_percentile': 50}

kwrgs_events = kwrgs_events
precur_aggr = 60
use_fold = None
n_boot = 0
lags_i = np.array([0,15,30])

list_of_fc = [fcev(path_data=path_allNH, precur_aggr=precur_aggr,
                    use_fold=use_fold, start_end_TVdate=None,
                    stat_model= ('logitCV',
                                {'Cs':10, #np.logspace(-4,1,10)
                                'class_weight':{ 0:1, 1:1},
                                  'scoring':'neg_brier_score',
                                  'penalty':'l2',
                                  'solver':'lbfgs',
                                  'max_iter':100,
                                  'kfold':5,
                                  'seed':1}),
                    kwrgs_pp={'add_autocorr':False, 'normalize':'datesRV'},
                    dataset='',
                    keys_d=('SST pattern', dict(zip(np.arange(10), [['0..1..NorthPacAtl', '0..2..NorthPacAtl']]*10)))),
               fcev(path_data=path_data, precur_aggr=precur_aggr,
                    use_fold=use_fold, start_end_TVdate=None,
                    stat_model= ('logitCV',
                                {'Cs':10, #np.logspace(-4,1,10)
                                'class_weight':{ 0:1, 1:1},
                                  'scoring':'neg_brier_score',
                                  'penalty':'l2',
                                  'solver':'lbfgs',
                                  'max_iter':100,
                                  'kfold':5,
                                  'seed':1}),
                    kwrgs_pp={'add_autocorr':False, 'normalize':'datesRV'},
                    dataset='',
                    keys_d=('SST clusters', dict(zip(np.arange(10), [['0..1..sst', '0..2..sst']]*10))))]



fc = list_of_fc[0]

#%%
times = []
t00 = time()
for fc in list_of_fc:
    t0 = time()
    fc.get_TV(kwrgs_events=kwrgs_events, detrend=False) # detrending already done on gridded data

    fc.fit_models(lead_max=lags_i, verbosity=1)

    # fc.TV.prob_clim = pd.DataFrame(np.repeat(fc.TV.prob_clim.mean(), fc.TV.prob_clim.size).values, index=fc.TV.prob_clim.index)

    fc.perform_validation(n_boot=n_boot, blocksize='auto', alpha=0.05,
                          threshold_pred='upper_clim')

    single_run_time = int(time()-t0)
    times.append(single_run_time)
    total_n_runs = len(list_of_fc)
    ETC = (int(np.mean(times) * total_n_runs))
    print(f'Time elapsed single run in {single_run_time} sec\t'
          f'ETC {int(ETC/60)} min \t Progress {int(100*(time()-t00)/ETC)}% ')


# In[8]:
working_folder, pathexper = list_of_fc[0]._print_sett(list_of_fc=list_of_fc)

store = False
if __name__ == "__main__":
    pathexper = list_of_fc[0].pathexper
    store = True

import valid_plots as dfplots
import functions_pp


dict_all = dfplots.merge_valid_info(list_of_fc, store=store)
if store:
    dict_merge_all = functions_pp.load_hdf5(pathexper+'/data.h5')


lag_rel = 15
kwrgs = {'wspace':0.16, 'hspace':.25, 'col_wrap':2, 'skip_redundant_title':True,
         'lags_relcurve':[lag_rel], 'fontbase':14, 'figaspect':2}
#kwrgs = {'wspace':0.25, 'col_wrap':3, 'threshold_bin':fc.threshold_pred}
met = ['AUC-ROC', 'AUC-PR', 'Precision', 'BSS', 'Accuracy', 'Rel. Curve']
line_dim = 'exper'
group_line_by = None

fig = dfplots.valid_figures(dict_merge_all,
                          line_dim=line_dim,
                          group_line_by=group_line_by,
                          met=met, **kwrgs)


f_format = '.pdf'
pathfig_valid = os.path.join(pathexper,'verification' + f_format)
fig.savefig(pathfig_valid,
            bbox_inches='tight') # dpi auto 600
fc = list_of_fc[0]
df, fig = fc.plot_feature_importances()
path_feat = pathexper + f'/ifc{1}_logitregul' + f_format
fig.savefig(path_feat, bbox_inches='tight')


fc.dict_sum[0].loc['Precision'].loc['Precision']

fc.dict_sum[0].loc['Accuracy'].loc['Accuracy']