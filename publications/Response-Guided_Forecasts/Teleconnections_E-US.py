#!/usr/bin/env python
# coding: utf-8

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
start_end_TVdate = ('07-01', '08-31')
start_end_date = ('1-1', '12-31')

tfreq = 15
# In[5]:


list_of_name_path = [(name_or_cluster_label, TVpath),
                      ('sm2', '/Users/semvijverberg/surfdrive/ERA5/input_raw/sm2_1979-2018_1_12_daily_1.0deg.nc'),
                      ('sm3', '/Users/semvijverberg/surfdrive/ERA5/input_raw/sm3_1979-2018_1_12_daily_1.0deg.nc'),
                      ('sst', '/Users/semvijverberg/surfdrive/ERA5/input_raw/sst_1979-2018_1_12_daily_1.0deg.nc')]



list_for_MI   = [BivariateMI(name='sm2', func=BivariateMI.corr_map,
                             kwrgs_func={'alpha':.05, 'FDR_control':True},
                             distance_eps=800, min_area_in_degrees2=5,
                             use_coef_wghts=True),
                 BivariateMI(name='sm3', func=BivariateMI.corr_map,
                              kwrgs_func={'alpha':.05, 'FDR_control':True},
                              distance_eps=800, min_area_in_degrees2=7,
                              use_coef_wghts=True),
                   BivariateMI(name='sst', func=BivariateMI.corr_map,
                                kwrgs_func={'alpha':1E-3, 'FDR_control':True},
                                distance_eps=800, min_area_in_degrees2=5,
                                use_coef_wghts=True)]



rg = RGCPD(list_of_name_path=list_of_name_path,
               list_for_MI=list_for_MI,
               start_end_TVdate=start_end_TVdate,
               start_end_date=start_end_date,
               tfreq=tfreq, lags_i=np.array([-1]),
               path_outmain=path_out_main)

rg.plot_df_clust()

# ### Post-processing Target Variable
rg.pp_TV(name_ds=name_ds, detrend=False, anomaly=False)


rg.traintest(method='random10')

subtitles = np.array([['Clustered simultaneous high temp. events']])
plot_maps.plot_labels(rg.ds['xrclustered'],
                      zoomregion=(230,300,25,60),
                      kwrgs_plot={'subtitles':subtitles,
                                  'y_ticks':np.array([30, 40,50,60]),
                                  'x_ticks':np.arange(230, 310, 10),
                                  'cbar_vert':-.03,
                                  'add_cfeature':'OCEAN'})

plt.savefig(os.path.join(rg.path_outsub1, 'TV_cluster.pdf'),
            bbox_inches='tight') # dpi auto 600


rg.pp_precursors(anomaly=[True, {'sm2':False, 'sm3':False}])





# In[166]:


rg.calc_corr_maps()


# In[167]:


rg.cluster_list_MI()


# In[168]:


rg.quick_view_labels()

#%% Get PDO & ENSO

# import climate_indices

# df_PDO, PDO_pattern = climate_indices.PDO(rg.list_precur_pp[-1][1], rg.df_splits)
# df_ENSO = climate_indices.ENSO_34(rg.list_precur_pp[-1][1], rg.df_splits)
# df = df_ENSO.merge(df_PDO,
#                     left_index=True,
#                     right_index=True).merge(rg.df_splits,
#                                             left_index=True,
#                                             right_index=True)
# file_path = os.path.join(rg.path_outsub1, 'PDO_ENSO34_ERA5_1979_2018.h5')
# import functions_pp
# functions_pp.store_hdf_df({'df_data':df}, file_path)
#%%
rg.list_import_ts = None
rg.get_ts_prec()

rg.PCMCI_df_data(keys=None,
                 pc_alpha=.1,
                 tau_max=2,
                 max_conds_dim=10,
                 max_combinations=10)
rg.PCMCI_get_links(alpha_level=.05)
#%%

# rg.plot_maps_corr(var=['sm2'], mean=True, save=False, aspect=2, cbar_vert=-.1,
#                   subtitles=np.array([['SM2 Correlated']]))

# rg.plot_maps_corr(var=['sm3'], mean=True, save=False, aspect=2, cbar_vert=-.1,
#                   subtitles=np.array([['SM3 Correlated']]))

rg.plot_maps_sum(cols=['corr'])

rg.plot_maps_sum(cols=['C.D.'])
#%% Add PDO ENSO

rg.list_import_ts = [('PDO_ENSO', '/Users/semvijverberg/surfdrive/output_RGCPD/Response-Guided/1ts_5e87d_1jun-31aug_lag0-0_random10s1/PDO_ENSO34_ERA5_1979_2018.h5')]
rg.get_ts_prec()

#%% Ridge regression
import df_ana; import sklearn, functions_pp, func_models, stat_models

# experiment = 'only C.D.'
experiment = 'all correlated'
# experiment = 'expert knowledge'
# experiment = 'climate indices + local sm'

lag = 3
transformer=func_models.standardize_on_train
# transformer=None

if experiment == 'all correlated':
    keys = np.array(rg.df_MCIa.loc[0].index)
    experiment += f' ({len(keys)})'
elif experiment == 'only C.D.':
    keys = rg.df_links.mean(0,level=1).index[(rg.df_links.mean(0,level=1) > .5).sum(axis=1) == 1]
    experiment += f' ({len(keys)})'
elif experiment == 'expert knowledge':
    keys = ['1ts', '-15..3..sm2', '-15..2..sm3', '-15..1..sst', '-15..2..sst',
            '-15..6..sst', '-15..8..sst']
    experiment += f' ({len(keys)})'
elif experiment == 'climate indices + local sm':
    keys = ['1ts', 'ENSO34', 'PDO', '-15..3..sm2', '-15..2..sm3']


kwrgs_model = {'scoring':'neg_mean_squared_error',
               'alphas':np.logspace(0.3, 1.8, num=20)}
s = 1
target_ts = rg.df_data.loc[s].iloc[:,[0]][rg.df_data.loc[s]['RV_mask']].copy()
target_mean = target_ts.mean().squeeze()
target_std = target_ts.std().squeeze()
# standardize :
target_ts = (target_ts - target_mean) / target_std
predict, coef, models = rg.fit_df_data_ridge(keys=keys, target=target_ts, tau_min=lag,
                                            tau_max=lag,
                                            transformer=transformer,
                                            kwrgs_model=kwrgs_model)
models['lag_30'] = models.pop('lag_3')


prediction = predict.rename({lag:'Prediction', 'RVz5000..0..z500_sp':'Rossby wave (z500)'}, axis=1)
stat_models.plot_importances(models, lag=30)
# AR1
AR1, c, m = rg.fit_df_data_ridge(keys=[rg.TV.name], target=target_ts,
                                 tau_min=lag,
                                 tau_max=lag,
                                 transformer=transformer,
                                 kwrgs_model=kwrgs_model)
AR1 = AR1.rename({lag:'AR1 fit'}, axis=1)
#%%
import matplotlib.dates as mdates
import stat_models_cont as sm
import pandas as pd

df_splits = rg.df_data[['TrainIsTrue', 'RV_mask']]
df_AR1test = functions_pp.get_df_test(AR1.merge(df_splits,
                                                    left_index=True,
                                                    right_index=True)).iloc[:,:2]
df_test = functions_pp.get_df_test(prediction.merge(df_splits,
                                                    left_index=True,
                                                    right_index=True)).iloc[:,:2]

# Plot
fig, ax = plt.subplots(1, 1, figsize = (15,5), facecolor='lightgrey')
ax.set_facecolor('white')
ax.grid(which='major', color='grey', alpha=.2)
ax.plot(df_test[['Prediction']], label=f'{experiment} regions',
        color='red',#ax.lines[0].get_color(),
        linewidth=1)

y = prediction['Prediction']
for fold in y.index.levels[0]:
    label = None ; color = 'red' ;
    ax.plot(y.loc[fold].index, y.loc[fold], alpha=.1,
            label=label, color=color)


ax.xaxis.set_major_locator(mdates.YearLocator(5, month=6, day=3))   # every year)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_xlim(pd.to_datetime('1979-01-01'), xmax=pd.to_datetime('2020-12-31'))
ax.tick_params(axis='both', labelsize=14)
fig.autofmt_xdate()

ax.scatter(df_test[['Prediction']].index,
           df_test[['Prediction']].values, label=None,
           color=ax.lines[0].get_color(),
           s=15)
ax.plot(df_test[[rg.TV.name]], label='truth', color='black',
        linewidth=1)
ax.scatter(df_test[[rg.TV.name]].index,
           df_test[[rg.TV.name]].values, label=None,
           color='black',
           s=10)

# ax.plot(df_AR1test[['AR1 fit']], label='AR1', color='grey',
#         linewidth=1, linestyle='--')
ax.set_title(f'30-day ahead forecast of onset 15-day period (using 10-fold cross-validation)',
             fontsize=16)
ax.hlines(y=0,xmin=pd.to_datetime('1979-06-03'),
          xmax=pd.to_datetime('2018-08-02'), color='grey')


ax.legend(loc='upper right', fontsize=16, frameon=True)
ax.set_ylabel('Standardized temperature timeseries', fontsize=16)
ax.set_ylim(-3,3)



score_func_list = [sklearn.metrics.mean_squared_error, np.corrcoef]
df_train_m, df_test_s_m, df_test_m = sm.get_scores(prediction, df_splits,
                                            score_func_list)
df_train_AR, df_test_s_AR, df_test_AR = sm.get_scores(AR1, df_splits,
                                            score_func_list)


# fullts = rg.df_data.loc[0].iloc[:,0]
# Persistence = fullts.shift(1)[rg.df_data.loc[0]['RV_mask']]
# Persistence = (Persistence - target_mean) / target_std


text1 =  'Corr. coeff. model  : {:.2f} ({:.2f})\n'.format(float(df_test_m['corrcoef']),
                                                          df_train_m['corrcoef'].mean())
# text1 +=  'Corr. coeff. persistence : {:.2f}\n'.format(Corr_pers[0][1])
text1 +=  'Corr. coeff. AR       : {:.2f} ({:.2f})'.format(float(df_test_AR['corrcoef']),
                                                  df_train_AR['corrcoef'].mean())
text2 = r'RMSE model   : {:.2f} ({:.2f}) $\sigma$'.format(float(df_test_m['mean_squared_error']**.5),
                                          (df_train_m['mean_squared_error']**.5).mean()) + '\n'
# text2 += r'RMSE persistence : {:.2f} $\sigma$'.format(MSE_pers) + '\n'
text2 += r'RMSE AR        : {:.2f} ({:.2f}) $\sigma$'.format(float(df_test_AR['mean_squared_error']**.5),
                                          (df_train_AR['mean_squared_error']**.5).mean())

ax.text(.038, .019, text1,
        transform=ax.transAxes, horizontalalignment='left',
        verticalalignment='bottom',
        fontdict={'fontsize':14},
        bbox = dict(boxstyle='round', facecolor='red', alpha=.5,
                    edgecolor='black'))
ax.text(.35, .019, text2,
        transform=ax.transAxes, horizontalalignment='left',
        verticalalignment='bottom',
        fontdict={'fontsize':14},
        bbox = dict(boxstyle='round', facecolor='red', alpha=.5,
                    edgecolor='black'))

exp_str = experiment.split(' ')[1]
figname = rg.path_outsub1 +f'/forecast_{exp_str}_tf{rg.precur_aggr}.pdf'
plt.savefig(figname, bbox_inches='tight')

# In[ ]:

rg.get_ts_prec(precur_aggr=1)
rg.store_df()

from class_fc import fcev


#%%
from time import time
start_time = time()

ERA_data = rg.path_df_data

kwrgs_events = {'event_percentile': 66, 'window':'mean',
                'min_dur':1, 'max_break': 1}

# kwrgs_events = {'event_percentile': 66}

kwrgs_events = kwrgs_events
precur_aggr = 15
use_fold = None
n_boot = 10
lags_i = np.array([0, 15, 35, 45, 60])
start_end_TVdate = None # ('7-04', '8-22')


fc = fcev(path_data=ERA_data, precur_aggr=precur_aggr,
                    use_fold=use_fold, start_end_TVdate=None,
                    stat_model= ('logitCV',
                                {'Cs':10, #np.logspace(-4,1,10)
                                'class_weight':{ 0:1, 1:1},
                                  'scoring':'neg_brier_score',
                                  'penalty':'l2',
                                  'solver':'lbfgs',
                                  'max_iter':100,
                                  'kfold':5,
                                  'seed':2}),
                    kwrgs_pp={'add_autocorr':True, 'normalize':'datesRV'},
                    dataset='',
                    keys_d='CPPA+PEP+sm')

fc.get_TV(kwrgs_events=kwrgs_events, detrend=False)

fc.fit_models(lead_max=lags_i, verbosity=1)

fc.perform_validation(n_boot=n_boot, blocksize='auto', alpha=0.05,
                          threshold_pred=50)

working_folder, filename = fc._print_sett(list_of_fc=[fc])

store = False
if __name__ == "__main__":
    filename = fc.filename
    store = True

import valid_plots as dfplots
import functions_pp


dict_all = dfplots.merge_valid_info([fc], store=store)
if store:
    dict_merge_all = functions_pp.load_hdf5(filename+'.h5')


lag_rel = 35
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
pathfig_valid = os.path.join(filename + f_format)
fig.savefig(pathfig_valid,
            bbox_inches='tight') # dpi auto 600

# In[ ]:


# RV = rg.fullts.drop('n_clusters').drop('q').drop('cluster').to_dataframe()
# import df_ana

# df_g = df_ana.load_hdf5('/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/Sub-seasonal_statistical_forecasts_of_eastern_United_States_extreme_temperature_events/data/df_data_sst_CPPAs30_sm2_sm3_dt1_c378f_good.h5')['df_data']
# RV['1_good'] = df_g.loc[0][['1']]
# RV.corr()




# In[166]:


# import pandas as pd; import df_ana
# name_ds_list = ['q95', 'q90tail', 'q75tail', 'q65tail', 'q50tail', 'ts']
# list_rg = []
# for name_ds in name_ds_list:

#     for detr in [True, False]:
#         list_rg.append(RGCPD(list_of_name_path=list_of_name_path,
#                    list_for_MI=list_for_MI,
#                    list_import_ts=list_import_ts,
#                    start_end_TVdate=start_end_TVdate,
#                    start_end_date=start_end_date,
#                    tfreq=1, lags_i=np.array([0]),
#                    path_outmain=user_dir+'/surfdrive/output_RGCPD'))


#         list_rg[-1].pp_TV(name_ds=name_ds, detrend=detr, anomaly=False)
#         kwrgs_events=None
#         list_rg[-1].traintest(method='random10', kwrgs_events=kwrgs_events)

# dfs = pd.concat([i.TV.RV_ts for i in list_rg],axis=1)
# dfs.corr()
# dfs.mean()
# list_rg[0].apply_df_ana_plot(df=dfs,func=df_ana.plot_timeseries, kwrgs_func={'selyears':[2012]})