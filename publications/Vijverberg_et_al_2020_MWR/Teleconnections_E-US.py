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



import plot_maps
import cartopy.crs as ccrs
from RGCPD import RGCPD
from RGCPD import BivariateMI
# In[5]:

# CPPA_s30_21march= [('sst_CPPAs30', user_dir + '/surfdrive/output_RGCPD/easternUS/ERA5_mx2t_sst_Northern/ff393_ran_strat10_s30/data/ERA5_21-03-20_12hr_lag_0_ff393.h5')]
# RV = user_dir + '/surfdrive/output_RGCPD/easternUS/tf1_n_clusters4_q90_dendo_ff393.nc'

CPPA_s30_14may  = [('sst_CPPAs30', user_dir + '/surfdrive/output_RGCPD/easternUS/ERA5_CPPA/ERA5_14-05-20_08hr_lag_0_c378f.h5')]
RV = user_dir + '/surfdrive/output_RGCPD/easternUS/tf1_n_clusters5_q95_dendo_c378f.nc'
RV_EC = user_dir + '/surfdrive/output_RGCPD/easternUS_EC/EC_tas_tos_Northern/tf1_n_clusters5_q95_dendo_958dd.nc'

list_of_name_path = [(1 , RV)]
                     # ('sm2', '/Users/semvijverberg/surfdrive/ERA5/input_raw/sm2_1979-2018_1_12_daily_1.0deg.nc'),
                     # ('sm3', '/Users/semvijverberg/surfdrive/ERA5/input_raw/sm3_1979-2018_1_12_daily_1.0deg.nc')]
                      # ('sst', '/Users/semvijverberg/surfdrive/ERA5/input_raw/sst_1979-2018_1_12_daily_1.0deg.nc')]

list_import_ts = CPPA_s30_14may


list_for_MI   = [BivariateMI(name='sm2', func=BivariateMI.corr_map,
                             kwrgs_func={'alpha':.05, 'FDR_control':True},
                             distance_eps=600, min_area_in_degrees2=5),
                 BivariateMI(name='sm3', func=BivariateMI.corr_map,
                              kwrgs_func={'alpha':.05, 'FDR_control':True},
                              distance_eps=600, min_area_in_degrees2=7)]
                  # BivariateMI(name='sst', func=BivariateMI.corr_map,
                  #              kwrgs_func={'alpha':.001, 'FDR_control':True},
                  #              distance_eps=800, min_area_in_degrees2=5)]



start_end_TVdate = ('06-24', '08-22')
start_end_date = ('1-1', '12-31')

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
# In[6]:
list_of_name_path = [(1 , RV),
                      ('sm2', '/Users/semvijverberg/surfdrive/ERA5/input_raw/sm2_1979-2018_1_12_daily_1.0deg.nc'),
                      ('sm3', '/Users/semvijverberg/surfdrive/ERA5/input_raw/sm3_1979-2018_1_12_daily_1.0deg.nc')]

name_ds = 'q65tail'

rg = RGCPD(list_of_name_path=list_of_name_path,
               list_for_MI=list_for_MI,
               list_import_ts=list_import_ts,
               start_end_TVdate=start_end_TVdate,
               start_end_date=start_end_date,
               tfreq=15, lags_i=np.array([0]),
               path_outmain=user_dir+'/surfdrive/output_RGCPD')

rg.plot_df_clust()

# plot_maps.plot_labels(rg.ds['xrclustered'], zoomregion=(235, 295, 25, 50),
#                       kwrgs_plot={'aspect':2, 'map_proj':ccrs.PlateCarree(central_longitude=240)})
# plt.savefig(os.path.join(rg.path_outsub1, 'TV_cluster.pdf'),
#             bbox_inches='tight') # dpi auto 600

#selbox = [None, {'sst':[-180,360,-10,90]}]
selbox = None
#anomaly = [True, {'sm1':False, 'sm2':False, 'sm3':False}]
anomaly = [True, {'sm1':False, 'sm2':False, 'sm3':False, 'st2':False}]

rg.pp_precursors(selbox=selbox, anomaly=anomaly)


# ### Post-processing Target Variable

# In[7]:


rg.pp_TV(name_ds=name_ds, detrend=False, anomaly=False)



# In[165]:

#kwrgs_events={'event_percentile':66}
kwrgs_events=None
rg.traintest(method='random10', kwrgs_events=kwrgs_events)



# In[166]:


rg.calc_corr_maps()


# In[167]:


rg.cluster_list_MI()


# In[168]:


rg.quick_view_labels()

#%%

rg.get_ts_prec()

rg.PCMCI_df_data(keys=['1q65tail', '0..2..sm2', 'RV_mask', 'TrainIsTrue'],
                 pc_alpha=None,
                 tau_max=1,
                 max_conds_dim=1,
                 max_combinations=1)
rg.PCMCI_get_links(alpha_level=.05)
#%%

rg.plot_maps_corr(var=['sm2'], mean=True, save=False, aspect=2, cbar_vert=-.1,
                  subtitles=np.array([['SM2 Correlated']]))

rg.plot_maps_corr(var=['sm3'], mean=True, save=False, aspect=2, cbar_vert=-.1,
                  subtitles=np.array([['SM3 Correlated']]))

rg.plot_maps_sum(cols=['corr'])
# In[ ]:
keys = ['ENSO34', 'PDO', '0..CPPAsv', '0..PEPsv', '0..1..sst', '0..2..sst',
        '0..3..sst', '0..4..sst', '0..5..sst', '0..6..sst', '0..7..sst', '0..8..sst']
rg.get_ts_prec(precur_aggr=1, keys_ext=keys)
rg.store_df()

from class_fc import fcev


#%%
from time import time
start_time = time()

ERA_data = rg.path_df_data

kwrgs_events = {'event_percentile': 'std', 'window':'single_event',
                'min_dur':3, 'max_break': 1}

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
#%% Correlation maps SST

list_of_name_path = [(1 , RV),
                      ('sst', '/Users/semvijverberg/surfdrive/ERA5/input_raw/sst_1979-2018_1_12_daily_1.0deg.nc')]

list_for_MI   = [BivariateMI(name='sst', func=BivariateMI.corr_map,
                             kwrgs_func={'alpha':.0001, 'FDR_control':True},
                             distance_eps=700, min_area_in_degrees2=5)]

name_ds = 'q90tail'

rg = RGCPD(list_of_name_path=list_of_name_path,
               list_for_MI=list_for_MI,
               list_import_ts=None,
               start_end_TVdate=start_end_TVdate,
               start_end_date=start_end_date,
               tfreq=15, lags_i=np.array([0]),
               path_outmain=user_dir+'/surfdrive/output_RGCPD')
selbox = [None, {'sst':[-180,360,-10,90]}]
anomaly = True

rg.pp_precursors(selbox=selbox, anomaly=anomaly)


# ### Post-processing Target Variable

# In[7]:


rg.pp_TV(name_ds=name_ds, detrend=False, anomaly=False)



# In[165]:

#kwrgs_events={'event_percentile':66}
kwrgs_events=None
rg.traintest(method='random10', kwrgs_events=kwrgs_events)



# In[166]:


rg.calc_corr_maps()
rg.cluster_list_MI()
rg.get_ts_prec()

rg.PCMCI_df_data(keys=['1q90tail', '0..1..sst', 'RV_mask', 'TrainIsTrue'],
                 pc_alpha=None,
                 tau_max=1,
                 max_conds_dim=1,
                 max_combinations=1)
rg.PCMCI_get_links(alpha_level=.05)



rg.plot_maps_sum(cols=['corr'], kwrgs_plot={'aspect':2.5, 'cbar_vert':.13})
