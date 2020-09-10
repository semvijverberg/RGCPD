#!/usr/bin/env python
# coding: utf-8

import os, inspect, sys
import numpy as np

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
# from RGCPD import EOF


TVpath = os.path.join(main_dir, 'publications/paper_Raed/clustering/q50_nc4_dendo_707fb.nc')
cluster_label = 3
name_ds='ts'

start_end_date = ('01-01','12-31') # not working
start_end_TVdate = start_end_date
start_end_year = (1980, 2015)
tfreq = 'annual'
lags=np.array(['4', '5', '6', '7', '8'])
lags=np.array(['45678'])
#%%
list_of_name_path = [(cluster_label, TVpath),
                      ('sst', os.path.join(path_raw, 'sst_1980-2015_1_12_daily_1.0deg.nc'))]
                      # ('sm', os.path.join(path_raw, 'sm23add_1979-2018_1_12_daily_1.0deg.nc'))]

list_for_MI   = [BivariateMI(name='sst', func=BivariateMI.corr_map,
                             kwrgs_func={'alpha':.05, 'FDR_control':True},
                             distance_eps=1000, min_area_in_degrees2=5,
                             selbox=(-180,360,-10,90),
                             lags=lags)]
                 # BivariateMI(name='sm', func=BivariateMI.corr_map,
                 #             kwrgs_func={'alpha':.05, 'FDR_control':True},
                 #             distance_eps=1000, min_area_in_degrees2=5)]

rg = RGCPD(list_of_name_path=list_of_name_path,
               list_for_MI=list_for_MI,
               start_end_date=start_end_date,
               start_end_year=start_end_year,
               tfreq=tfreq,
               path_outmain=curr_dir+'/output/')

rg.plot_df_clust()


#%%
# prec_dates_dict = {'MJJA':('05-01', '08-31'),
#                    'JJAS':('06-01', '09-30'),
#                    'JASO':('06-01', '10-31')}

# for name_dates, prec_dates in prec_dates_dict.items():
#     print(name_dates)
#     rg.start_end_TVdate = prec_dates

#     rg.pp_precursors(selbox=[None,{'sst':(-180,360,-10,90)}], anomaly=[True, {'sm':False}])

# name_dates = 'JJAS'
# rg.start_end_TVdate = prec_dates_dict[name_dates]
rg.pp_precursors(selbox=[None,{'sst':(-180,360,-10,90)}], anomaly=[True, {'sm':False}])
# rg.path_outmain = curr_dir+f'/output/{name_dates}'
# if os.path.isdir(rg.path_outmain) != True : os.makedirs(rg.path_outmain)
# Post-processing Target Variable
rg.pp_TV(name_ds=name_ds, detrend=False, anomaly=False)
#%% Pre-processing precursors

# plot_maps.plot_labels(rg.ds['xrclustered'], zoomregion=(235, 295, 25, 50),
#                       kwrgs_plot={'aspect':2, 'map_proj':ccrs.PlateCarree(central_longitude=240)})
# plt.savefig(os.path.join(rg.path_outsub1, 'TV_cluster.pdf'),
#             bbox_inches='tight') # dpi auto 600








# In[165]:

rg.traintest(method='leave_2', kwrgs_events=None)
precur = rg.list_for_MI[0]
kwrgs_load = rg.kwrgs_load
TV = rg.TV ; self = rg
df_splits = rg.df_splits
rg.calc_corr_maps('sst')
rg.plot_maps_corr('sst', save=True)

# In[167]:


rg.cluster_list_MI()


# In[168]:


rg.quick_view_labels(save=True)

#%%

rg.get_ts_prec()

#%%
from sklearn import metrics
import stat_models_cont as sm
import functions_pp
import matplotlib.pyplot as plt
kwrgs_model = {'scoring':'neg_mean_squared_error',
               'alphas':np.logspace(.4, 2.5, num=25)}
lag = 1
for monthlag in precur.lags:
    keys = rg.df_data.columns[1:-2]
    if type(monthlag) == np.str_:
        i_to_m = {'1':'J','2':'F','3':'M','4':'A','5':'M', '6':'J','7':'J','8':'A'}
        exper = ''.join([kv[1] for kv in i_to_m.items() if kv[0] in monthlag])
        keys = [k for k in keys if k.split('..')[0]==monthlag]
    else:
        keys = [k for k in keys if int(k.split('..')[0])==monthlag]
        exper = 'lag'+str(lag)
    target_ts = rg.TV.RV_ts
    target_ts = (target_ts - target_ts.mean()) / target_ts.std()
    out = rg.fit_df_data_ridge(target=target_ts,
                               keys=keys,
                               tau_min=lag, tau_max=lag,
                               kwrgs_model=kwrgs_model)
    predict, weights, models_lags = out
    prediction = predict.rename({'RV3ts':'Crop Yield [1/y]',lag:'Prediction'},axis=1)
    score_func_list = [metrics.mean_squared_error, np.corrcoef]
    df_train_m, df_test_s_m, df_test_m = sm.get_scores(prediction, rg.df_splits,
                                                score_func_list)

    print(df_test_s_m.mean())
    df_test = functions_pp.get_df_test(prediction.merge(rg.df_splits,
                                                        left_index=True,
                                                        right_index=True)).iloc[:,:2]


    fig, ax = plt.subplots(1, 1, figsize = (10,3), facecolor='lightgrey')
    ax.set_facecolor('white')
    y = prediction['Prediction']
    for fold in y.index.levels[0]:
        label = None ; color = 'red' ;
        ax.plot(y.loc[fold].index, y.loc[fold], alpha=.1,
                label=label, color=color)
    ax.plot(df_test[['Prediction']], label=f'{exper} precursors',
            color='red',#ax.lines[0].get_color(),
            linewidth=1)

    ax.scatter(df_test[['Prediction']].index,
               df_test[['Prediction']].values, label=None,
               color=ax.lines[0].get_color(),
               s=15)
    ax.plot(df_test.iloc[:,0], label='truth', color='black',
            linewidth=1)
    ax.scatter(df_test.index,
               df_test.iloc[:,0].values, label=None,
               color='black',
               s=10)
    ax.hlines(y=0,xmin=df_test[['Prediction']].index[0],
              xmax=df_test[['Prediction']].index[-1], color='grey')
    ax.tick_params(axis='both', labelsize=14)

    text1 =  'Corr. coeff. model  : {:.2f} ({:.2f})'.format(float(df_test_m['corrcoef']),
                                                              df_train_m['corrcoef'].mean())
    text2 = r'RMSE model   : {:.2f} ({:.2f}) $\sigma$'.format(float(df_test_m['mean_squared_error']**.5),
                                              (df_train_m['mean_squared_error']**.5).mean())
    ax.set_title(exper)
    ax.text(.02, .019, text1,
            transform=ax.transAxes, horizontalalignment='left',
            verticalalignment='bottom',
            fontdict={'fontsize':12},
            bbox = dict(boxstyle='round', facecolor='red', alpha=.5,
                        edgecolor='black'))
    ax.text(.4, .019, text2,
            transform=ax.transAxes, horizontalalignment='left',
            verticalalignment='bottom',
            fontdict={'fontsize':12},
            bbox = dict(boxstyle='round', facecolor='red', alpha=.5,
                        edgecolor='black'))
    ax.legend(loc='upper right', fontsize=16, frameon=True, facecolor='grey',
              framealpha=.5)
    ax.set_ylabel('Standardized Crop Yield [1/y]', fontsize=16)
    ax.set_ylim(-3,3)

    m = models_lags['lag_0']['split_0']
    print(m.alpha_)
    figname = rg.path_outsub1 +f'/forecast_{exper}.pdf'
    plt.savefig(figname, bbox_inches='tight')

#%%
# rg.get_ts_prec(precur_aggr=1)


rg.PCMCI_df_data(keys=keys,
                 pc_alpha=None,
                  tau_max=0,
                  max_conds_dim=5,
                  max_combinations=5)
rg.PCMCI_get_links(alpha_level=.05)
# print(rg.df_MCIc.mean(0,level=1))
# print(rg.df_links.mean(0,level=1))
#%%

# rg.plot_maps_corr(var=['sm2'], mean=True, save=False, aspect=2, cbar_vert=-.1,
#                   subtitles=np.array([['SM2 Correlated']]))

# rg.plot_maps_corr(var=['sm3'], mean=True, save=False, aspect=2, cbar_vert=-.1,
#                   subtitles=np.array([['SM3 Correlated']]))

# rg.plot_maps_sum(cols=['corr'])

#%%
# rg.get_ts_prec(precur_aggr=1)
# rg.store_df()




#%%
# from class_fc import fcev
# from time import time
# start_time = time()

# # ERA_data = rg.path_df_data
# ERA_data = '/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/paper_Raed/output/3ts_707fb_18mar-18mar_lag0-0_leave_2s1/2020-07-13_12hr_08min_df_data_sst_dt1_707fb.h5'
# ERA_data2lags = '/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/paper_Raed/output/3ts_707fb_12aug-12aug_lag0-100_leave_4s1/2020-07-13_15hr_51min_df_data_sst_sm_dt1_707fb.h5'


# kwrgs_events = {'event_percentile': 50}

# kwrgs_events = kwrgs_events
# use_fold = None
# n_boot = 2000
# start_end_TVdate = ('07-01', '10-31')
# start_end_PRdate_JJAS = ('06-01', '09-30')
# start_end_PRdate_FMAM = ('02-01', '05-31')

# SST_L0 = ['0..1..sst', '0..2..sst', '0..3..sst', '0..4..sst', '0..5..sst',
#           '0..8..sst', '0..9..sst', '0..10..sst', '0..11..sst', '0..6..sst',
#           '0..12..sst', '0..13..sst', '0..7..sst']
# SST_L1 = ['100..1..sst', '100..2..sst', '100..3..sst', '100..4..sst',
#           '100..5..sst', '100..9..sst', '100..10..sst', '100..11..sst', '100..12..sst']
# list_of_fc = [fcev(path_data=ERA_data2lags,
#                     use_fold=use_fold,
#                     start_end_TVdate=start_end_PRdate_JJAS,
#                     start_end_PRdate=start_end_PRdate_JJAS,
#                     stat_model= ('logitCV',
#                                 {'Cs':10, #np.logspace(-4,1,10)
#                                 'class_weight':{ 0:1, 1:1},
#                                   'scoring':'neg_brier_score',
#                                   'penalty':'l2',
#                                   'solver':'lbfgs',
#                                   'max_iter':100,
#                                   'kfold':5,
#                                   'seed':1}),
#                     kwrgs_pp={'add_autocorr':False, 'normalize':'all'},
#                     dataset='',
#                     keys_d=('SST from JJAS', dict(zip(np.arange(9), [SST_L0]*9)))),
#               fcev(path_data=ERA_data2lags,
#                     use_fold=use_fold,
#                     start_end_TVdate=start_end_PRdate_FMAM,
#                     start_end_PRdate=start_end_PRdate_FMAM,
#                     stat_model= ('logitCV',
#                                 {'Cs':10, #np.logspace(-4,1,10)
#                                 'class_weight':{ 0:1, 1:1},
#                                   'scoring':'neg_brier_score',
#                                   'penalty':'l2',
#                                   'solver':'lbfgs',
#                                   'max_iter':100,
#                                   'kfold':5,
#                                   'seed':1}),
#                     kwrgs_pp={'add_autocorr':False, 'normalize':'all'},
#                     dataset='',
#                     keys_d=('SST from FMAM', dict(zip(np.arange(9), [SST_L1]*9))))]

# for fc in list_of_fc:
#     fc.get_TV(kwrgs_events=kwrgs_events, detrend=False)

#     fc.fit_models(lead_max=np.array([0]), verbosity=1)

#     fc.perform_validation(n_boot=n_boot, blocksize='auto', alpha=0.05,
#                               threshold_pred=50)

# # In[8]:
# # list_of_fc = [fc]
# working_folder, pathexper = list_of_fc[0]._print_sett(list_of_fc=list_of_fc)

# store = False
# if __name__ == "__main__":
#     pathexper = list_of_fc[0].pathexper
#     store = True

# import valid_plots as dfplots
# import functions_pp


# dict_all = dfplots.merge_valid_info(list_of_fc, store=store)
# if store:
#     dict_merge_all = functions_pp.load_hdf5(pathexper+'/data.h5')


# lag_rel = 0
# kwrgs = {'wspace':0.16, 'hspace':.25, 'col_wrap':2, 'skip_redundant_title':True,
#          'lags_relcurve':[lag_rel], 'fontbase':14, 'figaspect':2}
# #kwrgs = {'wspace':0.25, 'col_wrap':3, 'threshold_bin':fc.threshold_pred}
# met = ['AUC-ROC', 'AUC-PR', 'Precision', 'BSS', 'Accuracy', 'Rel. Curve']
# line_dim = 'exper'
# group_line_by = None

# fig = dfplots.valid_figures(dict_merge_all,
#                           line_dim=line_dim,
#                           group_line_by=group_line_by,
#                           met=met, **kwrgs)


# f_format = '.pdf'
# pathfig_valid = os.path.join(pathexper,'verification' + f_format)
# fig.savefig(pathfig_valid,
#             bbox_inches='tight') # dpi auto 600

# # In[ ]:


# # RV = rg.fullts.drop('n_clusters').drop('q').drop('cluster').to_dataframe()
# # import df_ana

# # df_g = df_ana.load_hdf5('/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/Sub-seasonal_statistical_forecasts_of_eastern_United_States_extreme_temperature_events/data/df_data_sst_CPPAs30_sm2_sm3_dt1_c378f_good.h5')['df_data']
# # RV['1_good'] = df_g.loc[0][['1']]
# # RV.corr()
# #%% Correlation maps SST

# list_of_name_path = [(1 , RV),
#                       ('sst', '/Users/semvijverberg/surfdrive/ERA5/input_raw/sst_1979-2018_1_12_daily_1.0deg.nc')]

# list_for_MI   = [BivariateMI(name='sst', func=BivariateMI.corr_map,
#                              kwrgs_func={'alpha':.0001, 'FDR_control':True},
#                              distance_eps=700, min_area_in_degrees2=5)]

# name_ds = 'q90tail'

# rg = RGCPD(list_of_name_path=list_of_name_path,
#                list_for_MI=list_for_MI,
#                list_import_ts=None,
#                start_end_TVdate=start_end_TVdate,
#                start_end_date=start_end_date,
#                tfreq=15, lags_i=np.array([0]),
#                path_outmain=user_dir+'/surfdrive/output_RGCPD')
# selbox = [None, {'sst':[-180,360,-10,90]}]
# anomaly = True

# rg.pp_precursors(selbox=selbox, anomaly=anomaly)


# # ### Post-processing Target Variable

# # In[7]:


# rg.pp_TV(name_ds=name_ds, detrend=False, anomaly=False)



# # In[165]:

# #kwrgs_events={'event_percentile':66}
# kwrgs_events=None
# rg.traintest(method='random10', kwrgs_events=kwrgs_events)



# # In[166]:


# rg.calc_corr_maps()
# rg.cluster_list_MI()
# rg.get_ts_prec()

# rg.PCMCI_df_data(keys=['1q90tail', '0..1..sst', 'RV_mask', 'TrainIsTrue'],
#                  pc_alpha=None,
#                  tau_max=1,
#                  max_conds_dim=1,
#                  max_combinations=1)
# rg.PCMCI_get_links(alpha_level=.05)



# rg.plot_maps_sum(cols=['corr'], kwrgs_plot={'aspect':2.5, 'cbar_vert':.13})
