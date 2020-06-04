#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:03:30 2020

@author: semvijverberg
"""


#%%
import os, inspect, sys
import numpy as np
import cartopy.crs as ccrs

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
from RGCPD import EOF
import plot_maps

TVpath = '/Users/semvijverberg/surfdrive/output_RGCPD/circulation_US_HW/tf15_nc3_dendo_0ff31.nc'
cluster_label = 1
name_ds='ts'
start_end_TVdate = ('06-01', '08-31')
start_end_date = ('1-1', '12-31')
tfreq = 15
#%%
list_of_name_path = [(cluster_label, TVpath),
                      ('v200', os.path.join(path_raw, 'v200hpa_1979-2018_1_12_daily_2.5deg.nc')),
                       ('z500', os.path.join(path_raw, 'z500hpa_1979-2018_1_12_daily_2.5deg.nc'))]




list_for_MI   = [BivariateMI(name='v200', func=BivariateMI.corr_map,
                              kwrgs_func={'alpha':.01, 'FDR_control':True},
                              distance_eps=600, min_area_in_degrees2=1,
                              calc_ts='pattern cov'),
                   BivariateMI(name='z500', func=BivariateMI.corr_map,
                                kwrgs_func={'alpha':.01, 'FDR_control':True},
                                distance_eps=600, min_area_in_degrees2=1,
                                calc_ts='pattern cov')]

list_for_EOFS = [EOF(name='v200', neofs=2, selbox=[-180, 360, 10, 90],
                     n_cpu=1)]



rg = RGCPD(list_of_name_path=list_of_name_path,
            list_for_MI=list_for_MI,
            list_for_EOFS=list_for_EOFS,
            start_end_TVdate=start_end_TVdate,
            start_end_date=start_end_date,
            start_end_year=None,
            tfreq=tfreq, lags_i=np.array([0]),
            path_outmain=user_dir+'/surfdrive/output_RGCPD/circulation_US_HW',
            append_pathsub='_' + name_ds)


rg.pp_TV(name_ds=name_ds, detrend=False)

rg.pp_precursors(selbox=(0,360,10,90))

rg.traintest('no_train_test_split')

rg.calc_corr_maps()



subtitles = np.array([['Western U.S. one-point correlation map v-wind 200hpa']])
units = 'Corr. Coeff. [-]'
rg.plot_maps_corr(var='v200', aspect=2, size=5, cbar_vert=.19, save=True,
                  subtitles=subtitles, units=units, zoomregion=(-180,360,10,75),
                  map_proj=ccrs.PlateCarree(central_longitude=220), n_yticks=5,
                  clim=(-.6,.6))

rg.get_EOFs()
E = rg.list_for_EOFS[0]
secondEOF = E.eofs[0][1]
plot_maps.plot_corr_maps(secondEOF, aspect=2, size=5, cbar_vert=.19,
                  subtitles=subtitles, units=units, zoomregion=(-180,360,10,75),
                  map_proj=ccrs.PlateCarree(central_longitude=220), n_yticks=5)

greenrectangle_bb = (150,330,23,68)
subtitles = np.array([['Western U.S. one-point correlation map Z 500hpa']])
units = 'Corr. Coeff. [-]'
rg.plot_maps_corr(var='z500', aspect=2, size=5, cbar_vert=.19, save=True,
                  subtitles=subtitles, units=units, zoomregion=(-180,360,10,75),
                  map_proj=ccrs.PlateCarree(central_longitude=220), n_yticks=5,
                  drawbox=['all', greenrectangle_bb],
                  clim=(-.6,.6))

#%%
rg.cluster_list_MI(var='v200')

rg.quick_view_labels(var='v200')

rg.get_ts_prec(precur_aggr=None)

import df_ana
rg.df_data.loc[0].columns
df_sub = rg.df_data.loc[0][['1ts', '0..0..v200_sp', '0..2..EOF_v200']][rg.df_data.loc[0]['RV_mask']]
df_ana.plot_ts_matric(df_sub)

#%% Determine Rossby wave target variable within green rectangle

list_of_name_path = [(cluster_label, TVpath),
                     ('z500',os.path.join(path_raw, 'z500hpa_1979-2018_1_12_daily_2.5deg.nc'))]

list_for_MI   = [BivariateMI(name='z500', func=BivariateMI.corr_map,
                             kwrgs_func={'alpha':.01, 'FDR_control':True},
                             distance_eps=500, min_area_in_degrees2=1,
                             calc_ts='pattern cov')]

rg = RGCPD(list_of_name_path=list_of_name_path,
           list_for_MI=list_for_MI,
           start_end_TVdate=start_end_TVdate,
           start_end_date=start_end_date,
           tfreq=tfreq, lags_i=np.array([0]),
           path_outmain=user_dir+'/surfdrive/output_RGCPD/circulation_US_HW',
           append_pathsub='_' + name_ds)


rg.pp_precursors(selbox=greenrectangle_bb, anomaly=True)
rg.pp_TV(name_ds=name_ds)

rg.traintest(method='no_train_test_split')

rg.calc_corr_maps()
rg.plot_maps_corr(var='z500')
rg.cluster_list_MI(var='z500')
# rg.get_ts_prec(precur_aggr=None)
rg.get_ts_prec(precur_aggr=1)
rg.store_df()
#%% store data
rg.cluster_list_MI(var='z500')
# rg.get_ts_prec(precur_aggr=None)
rg.get_ts_prec(precur_aggr=1)
rg.store_df()




#%% Remnants past



list_of_name_path = [(cluster_label, TVpath),
                     ('z500',os.path.join(path_raw, 'z500hpa_1979-2018_1_12_daily_2.5deg.nc')),
                     ('NorthPac', os.path.join(path_raw, 'sst_1979-2018_1_12_daily_1.0deg.nc')),
                     ('NorthAtl', os.path.join(path_raw, 'sst_1979-2018_1_12_daily_1.0deg.nc')),
                     ('sm12', os.path.join(path_raw, 'sm12_1979-2018_1_12_daily_1.0deg.nc'))]
                     # ('snow',os.path.join(path_raw, 'snow_1979-2018_1_12_daily_1.0deg.nc')),
                     # ('OLRtrop',  os.path.join(path_raw, 'OLRtrop_1979-2018_1_12_daily_2.5deg.nc'))]
                     # ('st2',  os.path.join(path_raw, 'lsm_st2_1979-2018_1_12_daily_1.0deg.nc'))]

list_for_MI   = [BivariateMI(name='z500', func=BivariateMI.corr_map,
                             kwrgs_func={'alpha':.01, 'FDR_control':True},
                             distance_eps=700, min_area_in_degrees2=7,
                             calc_ts='pattern cov'),
                   BivariateMI(name='sm12', func=BivariateMI.corr_map,
                                 kwrgs_func={'alpha':.01, 'FDR_control':True},
                                 distance_eps=700, min_area_in_degrees2=5),
                 # BivariateMI(name='snow', func=BivariateMI.corr_map,
                 #               kwrgs_func={'alpha':.01, 'FDR_control':True},
                 #               distance_eps=700, min_area_in_degrees2=7),
                 BivariateMI(name='NorthPac', func=BivariateMI.corr_map,
                              kwrgs_func={'alpha':1E-3, 'FDR_control':True},
                              distance_eps=1000, min_area_in_degrees2=5,
                              calc_ts='pattern cov'),
                  BivariateMI(name='NorthAtl', func=BivariateMI.corr_map,
                              kwrgs_func={'alpha':1E-3, 'FDR_control':True},
                              distance_eps=700, min_area_in_degrees2=5,
                              calc_ts='pattern cov')]
                 # BivariateMI(name='st2', func=BivariateMI.corr_map,
                 #               kwrgs_func={'alpha':.01, 'FDR_control':True},
                 #               distance_eps=700, min_area_in_degrees2=5)]

# list_for_EOFS = [EOF(name='OLRtrop', neofs=2, selbox=[-180, 360, -15, 30])]

list_import_ts = [('OMI', '/Users/semvijverberg/surfdrive/output_RGCPD/circulation_US_HW/OMI.h5')]


rg = RGCPD(list_of_name_path=list_of_name_path,
           list_for_MI=list_for_MI,
           list_import_ts=list_import_ts,
           start_end_TVdate=start_end_TVdate,
           start_end_date=start_end_date,
           tfreq=tfreq, lags_i=np.array([1]),
           path_outmain=user_dir+'/surfdrive/output_RGCPD/circulation_US_HW',
           append_pathsub='_' + name_ds)


selbox = [None, {'NorthPac':(115, 250, 0, 70),
                 'NorthAtl':(360-83, 6, 0, 70),
                 'v200':[130,350,10,90]}]

anomaly = [True, {'sm12':False, 'OLRtrop':False}]
rg.pp_precursors(selbox=selbox, anomaly=anomaly)

rg.pp_TV(name_ds=name_ds)

rg.traintest(method='random10')

rg.calc_corr_maps()

 #%%
rg.cluster_list_MI()

rg.quick_view_labels()

rg.get_ts_prec(precur_aggr=None)


# keys = ['0..1..st2', '0..2..sm12']
# rg.reduce_df_data_ridge(keys=keys, newname='SM_ST')

# merge_sst = [k for k in rg.df_data.columns if 'sst' in k]
# merge_sst = ['0..2..sst',
#              '0..6..sst']
# predict, weights = rg.reduce_df_data_ridge(keys=merge_sst, tau_max=5,newname='sst')

# merge_sst = ['0..1..sst',
#              '0..5..sst']
# predict, weights = rg.reduce_df_data_ridge(keys=merge_sst,
#                                            tau_min=0,
#                                            tau_max=7,newname='sst')

# zz = weights.swaplevel()
# # zz['splits'] = np.repeat(zz.index.levels[1], zz.index.levels[0].size)
# zz['var'] = np.repeat(zz.index.levels[0], zz.index.levels[1].size)
# axes = weights.swaplevel().T.groupby(axis=1, level=1).boxplot()
# axes[0].set_xticks(range(1,8))


keys = ['1', '5..0..z500_sp', 'PC2', '5..0..NorthPac_sp', '5..0..NorthAtl_sp',
        'TrainIsTrue', 'RV_mask']

rg.PCMCI_df_data(keys=keys,
                 pc_alpha=None,
                 tau_max=3,
                 max_conds_dim=10,
                 max_combinations=10)
rg.PCMCI_get_links(alpha_level=.05)
rg.df_links.mean(0, level=1)
rg.df_MCIc.mean(0, level=1)

rg.PCMCI_plot_graph(min_link_robustness=10)

rg.PCMCI_get_ParCorr_from_txt()

rg.quick_view_labels(var='NorthPac', median=False)


rg.plot_maps_corr(var=['sm12'], mean=False, save=False)

rg.plot_maps_sum(var='sm12',
                 kwrgs_plot={'aspect': 2, 'wspace': -0.02})
rg.plot_maps_sum(var='snow',
                 kwrgs_plot={'aspect': 2, 'wspace': -0.02})
rg.plot_maps_sum(var='sst',
                 kwrgs_plot={'cbar_vert':.02})
rg.plot_maps_sum(var='z500',
                 kwrgs_plot={'cbar_vert':.02})


#%%


#%%
rg.get_ts_prec(precur_aggr=1)
rg.store_df_PCMCI()





#%%
from tigramite import plotting as tp
import matplotlib as mpl
s = 5
mpl.rcParams.update(mpl.rcParamsDefault)
variable = '0..0..sm12st2'
idx = rg.pcmci_dict[s].var_names.index(variable)
link_only_RV = np.zeros_like(rg.parents_dict[s][2])
link_matrix = rg.parents_dict[s][2]
link_only_RV[:,idx] = link_matrix[:,idx]
tp.plot_graph(val_matrix=rg.pcmci_results_dict[s]['val_matrix'],
              var_names=rg.pcmci_dict[s].var_names,
              link_matrix=link_only_RV,
              link_colorbar_label='cross-MCI',
node_colorbar_label='auto-MCI')

#%%
# from class_fc import fcev
# import os
# logitCV = ('logitCV',
#           {'class_weight':{ 0:1, 1:1},
#            'scoring':'brier_score_loss',
#            'penalty':'l2',
#            'solver':'lbfgs',
#            'max_iter':125,
#            'refit':False})

# path_data = rg.df_data_filename
# name = rg.TV.name
# # path_data = '/Users/semvijverberg/surfdrive/output_RGCPD/circulation_US_HW/3_80d77_26jun-21aug_lag14-14_q75tail_random10s1/None_at0.05_tau_0-1_conds_dimNone_combin2_dt14_dtd1.h5'
# # name = '3'
# kwrgs_events = {'event_percentile': 66}


# lags_i = np.array([0, 10, 14, 21, 28, 35])
# precur_aggr = 16
# use_fold = -9


# list_of_fc = [fcev(path_data=path_data, precur_aggr=precur_aggr,
#                     use_fold=None, start_end_TVdate=None,
#                     stat_model=logitCV,
#                     kwrgs_pp={},
#                     dataset=f'{precur_aggr} day means exper 1',
#                     keys_d='persistence'),
#               fcev(path_data=path_data, precur_aggr=precur_aggr,
#                    use_fold=None, start_end_TVdate=None,
#                    stat_model=logitCV,
#                    kwrgs_pp={},
#                    dataset=f'{precur_aggr} day means exper 2',
#                    keys_d='all')]



# for i, fc in enumerate(list_of_fc):


#     fc.get_TV(kwrgs_events=kwrgs_events)

#     fc.fit_models(lead_max=lags_i, verbosity=1)

# for i, fc in enumerate(list_of_fc):
#     fc.perform_validation(n_boot=500, blocksize='auto', alpha=0.05,
#                           threshold_pred=(1.5, 'times_clim'))



# df_valid, RV, y_pred_all = fc.dict_sum



# import valid_plots as dfplots
# kwrgs = {'wspace':0.25, 'col_wrap':None, 'threshold_bin':fc.threshold_pred}
# #kwrgs = {'wspace':0.25, 'col_wrap':3, 'threshold_bin':fc.threshold_pred}
# met = ['AUC-ROC', 'AUC-PR', 'BSS', 'Rel. Curve', 'Precision']
# #met = ['AUC-ROC', 'AUC-PR', 'BSS', 'Rel. Curve']
# line_dim = 'dataset'

# fig = dfplots.valid_figures(list_of_fc,
#                           line_dim=line_dim,
#                           group_line_by=None,
#                           met=met, **kwrgs)


# working_folder, filename = fc._print_sett(list_of_fc=list_of_fc)

# f_format = '.pdf'
# pathfig_valid = os.path.join(filename + f_format)
# fig.savefig(pathfig_valid,
#             bbox_inches='tight') # dpi auto 600