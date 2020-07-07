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
path_out_main = os.path.join(main_dir, 'publications/paper2/output/west/')
cluster_label = 1
name_ds='ts'
start_end_TVdate = ('06-01', '08-31')
start_end_date = ('1-1', '12-31')
tfreq = 15
#%%
list_of_name_path = [(cluster_label, TVpath),
                      ('v200', os.path.join(path_raw, 'v200hpa_1979-2018_1_12_daily_2.5deg.nc')),
                       ('z500', os.path.join(path_raw, 'z500hpa_1979-2018_1_12_daily_2.5deg.nc')),
                       ('sst', os.path.join(path_raw, 'sst_1979-2018_1_12_daily_1.0deg.nc'))]




list_for_MI   = [BivariateMI(name='v200', func=BivariateMI.corr_map,
                              kwrgs_func={'alpha':.01, 'FDR_control':True},
                              distance_eps=600, min_area_in_degrees2=1,
                              calc_ts='pattern cov', selbox=(0,360,0,90)),
                   BivariateMI(name='z500', func=BivariateMI.corr_map,
                                kwrgs_func={'alpha':.01, 'FDR_control':True},
                                distance_eps=600, min_area_in_degrees2=1,
                                calc_ts='pattern cov', selbox=(0,360,0,90)),
                   BivariateMI(name='sst', func=BivariateMI.corr_map,
                                kwrgs_func={'alpha':.001, 'FDR_control':True},
                                distance_eps=600, min_area_in_degrees2=1,
                                calc_ts='pattern cov', selbox=(0,360,-10,90))]

list_for_EOFS = [EOF(name='v200', neofs=2, selbox=[-180, 360, 0, 80],
                     n_cpu=1)]



rg = RGCPD(list_of_name_path=list_of_name_path,
            list_for_MI=list_for_MI,
            list_for_EOFS=list_for_EOFS,
            start_end_TVdate=start_end_TVdate,
            start_end_date=start_end_date,
            start_end_year=None,
            tfreq=tfreq, lags_i=np.array([0,1]),
            path_outmain=path_out_main,
            append_pathsub='_' + name_ds)


rg.pp_TV(name_ds=name_ds, detrend=False)

rg.pp_precursors()

rg.traintest('no_train_test_split')

rg.calc_corr_maps()




rg.get_EOFs()
E = rg.list_for_EOFS[0]
secondEOF = E.eofs[0][1]
subtitles = np.array([['v-wind 200hpa 2nd EOF pattern']])
plot_maps.plot_corr_maps(secondEOF, aspect=2.5, size=5, cbar_vert=.07,
                  subtitles=subtitles, units='-', zoomregion=(-180,360,0,80),
                  map_proj=ccrs.PlateCarree(central_longitude=220), n_yticks=6)
plt.savefig(os.path.join(rg.path_outsub1, 'EOF_2_v_wind')+'pdf')

firstEOF = E.eofs[0][0]
subtitles = np.array([['v-wind 200hpa 1st EOF pattern']])
plot_maps.plot_corr_maps(firstEOF, aspect=2.5, size=5, cbar_vert=.07,
                  subtitles=subtitles, units='-', zoomregion=(-180,360,0,80),
                  map_proj=ccrs.PlateCarree(central_longitude=220), n_yticks=6)
plt.savefig(os.path.join(rg.path_outsub1, 'EOF_1_v_wind')+'pdf')


greenrectangle_WestUS_v200 = (100,330,24,70)
units = 'Corr. Coeff. [-]'
subtitles = np.array([[f'lag {l}: v-wind 200hpa vs western U.S. mx2t'] for l in rg.lags])
rg.plot_maps_corr(var='v200', row_dim='lag', col_dim='split',
                  aspect=2, size=5, hspace=-0.58, cbar_vert=.18, save=True,
                  subtitles=subtitles, units=units, zoomregion=(-180,360,0,80),
                  map_proj=ccrs.PlateCarree(central_longitude=220), n_yticks=6,
                  drawbox=['all', greenrectangle_WestUS_v200],
                  clim=(-.6,.6))

greenrectangle_WestUS_bb = (140,325,24,62)
subtitles = np.array([[f'lag {l}: z 500hpa vs western U.S. mx2t'] for l in rg.lags])
rg.plot_maps_corr(var='z500', row_dim='lag', col_dim='split',
                  aspect=2, size=5, hspace=-0.58, cbar_vert=.18, save=True,
                  subtitles=subtitles, units=units, zoomregion=(-180,360,0,80),
                  map_proj=ccrs.PlateCarree(central_longitude=220), n_yticks=6,
                  drawbox=['all', greenrectangle_WestUS_bb],
                  clim=(-.6,.6))

greenrectangle_WestSST_bb = (160,235,24,62)
subtitles = np.array([[f'lag {l}: SST vs western U.S. mx2t'] for l in rg.lags])
rg.plot_maps_corr(var='sst', row_dim='lag', col_dim='split',
                  aspect=2, hspace=-.57, size=5, cbar_vert=.175, save=True,
                  subtitles=subtitles, units=units, zoomregion=(-180,360,-10,70),
                  map_proj=ccrs.PlateCarree(central_longitude=220), n_yticks=6,
                  drawbox=['all', greenrectangle_WestSST_bb],
                  clim=(-.6,.6))

#%%
rg.cluster_list_MI(var='v200')

rg.quick_view_labels(var='v200')

rg.get_ts_prec(precur_aggr=None)

import df_ana
rg.df_data.loc[0].columns
df_sub = rg.df_data.loc[0][['1ts', '0..0..v200_sp', '0..2..EOF_v200']][rg.df_data.loc[0]['RV_mask']]
df_ana.plot_ts_matric(df_sub)

#%% Determine Rossby wave within green rectangle, become target variable

list_of_name_path = [(cluster_label, TVpath),
                     ('z500',os.path.join(path_raw, 'z500hpa_1979-2018_1_12_daily_2.5deg.nc'))]

dim_reduction = ['region mean', 'pattern cov']
list_for_MI   = [BivariateMI(name='z500', func=BivariateMI.corr_map,
                             kwrgs_func={'alpha':.01, 'FDR_control':True},
                             distance_eps=500, min_area_in_degrees2=1,
                             calc_ts=dim_reduction[0],
                             selbox=greenrectangle_WestUS_bb)]

rg = RGCPD(list_of_name_path=list_of_name_path,
           list_for_MI=list_for_MI,
           start_end_TVdate=start_end_TVdate,
           start_end_date=start_end_date,
           tfreq=tfreq, lags_i=np.array([0]),
           path_outmain=user_dir+'/surfdrive/output_RGCPD/circulation_US_HW',
           append_pathsub='_' + name_ds)


rg.pp_precursors(anomaly=True)
rg.pp_TV(name_ds=name_ds)

rg.traintest(method='no_train_test_split')

rg.calc_corr_maps()
rg.plot_maps_corr(var='z500')
rg.cluster_list_MI(var='z500')

# rg.get_ts_prec(precur_aggr=None)
rg.get_ts_prec(precur_aggr=1)
# rg.store_df()
#%% store data
# rg.cluster_list_MI(var='z500')
# rg.get_ts_prec(precur_aggr=None)
# rg.get_ts_prec(precur_aggr=1)
# rg.store_df()

#%%
import class_RV ; import matplotlib.pyplot as plt
RV_ts = rg.fulltso.sel(time=rg.TV.aggr_to_daily_dates(rg.dates_TV))
threshold = class_RV.Ev_threshold(RV_ts, event_percentile=85)
RV_bin, np_dur = class_RV.Ev_timeseries(RV_ts, threshold=threshold, grouped=True)
plt.hist(np_dur[np_dur!=0])


#%% Remnants past

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