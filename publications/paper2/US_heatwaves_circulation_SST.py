#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:03:30 2020

@author: semvijverberg
"""

import os, inspect, sys
import numpy as np
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
from RGCPD import EOF
import class_BivariateMI
import climate_indices
import plot_maps, core_pp, df_ana


TVpath = '/Users/semvijverberg/surfdrive/output_RGCPD/circulation_US_HW/tf15_nc3_dendo_0ff31.nc'

west_east = 'east'
if west_east == 'east':
    path_out_main = os.path.join(main_dir, 'publications/paper2/output/east/')
    cluster_label = 2
elif west_east == 'west':
    path_out_main = os.path.join(main_dir, 'publications/paper2/output/west/')
    cluster_label = 1


name_ds='ts'
start_end_TVdate = ('06-01', '08-31')
start_end_date = ('1-1', '12-31')
method='ran_strat10' ; seed = 1
tfreq = 15
min_detect_gc=1.

# z500_green_bb = (140,260,20,73) #: Pacific box
if west_east == 'east':
    z500_green_bb = (155,300,20,73) #: RW box
    v300_green_bb = (170,359,23,73)
elif west_east == 'west':
    z500_green_bb = (145,325,20,62)
    v300_green_bb = (100,330,24,70)

#%% Circulation vs temperature
list_of_name_path = [(cluster_label, TVpath),
                      ('v300', os.path.join(path_raw, 'v300hpa_1979-2018_1_12_daily_2.5deg.nc')),
                       ('z500', os.path.join(path_raw, 'z500hpa_1979-2018_1_12_daily_2.5deg.nc'))]

lags = np.array([0])

list_for_MI   = [BivariateMI(name='v300', func=class_BivariateMI.corr_map,
                              alpha=.05, FDR_control=True, lags=lags,
                              distance_eps=600, min_area_in_degrees2=5,
                              calc_ts='pattern cov', selbox=(0,360,-10,90),
                              use_sign_pattern=True),
                   BivariateMI(name='z500', func=class_BivariateMI.corr_map,
                                alpha=.05, FDR_control=True, lags=lags,
                                distance_eps=600, min_area_in_degrees2=5,
                                calc_ts='pattern cov', selbox=(0,360,-10,90),
                                use_sign_pattern=True)]



rg = RGCPD(list_of_name_path=list_of_name_path,
            list_for_MI=list_for_MI,
            start_end_TVdate=start_end_TVdate,
            start_end_date=start_end_date,
            start_end_year=None,
            tfreq=tfreq,
            path_outmain=path_out_main)


rg.pp_TV(name_ds=name_ds, detrend=False)

rg.pp_precursors()

rg.traintest(method=method, seed=seed)



#%%

rg.calc_corr_maps()

#%% Plot corr(z500, mx2t)
import matplotlib
# Optionally set font to Computer Modern to avoid common missing font errors
matplotlib.rc('font', family='serif', serif='cm10')

matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']




title = f'$corr(z500, {west_east.capitalize()[0]}$-$US\ mx2t)$'
subtitles = np.array([['']] )
kwrgs_plot = {'row_dim':'lag', 'col_dim':'split', 'aspect':3.8, 'size':2.5,
              'hspace':0.0, 'cbar_vert':-.08, 'units':'Corr. Coeff. [-]',
              'zoomregion':(-180,360,0,80), 'drawbox':[(0,0), z500_green_bb],
              'map_proj':ccrs.PlateCarree(central_longitude=220), 'n_yticks':6,
              'clim':(-.6,.6), 'title':title, 'subtitles':subtitles}
save = True
rg.plot_maps_corr(var='z500', save=save,
                  append_str=''.join(map(str, z500_green_bb)),
                  min_detect_gc=min_detect_gc,
                  kwrgs_plot=kwrgs_plot)

#%% Plot corr(v300, mx2t)


kwrgs_plot['title'] = f'$corr(v300, {west_east.capitalize()[0]}$-$US\ mx2t)$'
kwrgs_plot['drawbox'] = [(0,0), v300_green_bb]
rg.plot_maps_corr(var='v300', save=save,
                  kwrgs_plot=kwrgs_plot,
                  min_detect_gc=min_detect_gc)

#%% Determine Rossby wave within green rectangle, become target variable for feedback

rg.list_for_MI = [BivariateMI(name='v300', func=class_BivariateMI.corr_map,
                              alpha=.05, FDR_control=True,
                              distance_eps=600, min_area_in_degrees2=5,
                              calc_ts='pattern cov', selbox=v300_green_bb,
                              use_sign_pattern=True, lags = np.array([0])),
                   BivariateMI(name='z500', func=class_BivariateMI.corr_map,
                                alpha=.05, FDR_control=True,
                                distance_eps=600, min_area_in_degrees2=5,
                                calc_ts='pattern cov', selbox=z500_green_bb,
                                use_sign_pattern=True, lags = np.array([0]))]
rg.list_for_EOFS = None
rg.calc_corr_maps(['z500', 'v300'])
rg.cluster_list_MI(['z500'])
rg.get_ts_prec(precur_aggr=1)
rg.store_df(append_str='z500_'+'-'.join(map(str, z500_green_bb)))

#%% EOFs
if west_east == 'east':
    subtitles = np.array([['Clustered simultaneous warm temp. periods']])
    rg.get_clust()
    plot_maps.plot_labels(rg.ds['xrclustered'],
                          zoomregion=(230,300,25,60),
                          kwrgs_plot={'subtitles':subtitles,
                                      'y_ticks':np.array([30, 40,50,60]),
                                      'x_ticks':np.arange(230, 310, 10),
                                      'cbar_vert':-.03,
                                      'add_cfeature':'OCEAN'})
    plt.savefig(os.path.join(rg.path_outsub1, 'clusters')+'.pdf', bbox_inches='tight')

    # rg.list_for_EOFS = [EOF(name='z500', neofs=3, selbox=[140, 300, 10, 80],
    #                      n_cpu=1, start_end_date=start_end_TVdate),
    #                     EOF(name='v300', neofs=3, selbox=[140, 300, 10, 80],
    #                      n_cpu=1, start_end_date=start_end_TVdate)]

    # rg.get_EOFs()
    # firstEOF = rg.list_for_EOFS[1].eofs.mean(dim='split')[0]
    # subtitles = np.array([['z 500hpa 1st EOF pattern']])
    # plot_maps.plot_corr_maps(firstEOF, aspect=2, size=5, cbar_vert=.07,
    #                   subtitles=subtitles, units='-', #zoomregion=(-180,360,0,80),
    #                   map_proj=ccrs.PlateCarree(central_longitude=220), n_yticks=6)
    # plt.savefig(os.path.join(rg.path_outsub1, 'EOF_1_z500')+'.pdf')

elif west_east == 'west':
    # circumglobal for picture
    rg.list_for_EOFS = [EOF(name='v300', neofs=2, selbox=[-180, 360, 0, 80],
                        n_cpu=1, start_end_date=start_end_TVdate)]
    rg.get_EOFs()
    eofs = rg.list_for_EOFS[0].eofs.mean(dim='split') # mean over training sets
    subtitles = 'v-wind 300 hPa - 1st EOF loading pattern'
    kwrgs_plotEOF = kwrgs_plot.copy()
    kwrgs_plotEOF.update({'clim':None, 'units':None, 'title':subtitles,
                          'y_ticks':np.array([10,30,50,70]),
                          'drawbox':None})
    plot_maps.plot_corr_maps(eofs[0] , **kwrgs_plotEOF)
    plt.savefig(os.path.join(rg.path_outsub1, 'EOF_1_v_wind')+'pdf')

    subtitles = np.array([['v-wind 300 hPa - 2nd EOF loading pattern']])
    plot_maps.plot_corr_maps(eofs[1] , **kwrgs_plotEOF)
    plt.savefig(os.path.join(rg.path_outsub1, 'EOF_2_v_wind')+'pdf')

    # spatial correlation eof and v300 in between v300 green bb
    rg.list_for_EOFS[0].eofs = core_pp.get_selbox(rg.list_for_EOFS[0].eofs, selbox=v300_green_bb)
    rg.list_for_EOFS[0].selbox = v300_green_bb
    rg.cluster_list_MI()
    rg.get_ts_prec(precur_aggr=None)
    rg.df_data.loc[0].columns
    df_sub = rg.df_data.loc[0][['1ts', '0..0..v300_sp', '0..0..z500_sp',
                                '0..1..EOF_v300']][rg.df_data.loc[0]['RV_mask']]
    df_sub = df_sub.rename({'1ts':'w-U.S. mx2t', '0..0..v300_sp':'RW (v300)',
                   '0..0..z500_sp':'RW (z500)',
                   '0..1..EOF_v300': '1st EOF (v300)',
                   '0..2..EOF_v300':'2nd EOF (v300)'}, axis=1)
    # df_sub['2nd EOF (v300)'] *= -1
    df_ana.plot_ts_matric(df_sub)










#%% SST vs mx2tm
list_of_name_path = [(cluster_label, TVpath),
                     ('sst', os.path.join(path_raw, 'sst_1979-2018_1_12_daily_1.0deg.nc'))]

lags = np.array([0,2])

list_for_MI   = [BivariateMI(name='sst', func=class_BivariateMI.corr_map,
                                alpha=.05, FDR_control=True, lags=lags,
                                distance_eps=600, min_area_in_degrees2=1,
                                calc_ts='pattern cov', selbox=(120,260,-10,90))]


rg = RGCPD(list_of_name_path=list_of_name_path,
            list_for_MI=list_for_MI,
            start_end_TVdate=start_end_TVdate,
            start_end_date=start_end_date,
            start_end_year=None,
            tfreq=tfreq,
            path_outmain=path_out_main)


rg.pp_TV(name_ds=name_ds, detrend=False)
rg.traintest(method, seed=seed)
rg.pp_precursors()
rg.calc_corr_maps()


#%% Plot corr(SST, mx2t)
import matplotlib
# Optionally set font to Computer Modern to avoid common missing font errors
matplotlib.rc('font', family='serif', serif='cm10')

matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']

save=True
SST_green_bb = (140,235,20,59)#(170,255,11,60)
# subtitles = np.array([[f'lag {l}: SST vs E-U.S. mx2t' for l in rg.lags]])
title = f'$corr(SST, {west_east.capitalize()[0]}$-$US\ mx2t)$'
subtitles = np.array([['lag 0', f'lag 2 (15 day lead)']] )
kwrgs_plot = {'row_dim':'split', 'col_dim':'lag','aspect':2, 'hspace':-.47,
              'wspace':-.15, 'size':3, 'cbar_vert':-.08,
              'units':'Corr. Coeff. [-]', 'zoomregion':(130,260,-10,60),
              'clim':(-.6,.6), 'map_proj':ccrs.PlateCarree(central_longitude=220),
              'n_yticks':6, 'x_ticks':np.arange(130, 280, 25),
              'subtitles':subtitles, 'title':title}
rg.plot_maps_corr(var='sst', save=save,
                  min_detect_gc=min_detect_gc,
                  kwrgs_plot=kwrgs_plot)

# #%% RW vs SST feedback, SST based on SST vs mx2t # Deprecicated.
# should re-establish direct relationship between RW and SST,
# see feedback_wave_SST.py


# freqs = [1, 5, 15, 30, 60]
# for f in freqs:
#     rg.get_ts_prec(precur_aggr=f)
#     rg.df_data = rg.df_data.rename({'0..0..z500_sp':'Rossby wave (z500)',
#                                '0..0..sst_sp':'Pacific SST',
#                                '15..0..sst_sp':'Pacific SST (lag 15)',
#                                '0..0..v300_sp':'Rossby wave (v300)'}, axis=1)

#     keys = [['Rossby wave (z500)', 'Pacific SST'], ['Rossby wave (v300)', 'Pacific SST']]
#     k = keys[0]
#     name_k = ''.join(k[:2]).replace(' ','')
#     k.append('TrainIsTrue') ; k.append('RV_mask')

#     rg.PCMCI_df_data(keys=k,
#                      pc_alpha=None,
#                      tau_max=5,
#                      max_conds_dim=10,
#                      max_combinations=10)
#     rg.PCMCI_get_links(var=k[0], alpha_level=.01)

#     rg.PCMCI_plot_graph(min_link_robustness=5, figshape=(3,2),
#                         kwrgs={'vmax_nodes':1.0,
#                                'vmax_edges':.6,
#                                'vmin_edges':-.6,
#                                'node_ticks':.3,
#                                'edge_ticks':.3,
#                                'curved_radius':.5,
#                                'arrowhead_size':1000,
#                                'label_fontsize':10,
#                                'link_label_fontsize':12,
#                                'node_label_size':16},
#                         append_figpath=f'_tf{rg.precur_aggr}_{name_k}')

#     rg.PCMCI_get_links(var=k[1], alpha_level=.01)
#     rg.df_links.astype(int).sum(0, level=1)
#     MCI_ALL = rg.df_MCIc.mean(0, level=1)





#%% Compare E-RW with PNA index
if west_east == 'east':
    import pandas as pd
    list_of_name_path = [(cluster_label, TVpath),
                           ('z500', os.path.join(path_raw, 'z500hpa_1979-2018_1_12_daily_2.5deg.nc')),
                           ('u', os.path.join(path_raw, 'u432_1979-2018_1_12_daily_1.0deg.nc'))]



    list_for_MI   = [BivariateMI(name='z500', func=class_BivariateMI.corr_map,
                                    alpha=.05, FDR_control=True,
                                    distance_eps=600, min_area_in_degrees2=1,
                                    calc_ts='pattern cov', selbox=z500_green_bb,
                                    use_sign_pattern=True,
                                    lags=np.array([0]))]

    # list_for_EOFS = [EOF(name='u', neofs=1, selbox=[120, 255, 0, 87.5],
    #                      n_cpu=1, start_end_date=('01-12', '02-28'))]
    list_for_EOFS = [EOF(name='u', neofs=1, selbox=[120, 255, 0, 87.5],
                         n_cpu=1, start_end_date=start_end_TVdate)]

    rg = RGCPD(list_of_name_path=list_of_name_path,
                list_for_MI=list_for_MI,
                list_for_EOFS=list_for_EOFS,
                start_end_TVdate=start_end_TVdate,
                start_end_date=start_end_date,
                start_end_year=None,
                tfreq=tfreq,
                path_outmain=path_out_main,
                append_pathsub='_' + name_ds)


    rg.pp_TV(name_ds=name_ds, detrend=False)

    rg.pp_precursors()

    rg.traintest(method, seed=seed)

    rg.calc_corr_maps()
    rg.cluster_list_MI()

    rg.get_EOFs()

    rg.get_ts_prec(precur_aggr=1)


    PNA = climate_indices.PNA_z500(rg.list_precur_pp[0][1])


    df_c = rg.df_data.merge(pd.concat([PNA.loc[rg.df_data.loc[0].index]]*10,
                            axis=0, keys=range(10)),
                            left_index=True, right_index=True)


    # From Climate Explorer
    # https://climexp.knmi.nl/getindices.cgi?WMO=NCEPData/cpc_pna_daily&STATION=PNA&TYPE=i&id=someone@somewhere&NPERYEAR=366
    # on 20-07-2020
    PNA_cpc = core_pp.import_ds_lazy(main_dir+'/publications/paper2/data/icpc_pna_daily.nc',
                                     start_end_year=(1979, 2018),
                                     seldates=start_end_TVdate).to_dataframe('PNAcpc')
    PNA_cpc.index.name = None
    df_c = df_c.merge(pd.concat([PNA_cpc]*10, axis=0, keys=range(10)),
                      left_index=True, right_index=True)

    df_c = df_c.rename({'0..0..z500_sp':'RW (z500)',
                        '0..1..EOF_u':'1st EOF u',
                        'PNA':'PNAw'}, axis=1)
    columns = [rg.TV.name, 'RW (z500)', '1st EOF u', 'PNAw', 'PNAcpc']

    df_ana.plot_ts_matric(df_c, win=15, columns=columns)
    filename = os.path.join(rg.path_outsub1, 'cross_corr_RWz500_PNA_15d.pdf')
    plt.savefig(filename, bbox_inches='tight')

#%%
# #%% Determine Rossby wave within green rectangle, become target variable for HM

# list_of_name_path = [(cluster_label, TVpath),
#                      ('z500',os.path.join(path_raw, 'z500hpa_1979-2018_1_12_daily_2.5deg.nc')),
#                      ('v300', os.path.join(path_raw, 'v300hpa_1979-2018_1_12_daily_2.5deg.nc'))]

# list_for_MI   = [BivariateMI(name='z500', func=class_BivariateMI.corr_map,
#                              alpha=.05, FDR_control=True,
#                              distance_eps=500, min_area_in_degrees2=1,
#                              calc_ts='pattern cov', selbox=z500_green_bb),
#                  BivariateMI(name='v300', func=class_BivariateMI.corr_map,
#                              alpha=.05, FDR_control=True,
#                              distance_eps=500, min_area_in_degrees2=1,
#                              calc_ts='pattern cov', selbox=v300_green_bb)]

# rg = RGCPD(list_of_name_path=list_of_name_path,
#            list_for_MI=list_for_MI,
#            start_end_TVdate=start_end_TVdate,
#            start_end_date=start_end_date,
#            tfreq=tfreq, lags_i=np.array([0]),
#            path_outmain=path_out_main,
#            append_pathsub='_' + name_ds)


# rg.pp_precursors(anomaly=True)
# rg.pp_TV(name_ds=name_ds)

# rg.traintest(method='no_train_test_split')

# rg.calc_corr_maps()
# subtitles = np.array([['E-U.S. Temp. correlation map Z 500hpa green box']])
# rg.plot_maps_corr(var='z500', cbar_vert=-.05, subtitles=subtitles, save=False)
# subtitles = np.array([['E-U.S. Temp. correlation map v300 green box']])
# rg.plot_maps_corr(var='v300', cbar_vert=-.05, subtitles=subtitles, save=False)
# rg.cluster_list_MI()
# # rg.get_ts_prec(precur_aggr=None)
# rg.get_ts_prec(precur_aggr=1)
# rg.store_df(append_str='z500_'+'-'.join(map(str, z500_green_bb)))


# #%% interannual variability events?
# import class_RV
# RV_ts = rg.fulltso.sel(time=rg.TV.aggr_to_daily_dates(rg.dates_TV))
# threshold = class_RV.Ev_threshold(RV_ts, event_percentile=85)
# RV_bin, np_dur = class_RV.Ev_timeseries(RV_ts, threshold=threshold, grouped=True)
# plt.hist(np_dur[np_dur!=0])