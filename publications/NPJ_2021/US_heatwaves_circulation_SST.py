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
data_dir = os.path.join(main_dir,'publications/NPJ_2021/data')
if cluster_func not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(RGCPD_func)
    sys.path.append(cluster_func)
    sys.path.append(fc_dir)

path_raw = user_dir + '/surfdrive/ERA5/input_raw'


from RGCPD import RGCPD
from RGCPD import BivariateMI
import class_BivariateMI
import plot_maps, functions_pp, df_ana, find_precursors

west_east = 'west'
TV = 'USCAnew'
if TV == 'USCAnew':
    # mx2t 25N-70N
    TVpath = os.path.join(data_dir, 'tfreq15_nc7_dendo_57db0USCA.nc')
    # TVpath = user_dir+'/surfdrive/output_RGCPD/circulation_US_HW/one-point-corr_maps_clusters/tf15_nc8_dendo_57db0USCA.nc'
    if west_east == 'east':
        cluster_label = 4
    elif west_east == 'west':
        cluster_label = 1 # 8
    elif west_east == 'northwest':
        cluster_label = 7
# elif TV == 'init':
#     TVpath = user_dir + '/surfdrive/output_RGCPD/circulation_US_HW/tf15_nc3_dendo_0ff31.nc'
#     if west_east == 'east':
#         cluster_label = 2
#     elif west_east == 'west':
#         cluster_label = 1
# elif TV == 'USCA':
#     # large eastern US, small western US and a north-western NA
#     TVpath = user_dir + '/surfdrive/output_RGCPD/circulation_US_HW/one-point-corr_maps_clusters/tf30_nc5_dendo_5dbeeUSCA.nc'
#     # smaller south-eastern US, small western US and a north-western NA
#     TVpath = user_dir + '/surfdrive/output_RGCPD/circulation_US_HW/one-point-corr_maps_clusters/tf30_nc8_dendo_5dbeeUSCA.nc'
#     # mx2t small western US, large eastern US
#     # TVpath = user_dir+'/surfdrive/output_RGCPD/circulation_US_HW/one-point-corr_maps_clusters/tf10_nc5_dendo_0a6f6USCA.nc'
#     if west_east == 'east':
#         cluster_label = 1
#     elif west_east == 'west':
#         cluster_label = 5
#         # cluster_label = 4
#         cluster_label = 2



path_out_main = os.path.join(main_dir,
                             'publications/NPJ_2021/output/heatwave_circulation_v300_z500_SST/{}'.format(TVpath.split('_')[-1][:-3]))





name_ds='ts'
start_end_TVdate = ('06-01', '08-31')
start_end_date = None
method='ranstrat_10' ; seed = 1
tfreq = 15
min_detect_gc=.9
start_end_year = (1979, 2020)

# z500_green_bb = (140,260,20,73) #: Pacific box
if west_east == 'east' or west_east == 'northwest':
    z500_green_bb = (155,300,20,73) #: RW box
    v300_green_bb = (170,359,23,73)
elif west_east == 'west':
    z500_green_bb = (145,325,20,62)
    v300_green_bb = (100,330,24,70)

#%% Circulation vs temperature
list_of_name_path = [(cluster_label, TVpath),
                     ('z500', os.path.join(path_raw, 'z500_1979-2020_1_12_daily_2.5deg.nc')),
                     ('v300', os.path.join(path_raw, 'v300_1979-2020_1_12_daily_2.5deg.nc'))]


lags = np.array([0])

list_for_MI   = [BivariateMI(name='z500', func=class_BivariateMI.corr_map,
                                alpha=.05, FDR_control=True, lags=lags,
                                distance_eps=600, min_area_in_degrees2=5,
                                calc_ts='pattern cov', selbox=(0,360,-10,90),
                                use_sign_pattern=True),
                 BivariateMI(name='v300', func=class_BivariateMI.corr_map,
                                alpha=.05, FDR_control=True, lags=lags,
                                distance_eps=600, min_area_in_degrees2=5,
                                calc_ts='pattern cov', selbox=(0,360,20,90),
                                use_sign_pattern=True)]



rg = RGCPD(list_of_name_path=list_of_name_path,
            list_for_MI=list_for_MI,
            start_end_TVdate=start_end_TVdate,
            start_end_date=start_end_date,
            start_end_year=start_end_year,
            tfreq=tfreq,
            path_outmain=path_out_main)


rg.pp_TV(name_ds=name_ds, detrend=False)

rg.plot_df_clust()

rg.pp_precursors()

rg.traintest(method=method, seed=seed,
             subfoldername=None)



#%%

rg.calc_corr_maps()

#%% Plot corr(z500, mx2t)
import matplotlib
# Optionally set font to Computer Modern to avoid common missing font errors
matplotlib.rc('font', family='serif', serif='cm10')

matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'

# Adjusted box for Reviewer 1, z500_green_bb = (155,255,20,73) #: RW box

title = f'$corr(z500_t, T^{west_east.capitalize()[0]}_t)$'
subtitles = np.array([['']] )
kwrgs_plot = {'row_dim':'lag', 'col_dim':'split', 'aspect':3.8, 'size':2.5,
              'hspace':0.0, 'cbar_vert':-.08, 'units':'Corr. Coeff. [-]',
              'zoomregion':(-180,360,0,80), 'drawbox':[(0,0), z500_green_bb],
              'map_proj':ccrs.PlateCarree(central_longitude=220), 'n_yticks':6,
              'clim':(-.6,.6), 'title':title, 'subtitles':subtitles,
              'title_fontdict':{'y':1.0, 'fontsize':18}}
save = True
# rg.plot_maps_corr(var='z500', save=save,
#                   min_detect_gc=min_detect_gc,
#                   kwrgs_plot=kwrgs_plot,
#                   append_str=''.join(map(str, z500_green_bb))+TV+str(cluster_label))

z500 = rg.list_for_MI[0]
xrvals, xrmask = RGCPD._get_sign_splits_masked(z500.corr_xr,
                                               min_detect_gc,
                                               z500.corr_xr['mask'])
fig = plot_maps.plot_corr_maps(xrvals, xrmask, **kwrgs_plot)

ds = rg.get_clust()
xrclustered = find_precursors.view_or_replace_labels(ds['xrclustered'],
                                                     cluster_label)
fig.axes[0].contour(xrclustered.longitude, xrclustered.latitude,
           np.isnan(xrclustered), transform=ccrs.PlateCarree(),
           levels=[0, 2], linewidths=2, linestyles=['solid'], colors=['white'])
filename = os.path.join(rg.path_outsub1, 'z500vsmx2t_'+
                        rg.hash+'_'+str(cluster_label))

fig.savefig(filename + rg.figext, bbox_inches='tight')

#%% upon request of reviewer, using a smaller bounding box plot
# kwrgs_plot.update({'drawbox':[(0,0), (155,300,20,73)]})
# fig = plot_maps.plot_corr_maps(xrvals, xrmask, **kwrgs_plot)
# fig.axes[0].contour(xrclustered.longitude, xrclustered.latitude,
#            np.isnan(xrclustered), transform=ccrs.PlateCarree(),
#            levels=[0, 2], linewidths=1, linestyles=['solid'], colors=['white'])
# filename = os.path.join(rg.path_outsub1, 'z500vsmx2t_'+
#                         rg.hash+'_'+str(cluster_label)+'small_box')
# fig.savefig(filename + rg.figext, bbox_inches='tight')

#%% Plot corr(v300, mx2t)


kwrgs_plot['title'] = f'$corr(v300_t, T^{west_east.capitalize()[0]}_t)$'
kwrgs_plot['drawbox'] = [(0,0), v300_green_bb]
kwrgs_plot['zoomregion'] = (-180,360,20,90)
kwrgs_plot['cbar_vert'] = -0.025

v200 = rg.list_for_MI[1]
xrvals, xrmask = RGCPD._get_sign_splits_masked(v200.corr_xr,
                                               min_detect_gc,
                                               v200.corr_xr['mask'])
fig = plot_maps.plot_corr_maps(xrvals, xrmask, **kwrgs_plot)

fig.axes[0].contour(xrclustered.longitude, xrclustered.latitude,
                    np.isnan(xrclustered), transform=ccrs.PlateCarree(),
                    levels=[0, 2], linewidths=1, linestyles=['solid'],
                    colors=['white'])
filename = os.path.join(rg.path_outsub1, 'v300vsmx2t_'+
                        rg.hash+'_'+str(cluster_label))

fig.savefig(filename + rg.figext, bbox_inches='tight')



kwrgs_plot['drawbox'] = None
xrvals, xrmask = RGCPD._get_sign_splits_masked(v200.corr_xr,
                                               min_detect_gc,
                                               v200.corr_xr['mask'])
fig = plot_maps.plot_corr_maps(xrvals, xrmask, **kwrgs_plot)

fig.axes[0].contour(xrclustered.longitude, xrclustered.latitude,
                    np.isnan(xrclustered), transform=ccrs.PlateCarree(),
                    levels=[0, 2], linewidths=2, linestyles=['solid'],
                    colors=['white'])
filename = os.path.join(rg.path_outsub1, 'v300vsmx2t_nobox'+
                        rg.hash+'_'+str(cluster_label))

fig.savefig(filename + rg.figext, bbox_inches='tight')
#%%
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

#%% Determine Rossby wave within green rectangle, become target variable for feedback SST
if west_east == 'east':
    if TV == 'USCA':
        west_east_labels = [1,5,4]
        naming = {1:'east', 5:'northwest', 4:'west'}
    elif 'USCAnew':
        # hash 9ad1eUSCA1500
        west_east_labels = [4,1,7]
        naming = {4:'east', 1:'west', 7:'northwest'}
    elif 'init':
        west_east_labels = [1,2]
        naming = {1:'west', 2:'east'}

    # i, label = 0, 1
    list_df_target = []
    for i, label in enumerate(west_east_labels):
        list_of_name_path = [(label, TVpath),
                     ('z500', os.path.join(path_raw, 'z500_1979-2020_1_12_daily_2.5deg.nc')),
                     ('v300', os.path.join(path_raw, 'v300_1979-2020_1_12_daily_2.5deg.nc'))]
        rg.list_of_name_path = list_of_name_path

        rg.pp_TV()
        rg.traintest(method, seed=seed,
                      subfoldername=None)
        if 'east' in naming[label] or 'north' in naming[label]:
            z500_green_bb = (155,300,20,73) #: RW box
            v300_green_bb = (170,359,23,73)
        elif 'west' in naming[label]:
            z500_green_bb = (145,325,20,62)
            v300_green_bb = (100,330,24,70)
        rg.list_for_MI = [BivariateMI(name='z500', func=class_BivariateMI.corr_map,
                                        alpha=.05, FDR_control=True,
                                        distance_eps=600, min_area_in_degrees2=5,
                                        calc_ts='pattern cov', selbox=z500_green_bb,
                                        use_sign_pattern=True, lags = np.array([0])),
                          BivariateMI(name='v300', func=class_BivariateMI.corr_map,
                                              alpha=.05, FDR_control=True,
                                              distance_eps=600, min_area_in_degrees2=5,
                                              calc_ts='pattern cov', selbox=v300_green_bb,
                                              use_sign_pattern=True, lags=np.array([0]))]
        rg.list_for_EOFS = None
        rg.calc_corr_maps()
        rg.plot_maps_corr(save=True)
        rg.cluster_list_MI()
        rg.get_ts_prec(precur_aggr=1)
        if i == 0:
            df_data = rg.df_data.copy()
            df_data = df_data.rename({'0..0..z500_sp':naming[label]+'RW',
                                      '0..0..v300_sp':naming[label]+'RWv300',
                                      f'{label}ts':'mx2t'+naming[label]}, axis=1)
            print(df_data.columns)
        else:
            df_app = rg.df_data.copy().rename({'0..0..z500_sp':naming[label]+'RW',
                                               '0..0..v300_sp':naming[label]+'RWv300',
                                               f'{label}ts':'mx2t'+naming[label]}, axis=1)
            list_df_target.append(df_app)
            print(df_app.columns)
            df_data = rg.merge_df_on_df_data(df_app, df_data)


    cols = ['eastRW', 'westRW', 'northwestRW', 'mx2teast', 'mx2twest', 'mx2tnorthwest' ]
    df_ana.plot_ts_matric(df_data, win=15, columns=cols,
                          period='RV_mask', plot_sign_stars=False, fontsizescaler=-8)
    filepath = os.path.join(rg.path_outsub1, '15d_z500_'+'-'.join(map(str, z500_green_bb))+rg.hash)
    plt.savefig(filepath+'.png', dpi=200, bbox_inches='tight')

    df_ana.plot_ts_matric(df_data, win=30, columns=cols,
                          period='RV_mask', plot_sign_stars=False, fontsizescaler=-8)
    filepath = os.path.join(rg.path_outsub1, '30d_z500_'+'-'.join(map(str, z500_green_bb))+rg.hash)
    plt.savefig(filepath+'.png', dpi=200, bbox_inches='tight')


    filepath = os.path.join(path_out_main, 'z500_'+'-'.join(map(str, z500_green_bb))+rg.hash)
    functions_pp.store_hdf_df({'df_data':df_data}, filepath+'.h5')



#%% SST vs T
list_of_name_path = [(cluster_label, TVpath),
                     ('sst', os.path.join(path_raw, 'sst_1979-2020_1_12_daily_1.0deg.nc'))]

lags = np.array([0,2])

list_for_MI   = [BivariateMI(name='sst', func=class_BivariateMI.corr_map,
                                alpha=.05, FDR_control=True, lags=lags,
                                distance_eps=600, min_area_in_degrees2=1,
                                calc_ts='pattern cov', selbox=(120,260,-10,90))]


# start_end_TVdate = ('07-01', '08-31')
rg = RGCPD(list_of_name_path=list_of_name_path,
            list_for_MI=list_for_MI,
            start_end_TVdate=start_end_TVdate,
            start_end_date=('03-01', '08-31'),
            start_end_year=start_end_year,
            tfreq=tfreq,
            path_outmain=path_out_main)


rg.pp_TV(name_ds=name_ds, detrend=False)
rg.traintest(method, seed=seed,
             subfoldername=None)
rg.pp_precursors()
rg.calc_corr_maps()


#%% Plot corr(SST, T)
import matplotlib
# Optionally set font to Computer Modern to avoid common missing font errors
matplotlib.rc('font', family='serif', serif='cm10')
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'

save=True
SST_green_bb = (140,235,20,59)#(170,255,11,60)
# subtitles = np.array([[f'lag {l}: SST vs E-U.S. mx2t' for l in rg.lags]])
title = r'$corr(SST_{t-lag},$'+f'$T^{west_east.capitalize()[0]}_t)$'
subtitles = np.array([['lag 0', 'lag 2 (15-day gap)']] )
kwrgs_plot = {'row_dim':'split', 'col_dim':'lag','aspect':2, 'hspace':-.47,
              'wspace':-.15, 'size':3, 'cbar_vert':-.1,
              'units':'Corr. Coeff. [-]', 'zoomregion':(130,260,-10,60),
              'clim':(-.6,.6), 'map_proj':ccrs.PlateCarree(central_longitude=220),
              'n_yticks':6, 'x_ticks':np.arange(130, 280, 25),
              'subtitles':subtitles, 'title':title,
              'title_fontdict':{'y':1.0, 'fontsize':18}}
rg.plot_maps_corr(var='sst', save=save,
                  min_detect_gc=min_detect_gc,
                  kwrgs_plot=kwrgs_plot,
                  append_str=rg.hash+'_'+str(cluster_label))

#%%
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
#%%





#%% Quick forecast from SST
import func_models as fc_utils

sst = rg.list_for_MI[0]
sst.calc_ts = 'region mean'
rg.cluster_list_MI()
rg.get_ts_prec()
#%%
fc_mask = rg.df_data.iloc[:,-1].loc[0]#.shift(lag, fill_value=False)
# rg.df_data = rg._replace_RV_mask(rg.df_data, replace_RV_mask=(fc_mask))
target_ts = rg.df_data.iloc[:,[0]].loc[0][fc_mask]
target_ts = (target_ts - target_ts.mean()) / target_ts.std()
alphas = np.append(np.logspace(.1, 1.5, num=25), [250])
kwrgs_model = {'scoring':'neg_mean_squared_error',
               'alphas':alphas, # large a, strong regul.
               'normalize':False}

keys = [k for k in rg.df_data.columns[:-2] if k not in [rg.TV.name, 'PDO']]
keys = [k for k in keys if int(k.split('..')[0]) in [2]]
# keys = [k for k in keys if int(k.split('..')[1]) in [1,3]]

out_fit = rg.fit_df_data_ridge(target=target_ts,tau_min=2, tau_max=2,
                               keys=keys,
                               kwrgs_model=kwrgs_model)
predict, weights, models_lags = out_fit
df_train_m, df_test_s_m, df_test_m, df_boot = fc_utils.get_scores(predict,
                                                                  score_func_list=[fc_utils.corrcoef, fc_utils.ErrorSkillScore(0).RMSE])
print(df_test_m)



    # rg.store_df(append_str='z500_'+'-'.join(map(str, z500_green_bb))+TV+str(cluster_label))