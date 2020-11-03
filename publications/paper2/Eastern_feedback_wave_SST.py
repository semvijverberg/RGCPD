#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 15:33:52 2020

@author: semvijverberg
"""

import os, inspect, sys
if sys.platform == 'linux':
    import matplotlib as mpl
    mpl.use('Agg')
import numpy as np
import cartopy.crs as ccrs
import argparse
import csv

user_dir = os.path.expanduser('~')
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-2])
RGCPD_func = os.path.join(main_dir, 'RGCPD')
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

import functions_pp

periods = ['summer_center', 'summer_shiftright', 'summer_shiftleft',
           'spring_center', 'spring_shiftleft', 'spring_shiftright']

periods = ['summer_shiftleft']
targets = ['west', 'east']
seeds = np.array([1,2,3])
combinations = np.array(np.meshgrid(targets, seeds, periods)).T.reshape(-1,3)

i_default = 3



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
    west_east = out[0]
    seed = int(out[1])
    period = out[2]
    print(f'arg {args.intexper} - {out}')
else:
    seed = 0



if west_east == 'east':
    TVpathRW = os.path.join(data_dir, '2020-10-29_13hr_45min_east_RW.h5')
elif west_east =='west':
    TVpathRW = os.path.join(data_dir, '2020-10-29_10hr_58min_west_RW.h5')


path_out_main = os.path.join(main_dir, f'publications/paper2/output/{west_east}/')
name_or_cluster_label = 'z500'
name_ds = f'0..0..{name_or_cluster_label}_sp'
if period == 'summer_center':
    start_end_TVdate = ('06-01', '08-31')
elif period == 'summer_shiftleft':
    start_end_TVdate = ('05-25', '08-24')
elif period == 'summer_shiftright':
    start_end_TVdate = ('06-08', '09-06')
elif period == 'spring_center':
    start_end_TVdate = ('02-01', '05-31')
elif period == 'spring_shiftleft':
    start_end_TVdate = ('01-25', '05-24')
elif period == 'spring_shiftright':
    start_end_TVdate = ('02-08', '06-06')

start_end_date = ('1-1', '12-31')

tfreq         = 15
min_detect_gc = 1.0
method        = 'ran_strat10' ;

name_MCI_csv = 'strength.csv'
name_rob_csv = 'robustness.csv'

if tfreq > 15: sst_green_bb = (140,240,-9,59) # (180, 240, 30, 60): original warm-code focus
if tfreq <= 15: sst_green_bb = (140,235,20,59) # same as for West
#%%

# list_of_name_path = [(name_or_cluster_label, TVpathRW),
#                      ('z500', os.path.join(path_raw, 'z500hpa_1979-2018_1_12_daily_2.5deg.nc')),
#                      ('N-Pac. SST', os.path.join(path_raw, 'sst_1979-2018_1_12_daily_1.0deg.nc'))]
#                      # ('Trop. Pac. SST', os.path.join(path_raw, 'sst_1979-2018_1_12_daily_1.0deg.nc'))]


# list_for_MI   = [BivariateMI(name='z500', func=class_BivariateMI.corr_map,
#                                 alpha=.05, FDR_control=True,
#                                 distance_eps=600, min_area_in_degrees2=5,
#                                 calc_ts='pattern cov', selbox=(-180,360,-10,90),
#                                 use_sign_pattern=True, lags=np.array([0])),
#                  BivariateMI(name='N-Pac. SST', func=class_BivariateMI.parcorr_map_time,
#                               alpha=.05, FDR_control=True,
#                               distance_eps=500, min_area_in_degrees2=5,
#                               calc_ts='pattern cov', selbox=(130,260,-10,90),
#                               lags=np.array([0]))]
#                  # BivariateMI(name='Trop. Pac. SST', func=BivariateMI.corr_map,
#                  #              kwrgs_func={'alpha':.01, 'FDR_control':True},
#                  #              distance_eps=500, min_area_in_degrees2=5,
#                  #              calc_ts='pattern cov', selbox=selbox)]



# # list_import_ts = [('versusmx2t', TVpathall)]

# rg = RGCPD(list_of_name_path=list_of_name_path,
#            list_for_MI=list_for_MI,
#            list_import_ts=None,
#            start_end_TVdate=start_end_TVdate,
#            start_end_date=start_end_date,
#            tfreq=tfreq,
#            path_outmain=path_out_main)


# rg.pp_TV(name_ds=name_ds)

# rg.pp_precursors(anomaly=True)


# rg.traintest(method=method, seed=seed)

# rg.calc_corr_maps()



# # subtitles = np.array([['SST vs spat. covariance Rossby wave (z500)']])

# # rg.plot_maps_corr(var='Pacific SST',
# #                   aspect=2, size=5, cbar_vert=.19, save=True,
# #                   subtitles=subtitles, units=units, zoomregion=(-180,360,10,75),
# #                   map_proj=ccrs.PlateCarree(central_longitude=220), n_yticks=5,
# #                   drawbox=['all', sst_green_bb],
# #                   clim=(-.6,.6))

# save = True
# units = 'Corr. Coeff. [-]'
# subtitles = np.array([['SST vs eastern RW']])
# kwrgs_plot = {'row_dim':'split', 'col_dim':'lag',
#               'aspect':2, 'hspace':-.57, 'wspace':-.22, 'size':2, 'cbar_vert':-.02,
#               'subtitles':subtitles, 'units':units, 'zoomregion':(130,260,-10,60),
#               'map_proj':ccrs.PlateCarree(central_longitude=220),
#               'x_ticks':np.array([]), 'y_ticks':np.array([]),
#               'drawbox':[(0,0), sst_green_bb],
#               'clim':(-.6,.6)}
# rg.plot_maps_corr(var='N-Pac. SST', save=save, min_detect_gc=min_detect_gc,
#                   kwrgs_plot=kwrgs_plot)


# # sst_tropbox = (140, 250, 0, 30)
# # subtitles = np.array([[f'lag {l}: SST vs Rossby wave ({name_or_cluster_label})'] for l in rg.lags])
# # units = 'Corr. Coeff. [-]'
# # rg.plot_maps_corr(var='Trop. Pac. SST', row_dim='lag', col_dim='split',
# #                   aspect=2, hspace=-.57, size=5, cbar_vert=.175, save=save,
# #                   subtitles=subtitles, units=units, #zoomregion=(-180,360,-10,70),
# #                   map_proj=ccrs.PlateCarree(central_longitude=220), n_yticks=6,
# #                   drawbox=['all', sst_tropbox], clim=(-.6,.6),
# #                   append_str=''.join(map(str, sst_tropbox)))

# z500_green_bb = (155, 310, 10, 80)
# precur = rg.list_for_MI[0]
# subtitles = np.array([[f'lag {l}: z 500hpa vs Rossby wave ({name_or_cluster_label})'] for l in precur.lags])
# # row_dim='lag', col_dim='split',
# kwrgs_plot.update({'size':5, 'cbar_vert':.175, 'subtitles':subtitles,
#                    'zoomregion':(-180,360,10,80),
#                    'drawbox':['all', z500_green_bb]})
# rg.plot_maps_corr(var='z500', save=save, min_detect_gc=min_detect_gc,
#                   append_str=''.join(map(str, z500_green_bb)),
#                   kwrgs_plot=kwrgs_plot)



#%% Only SST
list_of_name_path = [(name_or_cluster_label, TVpathRW),
                     ('N-Pac. SST', os.path.join(path_raw, 'sst_1979-2018_1_12_daily_1.0deg.nc'))]

list_for_MI = [BivariateMI(name='N-Pac. SST', func=class_BivariateMI.corr_map,
                           alpha=.05, FDR_control=True,
                           distance_eps=500, min_area_in_degrees2=5,
                           calc_ts='pattern cov', selbox=sst_green_bb,
                           lags=np.array([0]))]

rg = RGCPD(list_of_name_path=list_of_name_path,
           list_for_MI=list_for_MI,
           list_import_ts=None,
           start_end_TVdate=start_end_TVdate,
           start_end_date=start_end_date,
           tfreq=tfreq,
           path_outmain=path_out_main)

rg.pp_TV(name_ds=name_ds)
rg.pp_precursors(anomaly=True)
RV_name_range = '{}-{}'.format(*list(rg.start_end_TVdate))
subfoldername = 'RW_SST_fb_{}_{}s{}'.format(RV_name_range,
                                                  method, seed)
rg.traintest(method=method, seed=seed, subfoldername=subfoldername)

rg.calc_corr_maps(var='N-Pac. SST')
rg.cluster_list_MI(var='N-Pac. SST')
rg.quick_view_labels(median=True)
# rg.get_ts_prec(precur_aggr=1)
# rg.store_df(append_str=f'RW_and_SST_fb_tf{rg.tfreq}')

def append_MCI(rg, dict_v, dict_rb):
    dkeys = [f'{f}-d', f'{f}-d SST->RW', f'{f}-d RW->SST']

    rg.PCMCI_get_links(var=keys[0], alpha_level=.01) # links toward RW
    SSTtoRW = rg.df_MCIc.mean(0,level=1).loc['SST'].iloc[1:].max().round(3) # select SST
    rg.PCMCI_get_links(var=keys[1], alpha_level=.01) # links toward SST
    RWtoSST = rg.df_MCIc.mean(0,level=1).loc[f'{west_east[0].capitalize()}-RW'].iloc[1:].max().round(3) # select RW
    lag0 = rg.df_MCIc.mean(0,level=1).loc[f'{west_east[0].capitalize()}-RW']['coeff l0'].round(3)
    append_dict = {dkeys[0]:lag0, dkeys[1]:SSTtoRW, dkeys[2]:RWtoSST}
    dict_v.update(append_dict)

    robustness = wPCMCI.get_traintest_links(rg.pcmci_dict,
                                     rg.parents_dict,
                                     rg.pcmci_results_dict,
                                     min_link_robustness=mlr)[2]
    rblag0 = int(robustness[0][1][0])
    rbSSTtoRW = int(max(robustness[1][0][1:])) # from i to j, SST to RW
    rbRWtoSST = int(max(robustness[0][1][1:])) # from i to j, RW to SST
    append_dict = {dkeys[0]:rblag0, dkeys[1]:rbSSTtoRW, dkeys[2]:rbRWtoSST}
    dict_rb.update(append_dict)
    return SSTtoRW, rbRWtoSST, rbSSTtoRW

#%%
import wrapper_PCMCI as wPCMCI

dict_v = {'period':period.split('_')[0],'Experiment':'s{}_{}'.format(rg.kwrgs_TV['seed'], period)}
dict_rb = dict_v.copy()
freqs = [1, 5, 10, 15, 30, 60]
for f in freqs[:]:
    rg.get_ts_prec(precur_aggr=f)
    rg.df_data = rg.df_data.rename({'z5000..0..z500_sp':f'{west_east[0].capitalize()}-RW',
                                    '0..0..N-Pac. SST_sp':'SST'}, axis=1)

    keys = [f'{west_east[0].capitalize()}-RW','SST']
    rg.PCMCI_df_data(keys=keys,
                      pc_alpha=None,
                      tau_max=5,
                      max_conds_dim=10,
                      max_combinations=10)


    lags = range(rg.kwrgs_pcmci['tau_min'], rg.kwrgs_pcmci['tau_max'])
    lags = np.array([l*f for i, l in enumerate(lags)])
    mlr=5
    SSTtoRW, rbRWtoSST, rbSSTtoRW = append_MCI(rg, dict_v, dict_rb)
    #%%
    rg.PCMCI_plot_graph(min_link_robustness=mlr, figshape=(12,6),
                        kwrgs={'vmax_nodes':.9,
                               'node_aspect':80,
                               'node_size':.008,
                               'node_ticks':.3,
                               'node_label_size':40,
                               'vmax_edges':.6,
                               'vmin_edges':0,
                               'cmap_edges':'plasma_r',
                               'edge_ticks':.2,
                               'lag_array':lags,
                               'curved_radius':.5,
                               'arrowhead_size':1000,
                               'link_label_fontsize':30,
                               'label_fontsize':10,
                               'weights_squared':1.5},
                        append_figpath=f'_tf{rg.precur_aggr}_{SSTtoRW}_rb{mlr}_rbRWSST{rbRWtoSST}_rbSSTRW{rbSSTtoRW}')
    #%%
    rg.PCMCI_get_links(var=keys[1], alpha_level=.01)
    rg.df_links.astype(int).sum(0, level=1)
    MCI_ALL = rg.df_MCIc.mean(0, level=1)
#%%
# write MCI strength

csvfilenameMCI = os.path.join(rg.path_outmain, name_MCI_csv)
csvfilenamerobust = os.path.join(rg.path_outmain, name_rob_csv)
for csvfilename, dic in [(csvfilenameMCI, dict_v), (csvfilenamerobust, dict_rb)]:
    # create .csv if it does not exists
    if os.path.exists(csvfilename) == False:
        with open(csvfilename, 'a', newline='') as csvfile:

            writer = csv.DictWriter(csvfile, list(dic.keys()))
            writer.writerows([{f:f for f in list(dic.keys())}])

    # write robustness
    with open(csvfilename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, list(dic.keys()))
        writer.writerows([dic])

    # # write robustness
    # with open(csvfilenamerobust, 'a', newline='') as csvfile:

    #     writer = csv.DictWriter(csvfile, list(dict_v.keys()))
    #     writer.writerows([dict_v])
#%%
# import func_models

# shift = 2
# mask_standardize = np.logical_and(rg.df_data.loc[0]['TrainIsTrue'], rg.df_data.loc[0]['RV_mask'])
# df = func_models.standardize_on_train(rg.df_data.loc[0], mask_standardize)
# RV_and_SST_mask = np.logical_and(rg.df_data.loc[0]['RV_mask'], df['N-Pacific SST'].shift(-shift) > .5)
# fig = df[RV_and_SST_mask][keys].hist(sharex=True)
# fig[0,0].set_xlim(-3,3)

#%% Adapt RV_mask
import matplotlib.pyplot as plt

quantilethreshold = .66
freqs = [1, 15, 30, 60]
for f in freqs:
    rg.get_ts_prec(precur_aggr=f)
    rg.df_data = rg.df_data.rename({'z5000..0..z500_sp':f'{west_east[0].capitalize()}-RW',
                                    '0..0..N-Pac. SST_sp':'SST'}, axis=1)

    keys = [f'{west_east[0].capitalize()}-RW','SST']

    # when both SST and RW above threshold
    RW_ts = rg.df_data.loc[0].iloc[:,0]
    RW_mask = RW_ts > float(rg.TV.RV_ts.quantile(q=quantilethreshold))
    new_mask = np.logical_and(rg.df_data.loc[0]['RV_mask'], RW_mask)
    sst = functions_pp.get_df_test(rg.df_data, cols=['SST'])
    sst_mask = (sst > sst.quantile(q=quantilethreshold).values).squeeze()
    new_mask = np.logical_and(sst_mask, new_mask)
    sumyears = new_mask.groupby(new_mask.index.year).sum()
    sumyears = list(sumyears.index[sumyears > 25])
    RV_mask = rg.df_data.loc[0]['RV_mask']
    m = np.array([True if y in sumyears else False for y in RV_mask.index.year])
    new_mask = np.logical_and(m, RV_mask)
    try:
        new_mask.astype(int).plot()
        plt.savefig(os.path.join(rg.path_outsub1, 'subset_dates_SST_and_RW.pdf'))
    except:
        pass
    print(f'{new_mask[new_mask].size} datapoints')

    # when both SST is anomalous
    RW_ts = rg.df_data.loc[0].iloc[:,0]
    RW_mask = RW_ts > float(rg.TV.RV_ts.quantile(q=quantilethreshold))
    new_mask = np.logical_and(rg.df_data.loc[0]['RV_mask'], RW_mask)
    sst = functions_pp.get_df_test(rg.df_data, cols=['SST'])
    sst_mask = (sst > sst.quantile(q=quantilethreshold).values).squeeze()
    new_mask = np.logical_and(sst_mask, new_mask)
    sumyears = new_mask.groupby(new_mask.index.year).sum()
    sumyears = list(sumyears.index[sumyears > 25])
    RV_mask = rg.df_data.loc[0]['RV_mask']
    m = np.array([True if y in sumyears else False for y in RV_mask.index.year])
    new_mask = np.logical_and(m, RV_mask)
    try:
        new_mask.astype(int).plot()
        plt.savefig(os.path.join(rg.path_outsub1, 'subset_dates_SST_and_RW.pdf'))
    except:
        pass
    print(f'{new_mask[new_mask].size} datapoints')

    rg.PCMCI_df_data(keys=keys,
                     replace_RV_mask=new_mask.values,
                     pc_alpha=None,
                     tau_max=5,
                     max_conds_dim=10,
                     max_combinations=10)

    rg.PCMCI_plot_graph(min_link_robustness=5, figshape=(8,2),
                        kwrgs={'vmax_nodes':1.0,
                               'vmax_edges':.6,
                               'vmin_edges':-.6,
                               'node_ticks':.2,
                               'edge_ticks':.1},
                        append_figpath=f'_subset_dates_tf{rg.precur_aggr}')

    rg.PCMCI_get_links(var=keys[1], alpha_level=.01)
    rg.df_links.mean(0, level=1)
    MCI_subset = rg.df_MCIc.mean(0, level=1)




