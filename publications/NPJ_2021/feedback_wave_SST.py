#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 15:33:52 2020

@author: semvijverberg
"""

import os, inspect, sys
import matplotlib as mpl
from matplotlib.colors import ListedColormap
if sys.platform == 'linux':
    mpl.use('Agg')
else:
    # Optionally set font to Computer Modern to avoid common missing font errors
    mpl.rc('font', family='serif', serif='cm10')

    mpl.rc('text', usetex=True)
    mpl.rcParams['text.latex.preamble'] = r'\boldmath'
import numpy as np
import cartopy.crs as ccrs
import argparse
import csv
from time import time

user_dir = os.path.expanduser('~')
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
curr_dir = user_dir + '/surfdrive/Scripts/RGCPD/publications/NPJ_2021'
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
import wrapper_PCMCI as wPCMCI
import functions_pp

periods = ['summer_center', 'summer_shiftright', 'summer_shiftleft',
           'spring_center', 'spring_shiftleft', 'spring_shiftright']

# periods = ['winter_center', 'winter_shiftright', 'winter_shiftleft']

# periods = ['summer_shiftleft']
remove_PDO = False
if remove_PDO:
    targets = ['east']
else:
    targets = ['east', 'west']
seeds = np.array([1,2,3])

combinations = np.array(np.meshgrid(targets, seeds, periods)).T.reshape(-1,3)

i_default = 0



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


# TVpathtemp = os.path.join(data_dir, 'tf15_nc3_dendo_0ff31.nc') # old TV
TVpathtemp = os.path.join(data_dir, 'tfreq15_nc7_dendo_57db0USCA.nc')
# TVpathtemp = user_dir+'/surfdrive/output_RGCPD/circulation_US_HW/one-point-corr_maps_clusters/tf15_nc8_dendo_57db0USCA.nc'
if west_east == 'east':
    # TVpathRW = os.path.join(data_dir, '2020-10-29_13hr_45min_east_RW.h5')
    cluster_label = 4 # 2
    z500_green_bb = (155,300,20,73) # bounding box for eastern RW
elif west_east =='west':
    # TVpathRW = os.path.join(data_dir, '2020-10-29_10hr_58min_west_RW.h5')
    cluster_label = 1 # 8 # 1
    z500_green_bb = (145,325,20,62) # bounding box for western RW


path_out_main = os.path.join(main_dir, f'publications/NPJ_2021/output/{west_east}_fb/')
if period == 'summer_center':
    start_end_TVdate = ('06-01', '08-31')
    start_end_TVdatet2mvsRW = start_end_TVdate
elif period == 'summer_shiftleft':
    start_end_TVdate = ('05-25', '08-24')
    start_end_TVdatet2mvsRW = start_end_TVdate
elif period == 'summer_shiftright':
    start_end_TVdate = ('06-08', '09-06')
    start_end_TVdatet2mvsRW = start_end_TVdate
elif period == 'spring_center':
    start_end_TVdate = ('02-01', '05-31')
    start_end_TVdatet2mvsRW = ('06-01', '08-31') # always focus on RW in summer
elif period == 'spring_shiftleft':
    start_end_TVdate = ('01-25', '05-24')
    start_end_TVdatet2mvsRW = ('05-25', '08-24')
elif period == 'spring_shiftright':
    start_end_TVdate = ('02-08', '06-06')
    start_end_TVdatet2mvsRW = ('06-08', '09-06')
elif period == 'winter_center':
    start_end_TVdate = ('12-01', '02-28')
    start_end_TVdatet2mvsRW = ('06-01', '08-31') # always focus on RW in summer
elif period == 'winter_shiftleft':
    start_end_TVdate = ('11-25', '02-24')
    start_end_TVdatet2mvsRW = ('05-25', '08-24')
elif period == 'winter_shiftright':
    start_end_TVdate = ('12-08', '03-07')
    start_end_TVdatet2mvsRW = ('06-08', '09-06')

if period.split('_') == 'winter':
    start_end_date = None
else:
    start_end_date = ('01-01', '12-31')

# =============================================================================
# CHANGE SED
# =============================================================================
# start_end_date = ('03-01', start_end_TVdatet2mvsRW[-1])

tfreq         = 15
min_detect_gc = 0.9
method        = 'RepeatedKFold_2_10' ;
use_sign_pattern_z500 = True
name_MCI_csv = f'strength_rPDO{remove_PDO}.csv'
name_rob_csv = f'robustness_rPDO{remove_PDO}.csv'

if tfreq > 15: sst_green_bb = (140,240,-9,59) # (180, 240, 30, 60): original warm-code focus
if tfreq <= 15: sst_green_bb = (140,235,20,59) # same as for West

name_or_cluster_label = 'z500'
name_ds = f'0..0..{name_or_cluster_label}_sp'
#%%
# def pipeline(cluster_label, TVpathtemp, seed=1, save = True):
#%% Circulation vs temperature

save = False
list_of_name_path = [(cluster_label, TVpathtemp),
                     ('z500', os.path.join(path_raw, 'z500_1979-2020_1_12_daily_2.5deg.nc'))]


# Adjusted box upon request Reviewer 1:
# z500_green_bb = (155,255,20,73) #: RW box
# use_sign_pattern_z500 = False

list_for_MI   = [BivariateMI(name='z500', func=class_BivariateMI.corr_map,
                            alpha=.05, FDR_control=True,
                            distance_eps=600, min_area_in_degrees2=5,
                            calc_ts='pattern cov', selbox=z500_green_bb,
                            use_sign_pattern=use_sign_pattern_z500,
                            lags = np.array([0]), n_cpu=2)]

rg = RGCPD(list_of_name_path=list_of_name_path,
            list_for_MI=list_for_MI,
            start_end_TVdate=start_end_TVdatet2mvsRW,
            start_end_date=start_end_date,
            start_end_year=None,
            tfreq=tfreq,
            path_outmain=path_out_main)


rg.pp_TV(detrend=False)
rg.plot_df_clust(save=save)
rg.pp_precursors(anomaly=True, detrend=True)
RV_name_range = '{}-{}'.format(*list(rg.start_end_TVdate))
subfoldername = 'RW_SST_fb_{}_{}s{}'.format(RV_name_range, method, seed)
rg.traintest(method=method, seed=seed, subfoldername=subfoldername)

start_time = time()
rg.calc_corr_maps('z500')
print(f'End time: {int(time() - start_time)}')

rg.cluster_list_MI(['z500'])

start_time = time()
rg.get_ts_prec(precur_aggr=1)
print(f'End time: {int(time() - start_time)}')
#%%
TVpathRW = os.path.join(data_dir, f'{west_east}RW_{period}_s{seed}')
rg.store_df(filename=TVpathRW)


# Optionally set font to Computer Modern to avoid common missing font errors
# mpl.rc('font', family='serif', serif='cm10')

# matplotlib.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = r'\boldmath'



title = f'$corr(z500, {west_east.capitalize()[0]}$-$US\ mx2t)$'
subtitles = np.array([['']] )
kwrgs_plot = {'row_dim':'lag', 'col_dim':'split', 'aspect':3.8, 'size':2.5,
              'hspace':0.0, 'cbar_vert':-.08, 'units':'Corr. Coeff. [-]',
              'zoomregion':(135,330,15,80), 'drawbox':[(0,0), z500_green_bb],
              'map_proj':ccrs.PlateCarree(central_longitude=220), 'n_yticks':6,
              'clim':(-.6,.6), 'title':title, 'subtitles':subtitles}

rg.plot_maps_corr(var='z500', save=save,
                  append_str=f'vs{cluster_label}T'+''.join(map(str, z500_green_bb)),
                  min_detect_gc=min_detect_gc,
                  kwrgs_plot=kwrgs_plot)

#%% RW timeseries vs SST and RW timeseries vs RW
TVpathRW = os.path.join(data_dir, f'{west_east}RW_{period}_s{seed}')
list_of_name_path = [(name_or_cluster_label, TVpathRW+'.h5'),
                      ('z500', os.path.join(path_raw, 'z500_1979-2020_1_12_daily_2.5deg.nc')),
                      ('N-Pac. SST', os.path.join(path_raw, 'sst_1979-2020_1_12_daily_1.0deg.nc'))]
                      # ('Trop. Pac. SST', os.path.join(path_raw, 'sst_1979-2018_1_12_daily_1.0deg.nc'))]


list_for_MI   = [BivariateMI(name='z500', func=class_BivariateMI.corr_map,
                                alpha=.05, FDR_control=True,
                                distance_eps=600, min_area_in_degrees2=5,
                                calc_ts='pattern cov', selbox=(-180,360,-10,90),
                                use_sign_pattern=True, lags=np.array([0])),
                  BivariateMI(name='N-Pac. SST', func=class_BivariateMI.corr_map,
                              alpha=.05, FDR_control=True,
                              distance_eps=500, min_area_in_degrees2=5,
                              calc_ts='pattern cov', selbox=(130,260,-10,90),
                              lags=np.array([0]))]
list_import_ts = [('RW', TVpathRW+'.h5')]

rg = RGCPD(list_of_name_path=list_of_name_path,
            list_for_MI=list_for_MI,
            list_import_ts=list_import_ts,
            start_end_TVdate=start_end_TVdate,
            start_end_date=start_end_date,
            tfreq=tfreq,
            path_outmain=path_out_main)

rg.pp_TV(name_ds=name_ds)
rg.pp_precursors(anomaly=True, detrend=True)
RV_name_range = '{}-{}'.format(*list(rg.start_end_TVdate))
subfoldername = 'RW_SST_fb_{}_{}s{}'.format(RV_name_range, method, seed)
rg.traintest(method=method, seed=seed, subfoldername=subfoldername)
rg.get_ts_prec(keys_ext=['0..0..z500_sp'])
rg.calc_corr_maps(df_RVfull = rg.df_data[['z5000..0..z500_sp']])


#%%
units = 'Corr. Coeff. [-]'
subtitles = np.array([[f'$corr(SST_t,\ RW^{west_east[0].capitalize()}_t)$']])
kwrgs_plot = {'row_dim':'split', 'col_dim':'lag',
              'aspect':2, 'hspace':-.57, 'wspace':-.22, 'size':2.5, 'cbar_vert':-.02,
              'subtitles':subtitles, 'units':units, 'zoomregion':(130,260,-10,60),
              'map_proj':ccrs.PlateCarree(central_longitude=220),
              'x_ticks':np.array([]), 'y_ticks':np.array([]),
              'drawbox':[(0,0), sst_green_bb],
              'clevels':np.arange(-.6,.61,.075),
              'clabels':np.arange(-.6,.61,.3),
              'subtitle_fontdict':{'x':0.5, 'y':2}}
rg.plot_maps_corr(var='N-Pac. SST', save=save, min_detect_gc=min_detect_gc,
                  kwrgs_plot=kwrgs_plot, append_str='')

#%% Plot corr map versus z500


precur = rg.list_for_MI[0]
subtitles = np.array([[f'$corr(z500_t,\ RW^{west_east[0].capitalize()}_t)$']])
kwrgs_plot.update({'size':5, 'cbar_vert':.19, 'subtitles':subtitles,
                    'zoomregion':(-180,360,10,80),
                    'drawbox':['all', z500_green_bb]})
rg.plot_maps_corr(var='z500', save=save, min_detect_gc=min_detect_gc,
                  append_str=''.join(map(str, z500_green_bb)),
                  kwrgs_plot=kwrgs_plot)



#%% Only SST
# TVpathRW = os.path.join(data_dir, f'{west_east}RW_{period}_s{seed}')
# list_of_name_path = [(name_or_cluster_label, TVpathRW+'.h5'),
#                       ('N-Pac. SST', os.path.join(path_raw, 'sst_1979-2020_1_12_daily_1.0deg.nc'))]

# list_for_MI = [BivariateMI(name='N-Pac. SST', func=class_BivariateMI.corr_map,
#                             alpha=.05, FDR_control=True,
#                             distance_eps=500, min_area_in_degrees2=5,
#                             calc_ts='pattern cov', selbox=sst_green_bb,
#                             lags=np.array([0]))]

# rg = RGCPD(list_of_name_path=list_of_name_path,
#             list_for_MI=list_for_MI,
#             list_import_ts=None,
#             start_end_TVdate=start_end_TVdate,
#             start_end_date=start_end_date,
#             tfreq=tfreq,
#             path_outmain=path_out_main)

# rg.pp_TV(name_ds=name_ds)
# rg.pp_precursors(anomaly=True)
# RV_name_range = '{}-{}'.format(*list(rg.start_end_TVdate))
# subfoldername = 'RW_SST_fb_{}_{}s{}'.format(RV_name_range,
#                                                   method, seed)
# rg.traintest(method=method, seed=seed, subfoldername=subfoldername)

# rg.calc_corr_maps(var='N-Pac. SST')
rg.list_for_MI = [rg.list_for_MI[1]]
rg.cluster_list_MI(var='N-Pac. SST')
rg.quick_view_labels(min_detect_gc=min_detect_gc)
# rg.get_ts_prec(precur_aggr=1)
# rg.store_df(append_str=f'RW_and_SST_fb_tf{rg.tfreq}')


#%%
def append_MCI(rg, dict_v, dict_rb, alpha_level=.05):
    dkeys = [f'{f}-d', f'{f}-d SST->RW', f'{f}-d RW->SST']

    rg.PCMCI_get_links(var=keys[0], alpha_level=alpha_level) # links toward RW
    SSTtoRW = rg.df_MCIc.mean(0,level=1).loc[keys[1]].iloc[1:].max().round(3) # select SST
    rg.PCMCI_get_links(var=keys[1], alpha_level=alpha_level) # links toward SST
    RWtoSST = rg.df_MCIc.mean(0,level=1).loc[keys[0]].iloc[1:].max().round(3) # select RW
    lag0 = rg.df_MCIc.mean(0,level=1).loc[keys[0]]['coeff l0'].round(3)
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


if remove_PDO:
    lowpass = '2'
    keys_ext=[f'PDO{lowpass}bw']
    rg.list_import_ts = [('PDO', os.path.join(data_dir, 'df_PDOs.h5'))]
else:
    keys_ext = None

alpha_level = .05
dict_v = {'Target':west_east, 'Period':period,'Seed':'s{}'.format(rg.kwrgs_traintest['seed'])}
dict_rb = dict_v.copy()
freqs = [1, 5, 10, 15, 30, 60, 90]
for f in freqs[:]:
    rg.get_ts_prec(precur_aggr=f, keys_ext=keys_ext)
    keys = [f'$RW^{west_east[0].capitalize()}$',
            f'$SST^{west_east[0].capitalize()}$']
    rg.df_data = rg.df_data.rename({'z5000..0..z500_sp':keys[0],
                                    '0..0..N-Pac. SST_sp':keys[1]}, axis=1)




    if remove_PDO:
        rg.df_data[keys], fig = wPCMCI.df_data_remove_z(rg.df_data.copy(), z=['PDO'],
                                                         keys=keys,
                                                         standardize=False,
                                                         plot=True)
        fig_path = os.path.join(rg.path_outsub1, f'regressing_out_PDO_tf{f}')
        fig.savefig(fig_path+rg.figext, bbox_inches='tight')

    if f <= 5:
        tau_max = 5 ; n_cpu = 3
    elif f == 10:
        tau_max = 4 ; n_cpu = 2
    elif f == 15:
        tau_max = 3 ; n_cpu = 1
    elif f == 30:
        tau_max = 2 ; n_cpu = 1
    elif f == 60:
        tau_max = 1 ; n_cpu = 1

    kwrgs_tigr = {'tau_min':0, 'tau_max':tau_max, 'max_conds_dim':10,
                  'pc_alpha':0.05, 'max_combinations':10} # pc_alpha=None
    # start_time = time()
    rg.PCMCI_df_data(keys=keys,
                      kwrgs_tigr=kwrgs_tigr, n_cpu=n_cpu)
    # print(f'{int(time() - start_time)}')


    lags = range(rg.kwrgs_tigr['tau_min'], rg.kwrgs_tigr['tau_max']+1)
    lags = np.array([l*f for i, l in enumerate(lags)])
    mlr=9
    SSTtoRW, rbRWtoSST, rbSSTtoRW = append_MCI(rg, dict_v, dict_rb, alpha_level)
    AR1SST = rg.df_MCIc.mean(0,level=1).loc[keys[1]]['coeff l1'].round(2)

    # my_cmap = matplotlib.colors.ListedColormap(
    #     ["#f94144","#f3722c","#f8961e","#f9c74f","#90be6d","#43aa8b"][::-1])
    cmap_edges = ListedColormap(
        ["#8D0801","#bc4749", "#fb8500","#ffb703","#a7c957", "#b5dda4"][::-1])
    cmap_nodes = ["#9d0208",
                  "#dc2f02","#e85d04","#f48c06","#faa307", "#ffba08"][::-1]
    cmap_nodes = ListedColormap(cmap_nodes)

    rg.PCMCI_plot_graph(min_link_robustness=mlr, figshape=(10.5,4),
                        kwrgs={'vmax_nodes':.9,
                                'node_aspect':130,
                                'node_size':.008,
                                'node_ticks':.3,
                                'node_label_size':50,
                                'vmax_edges':.6,
                                'vmin_edges':0,
                                'cmap_edges':cmap_edges,
                                'cmap_nodes':cmap_nodes,
                                'edge_ticks':.2,
                                'lag_array':lags,
                                'curved_radius':.5,
                                'arrowhead_size':100000,
                                'link_label_fontsize':35,
                                'link_colorbar_label':'Link strength',
                                'node_colorbar_label':'Auto-strength',
                                'label_fontsize':15,
                                'weights_squared':1.5,
                                'network_lower_bound':.25},
                        append_figpath=f'_tf{rg.precur_aggr}_{AR1SST}_rb{mlr}_taumax{tau_max}_rPDO{remove_PDO}')
    #%%
    rg.PCMCI_get_links(var=keys[1], alpha_level=alpha_level)
    rg.df_links.astype(int).sum(0, level=1)
    MCI_ALL = rg.df_MCIc.mean(0, level=1)
    # return rg

#%%
# write MCI strength and robustness to csv

csvfilenameMCI = os.path.join(rg.path_outmain, name_MCI_csv)
csvfilenamerobust = os.path.join(rg.path_outmain, name_rob_csv)
for csvfilename, dic in [(csvfilenameMCI, dict_v), (csvfilenamerobust, dict_rb)]:
    # create .csv if it does not exists
    if os.path.exists(csvfilename) == False:
        with open(csvfilename, 'a', newline='') as csvfile:

            writer = csv.DictWriter(csvfile, list(dic.keys()))
            writer.writerows([{f:f for f in list(dic.keys())}])

    # write
    with open(csvfilename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, list(dic.keys()))
        writer.writerows([dic])
#%%
# s = 0
# tig = rg.pcmci_dict[s]
# functions_pp.get_oneyr(rg.dates_all) # dp per yr
# df_s = rg.df_data.loc[s][rg.df_data.loc[s]['TrainIsTrue'].values]
# print(f'{tig.T} total datapoints \ndf_data has shape {df_s.shape}')
# RVfs = rg.df_data.loc[s][np.logical_and(rg.df_data.loc[s]['RV_mask'], rg.df_data.loc[s]['TrainIsTrue']).values]

# print(f'df_data when datamask applied has shape {RVfs.shape}')
# # equal RV mask and tig.dataframe.mask
# all(np.equal(tig.dataframe.mask[:,s], ~rg.df_data.loc[s]['RV_mask'][rg.df_data.loc[s]['TrainIsTrue'].values] ))


# array = tig.dataframe.construct_array([(1,0)], [(0,0)], [(1,-1)], tau_max=5,
#                                       cut_off='max_lag',
#                                       mask=tig.dataframe.mask,
#                                       mask_type=tig.cond_ind_test.mask_type,
#                                       verbosity=3)[0]
# print(f'full array is loaded. array shape {array.shape}, 2*taumax=5 = 10' )


# array = tig.cond_ind_test._get_array([(1,0)], [(0,0)], [(1,-1)], tau_max=5)[0]
# array.shape


# #%%
# # import func_models

# # shift = 2
# # mask_standardize = np.logical_and(rg.df_data.loc[0]['TrainIsTrue'], rg.df_data.loc[0]['RV_mask'])
# # df = func_models.standardize_on_train(rg.df_data.loc[0], mask_standardize)
# # RV_and_SST_mask = np.logical_and(rg.df_data.loc[0]['RV_mask'], df['N-Pacific SST'].shift(-shift) > .5)
# # fig = df[RV_and_SST_mask][keys].hist(sharex=True)
# # fig[0,0].set_xlim(-3,3)

# # #%% Adapt RV_mask
# # import matplotlib.pyplot as plt

# # quantilethreshold = .66
# # freqs = [1, 15, 30, 60]
# # for f in freqs:
# #     rg.get_ts_prec(precur_aggr=f)
# #     rg.df_data = rg.df_data.rename({'z5000..0..z500_sp':f'{west_east[0].capitalize()}-RW',
# #                                     '0..0..N-Pac. SST_sp':'SST'}, axis=1)

# #     keys = [f'{west_east[0].capitalize()}-RW','SST']

# #     # when both SST and RW above threshold
# #     RW_ts = rg.df_data.loc[0].iloc[:,0]
# #     RW_mask = RW_ts > float(rg.TV.RV_ts.quantile(q=quantilethreshold))
# #     new_mask = np.logical_and(rg.df_data.loc[0]['RV_mask'], RW_mask)
# #     sst = functions_pp.get_df_test(rg.df_data, cols=['SST'])
# #     sst_mask = (sst > sst.quantile(q=quantilethreshold).values).squeeze()
# #     new_mask = np.logical_and(sst_mask, new_mask)
# #     sumyears = new_mask.groupby(new_mask.index.year).sum()
# #     sumyears = list(sumyears.index[sumyears > 25])
# #     RV_mask = rg.df_data.loc[0]['RV_mask']
# #     m = np.array([True if y in sumyears else False for y in RV_mask.index.year])
# #     new_mask = np.logical_and(m, RV_mask)
# #     try:
# #         new_mask.astype(int).plot()
# #         plt.savefig(os.path.join(rg.path_outsub1, 'subset_dates_SST_and_RW.pdf'))
# #     except:
# #         pass
# #     print(f'{new_mask[new_mask].size} datapoints')

# #     # when both SST is anomalous
# #     RW_ts = rg.df_data.loc[0].iloc[:,0]
# #     RW_mask = RW_ts > float(rg.TV.RV_ts.quantile(q=quantilethreshold))
# #     new_mask = np.logical_and(rg.df_data.loc[0]['RV_mask'], RW_mask)
# #     sst = functions_pp.get_df_test(rg.df_data, cols=['SST'])
# #     sst_mask = (sst > sst.quantile(q=quantilethreshold).values).squeeze()
# #     new_mask = np.logical_and(sst_mask, new_mask)
# #     sumyears = new_mask.groupby(new_mask.index.year).sum()
# #     sumyears = list(sumyears.index[sumyears > 25])
# #     RV_mask = rg.df_data.loc[0]['RV_mask']
# #     m = np.array([True if y in sumyears else False for y in RV_mask.index.year])
# #     new_mask = np.logical_and(m, RV_mask)
# #     try:
# #         new_mask.astype(int).plot()
# #         plt.savefig(os.path.join(rg.path_outsub1, 'subset_dates_SST_and_RW.pdf'))
# #     except:
# #         pass
# #     print(f'{new_mask[new_mask].size} datapoints')

# #     rg.PCMCI_df_data(keys=keys,
# #                      replace_RV_mask=new_mask.values,
# #                      pc_alpha=None,
# #                      tau_max=5,
# #                      max_conds_dim=10,
# #                      max_combinations=10)

# #     rg.PCMCI_plot_graph(min_link_robustness=5, figshape=(8,2),
# #                         kwrgs={'vmax_nodes':1.0,
# #                                'vmax_edges':.6,
# #                                'vmin_edges':-.6,
# #                                'node_ticks':.2,
# #                                'edge_ticks':.1},
# #                         append_figpath=f'_subset_dates_tf{rg.precur_aggr}')

# #     rg.PCMCI_get_links(var=keys[1], alpha_level=.01)
# #     rg.df_links.mean(0, level=1)
# #     MCI_subset = rg.df_MCIc.mean(0, level=1)




