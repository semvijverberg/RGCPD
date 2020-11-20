#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 15:33:52 2020

@author: semvijverberg
"""

import os, inspect, sys
import matplotlib as mpl
if sys.platform == 'linux':
    mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import cartopy.crs as ccrs
import argparse
import csv
from sklearn import preprocessing

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
import climate_indices
import functions_pp, filters

periods = ['summer_center', 'summer_shiftright', 'summer_shiftleft',
           'spring_center', 'spring_shiftleft', 'spring_shiftright']

# periods = ['summer_shiftleft']
targets = ['east'] # ['west', 'east']
remove_PDOyesno = np.array([0, 1])
seeds = np.array([1,2,3])
combinations = np.array(np.meshgrid(targets, seeds, periods, remove_PDOyesno)).T.reshape(-1,4)

i_default = 18



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
    remove_PDO = bool(int(out[3]))
    print(f'arg {args.intexper} - {out}')
else:
    seed = 0


TVpathtemp = os.path.join(data_dir, 'tf15_nc3_dendo_0ff31.nc')
if west_east == 'east':
    # TVpathRW = os.path.join(data_dir, '2020-10-29_13hr_45min_east_RW.h5')
    cluster_label = 2
    z500_green_bb = (155,300,20,73) # bounding box for eastern RW
elif west_east =='west':
    # TVpathRW = os.path.join(data_dir, '2020-10-29_10hr_58min_west_RW.h5')
    cluster_label = 1
    z500_green_bb = (145,325,20,62) # bounding box for western RW


path_out_main = os.path.join(main_dir, f'publications/paper2/output/SST_RW_T/')
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

if period.split('_')[0] == 'summer':
    start_end_date = ('03-01', start_end_TVdatet2mvsRW[-1])
elif period.split('_')[0] == 'spring':
    start_end_date = ('01-01', start_end_TVdatet2mvsRW[-1])

tfreq         = 15
min_detect_gc = 1.0
method        = 'ran_strat10' ;

if remove_PDO:
    append_pathsub = 'rmPDO'
else:
    append_pathsub = ''

name_MCI_csv = 'strength_SST_RW_T.csv'
name_rob_csv = 'robustness_SST_RW_T.csv'

if tfreq > 15: sst_green_bb = (140,240,-9,59) # (180, 240, 30, 60): original warm-code focus
if tfreq <= 15: sst_green_bb = (140,235,20,59) # same as for West

name_or_cluster_label = 'z500'
name_ds = f'0..0..{name_or_cluster_label}_sp'

#%% Circulation vs temperature
list_of_name_path = [(cluster_label, TVpathtemp),
                     ('z500', os.path.join(path_raw, 'z500hpa_1979-2018_1_12_daily_2.5deg.nc')),
                     ('SST', os.path.join(path_raw, 'sst_1979-2018_1_12_daily_1.0deg.nc'))]

list_for_MI   = [BivariateMI(name='z500', func=class_BivariateMI.corr_map,
                            alpha=.05, FDR_control=True,
                            distance_eps=600, min_area_in_degrees2=5,
                            calc_ts='pattern cov', selbox=z500_green_bb,
                            use_sign_pattern=True, lags = np.array([0])),
                 BivariateMI(name='SST', func=class_BivariateMI.corr_map,
                              alpha=.05, FDR_control=True,
                              distance_eps=500, min_area_in_degrees2=5,
                              calc_ts='pattern cov', selbox=sst_green_bb,#(130,340,-10,60),
                              lags=np.array([0]))]
                 # BivariateMI(name='sm', func=class_BivariateMI.parcorr_map_time,
                 #            alpha=.05, FDR_control=True,
                 #            distance_eps=1200, min_area_in_degrees2=10,
                 #            calc_ts='region mean', selbox=(200,300,20,73),
                 #            lags=np.array([0]))]

rg = RGCPD(list_of_name_path=list_of_name_path,
            list_for_MI=list_for_MI,
            start_end_TVdate=start_end_TVdatet2mvsRW,
            start_end_date=start_end_date,
            start_end_year=None,
            tfreq=tfreq,
            path_outmain=path_out_main,
            append_pathsub=append_pathsub)


rg.pp_TV(detrend=False)

rg.pp_precursors()
rg.traintest(method=method, seed=seed)
rg.calc_corr_maps()
rg.cluster_list_MI()

# Optionally set font to Computer Modern to avoid common missing font errors
# mpl.rc('font', family='serif', serif='cm10')

# matplotlib.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\boldmath']



title = '' #f'$parcorr(z500_t, {west_east.capitalize()[0]}$-$mx2t_t\ |\ $'+r'$z500_{t-1},$'+f'${west_east.capitalize()[0]}$-'+r'$mx2t_{t-1})$'
subtitles = np.array([[f'$parcorr(z500_t, {west_east.capitalize()[0]}$-$mx2t_t\ |\ $'+r'$z500_{t-1},$'+f'${west_east.capitalize()[0]}$-'+r'$mx2t_{t-1})$']] )
kwrgs_plot = {'row_dim':'lag', 'col_dim':'split', 'aspect':4, 'size':2.5,
              'hspace':0.0, 'cbar_vert':-.08, 'units':'Corr. Coeff. [-]',
              'zoomregion':z500_green_bb, #'drawbox':[(0,0), z500_green_bb],
              'map_proj':ccrs.PlateCarree(central_longitude=220), 'n_yticks':6,
              'clim':(-.6,.6), 'title':title, 'subtitles':subtitles}
save = True
rg.plot_maps_corr(var='z500', save=save,
                  append_str=''.join(map(str, z500_green_bb)),
                  min_detect_gc=min_detect_gc,
                  kwrgs_plot=kwrgs_plot)


subtitles = np.array([[f'$parcorr(SST_t, {west_east.capitalize()[0]}$-$mx2t_t\ |\ $'+r'$SST_{t-1},$'+f'${west_east.capitalize()[0]}$-'+r'$mx2t_{t-1})$']] )
kwrgs_plot = {'row_dim':'split', 'col_dim':'lag',
              'aspect':2, 'hspace':-.57, 'wspace':-.22, 'size':4, 'cbar_vert':-.02,
              'subtitles':subtitles, 'units':'Corr. Coeff. [-]',
              'zoomregion':(130,260,-10,60),
              'map_proj':ccrs.PlateCarree(central_longitude=220),
              'x_ticks':np.array([]), 'y_ticks':np.array([]),
              'drawbox':[(0,0), sst_green_bb],
              'clim':(-.6,.6)}
rg.plot_maps_corr(var='SST', save=save, min_detect_gc=min_detect_gc,
                  kwrgs_plot=kwrgs_plot)
rg.list_for_MI[1].selbox = sst_green_bb
#%% Get PDO
df_PDO, PDO_patterns = climate_indices.PDO(rg.list_precur_pp[0][1],
                                           None) #rg.df_splits)

from func_models import standardize_on_train
# summerdates = core_pp.get_subdates(dates, start_end_TVdate)
df_PDOsplit = df_PDO.loc[0]#.loc[summerdates]
# standardize = preprocessing.StandardScaler()
# standardize.fit(df_PDOsplit[df_PDOsplit['TrainIsTrue'].values].values.reshape(-1,1))
# df_PDOsplit = pd.DataFrame(standardize.transform(df_PDOsplit['PDO'].values.reshape(-1,1)),
#                 index=df_PDOsplit.index, columns=['PDO'])
df_PDOsplit = df_PDOsplit[['PDO']].apply(standardize_on_train,
                         args=[df_PDO.loc[0]['TrainIsTrue']],
                         result_type='broadcast')

# Butter Lowpass
yr = 2
dates = df_PDOsplit.index
freqraw = (dates[1] - dates[0]).days
window = int(yr*functions_pp.get_oneyr(dates).size) # 2 year
fig, ax = plt.subplots(1,1)

ax.plot_date(dates, df_PDOsplit.values, label=f'raw ({freqraw} daymeans)',
              alpha=.2, linestyle='solid', marker=None)
ax.plot_date(dates, filters.lowpass(df_PDOsplit, period=window), label='Butterworth',
        linestyle='solid', linewidth=1, marker=None)
df_PDOrm = df_PDOsplit.rolling(window=window, center=True, min_periods=1).mean()
# ax.plot_date(dates, filters.lowpass(df_PDOrm, period=window), label='Butterworth on rolling mean',
#         linestyle='solid', linewidth=1, marker=None)
ax.plot_date(dates, df_PDOrm,
             label='rolling mean', color='green', linestyle='solid', linewidth=1, marker=None)

ax.legend()




#%%
import wrapper_PCMCI as wPCMCI

def append_MCI(rg, dict_v, dict_rb):
    dkeys = [f'{f}-d RW--T', f'{f}-d RW--SST', f'{f}-d RW->SST']


    RWtoT = rg.df_MCIc.mean(0,level=1).loc[keys[1]].iloc[0].round(3) # links to temp
    rg.PCMCI_get_links(var='SST', alpha_level=.01) # links toward SST
    lag0 = rg.df_MCIc.mean(0,level=1).loc[f'{west_east[0].capitalize()}-RW']['coeff l0'].round(3)
    RWtoSST = rg.df_MCIc.mean(0,level=1).loc[f'{west_east[0].capitalize()}-RW'].iloc[1:].max().round(3) # select RW
    append_dict = {dkeys[0]:RWtoT, dkeys[1]:lag0, dkeys[2]:RWtoSST}
    dict_v.update(append_dict)

    robustness = wPCMCI.get_traintest_links(rg.pcmci_dict,
                                      rg.parents_dict,
                                      rg.pcmci_results_dict,
                                      min_link_robustness=mlr)[2]
    rbRWTlag0 = int(robustness[0][1][0])
    rbRWSSTlab0 = int(robustness[1][2][0]) # from i to j, RW to SST
    rbRWtoSST = int(max(robustness[1][2][1:])) # from i to j, RW to SST
    append_dict = {dkeys[0]:rbRWTlag0, dkeys[1]:rbRWSSTlab0, dkeys[2]:rbRWtoSST}
    dict_rb.update(append_dict)
    return

if remove_PDO:
    lowpass = '2y'
    rg.list_import_ts = [('PDO', os.path.join(data_dir, f'PDO_{lowpass}_rm_25-09-20_15hr.h5'))]



dict_v = {'rmPDO':str(remove_PDO), 'Period':period.split('_')[0], 'Shift':period,'Seed':'s{}'.format(rg.kwrgs_TV['seed'])}
dict_rb = dict_v.copy()
freqs = [15, 30, 60]
for f in freqs[:]:
    rg.get_ts_prec(precur_aggr=f)
    rg.df_data = rg.df_data.rename({'2ts':f'{west_east[0].capitalize()}-T',
                                    '0..0..z500_sp':f'{west_east[0].capitalize()}-RW',
                                    '0..0..SST_sp':'SST',
                                    '0..2..sm':'SM'}, axis=1)
    keys = [f'{west_east[0].capitalize()}-T', f'{west_east[0].capitalize()}-RW',
            'SST']
    if remove_PDO:
        rg.df_data['SST'], fig = wPCMCI.df_data_remove_z(rg.df_data, z=['PDO'],
                                                         keys=['SST'],
                                                         standardize=False,
                                                         plot=True)
        fig_path = os.path.join(rg.path_outsub1, f'regressing_out_PDO_tf{f}')
        fig.savefig(fig_path+rg.figext, bbox_inches='tight')

    # interannualSST = rg.df_data[['SST']].rolling(int((365*2)/f), min_periods=1,center=True).mean()
    # interannualSST = interannualSST.rename({'SST':r'$SST_{lwp}$'}, axis=1)
    # diff = rg.df_data[['SST']].sub(interannualSST.values, level=1)
    # diff = diff.rename({'SST':'SST-SST_lwp'},axis=1)
    # interannualSST.merge(diff.merge(rg.df_data[['SST', 'PDO']], left_index=True, right_index=True),
    #                      left_index=True, right_index=True).loc[0].plot()
    # rg.df_data = rg.df_data.merge(diff, left_index=True, right_index=True)
    # keys = [f'{west_east[0].capitalize()}-RW','SST']
    # rg.df_data[keys] = wPCMCI.df_data_remove_z(rg.df_data, z=['PDO'], keys=keys)
    tigr_function_call='run_pcmciplus'
    rg.PCMCI_df_data(keys=keys,
                     tigr_function_call=tigr_function_call,
                      pc_alpha=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
                      tau_min=0,
                      tau_max=5,
                      max_conds_dim=10,
                      max_combinations=10,
                      update_dict={})
    rg.PCMCI_get_links(var=keys[0], alpha_level=.01) # links toward RW

    lags = range(rg.kwrgs_tigr['tau_min'], rg.kwrgs_tigr['tau_max']+1)
    lags = np.array([l*f for i, l in enumerate(lags)])
    mlr=5
    append_MCI(rg, dict_v, dict_rb)

    rg.PCMCI_plot_graph(min_link_robustness=mlr, figshape=(8,8),
                        kwrgs={'vmax_nodes':.9,
                                'node_size':.75,
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
                                'label_fontsize':20,
                                'weights_squared':1.5},
                        append_figpath=f'{tigr_function_call}_tf{rg.precur_aggr}_mlr{mlr}')

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
# #                                     '0..0..SST_sp':'SST'}, axis=1)

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




