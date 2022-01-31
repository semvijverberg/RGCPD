#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 15:03:30 2020

Step 4 to reproduce results of NPJ paper:
"The role of the Pacific Decadal Oscillation and
ocean-atmosphere interactions in driving US temperature variability"

It loads the RW and temperature timeseries that are stored by step 2.
This script calculates and plots the (partial) correlation maps of
Figure 3 and 4.

@author: semvijverberg
"""

import os, inspect, sys
import numpy as np
import cartopy.crs as ccrs ;
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from time import sleep

user_dir = os.path.expanduser('~')
# curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
curr_dir = user_dir + '/surfdrive/Scripts/RGCPD/publications/Vijverberg_Coumou_2022_NPJ'
main_dir = '/'.join(curr_dir.split('/')[:-2])
RGCPD_func = os.path.join(main_dir, 'RGCPD')
cluster_func = os.path.join(main_dir, 'clustering/')
fc_dir = os.path.join(main_dir, 'forecasting')
path_data = os.path.join(main_dir, 'publications/Vijverberg_Coumou_2022_NPJ/data/')
if cluster_func not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(RGCPD_func)
    sys.path.append(cluster_func)
    sys.path.append(fc_dir)

path_raw = user_dir + '/surfdrive/ERA5/input_raw'

from RGCPD import RGCPD
from RGCPD import BivariateMI
from class_BivariateMI import corr_map
from class_BivariateMI import parcorr_map
import climate_indices, filters, functions_pp, core_pp, plot_maps
from func_models import standardize_on_train
import func_models as fc_utils



expers = np.array(['parcorr', 'parcorrENSO','parcorr_SSTlag1',
                   'parcorrtime_target', 'parcorrtime_precur', 'corr']) # np.array(['fixed_corr', 'adapt_corr'])
combinations = np.array(np.meshgrid(expers)).T.reshape(-1,1)

i_default = 5
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
    exper = out[0]
    print(f'arg {args.intexper} f{out}')
else:
    exper = 'parcorr'


# path to RW timeseries
west_east = 'east'
mainpath_df = os.path.join(main_dir, 'publications/Vijverberg_Coumou_2022_NPJ/output/heatwave_circulation_v300_z500_SST/57db0USCA/')
# # t2m
# TVpath = 'z500_145-325-20-62USCA.h5'
# # mx2t
# TVpath = 'z500_145-325-20-620a6f6USCA.h5'
# mx2t 25N-70N
TVpath = 'z500_155-300-20-7357db0USCA.h5'

TVpath  = os.path.join(mainpath_df, TVpath)

TVpath = os.path.join(path_data, 'eastRW_summer_center_s1_RepeatedKFold_10_7_tf1.h5')

period = 'summer'
if period == 'spring':
    start_end_TVdate = ('03-01', '05-31')
    lags = np.array([0,1])
elif period == 'summer':
    start_end_TVdate = ('06-01', '08-31')
    lags = np.array([0, 1, 2])



path_out_main = os.path.join(main_dir, f'publications/Vijverberg_Coumou_2022_NPJ/output/{west_east}_parcorrmaps')
if os.path.isdir(path_out_main) != True:
    os.makedirs(path_out_main)
cluster_label = '' # 'z500'
# name_or_cluster_label = ''
# name_ds = west_east + 'RW' # f'0..0..{name_or_cluster_label}_sp'
name_or_cluster_label = 'z500'
name_ds = f'0..0..{name_or_cluster_label}_sp'
start_end_date = ('1-1', start_end_TVdate[-1])
filepath_df_PDOs = os.path.join(path_data, 'df_PDOs_monthly.h5')
filepath_df_ENSO = os.path.join(path_data, 'df_ENSOs_monthly.h5')
filepath_df_SSTlag1 = os.path.join(path_data, 'df_SST_lag1.h5')

#%% Get PDO and apply low-pass filter
if 'parcorr' == exper:
    try:
        df_PDOs = functions_pp.load_hdf5(filepath_df_PDOs)['df_data']
    except:

        SST_pp_filepath = user_dir + '/surfdrive/ERA5/input_raw/preprocessed/sst_1979-2020_jan_dec_monthly_1.0deg.nc'

        if 'df_PDOs' not in globals():
            df_PDO, PDO_patterns = climate_indices.PDO(SST_pp_filepath,
                                                       None)
            PDO_plot_kwrgs = {'units':'[-]', 'cbar_vert':-.1,
                              # 'zoomregion':(130,260,20,60),
                              'map_proj':ccrs.PlateCarree(central_longitude=220),
                              'y_ticks':np.array([25,40,50,60]),
                              'x_ticks':np.arange(130, 280, 25),
                              'clevels':np.arange(-.6,.61,.075),
                              'clabels':np.arange(-.6,.61,.3),
                              'subtitles':np.array([['PDO loading pattern']])}
            fig = plot_maps.plot_corr_maps(PDO_patterns[0], **PDO_plot_kwrgs)
            filepath = os.path.join(path_out_main, 'PDO_pattern')
            fig.savefig(filepath + '.pdf', bbox_inches='tight')
            fig.savefig(filepath + '.png', bbox_inches='tight')

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
        dates = df_PDOsplit.index
        freqraw = (dates[1] - dates[0]).days
        ls = ['solid', 'dotted', 'dashdot', 'dashed']
        fig, ax = plt.subplots(1,1, figsize=(10,5))
        list_dfPDO = [df_PDOsplit]
        lowpass_yrs = [.25, .5, 1.0, 2.0]
        for i, yr in enumerate(lowpass_yrs):
            window = int(yr*functions_pp.get_oneyr(dates).size) # 2 year
            if i ==0:
                ax.plot_date(dates, df_PDOsplit.values, label=f'Raw ({freqraw} day means)',
                          alpha=.3, linestyle='solid', marker=None)
            df_PDObw = pd.Series(filters.lowpass(df_PDOsplit, period=window).squeeze(),
                                 index=dates, name=f'PDO{yr}bw')
            ax.plot_date(dates, df_PDObw, label=f'Butterworth {yr}-year low-pass',
                    color='red',linestyle=ls[i], linewidth=1, marker=None)
            df_PDOrm = df_PDOsplit.rolling(window=window, closed='right', min_periods=window).mean()
            df_PDOrm = df_PDOrm.rename({'PDO':f'PDO{yr}rm'}, axis=1)
            ax.plot_date(dates, df_PDOrm,
                         label=f'Rolling mean {yr}-year low-pass (closed right)', color='green',linestyle=ls[i],
                         linewidth=1, marker=None)
            list_dfPDO.append(df_PDObw) ; list_dfPDO.append(df_PDOrm)
            ax.legend()

        filepath = os.path.join(path_out_main, 'Low-pass_filter.pdf')
        plt.savefig(filepath, bbox_inches='tight')
        df_PDOs = pd.concat(list_dfPDO,axis=1)

    functions_pp.store_hdf_df({'df_data':df_PDOs},
                              file_path=filepath_df_PDOs)
#%% Get ENSO 3.4 index
if 'parcorrENSO' == exper:
    try:
        df_ENSOs = functions_pp.load_hdf5(filepath_df_ENSO)['df_data']
    except:

        SST_pp_filepath = user_dir + '/surfdrive/ERA5/input_raw/preprocessed/sst_1979-2020_jan_dec_monthly_1.0deg.nc'

        if 'df_PDOsplit' not in globals():
            df_ENSO, ENSO_years, ENSO_cycle = climate_indices.ENSO_34(SST_pp_filepath)

        df_ENSO = (df_ENSO - df_ENSO.mean()) / df_ENSO.std()

        # Butter Lowpass
        dates = df_ENSO.index
        freqraw = (dates[1] - dates[0]).days
        ls = ['solid', 'dotted', 'dashdot', 'dashed']
        fig, ax = plt.subplots(1,1, figsize=(10,5))
        list_dfENSO = [df_ENSO]
        lowpass_yrs = [.25, .5, 1.0, 2.0]
        for i, yr in enumerate(lowpass_yrs):
            window = int(yr*functions_pp.get_oneyr(dates).size) # 2 year
            if i ==0:
                ax.plot_date(dates, df_ENSO.values, label=f'Raw ({freqraw} day means)',
                          alpha=.3, linestyle='solid', marker=None)
            df_ENSObw = pd.Series(filters.lowpass(df_ENSO, period=window).squeeze(),
                                 index=dates, name=f'ENSO{yr}bw')
            ax.plot_date(dates, df_ENSObw, label=f'Butterworth {yr}-year low-pass',
                    color='red',linestyle=ls[i], linewidth=1, marker=None)
            df_ENSOrm = df_ENSO.rolling(window=window, closed='right', min_periods=window).mean()
            df_ENSOrm = df_ENSOrm.rename({'ENSO34':f'ENSO{yr}rm'}, axis=1)
            ax.plot_date(dates, df_ENSOrm,
                         label=f'Rolling mean {yr}-year low-pass (closed right)', color='green',linestyle=ls[i],
                         linewidth=1, marker=None)
            list_dfENSO.append(df_ENSObw) ; list_dfENSO.append(df_ENSOrm)
            ax.legend()

        filepath = os.path.join(path_out_main, 'Low-pass_filter_ENSO.pdf')
        plt.savefig(filepath, bbox_inches='tight')
        df_ENSOs = pd.concat(list_dfENSO,axis=1)

    functions_pp.store_hdf_df({'df_data':df_ENSOs},
                              file_path=filepath_df_ENSO)

#%% Only SST (Parcorrtime and parcorr on PDO)

list_of_name_path = [(name_or_cluster_label, TVpath),
                       ('sst', os.path.join(path_raw, 'sst_1979-2020_1_12_monthly_1.0deg.nc'))]

def get_lagged_ts(df_data, lag, keys=None):
    if keys is None:
        keys = df_data.columns[df_data.dtypes != bool]
    df_lagmask = []
    for s in df_data.index.levels[0]:
        lagmask = fc_utils.apply_shift_lag(df_data.loc[s][['TrainIsTrue', 'RV_mask']], lag)
        df_lagmask.append(lagmask)
    df_lagmask = pd.concat(df_lagmask, keys=df_data.index.levels[0])
    # persPDO = functions_pp.get_df_test(rgPDO.df_data[keys_ext+['TrainIsTrue']])[keys_ext]
    df_lag = df_data[df_lagmask['x_fit']]
    df_lag.index = df_data[df_lagmask['y_fit']].index
    return df_lag[keys].rename({k:k+f'_{lag}' for k in keys}, axis=1), df_lagmask

# exper = 'parcorr'
lowpass = 0.5
if 'parcorr' == exper:
    # lowpass = float(exper.split('__')[1])
    func = parcorr_map
    # z_filepath = os.path.join(path_data, 'PDO_ENSO34_ERA5_1979_2018.h5')
    z_filepath = filepath_df_PDOs
    keys_ext = [f'PDO{lowpass}rm']
    kwrgs_func = {'filepath':z_filepath,
                  'keys_ext':keys_ext,
                  'lag_z':1}

    # lowpass = 0.5 #!!!
    # keys_ext = ['PDO']
    rgPDO = RGCPD(list_of_name_path=list_of_name_path,
                  list_import_ts=[('PDO', z_filepath)],
                  start_end_TVdate=('05-01', '08-01'),
                  start_end_date=None,
                  start_end_year=(1979+int(round(lowpass+0.49)), 2020),
                  tfreq=2,
                  path_outmain=path_out_main,
                  append_pathsub='_' + exper)
    rgPDO.pp_TV(name_ds, anomaly=True, kwrgs_core_pp_time={'dailytomonths':True})

    rgPDO.pp_precursors()

    rgPDO.traintest('random_10')
    rgPDO.get_ts_prec()
    # Predicting PDO at lag 1 vs start_end_data of RW
    PDO1, df_lagmask1 = get_lagged_ts(rgPDO.df_data.copy() , 0, keys_ext)
    target = functions_pp.get_df_test(PDO1,
                                      df_splits=rgPDO.df_data[['TrainIsTrue']].loc[PDO1.index])
    PDO2, df_lagmask2 = get_lagged_ts(rgPDO.df_data.copy() , 2, keys_ext)
    # PDO3, df_lagmask3 = get_lagged_ts(rgPDO.df_data.copy() , 3, keys_ext)
    # PDO4, df_lagmask4 = get_lagged_ts(rgPDO.df_data.copy() , 4, keys_ext)
    # PDO5, df_lagmask5 = get_lagged_ts(rgPDO.df_data.copy() , 5, keys_ext)
    df_prec = PDO2 # AR1 model to predict PDO at lag 1 vs RW
    # df_prec = df_prec.merge(PDO3, left_index=True, right_index=True)
    # df_prec = df_prec.merge(PDO4, left_index=True, right_index=True)
    # df_prec = df_prec.merge(PDO5, left_index=True, right_index=True)


    out = rgPDO.fit_df_data_ridge(target=target,
                                  df_data = df_prec,
                                  tau_min=0, tau_max=0,
                                  kwrgs_model={'alphas':np.array([.01,.1,1,5,10])})
    predict = out[0].rename({0:'AR1'}, axis=1)

    lowPDO, df_lagmask = get_lagged_ts(rgPDO.df_data.copy(), 1, ['PDO0.5rm'])
    # perPDO = rgPDO.df_data[keys_ext][persmask['x_fit']]
    # persPDO[persmask['x_fit']] = persPDO[persmask['x_fit']]
    # perPDO.index = rgPDO.df_data[rgPDO.df_data['RV_mask']].index
    perPDO = lowPDO.rename({'PDO1.0rm_2':'persistence'}, axis=1)
    perPDO = perPDO.loc[df_prec.index]
    predict = predict.merge(perPDO, left_index=True, right_index=True)

    dates = core_pp.get_subdates(rgPDO.dates_TV, start_end_year=(1980,2020))
    predict = predict.loc[pd.IndexSlice[:, dates], :]
    test = fc_utils.get_scores(predict,
                               score_func_list=[fc_utils.corrcoef,
                                                fc_utils.metrics.mean_squared_error])[2]
    df_test = functions_pp.get_df_test(predict,
                                       df_splits=rgPDO.df_data.loc[predict.index][['TrainIsTrue', 'RV_mask']])
    df_z = df_test[['AR1']]
    df_z = lowPDO
    # df_z = functions_pp.get_df_test(df_prec,
    #                                 df_splits=rgPDO.df_data.loc[predict.index][['TrainIsTrue', 'RV_mask']])
    # years = functions_pp.get_oneyr(df_z, *list(range(1980, 2020+1)))
    # df_z = df_z.loc[years]

    kwrgs_func = {'filepath':df_z,
                  'lag_z':0}



#%%
lowpass = 0.5
func = parcorr_map
if 'parcorr' == exper:
    # lags = np.array([1])
    # PDO1, df_lagmask1 = get_lagged_ts(rgPDO.df_data.copy() , lags[0], keys_ext)
    # years = functions_pp.get_oneyr(df_lagmask1.loc[0], *list(range(1980, 2020+1)))
    # df_z.index = df_lagmask1.loc[0].loc[years][df_lagmask1.loc[0]['x_pred'].loc[years]].index
    kwrgs_func = {'filepath':filepath_df_PDOs,
                  'keys_ext':[f'PDO{lowpass}rm'],
                  'lag_z':[1]} # lag_z is defined wrt precursor dates
elif 'parcorrENSO' == exper:
    kwrgs_func = {'filepath':filepath_df_ENSO,
                  'keys_ext':[f'ENSO{lowpass}rm'],
                  'lag_z':[1]} # lag_z is defined wrt precursor dates
elif 'parcorr_SSTlag1' == exper:
    df_z = functions_pp.load_hdf5(filepath_df_SSTlag1)['df_data']
    kwrgs_func = {'filepath':df_z.mean(axis=0, level=1),
                  'lag_z':[1]} # lag_z is defined wrt precursor dates

elif exper == 'corr':
    kwrgs_func = {} ; func = corr_map
elif 'parcorrtime' in exper:
    if exper.split('_')[1] == 'target':
        kwrgs_func = {'lag_y':[2]}
    elif exper.split('_')[1] == 'precur':
        kwrgs_func = {'lag_x':[1]}
    elif exper.split('_')[1] == 'both':
        kwrgs_func = {'lag_y':[1], 'lag_x':[1]}

# kwrgs_func['lagzxrelative'] = False

sst_dailytomonths = False
if sst_dailytomonths:
    list_of_name_path = [(name_or_cluster_label, TVpath),
                         ('sst', os.path.join(path_raw, 'sst_1979-2020_1_12_daily_1.0deg.nc'))]
else:
    list_of_name_path = [(name_or_cluster_label, TVpath),
                         ('sst', os.path.join(path_raw, 'sst_1979-2020_1_12_monthly_1.0deg.nc'))]


list_for_MI   = [BivariateMI(name='sst', func=func,
                            alpha=.05, FDR_control=True,
                            kwrgs_func=kwrgs_func,
                            distance_eps=1000, min_area_in_degrees2=1,
                            calc_ts='pattern cov', selbox=(130,260,-10,60),
                            lags=lags, dailytomonths=sst_dailytomonths)]



rg = RGCPD(list_of_name_path=list_of_name_path,
            list_for_MI=list_for_MI,
            start_end_TVdate=('05-01', '08-01'),
            start_end_date=None,
            start_end_year=(1980, 2020),
            tfreq=2,
            path_outmain=path_out_main,
            append_pathsub='_' + exper)


rg.pp_TV(name_ds=name_ds, detrend=False, anomaly=False, # detrending already done on z500
         kwrgs_core_pp_time={'dailytomonths':True})

rg.pp_precursors()

rg.traintest('random_10')


rg.calc_corr_maps()
precur = rg.list_for_MI[0]
#%%
import matplotlib
# Optionally set font to Computer Modern to avoid common missing font errors
matplotlib.rc('font', family='serif', serif='cm10')

matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'
min_detect_gc = 0.5
save = True
# Plot lag 0 and 1
# subtitles = np.array([['lag 0'], [f'lag 1 ({1*rg.tfreq} days)']] )
title = '' ; hspace = .2
if 'parcorr' == exper and west_east == 'east':
    z_ts = '$\overline{PDO_{t-1}}$'
    # title0 = r'$parcorr(SST_{t},\ $'+'$RW^E_t\ |\ $Z)'+'\nZ='+'('+z_ts+')'
    title0 = r'$parcorr(SST_{t},\ $'+'$RW^E_t\ |\ $'+z_ts+')'
    z_ts = '$\overline{PDO_{t-2}}$'
    title1 = r'$parcorr(SST_{t-1},\ $'+'$RW^E_t\ |\ $'+z_ts+')'
    subtitles = np.array([[title0],[title1]])
    tscol = ''.join(precur.kwrgs_func['df_z'].columns)
    kw = [k for k in kwrgs_func.keys() if k != 'df_z']
    val = ''.join([str(kwrgs_func[k]) for k in kw])
    append_str='parcorrtime_{}_'.format(period) + ''.join(kw) + val
    fontsize = 12
if 'parcorr_SSTlag1' == exper and west_east == 'east':
    z_ts = '$SST^{pattern}_{t-1}$'
    # title0 = r'$parcorr(SST_{t},\ $'+'$RW^E_t\ |\ $Z)'+'\nZ='+'('+z_ts+')'
    title0 = r'$parcorr(SST_{t},\ $'+'$RW^E_t\ |\ $'+z_ts+')'
    z_ts = '$SST^{pattern}_{t-2}$'
    title1 = r'$parcorr(SST_{t-1},\ $'+'$RW^E_t\ |\ $'+z_ts+')'
    subtitles = np.array([[title0],[title1]])
    tscol = ''.join(precur.kwrgs_func['df_z'].columns)
    kw = [k for k in kwrgs_func.keys() if k != 'df_z']
    val = ''.join([str(kwrgs_func[k]) for k in kw])
    append_str='parcorr_{}_{}_'.format(tscol, period) + ''.join(kw) + val
    fontsize = 14
if 'parcorrENSO' == exper and west_east == 'east':
    z_ts = '$\overline{ENSO_{t-1}}$'
    # title0 = r'$parcorr(SST_{t},\ $'+'$RW^E_t\ |\ $Z)'+'\nZ='+'('+z_ts+')'
    title0 = r'$parcorr(SST_{t},\ $'+'$RW^E_t\ |\ $'+z_ts+')'
    z_ts = '$\overline{ENSO_{t-2}}$'
    title1 = r'$parcorr(SST_{t-1},\ $'+'$RW^E_t\ |\ $'+z_ts+')'
    # title1 = r'$parcorr(SST_{t-1},\ $'+'$RW^E_t\ |\ $Z)'+'\nZ='+'('+z_ts+')'
    subtitles = np.array([[title0],[title1]])
    tscol = ''.join(precur.kwrgs_func['df_z'].columns)
    kw = [k for k in precur.kwrgs_func.keys() if k != 'df_z']
    val = ''.join([str(kwrgs_func[k]) for k in kw])
    append_str='parcorrENSO_{}_{}_'.format(tscol, period) + ''.join(kw) + val
    fontsize = 14
elif exper == 'corr' and west_east == 'east':
    title0 =  r'$corr(SST_{t},\ $'+'$RW^E_t\ )$'
    title1 =  r'$corr(SST_{t-1},\ $'+'$RW^E_t\ )$'
    subtitles = np.array([[title0],[title1]])
    append_str='' ; fontsize = 14 ; hspace = .2
elif 'parcorrtime' in exper and west_east == 'east':
    if 'lag_y' not in list(kwrgs_func.keys()) and 'lag_x' in list(kwrgs_func.keys()):
        # regress out past precursor
        z_ts = ', '.join([f'$SST_{{t-{l}}}$' for l in kwrgs_func['lag_x']])
        # title0 = r'$parcorr(SST_{t},\ $'+'$RW^E_t\ |\ $Z)'+'\nZ='+'('+z_ts+')'
        title0 = r'$parcorr(SST_{t},\ $'+'$RW^E_t\ |\ $'+z_ts+')'
        z_ts = ', '.join([f'$SST_{{t-{l+1}}}$' for l in kwrgs_func['lag_x']])
        title1 = r'$parcorr(SST_{t-1},\ $'+'$RW^E_t\ |\ $'+z_ts+')'
        subtitles = np.array([[title0],[title1]])
    elif 'lag_y' in list(kwrgs_func.keys()) and 'lag_x' not in list(kwrgs_func.keys()):
        # regress out past target variable
        z_ts = ', '.join([f'$RW^E_{{t-{l}}}$' for l in kwrgs_func['lag_y']])
        # title0 = r'$parcorr(SST_{t},\ $'+'$RW^E_t\ |\ $Z)'+'\nZ='+'('+z_ts+')'
        title0 = r'$parcorr(SST_{t},\ $'+'$RW^E_t\ |\ $'+z_ts+')'
        z_ts = ', '.join([f'$RW^E_{{t-{l}}}$' for l in kwrgs_func['lag_y']])
        title1 = r'$parcorr(SST_{t-1},\ $'+'$RW^E_t\ |\ $'+z_ts+')'
        # title1 = r'$parcorr(SST_{t-1},\ $'+'$RW^E_t\ |\ $Z)'+'\nZ='+'('+z_ts+')'
        subtitles = np.array([[title0],[title1]])
    elif 'lag_y' in list(kwrgs_func.keys()) and 'lag_x' in list(kwrgs_func.keys()):
        # regress out past target variable
        z_tsy = ', '.join([f'$RW^E_{{t-{l}}}$' for l in kwrgs_func['lag_y']])
        z_tsx = ', '.join([f'$SST_{{t-{l}}}$' for l in kwrgs_func['lag_x']])
        z_ts = z_tsy + ', ' + z_tsx
        # title0 = r'$parcorr(SST_{t},\ $'+'$RW^E_t\ |\ $Z)'+'\nZ='+'('+z_ts+')'
        title0 = r'$parcorr(SST_{t},\ $'+'$RW^E_t\ |\ $'+z_ts+')'
        z_tsy = ', '.join([f'$RW^E_{{t-{l}}}$' for l in kwrgs_func['lag_y']])
        z_tsx = ', '.join([f'$SST_{{t-{l+1}}}$' for l in kwrgs_func['lag_x']])
        z_ts = z_tsy + ', ' + z_tsx
        title1 = r'$parcorr(SST_{t-1},\ $'+'$RW^E_t\ |\ $'+z_ts+')'
        subtitles = np.array([[title0],[title1]])
    # else:
    #     title = r'$parcorr(SST_{t-lag}, $'+'$RW^E_t\ |\ $'+r'$SST_{t-lag-1},$'+'$RW^E_{t-1})$'
    kw = [k for k in kwrgs_func.keys() if k != 'df_z']
    val = ''.join([str(kwrgs_func[k]) for k in kw])
    append_str='parcorrtime_{}_'.format(period) + ''.join(kw) + val
    fontsize = 12


kwrgs_plot = {'row_dim':'lag', 'col_dim':'split',
              'aspect':2,  'hspace':hspace, 'size':2.5, 'cbar_vert':-.02,
              'units':'Corr. Coeff. [-]', 'zoomregion':(130,260,-10,60),
              'map_proj':ccrs.PlateCarree(central_longitude=220),
              'y_ticks':np.array([-10,10,30,50]),
              'x_ticks':np.arange(130, 280, 25),
              'clevels':np.arange(-.6,.61,.075),
              'clabels':np.arange(-.6,.61,.3), 'title':title,
              'subtitles':subtitles, 'subtitle_fontdict':{'fontsize':13},
              'title_fontdict':{'fontsize':fontsize, 'fontweight':'bold'}}
#%%
plotlags= [0,1]
if sys.platform == 'linux':
    for min_detect_gc in [.5,.6,.7,.8,.9,1.]:
        rg.plot_maps_corr(var='sst', save=save,
                          kwrgs_plot=kwrgs_plot,
                          min_detect_gc=min_detect_gc,
                          plotlags=plotlags,
                          append_str=append_str)
else:
    rg.plot_maps_corr(var='sst', save=save,
                      kwrgs_plot=kwrgs_plot,
                      min_detect_gc=min_detect_gc,
                      plotlags=plotlags,
                      append_str=append_str)

#%% plot lag 1
# if exper == 'parcorrtime' and west_east == 'east':
#     if kwrgs_func['target'] == False and kwrgs_func['precursor'] == True:
#         title = r'$parcorr(SST_{t-1},\ RW^E_t\ |\ SST_{t-2})$'
#     elif kwrgs_func['target'] == True and kwrgs_func['precursor'] == False:
#         title = r'$parcorr(SST_{t-1},\ RW^E_t\ |\ RW^E_{t-1})$'
#     else:
#         title = r'$parcorr(SST_{t-1},\ RW^E_t\ |\ SST_{t-2}, RW^E_{t-1})$'


kwrgs_plot['subtitles'] = subtitles[[1]]
kwrgs_plot['cbar_vert'] = -.1
kwrgs_plot['title'] = title
kwrgs_plot['title_fontdict'] = {'y':1,'fontsize':fontsize, 'fontweight':'bold'}
if sys.platform == 'linux':
    for min_detect_gc in [.1,.5,.6,.7,.8,.9,1.]:
        rg.plot_maps_corr(var='sst', plotlags=[1], save=save,
                          kwrgs_plot=kwrgs_plot,
                          min_detect_gc=min_detect_gc,
                          append_str=append_str+'Lag1')
else:
    rg.plot_maps_corr(var='sst', plotlags=[1], save=save,
                      splits='mean',
                      kwrgs_plot=kwrgs_plot,
                      min_detect_gc=min_detect_gc,
                      append_str=append_str+'Lag1')
#%%
if exper == 'parcorrtime_precur' or exper == 'parcorrtime_target':
    kwrgs_plot['scatter']= [['all', [[np.array([[ 15.5, 204.5]]), {'color':'red', 'edgecolors':"black"}],
                                     [np.array([[ 16.5, 204.5]]), {'color':'red', 'edgecolors':"black"}],
                                     [np.array([[ 16.5, 205.5]]), {'color':'red', 'edgecolors':"black"}],
                                     [np.array([[ 16.5, 206.5]]), {'color':'red', 'edgecolors':"black"}],
                                     [np.array([[ 17.5, 204.5]]), {'color':'red', 'edgecolors':"black"}],
                                     [np.array([[ 17.5, 205.5]]), {'color':'red', 'edgecolors':"black"}],
                                     [np.array([[ 17.5, 206.5]]), {'color':'red', 'edgecolors':"black"}],
                                     [np.array([[ 18.5, 204.5]]), {'color':'red', 'edgecolors':"black"}],
                                     [np.array([[ 18.5, 205.5]]), {'color':'red', 'edgecolors':"black"}],
                                     [np.array([[ 19.5, 205.5]]), {'color':'red', 'edgecolors':"black"}],
                                     [np.array([[ 45.5, 167.5]]), {'color':'purple', 'edgecolors':"black"}],
                                     [np.array([[ 46.5, 170.5]]), {'color':'purple', 'edgecolors':"black"}],
                                     [np.array([[ 46.5, 171.5]]), {'color':'purple', 'edgecolors':"black"}],
                                     [np.array([[ 46.5, 172.5]]), {'color':'purple', 'edgecolors':"black"}],
                                     [np.array([[ 46.5, 182.5]]), {'color':'purple', 'edgecolors':"black"}],
                                     [np.array([[ 46.5, 183.5]]), {'color':'purple', 'edgecolors':"black"}],
                                     [np.array([[ 47.5, 173.5]]), {'color':'purple', 'edgecolors':"black"}],
                                     [np.array([[ 47.5, 174.5]]), {'color':'purple', 'edgecolors':"black"}],
                                     [np.array([[ 47.5, 175.5]]), {'color':'purple', 'edgecolors':"black"}],
                                     [np.array([[ 47.5, 176.5]]), {'color':'purple', 'edgecolors':"black"}],
                                     [np.array([[ 47.5, 177.5]]), {'color':'purple', 'edgecolors':"black"}],
                                     [np.array([[ 47.5, 178.5]]), {'color':'purple', 'edgecolors':"black"}],
                                     [np.array([[ 47.5, 179.5]]), {'color':'purple', 'edgecolors':"black"}],
                                     [np.array([[ 48.5, 177.5]]), {'color':'purple', 'edgecolors':"black"}],
                                     [np.array([[ 50.5, 162.5]]), {'color':'purple', 'edgecolors':"black"}],
                                     [np.array([[ 50.5, 163.5]]), {'color':'purple', 'edgecolors':"black"}]]]]


    rg.plot_maps_corr(var='sst', plotlags=[1], save=save,
                      splits=9,
                      kwrgs_plot=kwrgs_plot,
                      min_detect_gc=min_detect_gc,
                      append_str=append_str+'Lag1_scatter')
#%%
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
#%% get Correlation between pattern and PDO

# rg.list_for_MI[0].calc_ts = 'pattern cov'
# rg.cluster_list_MI()
# rg.get_ts_prec()

# df_test = functions_pp.get_df_test(rg.df_data)

# df_PDO_and_SST = df_PDOs.merge(df_test, left_index=True, right_index=True)[['PDO', '1..0..sst_sp']]


# RV_mask = fc_utils.apply_shift_lag(rg.df_splits, 1)['x_pred'].loc[0]
# df_PDO_and_SST = df_PDO_and_SST[RV_mask.values]
# df_PDO_and_SST.corr()

# rg.cluster_list_MI() ; rg.get_ts_prec() ;
# out = rg.fit_df_data_ridge()

# s = 0
# X_pred = out[2]['lag_1'][f'split_{s}'].X_pred
# X_pred.index = df_prec.loc[s].index
# df = X_pred.merge(df_prec.loc[s], left_index=True, right_index=True)
# df = df.merge(PDO1.loc[s], left_index=True, right_index=True)
# df = rg.TV.RV_ts.merge(df, left_index=True, right_index=True)

#%% Make plot to show eastern RW is in phase, while western RW is not.

# first get PDO
SST_pp_filepath = user_dir + '/surfdrive/ERA5/input_raw/preprocessed/sst_1979-2020_jan_dec_monthly_1.0deg.nc'
df_PDO, PDO_patterns = climate_indices.PDO(SST_pp_filepath, None)
neg_PDO = (-1*PDO_patterns[0]).drop_vars('split')

PDO_plot_kwrgs = {'units':'[-]', 'cbar_vert':-.1,
                  # 'zoomregion':(130,260,20,60),
                  'map_proj':ccrs.PlateCarree(central_longitude=220),
                  'y_ticks':np.array([25, 35, 45, 55, 65]),
                  'x_ticks':np.arange(130, 280, 25),
                  'clevels':np.arange(-.6,.61,.075),
                  'clabels':np.arange(-.6,.61,.3),
                  'subtitles':np.array([['PDO loading pattern']])}
fig = plot_maps.plot_corr_maps(PDO_patterns[0], **PDO_plot_kwrgs)


#%%
list_of_name_path = [(name_or_cluster_label, TVpath),
                         ('sst', os.path.join(path_raw, 'sst_1979-2020_1_12_monthly_1.0deg.nc'))]


list_for_MI   = [BivariateMI(name='sst', func=corr_map,
                            alpha=.05, FDR_control=True,
                            distance_eps=1000, min_area_in_degrees2=1,
                            calc_ts='pattern cov', selbox=(130,260,-10,60),
                            lags=np.array([0,1]))]
rg = RGCPD(list_of_name_path=list_of_name_path,
            list_for_MI=list_for_MI,
            start_end_TVdate=('05-01', '08-01'),
            start_end_date=None,
            start_end_year=(1980, 2020),
            tfreq=2,
            path_outmain=path_out_main,
            append_pathsub='_' + exper)
rg.pp_TV(name_ds=name_ds, detrend=False, anomaly=False, # detrending already done on z500
         kwrgs_core_pp_time={'dailytomonths':True})
rg.pp_precursors()
rg.traintest('random_10')
rg.calc_corr_maps()
#%%
from plot_maps import ccrs
# Optionally set font to Computer Modern to avoid common missing font errors
matplotlib.rc('font', family='serif', serif='cm10')

matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'

kwrgs_plot = PDO_plot_kwrgs.copy()
kwrgs_plot['units'] = None ; kwrgs_plot['hspace'] = .05
kwrgs_plot['subtitles'] = [[''], ['']]
kwrgs_plot['row_dim'] = 'lag' ; kwrgs_plot['col_dim'] = 'split'
kwrgs_plot['cbar_vert'] = -.015
kwrgs_plot['zoomregion'] = (130, 250, 20, 60)
g = rg.plot_maps_corr(min_detect_gc=.5, return_fig=True,
                      mask_xr=False, kwrgs_plot=kwrgs_plot)
for ax in g.axes.flatten():
    neg_PDO.plot.contour(ax=ax, linewidths=3,
                                      levels=[-0.7, -0.5, -0.35, 0., 0.35, 0.5, 0.7],
                                      linestyles=['--', '--', '--', 'solid', '-', '-', '-'],
                                      colors=['black', 'black', 'black', 'green', 'black', 'black', 'black'],
                                      transform=ccrs.PlateCarree())
    ax.set_xlabel(None) ; ax.set_ylabel(None)
g.axes[0,0].set_title('$corr(SST_t, RW_t^E)$',
                       fontdict={'fontsize' : 16},
                      loc='center')
g.axes[1,0].set_title('$corr(SST_{t-1}, RW_t^E)$',
                       fontdict={'fontsize' : 16},
                      loc='center')
g.fig.savefig(os.path.join(rg.path_outsub1, 'east_2monthmean_corr_map_with_PDO.pdf'),
              bbox_inches='tight')
#%%
TVpathwest = '/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/Vijverberg_Coumou_2022_NPJ/data/westRW_summer_center_s1_RepeatedKFold_10_7_tf1.h5'
list_of_name_path = [(name_or_cluster_label, TVpathwest),
                         ('sst', os.path.join(path_raw, 'sst_1979-2020_1_12_monthly_1.0deg.nc'))]


list_for_MI   = [BivariateMI(name='sst', func=corr_map,
                            alpha=.05, FDR_control=True,
                            distance_eps=1000, min_area_in_degrees2=1,
                            calc_ts='pattern cov', selbox=(130,260,-10,60),
                            lags=np.array([0, 1]), dailytomonths=sst_dailytomonths)]
rgw = RGCPD(list_of_name_path=list_of_name_path,
            list_for_MI=list_for_MI,
            start_end_TVdate=('05-01', '08-01'),
            start_end_date=None,
            start_end_year=(1980, 2020),
            tfreq=2,
            path_outmain=path_out_main,
            append_pathsub='_' + exper)
rgw.pp_TV(name_ds=name_ds, detrend=False, anomaly=False, # detrending already done on z500
         kwrgs_core_pp_time={'dailytomonths':True})
rgw.pp_precursors()
rgw.traintest('random_10')
rgw.calc_corr_maps()

#%%
from plot_maps import ccrs

g = rgw.plot_maps_corr(min_detect_gc=.9, return_fig=True, kwrgs_plot=kwrgs_plot)
for ax in g.axes.flatten():
    neg_PDO.plot.contour(ax=ax, linewidths=3,
                                      levels=[-0.7, -0.5, -0.35, 0., 0.35, 0.5, 0.7],
                                      linestyles=['--', '--', '--', 'solid', '-', '-', '-'],
                                      colors=['black', 'black', 'black', 'green', 'black', 'black', 'black'],
                                      transform=ccrs.PlateCarree())
    ax.set_xlabel(None) ; ax.set_ylabel(None)
g.axes[0,0].set_title('$corr(SST_t, RW_t^W)$',
                       fontdict={'fontsize' : 16},
                      loc='center')
g.axes[1,0].set_title('$corr(SST_{t-1}, RW_t^W)$',
                       fontdict={'fontsize' : 16},
                      loc='center')
g.fig.savefig(os.path.join(rg.path_outsub1, 'west_2monthmean_corr_map_with_PDO.pdf'),
              bbox_inches='tight')

#%%
# # remove PDO df
# if 'parcorr__' in exper:
#     sleep(90)
#     os.remove(os.path.join(path_data, 'df_PDOs.h5'))

#%%
# #%% Store data

# rg.cluster_list_MI()
# rg.list_for_MI[0].calc_ts = 'region mean'
# rg.get_ts_prec()

# rename = {'z5000..0..z500_sp': 'Rossby wave (z500)',
#           '0..0..z500_sp': 'Rossby wave (z500) lag 0',
#           '60..0..z500_sp':'Rossby wave (z500) lag 1',
#           '0..0..sst_sp': 'SST lag 0',
#           f'{rg.tfreq}..0..sst_sp': 'SST lag 1',
#           f'{rg.tfreq}..1..sst': 'SST r1 lag 1',
#           f'{rg.tfreq}..2..sst': 'SST r2 lag 1',
#           '0..1..sst': 'SST r1 lag 0',
#           '0..2..sst': 'SST r2 lag 0'}
# rg.df_data = rg.df_data.rename(rename, axis=1)
# rg.store_df()

# #%% Ridge
# import df_ana; import sklearn, functions_pp, func_models
# keys = ['SST r1 lag 1', 'SST r2 lag 1'] # lag 1
# keys = ['SST lag 1']
# kwrgs_model = {'scoring':'neg_mean_squared_error',
#                'alphas':np.logspace(-3, 1, num=10)}
# s = 0
# target_ts = rg.df_data.loc[s].iloc[:,[0]][rg.df_data.loc[s]['RV_mask']].copy()
# target_mean = target_ts.mean().squeeze()
# target_std = target_ts.std().squeeze()
# # standardize :
# target_ts = (target_ts - target_mean) / target_std
# predict, coef, model = rg.fit_df_data_ridge(keys=keys, target=target_ts, tau_min =1,
#                                             tau_max=1,
#                                             transformer=func_models.standardize_on_train,
#                                             kwrgs_model=kwrgs_model)
# prediction = predict.rename({1:'Prediction', 'RVz5000..0..z500_sp':'Rossby wave (z500)'}, axis=1)

# # AR1
# AR1, c, m = rg.fit_df_data_ridge(keys=['Rossby wave (z500)'], target=target_ts,
#                                  tau_min =1,
#                                  tau_max=1,
#                                  transformer=func_models.standardize_on_train,
#                                  kwrgs_model=kwrgs_model)
# AR1 = AR1.rename({1:'AR1 fit'}, axis=1)

# #%%
# import matplotlib.dates as mdates

# df_splits = rg.df_data[['TrainIsTrue', 'RV_mask']]
# df_AR1test = functions_pp.get_df_test(AR1.merge(df_splits,
#                                                     left_index=True,
#                                                     right_index=True)).iloc[:,:2]
# df_test = functions_pp.get_df_test(prediction.merge(df_splits,
#                                                     left_index=True,
#                                                     right_index=True)).iloc[:,:2]

# # Plot
# fig, ax = plt.subplots(1, 1, figsize = (15,5))
# ax.plot(df_test[['Prediction']], label='SST pattern lag 1',
#         color='red',#ax.lines[0].get_color(),
#         linewidth=1)

# y = prediction['Prediction']
# for fold in y.index.levels[0]:
#     label = None ; color = 'red' ;
#     ax.plot(y.loc[fold].index, y.loc[fold], alpha=.1,
#             label=label, color=color)


# ax.xaxis.set_major_locator(mdates.YearLocator(5, month=6, day=3))   # every year)
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# ax.set_xlim(pd.to_datetime('1979-01-01'), xmax=pd.to_datetime('2020-12-31'))
# ax.tick_params(axis='both', labelsize=14)
# fig.autofmt_xdate()

# ax.scatter(df_test[['Prediction']].index,
#            df_test[['Prediction']].values, label=None,
#            color=ax.lines[0].get_color(),
#            s=15)
# ax.plot(df_test[['Rossby wave (z500)']], label='Truth', color='black',
#         linewidth=1)

# ax.plot(df_AR1test[['AR1 fit']], label='AR1', color='grey',
#         linewidth=1, linestyle='--')
# ax.set_title('Out of sample (10-fold) 60-day aggr. RW prediction from lagged SST pattern',
#              fontsize=16)
# ax.hlines(y=0,xmin=pd.to_datetime('1979-06-03'),
#           xmax=pd.to_datetime('2018-08-02'), color='black')


# ax.legend()
# ax.set_ylabel('Standardized E-U.S. RW timeseries', fontsize=16)
# ax.set_ylim(-3,3)


# MSE_func = sklearn.metrics.mean_squared_error
# fullts = rg.df_data.loc[0].iloc[:,0]
# Persistence = fullts.shift(1)[rg.df_data.loc[0]['RV_mask']]
# Persistence = (Persistence - target_mean) / target_std
# MSE_model = MSE_func(df_test.iloc[:,0],df_test.iloc[:,1], squared=False)
# MSE_pers  = MSE_func(df_test.iloc[:,0],Persistence, squared=False)
# MSE_AR1  = MSE_func(df_test.iloc[:,0],df_AR1test.iloc[:,1], squared=False)
# Corr_pers = np.corrcoef(Persistence.values.squeeze(), df_test.iloc[:,0].values)
# Corr_AR1 = np.corrcoef(df_test.iloc[:,0].values, df_AR1test.iloc[:,1].values)

# text1 =  'Corr. coeff. model          : {:.2f}\n'.format(df_test.corr().iloc[0,1])
# text1 +=  'Corr. coeff. persistence : {:.2f}\n'.format(Corr_pers[0][1])
# text1 +=  'Corr. coeff. AR1             : {:.2f}'.format(Corr_AR1[0][1])
# text2 = r'RMSE model          : {:.2f} $\sigma$'.format(MSE_model) + '\n'
# text2 += r'RMSE persistence : {:.2f} $\sigma$'.format(MSE_pers) + '\n'
# text2 += r'RMSE AR1            : {:.2f} $\sigma$'.format(MSE_AR1)

# ax.text(.038, .05, text1,
#         transform=ax.transAxes, horizontalalignment='left',
#         fontdict={'fontsize':14},
#         bbox = dict(boxstyle='round', facecolor='white', alpha=1,
#                     edgecolor='black'))
# ax.text(.395, .05, text2,
#         transform=ax.transAxes, horizontalalignment='left',
#         fontdict={'fontsize':14},
#         bbox = dict(boxstyle='round', facecolor='white', alpha=1,
#                     edgecolor='black'))

# figname = rg.path_outsub1 +f'/forecast_lagged_SST_tf{rg.precur_aggr}.pdf'
# plt.savefig(figname, bbox_inches='tight')


# #%% correlate with PDO part 1
# import climate_indices
# rg.tfreq = 1
# rg.pp_TV(name_ds=name_ds, detrend=False)
# rg.traintest('random10')
# df_PDO, PDO_patterns = climate_indices.PDO(rg.list_precur_pp[2][1], rg.df_splits)
# #%% correlate with PDO part 2
# rg.cluster_list_MI('sst')
# rg.get_ts_prec()
# df = rg.df_data.mean(axis=0, level=1)[['0..0..sst_sp']]
# df = df.rename({'0..0..sst_sp':'N-Pacifc SST (60d)'},axis=1)

# df_PDO_corr = df.merge(df_PDO.mean(axis=0, level=1),left_index=True, right_index=True)
# df_PDO_corr.corr()

# #%% Store daily data
# rg.get_ts_prec(precur_aggr=1)
# rg.df_data = rg.df_data.rename(rename, axis=1)
# rg.store_df()
# #%% interannual variability events?
# import class_RV
# RV_ts = rg.fulltso.sel(time=rg.TV.aggr_to_daily_dates(rg.dates_TV))
# threshold = class_RV.Ev_threshold(RV_ts, event_percentile=85)
# RV_bin, np_dur = class_RV.Ev_timeseries(RV_ts, threshold=threshold, grouped=True)
# plt.hist(np_dur[np_dur!=0])

# #%%


# freqs = [1, 5, 15, 30, 60]
# for f in freqs:
#     rg.get_ts_prec(precur_aggr=f)
#     rg.df_data = rg.df_data.rename({'0..0..z500_sp':'Rossby wave (z500)',
#                                '0..0..sst_sp':'Pacific SST',
#                                '15..0..sst_sp':'Pacific SST (lag 15)',
#                                '0..0..v200_sp':'Rossby wave (v200)'}, axis=1)

#     keys = [['Rossby wave (z500)', 'Pacific SST'], ['Rossby wave (v200)', 'Pacific SST']]
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

# #%% Conditional probability summer RW

# rg.cluster_list_MI()

# rg.get_ts_prec()
# rename = {'z5000..0..z500_sp': 'Rossby wave (z500)',
#           '0..0..v200_sp':'Rossby wave (v300) lag 0',
#           '60..0..v200_sp': 'Rossby wave (v300) lag 1',
#           '0..0..z500_sp': 'Rossby wave (z500) lag 0',
#           '60..0..z500_sp':'Rossby wave (z500) lag 1',
#           '0..0..sst_sp': 'SST lag 0',
#           '60..0..sst_sp': 'SST lag 1',
#           '60..1..sst': 'SST r1 lag 1',
#           '60..2..sst': 'SST r2 lag 1',
#           '0..1..sst': 'SST r1 lag 0',
#           '0..2..sst': 'SST r2 lag 0'}
# rg.df_data = rg.df_data.rename(rename, axis=1)
# #%% (Conditional) Probability Density Function

# import func_models
# import functions_pp

# k = list(rename.values())
# # s = 9
# # df_std = func_models.standardize_on_train(rg.df_data[k], np.logical_and(df_test['RV_mask']))
# df_test = functions_pp.get_df_test(rg.df_data)

# shift = 1
# mask_standardize = df_test['RV_mask']
# df = func_models.standardize_on_train(df_test[k], mask_standardize)
# SST_lag_summer = df['SST lag 60'].shift(shift)[df_test['RV_mask']]
# RV_and_SST_mask = SST_lag_summer > np.percentile(SST_lag_summer, 85)
# fig = df[df_test['RV_mask']][RV_and_SST_mask][k].hist(sharex=True)
# fig[0,0].set_xlim(-3,3)

# mask_summer = (df['SST lag 1'].shift(shift) > np.percentile(df['SST lag 1'], 85))[df_test['RV_mask']]
# #%% Test MJO signal in OLR
# tfreq = 60
# list_of_name_path = [(name_or_cluster_label, TVpathRW),
#                        ('OLR', os.path.join(path_raw, 'OLRtrop_1979-2018_1_12_daily_2.5deg.nc'))]



# list_for_MI   = [BivariateMI(name='OLR', func=BivariateMI.corr_map,
#                                 kwrgs_func={'alpha':.2, 'FDR_control':True},
#                                 distance_eps=600, min_area_in_degrees2=1,
#                                 calc_ts='pattern cov', selbox=(150,360,-30,30))]

# list_for_EOFS = [EOF(name='OLR', neofs=2, selbox=(150,360,-30,30),
#                      n_cpu=1, start_end_date=start_end_TVdate)]

# rg = RGCPD(list_of_name_path=list_of_name_path,
#             list_for_MI=list_for_MI,
#             list_for_EOFS=list_for_EOFS,
#             start_end_TVdate=start_end_TVdate,
#             start_end_date=start_end_date,
#             start_end_year=None,
#             tfreq=tfreq, lags_i=np.array([0,1]),
#             path_outmain=path_out_main,
#             append_pathsub='_' + name_ds)


# rg.pp_TV(name_ds=name_ds, detrend=False)

# rg.pp_precursors(anomaly=False, detrend=False)

# rg.traintest('random10')


# import cartopy.crs as ccrs ; import matplotlib.pyplot as plt

# rg.calc_corr_maps()

# rg.get_EOFs()

# rg.cluster_list_MI()

# rg.get_ts_prec()


# df_test = functions_pp.get_df_test(rg.df_data)
# corr = df_test.corr()