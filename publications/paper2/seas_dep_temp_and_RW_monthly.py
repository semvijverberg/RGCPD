#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 09:59:06 2020

@author: semvijverberg
"""


import os, inspect, sys
import matplotlib as mpl
if sys.platform == 'linux':
    mpl.use('Agg')
else:
    # Optionally set font to Computer Modern to avoid common missing font errors
    mpl.rc('font', family='serif', serif='cm10')

    mpl.rc('text', usetex=True)
import numpy as np
from time import time
import cartopy.crs as ccrs ; import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import xarray as xr
import csv
# import sklearn.linear_model as scikitlinear
import argparse

user_dir = os.path.expanduser('~')
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-2])
RGCPD_func = os.path.join(main_dir, 'RGCPD')
assert main_dir.split('/')[-1] == 'RGCPD', 'main dir is not RGCPD dir'
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
import func_models as fc_utils
import functions_pp; import df_ana
import plot_maps; import core_pp

# Optionally set font to Computer Modern to avoid common missing font errors
# matplotlib.rc('font', family='serif', serif='cm10')

# matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'

# target = 'temperature'

# if region == 'eastern':
#     targets = ['easterntemp', 'easternRW']
#     targets = ['easterntemp']
# elif region == 'western':
#     targets = ['westerntemp', 'westernRW']

targets = ['westerntemp', 'easterntemp']


expers = np.array(['adapt_corr']) # np.array(['fixed_corr', 'adapt_corr'])
remove_PDOyesno = np.array([0])
seeds = np.array([1,2,3])
combinations = np.array(np.meshgrid(targets, expers, seeds, remove_PDOyesno)).T.reshape(-1,4)

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
    target = out[0]
    experiment = out[1]
    seed = int(out[2])
    remove_PDO = bool(int(out[3]))
    if target[-4:]=='temp':
        tfreq = 2
    else:
        tfreq = 2
    print(f'arg {args.intexper} f{out}')
else:
    target = targets[2]
    tfreq = 15
    experiment = 'fixed_corr'
    experiment = 'adapt_corr'
    remove_PDO = False

mainpath_df = os.path.join(main_dir, 'publications/paper2/output/heatwave_circulation_v300_z500_SST/57db0USCA/')
calc_ts='region mean' # pattern cov
# t2m
TVpath = 'z500_145-325-20-62USCA.h5'
# mx2t
TVpath = 'z500_145-325-20-620a6f6USCA.h5'
# mx2t 25N-70N
TVpath = 'z500_155-300-20-7357db0USCA.h5'

TVpath  = os.path.join(mainpath_df, TVpath)



if target[-4:] == 'temp':
    # TVpath = user_dir + '/surfdrive/output_RGCPD/circulation_US_HW/tf15_nc3_dendo_0ff31.nc'
    alpha_corr = .05

    if target == 'westerntemp':
        cluster_label = 'mx2twest' #4 # 1
        corlags = np.array([1])
    elif target == 'easterntemp':
        cluster_label = 'mx2teast' # 2
        corlags = np.array([1])
    name_ds=f'{cluster_label}'
elif target[-2:] == 'RW':
    cluster_label = '' # 'z500'
    name_ds = target[:4]
    alpha_corr = .05
    # if target == 'easternRW':
    #     TVpath = os.path.join(data_dir, '2020-10-29_13hr_45min_east_RW.h5')
    # elif target == 'westernRW':
    #     TVpath = os.path.join(data_dir, '2020-10-29_10hr_58min_west_RW.h5')

precur_aggr = tfreq
method     = 'ranstrat_10' ;
n_boot = 2000

append_main = ''
name_csv = f'skill_scores_tf{tfreq}.csv'


#%% run RGPD
start_end_TVdate = ('06-01', '08-31')
start_end_date = ('1-1', '12-31')
list_of_name_path = [(cluster_label, TVpath),
                     ('sst', os.path.join(path_raw, 'sst_1979-2020_1_12_monthly_1.0deg.nc'))]

list_for_MI   = [BivariateMI(name='sst', func=class_BivariateMI.corr_map,
                            alpha=alpha_corr, FDR_control=True,
                            kwrgs_func={},
                            distance_eps=1700, min_area_in_degrees2=3,
                            calc_ts=calc_ts, selbox=(130,260,-10,60),
                            lags=corlags)]
if calc_ts == 'region mean':
    s = ''
else:
    s = '_' + calc_ts.replace(' ', '')

path_out_main = os.path.join(main_dir, f'publications/paper2/output/{target}{s}{append_main}/')

rg = RGCPD(list_of_name_path=list_of_name_path,
           list_for_MI=list_for_MI,
           list_import_ts=None,
           start_end_TVdate=start_end_TVdate,
           start_end_date=start_end_date,
           start_end_year=None,
           tfreq=tfreq,
           path_outmain=path_out_main,
           append_pathsub='_' + experiment)
precur = rg.list_for_MI[0] ; lag = precur.lags[0]
if precur.func.__name__ == 'corr_map':

    title = '$corr(SST_{t-1},\ $'+f'$T^{target[0].capitalize()}_t)$'
else:
    title = r'$parcorr(SST_t, mx2t_t\ |\ SST_{t-1},mx2t_{t-1})$'
subtitles = np.array([['']]) #, f'lag 2 (15 day lead)']] )
kwrgs_plotcorr = {'row_dim':'split', 'col_dim':'lag','aspect':2, 'hspace':-.47,
              'wspace':-.15, 'size':3, 'cbar_vert':-.1,
              'units':'Corr. Coeff. [-]', 'zoomregion':(130,260,-10,60),
              'clim':(-.60,.60), 'map_proj':ccrs.PlateCarree(central_longitude=220),
              'y_ticks':np.arange(-10,61,20), 'x_ticks':np.arange(130, 280, 25),
              'subtitles':subtitles, 'title':title,
              'title_fontdict':{'fontsize':16, 'fontweight':'bold'}}

#%%
append_str = experiment + f'lag{lag}'
if experiment == 'fixed_corr':
    rg.pp_TV(name_ds=name_ds, detrend=False,
             anomaly=True, kwrgs_core_pp_time={'dailytomonths':True})

    subfoldername = '_'.join([target,rg.hash, experiment.split('_')[0],
                          str(precur_aggr), str(alpha_corr), method,
                          str(seed)])

    if remove_PDO:
        subfoldername += '_rmPDO'
    rg.pp_precursors()
    rg.traintest(method=method, seed=seed, subfoldername=subfoldername)
    rg.calc_corr_maps()
    rg.cluster_list_MI()
    rg.quick_view_labels(save=True, append_str=append_str)
    # plotting corr_map
    rg.plot_maps_corr(var='sst', save=True,
                      kwrgs_plot=kwrgs_plotcorr,
                      min_detect_gc=1.0,
                      append_str=append_str)



# rg.get_ts_prec()
#%% (Adaptive) forecasting


def append_dict(month, df_test_m):
    dkeys = [f'{month} RMSE-SS', f'{month} Corr.']
    append_dict = {dkeys[0]:float(df_test_m.iloc[:,0].round(3)),
                   dkeys[1]:float(df_test_m.iloc[:,1].round(3))}
    dict_v.update(append_dict)
    return

months = {'May-June'    : ('05-01', '06-30'),
          'June-July'   : ('06-01', '07-30'),
           'July-Aug'    : ('07-01', '08-31'),
           'Aug-Sept'    : ('08-01', '09-30'),
           'Sept-Okt'    : ('09-01', '10-31')}


monthkeys= list(months.keys()) ; oneyrsize = 0

if remove_PDO:
    import wrapper_PCMCI as wPCMCI
    lowpass = '2y'
    rg.list_import_ts = [('PDO', os.path.join(data_dir, f'PDO_{lowpass}_rm_25-09-20_15hr.h5'))]


blocksize=1
lag = 1
# only needed such that score_func_list exists
RMSE_SS = fc_utils.ErrorSkillScore(constant_bench=0).RMSE
MAE_SS = fc_utils.ErrorSkillScore(constant_bench=0).MAE
CRPSS = fc_utils.CRPSS_vs_constant_bench(constant_bench=0).CRPSS
score_func_list = [RMSE_SS, fc_utils.corrcoef, MAE_SS]

dict_v = {'target':target, 'lag':lag,'rmPDO':str(remove_PDO), 'exper':experiment,
          'Seed':f's{seed}'}


list_test = []
list_test_b = []
no_info_fc = []
dm = {} # dictionairy months
dmc = {} # dictionairy months clusters
for month, start_end_TVdate in months.items():
    month, start_end_TVdate = list(months.items())[1]
    if experiment == 'fixed_corr':
        # overwrite RV_mask
        rg.get_ts_prec(precur_aggr=precur_aggr,
                       start_end_TVdate=start_end_TVdate)
    elif experiment == 'adapt_corr':
        rg.start_end_TVdate = start_end_TVdate # adapt target period
        rg.pp_TV(name_ds=name_ds, anomaly=True, kwrgs_core_pp_time={'dailytomonths':True})
        subfoldername = '_'.join([target,rg.hash, experiment.split('_')[0],
                          str(precur_aggr), str(alpha_corr), method,
                          str(seed)])
        if remove_PDO:
            subfoldername += '_rmPDO'
        rg.pp_precursors()
        rg.traintest(method=method, seed=seed, subfoldername=subfoldername)
        finalfolder = rg.path_outsub1.split('/')[-1]

        rg.calc_corr_maps()
        rg.cluster_list_MI()
        rg.quick_view_labels(save=True, append_str=append_str+'+'+month)
        rg.get_ts_prec(precur_aggr=precur_aggr)

        # plotting corr_map
        rg.plot_maps_corr(var='sst', save=True,
                          kwrgs_plot=kwrgs_plotcorr,
                          min_detect_gc=0.9,
                          append_str=append_str+'_'+month)
        dm[month] = rg.list_for_MI[0].corr_xr.copy()
        dmc[month] = rg.list_for_MI[0].prec_labels.copy()

    alphas = np.append([1E-3, 1E-2, 1E-1], np.logspace(.1, 2, num=25))
    kwrgs_model = {'scoring':'neg_mean_squared_error',
                   'alphas':alphas, # large a, strong regul.
                   'normalize':False}
    print('\n', month, '\n')
    keys = [k for k in rg.df_data.columns[:-2] if k not in [rg.TV.name, 'PDO']]
    if target == 'easterntemp':
        keys = [k for k in keys if int(k.split('..')[1]) in [1,2]]
    if remove_PDO:
        y_keys = [k for k in keys if 'sst' in k]
        rg.df_data[y_keys], fig = wPCMCI.df_data_remove_z(rg.df_data, z=['PDO'], keys=y_keys,
                                                   standardize=False)
        fig_path = os.path.join(rg.path_outsub1, f'regressing_out_PDO_tf{month}')
        fig.savefig(fig_path+rg.figext, bbox_inches='tight')

    if any(rg._df_count == rg.n_spl): # at least one timeseries always present
        oneyr = functions_pp.get_oneyr(rg.df_data['RV_mask'].loc[0][rg.df_data['RV_mask'].loc[0]])
        oneyrsize = oneyr.size
        if monthkeys.index(month) >= 1:
            nextyr = functions_pp.get_oneyr(rg.df_data['RV_mask'].loc[0][rg.df_data['RV_mask'].loc[0]])
            if nextyr.size != oneyrsize:
                raise ValueError

        fc_mask = rg.df_data.iloc[:,-1].loc[0]#.shift(lag, fill_value=False)
        # rg.df_data = rg._replace_RV_mask(rg.df_data, replace_RV_mask=(fc_mask))
        target_ts = rg.df_data.iloc[:,[0]].loc[0][fc_mask]
        target_ts = (target_ts - target_ts.mean()) / target_ts.std()

        # ScikitModel = scikitlinear.LassoCV

        out_fit = rg.fit_df_data_ridge(target=target_ts,
                                   keys=keys,
                                   tau_min=lag, tau_max=lag,
                                   kwrgs_model=kwrgs_model,
                                   transformer=None)

        predict, weights, models_lags = out_fit
        prediction = predict.rename({predict.columns[0]:'target',lag:'Prediction'},
                                    axis=1)

        if monthkeys.index(month)==0:
            weights_norm = weights.mean(axis=0, level=1)
            weights_norm.div(weights_norm.max(axis=0)).T.plot(kind='box')

        clim_mean_temp = float(target_ts.mean())
        RMSE_SS = fc_utils.ErrorSkillScore(constant_bench=clim_mean_temp).RMSE
        MAE_SS = fc_utils.ErrorSkillScore(constant_bench=clim_mean_temp).MAE
        CRPSS = fc_utils.CRPSS_vs_constant_bench(constant_bench=clim_mean_temp).CRPSS
        score_func_list = [RMSE_SS, fc_utils.corrcoef, MAE_SS]

        df_train_m, df_test_s_m, df_test_m, df_boot = fc_utils.get_scores(prediction,
                                                                 rg.df_data.iloc[:,-2:],
                                                                 score_func_list,
                                                                 n_boot = n_boot,
                                                                 blocksize=blocksize,
                                                                 rng_seed=1)
        df_test_m = df_test_m['Prediction'] ; df_boot = df_boot['Prediction']
        # # Benchmark prediction

        # observed = pd.concat(n_splits*[target_ts], keys=range(n_splits))
        # benchpred = observed.copy()
        # benchpred[:] = np.zeros_like(observed) # fake pred
        # benchpred = pd.concat([observed, benchpred], axis=1)

        # bench_MSE = fc_utils.get_scores(benchpred,
        #                                rg.df_data.iloc[:,-2:][rg.df_data.iloc[:,-1:].values].dropna(),
        #                                score_func_list,
        #                                n_boot = 0,
        #                                blocksize=blocksize,
        #                                rng_seed=1)[2]
        # bench_MSE = float(bench_MSE.values)


        # print(df_test_m)
        # df_boot['mean_squared_error'] = (bench_MSE-df_boot['mean_squared_error'])/ \
        #                                         bench_MSE

        # df_test_m['mean_squared_error'] = (bench_MSE-df_test_m['mean_squared_error'])/ \
        #                                         bench_MSE
        n_splits = rg.df_data.index.levels[0].size
        cvfitalpha = [models_lags[f'lag_{lag}'][f'split_{s}'].alpha_ for s in range(n_splits)]
        print('mean alpha {:.2f}'.format(np.mean(cvfitalpha)))
        maxalpha_c = list(cvfitalpha).count(alphas[-1])
        if maxalpha_c > n_splits/3:
            print(f'\n{month} alpha {int(np.mean(cvfitalpha))}')
            print(f'{maxalpha_c} splits are max alpha\n')
            # maximum regularization selected. No information in timeseries
            # df_test_m['Prediction']['corrcoef'][:] = 0
            # df_boot['Prediction']['corrcoef'][:] = 0
            no_info_fc.append(month)
        df_test = functions_pp.get_df_test(prediction.merge(rg.df_data.iloc[:,-2:],
                                                        left_index=True,
                                                        right_index=True)).iloc[:,:2]

    else:
        print('no precursor timeseries found, scores all 0')
        df_boot = pd.DataFrame(data=np.zeros((n_boot, len(score_func_list))),
                            columns=['RMSE', 'corrcoef', 'MAE'])
        df_test_m = pd.DataFrame(np.zeros((1,len(score_func_list))),
                                 columns=['RMSE', 'corrcoef', 'MAE'])


    list_test_b.append(df_boot)
    list_test.append(df_test_m)
    append_dict(month, df_test_m)
    # df_ana.loop_df(df=rg.df_data[keys], colwrap=1, sharex=False,
    #                       function=df_ana.plot_timeseries,
    #                       kwrgs={'timesteps':rg.fullts.size,
    #                                   'nth_xyear':5})



#%%
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.patches as mpatches
corrvals = [test.values[0,1] for test in list_test]
MSE_SS_vals = [test.values[0,0] for test in list_test]
MAE_SS_vals = [test.values[0,-1] for test in list_test]
rename_metrics = {'RMSE-SS':'RMSE',
                  'Corr. Coef.':'corrcoef',
                  'MAE-SS':'MAE'}
df_scores = pd.DataFrame({'MAE-SS':MAE_SS_vals,'Corr. Coef.':corrvals,
                          # 'RMSE-SS':MSE_SS_vals,
                           },
                         index=monthkeys)
df_test_b = pd.concat(list_test_b, keys = monthkeys,axis=1)

yerr = [] ; quan = [] ; alpha = .05
monmet = np.array(np.meshgrid(monthkeys,
                              df_scores.columns)).T.reshape(-1,2) ;
for i, (mon, met) in enumerate(monmet):
    Eh = 1 - alpha/2 ; El = alpha/2
    met = rename_metrics[met]
    _scores = df_test_b[mon][met]
    tup = [_scores.quantile(El), _scores.quantile(Eh)]
    quan.append(tup)
    mean = df_scores.values.flatten()[i] ;
    tup = abs(mean-tup)
    yerr.append(tup)


fig, ax = plt.subplots(1,1, figsize=(10,8))
# ax.set_facecolor('lightgrey')
plt.style.use('seaborn')
_yerr = np.array(yerr).reshape(df_scores.columns.size,len(monthkeys)*2,
                               order='F').reshape(df_scores.columns.size,2,len(monthkeys))
df_scores.plot.bar(ax=ax, rot=0, yerr=_yerr,
                   capsize=8, error_kw=dict(capthick=1),
                   color=['blue', 'green', 'purple'],
                   legend=False)
# ax.grid(color='white', lw=1.5)
# for noinfo in no_info_fc:
#     # first two children are not barplots
#     idx = monthkeys.index(noinfo) + 3
#     ax.get_children()[idx].set_color('r') # RMSE bar
#     idx = monthkeys.index(noinfo) + 15
#     ax.get_children()[idx].set_color('r') # RMSE bar



ax.set_ylabel('Skill Score', fontsize=20, labelpad=0)
# ax.set_xticklabels(months, fontdict={'fontsize':20})
if target[-4:] == 'temp':
    title = f'Seasonal dependence of $T^{target[0].capitalize()}$ predictions'
elif target[-2:] == 'RW':
    title = f'Seasonal dependence of {precur_aggr}-day mean RW predictions'
ax.set_title(title,
             fontsize=20)
ax.tick_params(axis='y', labelsize=18)
ax.tick_params(axis='x', labelsize=18)


if target=='westerntemp':
    # patch1 = mpatches.Patch(color='blue', label='RMSE-SS')
    patch2 = mpatches.Patch(color='green', label='Corr. Coef.')
    patch3 = mpatches.Patch(color='blue', label='MAE-SS')
    handles = [patch2, patch3]
    legend1 = ax.legend(handles=handles,
              fontsize=20, frameon=True, facecolor='grey',
              framealpha=.5)
    ax.add_artist(legend1)

    # # manually define a new patch
    # if len(no_info_fc) != 0:
    #     patch = mpatches.Patch(color='red', label=r'$\alpha=$'+f' ${int(alphas[-1])}$')
    #     legend2 = ax.legend(loc='upper left', handles=[patch],
    #           fontsize=16, frameon=True, facecolor='grey',
    #           framealpha=.5)
    #     ax.add_artist(legend2)

ax.set_ylim(-0.1, 1.0)
plt.savefig(os.path.join(rg.path_outsub1,
           f'skill_score_vs_months_a{alpha}+{precur_aggr}tf_lag{lag}_nb{n_boot}_blsz{blocksize}_ac{alpha_corr}_corlag{precur.lags}.pdf'),
            bbox_inches='tight')

csvfilename = os.path.join(rg.path_outmain, name_csv)
for csvfilename, dic in [(csvfilename, dict_v)]:
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
min_detect_gc = .9
mpl.rcParams.update(mpl.rcParamsDefault)
if experiment == 'adapt_corr':

    corr = dm[monthkeys[0]].mean(dim='split')
    list_xr = [corr.expand_dims('months', axis=0) for i in range(len(monthkeys))]
    corr = xr.concat(list_xr, dim = 'months')
    corr['months'] = ('months', monthkeys)

    np_data = np.zeros_like(corr.values)
    np_mask = np.zeros_like(corr.values)
    for i, f in enumerate(monthkeys):
        corr_xr = dm[f].copy()
        corr_, mask_ = RGCPD._get_sign_splits_masked(xr_in=corr_xr, min_detect=min_detect_gc,
                                                     mask=dm[f].mask)
        np_data[i] = corr_.values
        np_mask[i] = mask_.values

    corr.values = np_data
    mask = (('months', 'lag', 'latitude', 'longitude'), np_mask )
    corr.coords['mask'] = mask

    precur = rg.list_for_MI[0]
    f_name = 'corr_{}_a{}'.format(precur.name,
                                precur.alpha) + '_' + \
                                f'{experiment}_lag{corlags}_' + \
                                f'tf{precur_aggr}_{method}_gc{min_detect_gc}'
    # ncdf_filepath = os.path.join(rg.path_outsub1, f_name+'.nc')
    # if os.path.isfile(ncdf_filepath): os.remove(ncdf_filepath)
    # corr.to_netcdf(ncdf_filepath, mode='w')
    # import_ds = core_pp.import_ds_lazy
    # corr = import_ds(os.path.join(rg.path_outsub1, f_name+'.nc'))[precur.name]
    # Horizontal plot
    subtitles = np.array([['$t=$'+k+' mean' for k in monthkeys]])
    if corlags[0] == 0:
        title = r'$corr(SST_t, T^{}_t)$'.format('{'+f'{target[0].capitalize()}'+'}')
    else:
        title = r'$corr(SST_{}, T^{}_t)$'.format('{'+f't-{corlags[0]}'+'}', target[0].capitalize())
        # title = '$corr(SST_{'+f't-{corlags[0]}'+'}, T^{}_t)$'.format(target[0].capitalize())
    kwrgs_plot = {'aspect':2, 'hspace':.35,
                  'wspace':-.32, 'size':2, 'cbar_vert':-0.1,
                  'units':'Corr. Coeff. [-]',
                  'map_proj':ccrs.PlateCarree(central_longitude=220),
                  'clevels':np.arange(-.6,.61,.075),
                  'clabels':np.arange(-.6,.61,.3),
                  'y_ticks':False,
                  'x_ticks':False,
                  'subtitles':subtitles,
                  'title':title,
                  'title_fontdict':{'y':1.2, 'fontsize':20}}

    fig = plot_maps.plot_corr_maps(corr, mask_xr=corr.mask, col_dim='months',
                                   row_dim=corr.dims[1],
                                   **kwrgs_plot)
    fig_path = os.path.join(rg.path_outsub1, f_name+'hor')+rg.figext
    plt.savefig(fig_path, bbox_inches='tight')
    # %%
    # vertical plot
    f_name = 'corr_{}_a{}'.format(precur.name,
                                precur.alpha) + '_' + \
                                f'{experiment}_lag{corlags}_' + \
                                f'tf{precur_aggr}_{method}_gc{min_detect_gc}'
    subtitles = np.array([['$t=$'+k+' mean' for k in monthkeys]])
    subtitles = subtitles.reshape(-1,1)
    kwrgs_plot = {'aspect':2, 'hspace':.3,
                  'wspace':-.4, 'size':2, 'cbar_vert':0.06,
                  'units':'Corr. Coeff. [-]',
                  'map_proj':ccrs.PlateCarree(central_longitude=220),
                  'clevels':np.arange(-.6,.61,.075),
                  'clabels':np.arange(-.6,.61,.3),
                  'y_ticks':False,
                  'x_ticks':False,
                  'subtitles':subtitles,
                  'title':title,
                  'title_fontdict':{'y':0.96, 'fontsize':20}}
    fig = plot_maps.plot_corr_maps(corr, mask_xr=corr.mask, col_dim=corr.dims[1],
                                   row_dim='months',
                                   **kwrgs_plot)

    fig_path = os.path.join(rg.path_outsub1, f_name+'vert')+rg.figext
    plt.savefig(fig_path, bbox_inches='tight')
#%%
mpl.rcParams.update(mpl.rcParamsDefault)
if experiment == 'adapt_corr':

    labels = dmc[monthkeys[0]][0]
    list_xr = [labels.expand_dims('months', axis=0) for i in range(len(monthkeys))]
    labels = xr.concat(list_xr, dim = 'months')
    labels['months'] = ('months', monthkeys)

    np_data = np.zeros_like(labels.values)
    np_mask = np.zeros_like(labels.values) ; np_mask[:] = np.nan
    for i, f in enumerate(monthkeys):
        labels_xr = dmc[f].copy()
        labels_, mask_ = RGCPD._get_sign_splits_masked(xr_in=labels_xr, min_detect=min_detect_gc)

        labels_ = labels_.where(mask_)

        vals = labels_.values
        np_data[i] = labels_.values
        np_mask[i] = mask_.values

    labels.values = np_data
    mask = (('months', 'lag', 'latitude', 'longitude'), np_mask )
    labels.coords['mask'] = mask

    precur = rg.list_for_MI[0]
    f_name = 'labels_{}_a{}'.format(precur.name,
                                precur.alpha) + '_' + \
                                f'{experiment}_lag{corlags}_' + \
                                f'tf{precur_aggr}_{method}'

    labels.to_netcdf(os.path.join(rg.path_outsub1, f_name+'.nc'), mode='w')
    # import_ds = core_pp.import_ds_lazy
    # labels = import_ds(os.path.join(rg.path_outsub1, f_name+'.nc'))[precur.name+'_labels_init']
    # Horizontal plot
    subtitles = np.array([monthkeys])

    title = 'Clusters of correlating regions [from $corr(SST_{t-1},$'+f'$\ T_t^{target[0].capitalize()})$]'
    kwrgs_plot = {'aspect':2, 'hspace':.35,
                  'wspace':-.32, 'size':2, 'cbar_vert':-0.1,
                  'units':'',
                  'map_proj':ccrs.PlateCarree(central_longitude=220),
                  'y_ticks':False,
                  'x_ticks':False,
                  'subtitles':subtitles,
                  'title':title,
                  'title_fontdict':{'y':1.2, 'fontsize':20},
                  'col_dim':'months', 'row_dim':labels.dims[1]}

    fig = plot_maps.plot_labels(labels, kwrgs_plot=kwrgs_plot)

    fig_path = os.path.join(rg.path_outsub1, f_name+'hor')+rg.figext

    plt.savefig(fig_path, bbox_inches='tight')
    # %%
    # vertical plot
    title = 'Clusters of \ncorrelating regions'
    f_name = 'labels_{}_a{}'.format(precur.name,
                                precur.alpha) + '_' + \
                                f'{experiment}_lag{corlags}_' + \
                                f'tf{precur_aggr}_{method}'
    subtitles = np.array([monthkeys]).reshape(-1,1)
    kwrgs_plot = {'aspect':2, 'hspace':.3,
                  'wspace':-.4, 'size':2, 'cbar_vert':0.06,
                  'units':'Corr. Coeff. [-]',
                  'map_proj':ccrs.PlateCarree(central_longitude=220),
                  'y_ticks':False,
                  'x_ticks':False,
                  'subtitles':subtitles,
                  'title':title,
                  'title_fontdict':{'y':0.96, 'fontsize':18},
                  'col_dim':labels.dims[1], 'row_dim':'months'}
    fig = plot_maps.plot_labels(labels, kwrgs_plot=kwrgs_plot)

    fig_path = os.path.join(rg.path_outsub1, f_name+'vert')+rg.figext
    plt.savefig(fig_path, bbox_inches='tight')

#%%%
# df = df_test_b.stack().reset_index(level=1)
# dfx = df.groupby(['level_1'])
# axes = dfx.boxplot()
# axes[0].set_ylim(-0.5, 1)
#%%
# import seaborn as sns
# df_ap = pd.concat(list_test_b, axis=0, ignore_index=True)
# df_ap['months'] = np.repeat(monthkeys, list_test_b[0].index.size)
# # df_ap.boxplot(by='months')
# ax = sns.boxplot(x=df_ap['months'], y=df_ap['mean_squared_error'])
# ax.set_ylim(-0.5, 1)
# plt.figure()
# ax = sns.boxplot(x=df_ap['months'], y=df_ap['corrcoef'])
# ax.set_ylim(-0.5, 1)

# #%%
# columns_my_order = monthkeys
# fig, ax = plt.subplots()
# for position, column in enumerate(columns_my_order):
#     ax.boxplot(df_test_b.loc[column], positions=[position,position+.25])

# ax.set_xticks(range(position+1))
# ax.set_xticklabels(columns_my_order)
# ax.set_xlim(xmin=-0.5)
# plt.show()


#%%

