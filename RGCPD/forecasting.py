#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 10:58:26 2019

@author: semvijverberg
"""
#%%
import time
start_time = time.time()
import inspect, os, sys
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
script_dir = "/Users/semvijverberg/surfdrive/Scripts/RGCPD/RGCPD" # script directory
# To link modules in RGCPD folder to this script
os.chdir(script_dir)
sys.path.append(script_dir)
from importlib import reload as rel
import os, datetime
import pandas as pd
import numpy as np
import func_fc
import matplotlib.pyplot as plt
import validation as valid
import exp_fc

# =============================================================================
# load data 
# =============================================================================

strat_1d_CPPA_era5 = '/Users/semvijverberg/surfdrive/MckinRepl/era5_T2mmax_sst_Northern/ran_strat10_s30/data/era5_24-09-19_07hr_lag_0.h5'
strat_1d_CPPA_EC   = '/Users/semvijverberg/surfdrive/MckinRepl/EC_tas_tos_Northern/ran_strat10_s30/data/EC_16-09-19_19hr_lag_0.h5'
CPPA_v_sm_10d = '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/t2mmax_E-US_v200hpa_sm123_m01-09_dt10/18jun-17aug_lag0-0_ran_strat10_s30/pcA_none_ac0.05_at0.05_subinfo/fulldata_pcA_none_ac0.05_at0.05_2019-09-20.h5'
CPPA_sm_10d   = '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/t2mmax_E-US_sm123_m01-09_dt10/18jun-17aug_lag0-0_ran_strat10_s30/pcA_none_ac0.05_at0.05_subinfo/fulldata_pcA_none_ac0.05_at0.05_2019-09-24.h5'
CPPA_sm_30d = '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/t2mmax_E-US_sm123_m01-09_dt30/18jun-17aug_lag0-0_ran_strat10_s30/pcA_none_ac0.01_at0.01_subinfo/fulldata_pcA_none_ac0.01_at0.01_2019-10-18.h5'
RGCPD_sst_sm_10d = '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/t2mmax_E-US_sst_sm123_m01-09_dt10/18jun-17aug_lag0-0_ran_strat10_s30/pcA_none_ac0.01_at0.01_subinfo/fulldata_pcA_none_ac0.01_at0.01_2019-10-04.h5'

n_boot = 2000
verbosity = 0



#%%
logit = ('logit', None)

GBR_model = ('GBR', 
              {'max_depth':3,
               'learning_rate':1E-3,
               'n_estimators' : 1250,
               'max_features':'sqrt',
               'subsample' : 0.6} )
    
logitCV = ('logit-CV', { 'class_weight':{ 0:1, 1:1},
                'scoring':'brier_score_loss',
                'penalty':'l2',
                'solver':'lbfgs'})
    
GBR_logitCV = ('GBR-logitCV', 
              {'max_depth':3,
               'learning_rate':1E-3,
               'n_estimators' : 750,
               'max_features':'sqrt',
               'subsample' : 0.6} )  

GBR_logitCV_tuned = ('GBR-logitCV', 
          {'max_depth':[3,5,7],
           'learning_rate':1E-3,
           'n_estimators' : 750,
           'max_features':'sqrt',
           'subsample' : 0.6} ) 
    

# format {'dataset' : (path_data, list(keys_options) ) }

ERA_and_EC_daily  = {'ERA-5':(strat_1d_CPPA_era5, ['PEP', 'CPPA']),
                 'EC-earth 2.3':(strat_1d_CPPA_EC, ['PEP', 'CPPA'])}
stat_model_l = [GBR_logitCV, logit]

   
#
ERA5         = {'ERA-5:':(CPPA_sm_30d, ['sst(PEP)+sm', 'sst(PDO,ENSO)+sm', 'sst(CPPA)+sm'])}
#ERA_Bram         = {'ERA-5:':(CPPA_sm_10d, ['all'])}
stat_model_l = [GBR_logitCV, logit]

#RGCPD       = {'RGCPD:' : (RGCPD_sst_sm_10d, ['only_db_regs', 'causal only_db_regs'])}
#stat_model_l = [logit, GBR_logitCV]
#
#ERA_sp      = {'ERA-5:':(CPPA_sm_10d, ['CPPAregs+sm', 'CPPApattern+sm', 'sst(CPPA)+sm'])}
stat_model_l = [GBR_logitCV, logit]

datasets_path = ERA5

causal = False
experiments = {} #; keys_d_sets = {}
for dataset, path_key in datasets_path.items():
#    keys_d = exp_fc.compare_use_spatcov(path_data, causal=causal)
    
    path_data = path_key[0]
    keys_options = path_key[1]
    
#    keys_d = exp_fc.CPPA_precursor_regions(path_data, 
#                                           keys_options=keys_options)
    
    keys_d = exp_fc.normal_precursor_regions(path_data, 
                                             keys_options=keys_options,
                                             causal=causal)

    for master_key, feature_keys in keys_d.items():
        kwrgs_pp = {'EOF':False, 
                    'expl_var':0.5,
                    'fit_model_dates' : None}
        experiments[dataset+' '+master_key] = (path_data, {'keys':feature_keys,
                                           'kwrgs_pp':kwrgs_pp
                                           })

#%%
# import original Response Variable timeseries:
path_ts = '/Users/semvijverberg/surfdrive/MckinRepl/RVts'
RVts_filename = 'era5_t2mmax_US_1979-2018_averAggljacc0.25d_tf1_n4__to_t2mmax_US_tf1_selclus4_okt19.npy'
filename_ts = os.path.join(path_ts, RVts_filename)
kwrgs_events_daily =    (filename_ts, 
                         {  'event_percentile': 90,
                        'min_dur' : 1,
                        'max_break' : 0,
                        'grouped' : False   }
                         )

kwrgs_events = kwrgs_events_daily
    
kwrgs_events = {'event_percentile': 66,
                'min_dur' : 1,
                'max_break' : 0,
                'grouped' : False}



dict_experiments = {}
for dataset, tuple_sett in experiments.items():
    '''
    Format output is 
    dict(
            exper_name = dict( statmodel=tuple(df_valid, RV, y_pred) ) 
        )
    '''
    path_data = tuple_sett[0]
    kwrgs_exp = tuple_sett[1]
    dict_of_dfs = func_fc.load_hdf5(path_data)
    df_data = dict_of_dfs['df_data']
    splits  = df_data.index.levels[0]
    tfreq = (df_data.loc[0].index[1] - df_data.loc[0].index[0]).days
    if tfreq == 1:
        lags_i = np.arange(0, 70+1E-9, max(10,tfreq), dtype=int)
    else:
        lags_i = np.array(np.arange(0, 70+1E-9, max(10,tfreq))/max(10,tfreq), dtype=int)

#    lags_i = np.array([0], dtype=int)
    
    if 'keys' not in kwrgs_exp:
        # if keys not defined, getting causal keys
        kwrgs_exp['keys'] = exp_fc.normal_precursor_regions(path_data, causal=True)['normal_precursor_regions']

    print(kwrgs_events)
    df_data  = func_fc.load_hdf5(path_data)['df_data']
    dict_sum = func_fc.forecast_wrapper(df_data, kwrgs_exp=kwrgs_exp, kwrgs_events=kwrgs_events, 
                            stat_model_l=stat_model_l, 
                            lags_i=lags_i, n_boot=n_boot)

    dict_experiments[dataset] = dict_sum
    
df_valid, RV, y_pred = dict_sum[stat_model_l[-1][0]]





def print_sett(experiments, stat_model_l, filename):
    f= open(filename+".txt","w+")
    lines = []
    
    lines.append("\nEvent settings:")
    lines.append(kwrgs_events)
    
    lines.append(f'\nModels used:')  
    for m in stat_model_l:
        lines.append(m)
        
    lines.append(f'\nnboot: {n_boot}')
    e = 1
    for k, item in experiments.items():
        
        lines.append(f'\n\n***Experiment {e}***\n\n')
        lines.append(f'Title \t : {k}')
        lines.append(f'file \t : {item[0]}')
        for key, it in item[1].items():
            lines.append(f'{key} : {it}')
        e+=1
    
    [print(n, file=f) for n in lines]
    f.close()
    [print(n) for n in lines]

        
  
RV_name = 't2mmax'
working_folder = '/Users/semvijverberg/surfdrive/RGCPD_mcKinnon/forecasting'
pdfs_folder = os.path.join(working_folder,'pdfs')
if os.path.isdir(pdfs_folder) != True : os.makedirs(pdfs_folder)
today = datetime.datetime.today().strftime('%Hhr_%Mmin_%d-%m-%Y')
f_name = f'{RV_name}_{tfreq}d_{today}'


#%%
#import valid_plots as dfplots
#rename_ERA =    {'ERA-5: sst(PEP)+sm':'PEP+sm', 
#             'ERA-5: sst(PDO,ENSO)+sm':'PDO+ENSO+sm', 
#             'ERA-5: sst(CPPA)+sm':'CPPA+sm'}
#
#for old, new in rename_ERA.items():
#    if new not in dict_experiments.keys():
#        dict_experiments[new] = dict_experiments.pop(old)

rename_EC = {'ERA-5 PEP':'PEP', 
             'ERA-5 CPPA':'CPPA', 
             'EC-earth 2.3 PEP':'PEP ', 
             'EC-earth 2.3 CPPA':'CPPA '}

for old, new in rename_EC.items():
    if new not in dict_experiments.keys():
        dict_experiments[new] = dict_experiments.pop(old)

#rename_CPPA_comp =    {'ERA-5: CPPAregs+sm' : 'precursor regions + sm', 
#                       'ERA-5: CPPApattern+sm': 'precursor pattern + sm', 
#                       'ERA-5: sst(CPPA)+sm' : 'CPPA (all) + sm'}
#
#for old, new in rename_CPPA_comp.items():
#    if new not in dict_experiments.keys():
#        dict_experiments[new] = dict_experiments.pop(old)
f_formats = ['.png', '.pdf']
f_format = '.png' 
for f_format in f_formats:
    filename = os.path.join(working_folder, f_name)
    
    group_line_by = None
    #group_line_by = ['ERA-5', 'EC-Earth']
    col_wrap = None
    wspace = 0.05
    kwrgs = {'wspace':wspace, 'col_wrap':col_wrap}
#    kwrgs = {'wspace':0.25, 'col_wrap' : 3}
    met = ['AUC-ROC', 'AUC-PR', 'BSS', 'Rel. Curve'] #'Precision', 'Accuracy'
#    met = ['AUC-ROC', 'AUC-PR', 'BSS', 'Rel. Curve', 'Precision', 'Accuracy']
    expers = list(dict_experiments.keys())
    models   = list(dict_experiments[expers[0]].keys())
    line_dim = 'model'
    
    
    fig = dfplots.valid_figures(dict_experiments, expers=expers, models=models,
                              line_dim=line_dim, 
                              group_line_by=group_line_by,  
                              met=met, **kwrgs)
    if f_format == '.png':
        fig.savefig(os.path.join(filename + f_format), 
                    bbox_inches='tight') # dpi auto 600
    elif f_format == '.pdf':
        fig.savefig(os.path.join(pdfs_folder,f_name+ f_format), 
                    bbox_inches='tight')
    
print_sett(experiments, stat_model_l, filename)


np.save(filename + '.npy', dict_experiments)
#%%
# =============================================================================
# Cross-correlation matrix
# =============================================================================
#f_format = '.pdf' 
#
#path_data = strat_1d_CPPA_era5
#win = 1
#
#period = ['fullyear', 'summer60days', 'pre60days'][1]
#df_data = func_fc.load_hdf5(path_data)['df_data']
##df_data['0_104_PDO'] = df_data['0_104_PDO'] * -1
#f_name = f'Cross_corr_strat_1d_CPPA_era5_win{win}_{period}'
#columns = ['t2mmax', '0_100_CPPAspatcov', '0_101_PEPspatcov', '0_901_PDO', '0_900_ENSO34']
#rename = {'t2mmax':'T95', 
#          '0_100_CPPAspatcov':'CPPA', 
#          '0_101_PEPspatcov':'PEP',
#          '0_901_PDO' : 'PDO',
#          '0_900_ENSO34': 'ENSO'}
#dfplots.build_ts_matric(df_data, win=win, lag=0, columns=columns, rename=rename, period=period)
#if f_format == '.png':
#    plt.savefig(os.path.join(working_folder, f_name + f_format), 
#                bbox_inches='tight') # dpi auto 600
#elif f_format == '.pdf':
#    plt.savefig(os.path.join(pdfs_folder,f_name+ f_format), 
#                bbox_inches='tight')

# =============================================================================
#%% Translation to extremes
# =============================================================================

#
#all_expers = list(dict_experiments.keys())
#name_exp = all_expers[-1]
#models = list(dict_experiments[name_exp].keys())
#name_model = models[-1]
## import original timeseries:
#path_ts = '/Users/semvijverberg/surfdrive/MckinRepl/RVts'
#RVts_filename = 'era5_t2mmax_US_1979-2018_averAggljacc0.25d_tf1_n4__to_t2mmax_US_tf1_selclus4.npy'
#filename_ts = os.path.join(path_ts, RVts_filename)
#kwrgs_events_daily = {  'event_percentile': 90,
#                        'min_dur' : 1,
#                        'max_break' : 0,
#                        'grouped' : False   }



#def pers_ano_to_extr(filename_ts, RV, kwrgs_events_daily, dict_experiments, 
#                     name_exp, name_model, n_boot):
#    
#   
#    # loading in daily timeseries
#    RVfullts = np.load(filename_ts, encoding='latin1',
#                             allow_pickle=True).item()['RVfullts95']
#
#    # Retrieve information on input timeseries
#    import functions_pp
#    dates = functions_pp.get_oneyr(RV.RV_ts.index)
#    tfreq = (dates[1] - dates[0]).days
#    start_date = dates[0] - pd.Timedelta(f'{tfreq/2}d')
#    end_date   = dates[-1] + pd.Timedelta(f'{-1+tfreq/2}d')
#    yr_daily  = pd.DatetimeIndex(start=start_date, end=end_date,
#                                    freq=pd.Timedelta('1d'))
#    ext_dates = functions_pp.make_dates(RV.RV_ts.index, yr_daily, 
#                                        RV.RV_ts.index.year[-1])
#
#    df_RV_ts_e = pd.DataFrame(RVfullts.sel(time=ext_dates).values, 
#                              index=ext_dates, columns=['RV_ts'])
#    df_RVfullts = pd.DataFrame(RVfullts.values, 
#                              index=pd.to_datetime(RVfullts.time.values), 
#                              columns=['RVfullts'])
#    
#    # Make new class based on new kwrgs_events_daily
#    RV_d = func_fc.RV_class(df_RVfullts, df_RV_ts_e, kwrgs_events_daily)
#    # Ensure that the bins on the daily time series matches the original
#    ex = dict(sstartdate = f'{yr_daily[0].month}-{yr_daily[0].day}',
#              senddate   = f'{yr_daily[-1].month}-{yr_daily[-1].day}',
#              startyear  = ext_dates.year[0],
#              endyear    = ext_dates.year[-1])
#    RV_d.RV_bin, dates_gr = functions_pp.time_mean_bins(RV_d.RV_bin, ex, tfreq)
#    RV_d.RV_bin[RV_d.RV_bin>0] = 1
#    RV_d.TrainIsTrue = RV.TrainIsTrue
#    RV_d.RV_mask     = RV.RV_mask
#    # add new probability of event occurence     
#    RV_d.prob_clim = func_fc.get_obs_clim(RV_d)
#    
#
#    dict_comparison = {}
#    # loading model predicting pers. anomalies
#    orig_event_perc = np.round(1 - float(RV.prob_clim.mean()), 2)
#    new_name = '{}d mean +{}p to +{}p events'.format(tfreq, orig_event_perc, 
#                kwrgs_events_daily['event_percentile'])
#    
#    dict_sum = dict_experiments[name_exp]
#    df_valid, RV, y_pred = dict_sum[models[-1]]
#    
#    blocksize = valid.get_bstrap_size(RV.RVfullts, plot=False)
#    out = valid.get_metrics_sklearn(RV_d, y_pred, RV_d.prob_clim, n_boot=n_boot,
#                                    blocksize=blocksize)
#    df_valid, metrics_dict = out
#    dict_comparison[new_name] = {name_model : (df_valid, RV_d, y_pred)}
#    return dict_comparison
#
##%%
#
#dict_comparison = pers_ano_to_extr(filename_ts, RV, kwrgs_events_daily, dict_experiments, 
#                     name_exp, name_model, n_boot)
#
#df_valid_ex, RV_d, y_pred = dict_comparison[new_name][name_model]
#f_format = '.png' 
#f_name_extremes = '{}_{}d_{}p_to_{}p_events_{}'.format(RV_name, tfreq, 
#                   kwrgs_events['event_percentile'],
#                   kwrgs_events_daily['event_percentile'], today)
#filename = os.path.join(working_folder, f_name_extremes)
#
#group_line_by = None
##group_line_by = ['ERA-5', 'EC']
#kwrgs = {'wspace':0.08}
#met = ['AUC-ROC', 'AUC-PR', 'BSS', 'Precision', 'Rel. Curve']
#expers = list(dict_comparison.keys())
#models   = list(dict_comparison[expers[0]].keys())
#
#fig = dfplots.valid_figures(dict_comparison, expers=expers, models=models,
#                          line_dim='exper', 
#                          group_line_by=group_line_by,  
#                          met=met, **kwrgs)
#if f_format == '.png':
#    fig.savefig(os.path.join(filename + f_format), 
#                bbox_inches='tight') # dpi auto 600
#elif f_format == '.pdf':
#    fig.savefig(os.path.join(pdfs_folder, f_name_extremes + f_format), 
#                bbox_inches='tight')
#
#filename_extreme = os.path.join(working_folder, f_name_extremes)
#print_sett(experiments, stat_model_l, filename_extreme)
#
#
#np.save(filename + '.npy', dict_experiments)



#threshold = func_fc.Ev_threshold(RVts_daily, 90)
#RV_bin    = func_fc.Ev_timeseries(RVts_daily,
#                               threshold=threshold ,
#                               min_dur=4,
#                               max_break=1,
#                               grouped=True)[0]     
#df_fc_ext = pd.DataFrame(np.stack([RV_bin_gr.values.squeeze(), y_pred.loc[RV_bin_gr.index][0]], axis=1),
#             index=RV_bin_gr.index, columns=['RV', 'fc'])
#df_fc = y_pred[1]
#df_fc['RV'] = RV_bin_gr
##df_fc_ext['RV'] = RV_bin                                         
#fraction_of_positives,mean_predicted_values = calibration_curve(df_fc_ext['RV'], df_fc_ext['fc'], n_bins=5); 
#clim_prob = RV_bin_gr[RV_bin_gr.values==1.0].size/RV_bin_gr.size
#plt.hlines(clim_prob, 0, 1)
#plt.scatter(mean_predicted_values, fraction_of_positives) 
#dict_to_extremes