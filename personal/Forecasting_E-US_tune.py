#!/usr/bin/env python
# coding: utf-8

# # Forecasting
# Below done with test data, same format as df_data

# In[1]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
import os, inspect, sys
import numpy as np
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-1])
python_dir = os.path.join(main_dir, 'RGCPD')
df_ana_dir = os.path.join(main_dir, 'df_analysis/df_analysis/')
if main_dir not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(python_dir)
    sys.path.append(df_ana_dir)

user_dir = os.path.expanduser('~')
if sys.platform == 'linux':
    import matplotlib as mpl
    mpl.use('Agg')
# In[2]:


from func_fc import fcev


# In[3]:
old_CPPA = user_dir + '/surfdrive/MckinRepl/era5_T2mmax_sst_Northern/ran_strat10_s30/data/era5_24-09-19_07hr_lag_0.h5'
old = user_dir + '/Downloads/output_RGCPD/20jun-19aug_lag10-10/ran_strat10_s1/None_at0.001_tau_0-1_conds_dim4_combin1.h5'
era5_10d_CPPA_sm = user_dir + '/Downloads/output_RGCPD/Xzkup1_20jun-19aug_lag20-20/random10_s1/df_data_sst_CPPA_sm123_dt10_Xzkup1.h5'
CPPA_10d_sm1_2_3_OLR_l0 = user_dir + '/Downloads/output_RGCPD/Xzkup1_20jun-19aug_lag10-20/random10_s1/df_data_sst_CPPA_sm1_sm2_sm3_OLR_dt10_Xzkup1.h5'
era5_1d_CPPA_lag0 =  user_dir + '/surfdrive/MckinRepl/era5_T2mmax_sst_Northern/Xzkup1_ran_strat10_s30/data/era5_21-01-20_10hr_lag_0_Xzkup1.h5'
era5_1d_CPPA_l10 = user_dir + '/surfdrive/MckinRepl/era5_T2mmax_sst_Northern/Xzkup1_ran_strat10_s30/data/era5_21-01-20_10hr_lag_10_Xzkup1.h5'
era5_16d_CPPA_sm = user_dir + '/Downloads/output_RGCPD/Xzkup1_19jun-22aug_lag16-16/ran_strat10_s1/df_data_sst_CPPA_sm123_dt16_Xzkup1.h5'
era5_16d_RGCPD_sm = user_dir + '/Downloads/output_RGCPD/Xzkup1_19jun-22aug_lag16-16/ran_strat10_s1/df_data__sm123_sst_dt16_Xzkup1.h5'
era5_12d_RGCPD_sm = user_dir + '/Downloads/output_RGCPD/Xzkup1_24may-28aug_lag12-12/ran_strat20_s1/df_data__sm123_sst_dt12_Xzkup1.h5'
era5_10d_RGCPD_sm = user_dir + '/Downloads/output_RGCPD/Xzkup1_10jun-29aug_lag20-20/random10_s1/df_data__sm1_sm2_sm3_OLR_sst_dt10_Xzkup1.h5'
era5_10d_RGCPD_sm_uv = user_dir + '/Downloads/output_RGCPD/Xzkup1_10jun-29aug_lag20-20/random10_s1/df_data__sm123_u500_v200_sst_dt10_Xzkup1.h5'
# In[4]:


#ERA_and_EC_daily  = {'ERA-5':(strat_1d_CPPA_era5, ['PEP', 'CPPA']),
#                 'EC-earth 2.3':(strat_1d_CPPA_EC, ['PEP', 'CPPA'])}
ERA_10d = {'ERA-5':(era5_10d_CPPA_sm, ['sst(PEP)+sm', 'sst(PDO,ENSO)+sm', 'sst(CPPA)+sm'])}
#ERA_10d_sm = {'ERA-5':(era5_10d_CPPA_sm_n, ['sst(PDO,ENSO)', 'sst(CPPA)', 'sst(CPPA)+sm'] )}
ERA_10d_sm = {'ERA-5':(CPPA_10d_sm1_2_3_OLR_l0, ['all'] )}
ERA_1d_CPPA = {'ERA-5':(era5_1d_CPPA_lag0, ['sst(PDO,ENSO)', 'sst(CPPA)'])}
ERA_10d_RGCPD = {'ERA-5':(era5_10d_RGCPD_sm, ['all'])}
ERA_10d_RGCPD_all = {'ERA-5':(era5_10d_RGCPD_sm_uv, ['all'])}
ERA_16d_RGCPD = {'ERA-5':(era5_16d_RGCPD_sm, [None, 'sst(CPPA)'])}
ERA_12d_RGCPD = {'ERA-5':(era5_12d_RGCPD_sm, ['sst(CPPA)+sm', 'sst(CPPA)'])}
ERA_vs_PEP = {'ERA-5':(era5_1d_CPPA_lag0, ['sst(PEP)+sm', 'sst(PDO,ENSO)+sm', 'sst(CPPA)+sm'])}

datasets_path  = ERA_1d_CPPA


# Define statmodel:
logit = ('logit', None)

logitCV = ('logitCV',
          {'class_weight':{ 0:1, 1:1},
           'scoring':'brier_score_loss',
           'penalty':'l2',
           'solver':'lbfgs'})


logitCVfs = ('logitCV',
          {'class_weight':{ 0:1, 1:1},
           'scoring':'brier_score_loss',
           'penalty':'l2',
           'solver':'lbfgs',
           'feat_sel':{'model':None}})

GBC_tfs = ('GBC',
          {'max_depth':[1, 2, 3, 4],
           'learning_rate':[1E-2, 5E-3, 1E-3, 5E-4],
           'n_estimators' : [200, 300, 400, 500, 600, 700, 800, 1000],
           'min_samples_split':[.15, .25],
           'max_features':[.2,'sqrt', .5],
           'subsample' : [.3, .4, .5, 0.6],
           'random_state':60,
           'scoringCV':'brier_score_loss',
           'feat_sel':{'model':None} } )

GBC_t = ('GBC', 
         {'max_depth':[1, 2, 3, 4],
           'learning_rate':[.05, 1E-2, 5E-3],
           'n_estimators' : [100, 250, 400, 550, 700, 850, 1000],
           'min_samples_split':[.15, .25],
           'max_features':[.15, .2, 'sqrt'],
           'subsample' : [.3, .45, 0.6],
           'random_state':60,
           'scoringCV':'brier_score_loss' } )


GBC = ('GBC',
      {'max_depth':1,
       'learning_rate':.05,
       'n_estimators' : 500,
       'min_samples_split':.25,
       'max_features':.4,
       'subsample' : .6,
       'random_state':60,
       'n_iter_no_change':20,
       'tol':1E-4,
       'validation_fraction':.3,
       'scoringCV':'brier_score_loss'
       } )

# In[6]:
path_ts = '/Users/semvijverberg/surfdrive/MckinRepl/RVts'
RVts_filename = '/Users/semvijverberg/surfdrive/MckinRepl/RVts/era5_t2mmax_US_1979-2018_averAggljacc0.25d_tf1_n4__to_t2mmax_US_tf1_selclus4_okt19_Xzkup1.npy'
filename_ts = os.path.join(path_ts, RVts_filename)
kwrgs_events_daily =    (filename_ts,
                         {'event_percentile': 90})

kwrgs_events = {'event_percentile': 66}

kwrgs_events = kwrgs_events

#stat_model_l = [logitCVfs, logitCV, GBC_tfs, GBC_t, GBC]
stat_model_l = [logitCV]
kwrgs_pp     = {'add_autocorr' : True, 'normalize':'datesRV'}

lags_i = np.array([0, 5, 15])
precur_aggr = 10
use_fold = None



dict_experiments = {} ; list_of_fc = []
for dataset, tuple_sett in datasets_path.items():
    path_data = tuple_sett[0]
    keys_d_list = tuple_sett[1]
    for keys_d in keys_d_list:

        fc = fcev(path_data=path_data, precur_aggr=precur_aggr, use_fold=use_fold)
        fc.get_TV(kwrgs_events=kwrgs_events)
        fc.fit_models(stat_model_l=stat_model_l, lead_max=lags_i,
                           keys_d=keys_d, kwrgs_pp=kwrgs_pp, verbosity=1)

        fc.perform_validation(n_boot=500, blocksize='auto', alpha=0.05,
                              threshold_pred=(1.5, 'times_clim'))
        dict_experiments[dataset+'_'+str(keys_d)] = fc.dict_sum
        list_of_fc.append(fc)

y_pred_all, y_pred_c = fc.dict_preds[fc.stat_model_l[0][0]]

# In[8]:


import valid_plots as dfplots
kwrgs = {'wspace':0.25, 'col_wrap':None, 'threshold_bin':fc.threshold_pred}
#kwrgs = {'wspace':0.25, 'col_wrap':3, 'threshold_bin':fc.threshold_pred}
met = ['AUC-ROC', 'AUC-PR', 'BSS', 'Rel. Curve', 'Precision']
#met = ['AUC-ROC', 'AUC-PR', 'BSS', 'Rel. Curve']
expers = list(dict_experiments.keys())
models   = list(dict_experiments[expers[0]].keys())
line_dim = 'model'


fig = dfplots.valid_figures(dict_experiments, expers=expers, models=models,
                          line_dim=line_dim,
                          group_line_by=None,
                          met=met, **kwrgs)


working_folder, filename = fc._print_sett(list_of_fc=list_of_fc)

f_format = '.pdf'
pathfig_valid = os.path.join(filename + f_format)
fig.savefig(pathfig_valid,
            bbox_inches='tight') # dpi auto 600



#%%


import valid_plots as dfplots
if __name__ == "__main__":
    for fc in list_of_fc:
        models = [n[0] for n in fc.stat_model_l]
        for m in models:
            for l in fc.lags_i:
                # visual analysis
                fig = dfplots.visual_analysis(fc, lag=l, model=m)
                f_name = filename + f'_va_l{l}_{m}'
                f_format = '.pdf'
                pathfig_vis = os.path.join(working_folder, f_name) + f_format
                fig.savefig(pathfig_vis, bbox_inches='tight') # dpi auto 600
                # plot deviance
                if m[:3] == 'GBC':
                    f_name = filename +f'_l{l}_deviance'
                    fig = dfplots.plot_deviance(fc, lag=l, model=m)
                    f_format = '.pdf'
                    path_fig_GBC = os.path.join(working_folder, f_name) + f_format
                    fig.savefig(path_fig_GBC,
                            bbox_inches='tight') # dpi auto 600

#model = 'logitCV'
#model = None
#fig = dfplots.visual_analysis(fc, lag=2, model=model)
#if model is None:
#    model = fc.stat_model_l[0][0]
#f_name = f'{RV_name}_{tfreq}d_{percentile}p_fold{folds_used}_{today}_va_{model}'
#f_format = '.pdf'
#pathfig_vis = os.path.join(working_folder, f_name) + f_format
#fig.savefig(pathfig_vis, bbox_inches='tight') # dpi auto 600

#try:
#    f_name = f'{RV_name}_{tfreq}d_{percentile}p_fold{folds_used}_{today}_deviance'
#
#    fig = dfplots.plot_deviance(fc, lag=None, model=model)
#    f_format = '.pdf'
#    path_fig_GBC = os.path.join(working_folder, f_name) + f_format
#    fig.savefig(path_fig_GBC,
#            bbox_inches='tight') # dpi auto 600
#except:
#    pass

