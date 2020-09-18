#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 09:59:06 2020

@author: semvijverberg
"""


import os, inspect, sys
import numpy as np
from time import time
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


from class_fc import fcev

east_15data = '/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/paper2/output/east/z5000..0..z500_sp_140-300-20-73_10jun-24aug_lag0-15_0..0..z500_sp_random10s1/2020-07-14_15hr_44min_df_data_N-Pac. SST_dt1_140-300-20-73_RW_and_SST_feedback.h5'
west_15data = '/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/paper2/output/west/z5000..0..z500_sp_145-325-20-62_10jun-24aug_lag0-0_0..0..z500_sp_random10s1/2020-07-14_19hr_52min_df_data_Pacific SST_dt1_145-325-20-62_RW_and_SST_feedback.h5'
east_60data = '/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/paper2/output/east/z5000..0..z500_sp_140-300-20-73_3jun-2aug_lag0-60_0..0..z500_sp_random10s1/2020-07-14_16hr_56min_df_data_N-Pac. SST_dt1_140-300-20-73_RW_and_SST_fb_tf60.h5'
east_60data = '/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/paper2/output/east/z5000..0..z500_sp_140-300-20-73_3jun-2aug_lag0-60_0..0..z500_sp_random10s1/2020-07-23_09hr_43min_df_data_v200_z500_sst_dt1_tf60_140-300-20-73.h5'


kwrgs_events = {'event_percentile': 50}

kwrgs_events = kwrgs_events
use_fold = None
n_boot = 0
lags_i = np.array([0,10,20,30])


# rg.store_df()

#%% 15-day mean
start_time = time()


list_of_fc = [fcev(path_data=east_15data, precur_aggr=15,
                    use_fold=use_fold, start_end_TVdate=None,
                    stat_model= ('logitCV',
                                {'Cs':10, #np.logspace(-4,1,10)
                                'class_weight':{ 0:1, 1:1},
                                  'scoring':'neg_brier_score',
                                  'penalty':'l2',
                                  'solver':'lbfgs',
                                  'max_iter':100,
                                  'kfold':5,
                                  'seed':1}),
                    kwrgs_pp={'add_autocorr':False, 'normalize':'datesRV'},
                    dataset='',
                    keys_d=('east-RW <-- N-Pac. SST 15-d', dict(zip(np.arange(10), [['0..0..N-Pac. SST_sp']]*10)))),
              fcev(path_data=west_15data, precur_aggr=15,
                    use_fold=use_fold, start_end_TVdate=None,
                    stat_model= ('logitCV',
                                {'Cs':10, #np.logspace(-4,1,10)
                                'class_weight':{ 0:1, 1:1},
                                  'scoring':'neg_brier_score',
                                  'penalty':'l2',
                                  'solver':'lbfgs',
                                  'max_iter':100,
                                  'kfold':5,
                                  'seed':1}),
                    kwrgs_pp={'add_autocorr':False, 'normalize':'datesRV'},
                    dataset='',
                    keys_d=('west-RW <-- N-Pac. SST 15-d', dict(zip(np.arange(10), [['0..0..Pacific SST_sp']]*10))))
              ]




fc = list_of_fc[0]
#%% 60-day mean
lags_i = np.array([0,30])
start_time = time()

list_of_fc = [fcev(path_data=east_60data, precur_aggr=60,
                    use_fold=use_fold, start_end_TVdate=None,
                    stat_model= ('logitCV',
                                {'Cs':10, #np.logspace(-4,1,10)
                                'class_weight':{ 0:1, 1:1},
                                  'scoring':'neg_brier_score',
                                  'penalty':'l2',
                                  'solver':'lbfgs',
                                  'max_iter':100,
                                  'kfold':5,
                                  'seed':1}),
                    kwrgs_pp={'add_autocorr':False, 'normalize':'datesRV'},
                    dataset='',
                    keys_d=('N-Pac. SST 60-d', dict(zip(np.arange(10), [['SST lag 60']]*10))))]



fc = list_of_fc[0]

#%%
times = []
t00 = time()
for fc in list_of_fc:
    t0 = time()
    fc.get_TV(kwrgs_events=kwrgs_events, detrend=False) # detrending already done on gridded data

    fc.fit_models(lead_max=lags_i, verbosity=1)

    # fc.TV.prob_clim = pd.DataFrame(np.repeat(fc.TV.prob_clim.mean(), fc.TV.prob_clim.size).values, index=fc.TV.prob_clim.index)

    fc.perform_validation(n_boot=n_boot, blocksize='auto', alpha=0.05,
                          threshold_pred='upper_clim')

    single_run_time = int(time()-t0)
    times.append(single_run_time)
    total_n_runs = len(list_of_fc)
    ETC = (int(np.mean(times) * total_n_runs))
    print(f'Time elapsed single run in {single_run_time} sec\t'
          f'ETC {int(ETC/60)} min \t Progress {int(100*(time()-t00)/ETC)}% ')


# In[8]:
working_folder, pathexper = list_of_fc[0]._print_sett(list_of_fc=list_of_fc)

store = False
if __name__ == "__main__":
    pathexper = list_of_fc[0].pathexper
    store = True

import valid_plots as dfplots
import functions_pp


dict_all = dfplots.merge_valid_info(list_of_fc, store=store)
if store:
    dict_merge_all = functions_pp.load_hdf5(pathexper+'/data.h5')


lag_rel = 30
kwrgs = {'wspace':0.16, 'hspace':.25, 'col_wrap':2, 'skip_redundant_title':True,
         'lags_relcurve':[lag_rel], 'fontbase':14, 'figaspect':2}
#kwrgs = {'wspace':0.25, 'col_wrap':3, 'threshold_bin':fc.threshold_pred}
met = ['AUC-ROC', 'AUC-PR', 'Precision', 'BSS', 'Accuracy', 'Rel. Curve']
line_dim = 'exper'
group_line_by = None

fig = dfplots.valid_figures(dict_merge_all,
                          line_dim=line_dim,
                          group_line_by=group_line_by,
                          met=met, **kwrgs)


f_format = '.pdf'
pathfig_valid = os.path.join(pathexper,'verification' + f_format)
fig.savefig(pathfig_valid,
            bbox_inches='tight') # dpi auto 600
fc = list_of_fc[0]
df, fig = fc.plot_feature_importances()
path_feat = pathexper + f'/ifc{1}_logitregul' + f_format
fig.savefig(path_feat, bbox_inches='tight')


fc.dict_sum[0].loc['Precision'].loc['Precision']

fc.dict_sum[0].loc['Accuracy'].loc['Accuracy']