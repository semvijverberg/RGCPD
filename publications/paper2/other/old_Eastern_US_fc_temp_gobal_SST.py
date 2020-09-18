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

path_raw = user_dir + '/surfdrive/ERA5/input_raw'


from RGCPD import RGCPD
from RGCPD import BivariateMI

TVpath = '/Users/semvijverberg/surfdrive/output_RGCPD/circulation_US_HW/tf15_nc3_dendo_0ff31.nc'
path_out_main = os.path.join(main_dir, 'publications/paper2/output/east_testing/')
cluster_label = 2
name_ds='ts'
start_end_TVdate = ('06-01', '08-31')
start_end_date = ('1-1', '12-31')
tfreq = 15

#%%
list_of_name_path = [(cluster_label, TVpath),
                     ('sst', os.path.join(path_raw, 'sst_1979-2018_1_12_daily_1.0deg.nc'))]


list_import_ts = None#[('sstpattern', '/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/paper2/output/east/z5000..0..z500_sp_0ff31_10jun-24aug_lag0-0_0..0..z500_sp_random10s1/2020-07-02_11hr_52min_df_data_NorthPacAtl_dt1_0ff31.h5'),
                 # ('sstregions', '/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/paper2/output/east/z5000..0..z500_sp_0ff31_10jun-24aug_lag0-0_0..0..z500_sp_random10s1/2020-07-02_12hr_10min_df_data_NorthPacAtl_dt1_0ff31.h5')]

selboxsst  = None#(170,255,11,60)
list_for_MI = [BivariateMI(name='sst', func=BivariateMI.corr_map,
                        kwrgs_func={'alpha':.001, 'FDR_control':True},
                        distance_eps=1000, min_area_in_degrees2=5,
                        calc_ts='region mean', selbox=selboxsst)]



rg = RGCPD(list_of_name_path=list_of_name_path,
           list_for_MI=list_for_MI,
           list_import_ts=list_import_ts,
            start_end_TVdate=start_end_TVdate,
            start_end_date=start_end_date,
            start_end_year=None,
            tfreq=tfreq, lags_i=np.array([0]),
            path_outmain=path_out_main,
            append_pathsub='_' + name_ds)


rg.pp_TV(name_ds=name_ds, detrend=False)
rg.pp_precursors(selbox=(0,360,10,90))

rg.traintest(method='random10')
rg.calc_corr_maps()
rg.plot_maps_corr(save=True)
rg.cluster_list_MI()
rg.quick_view_labels(save=True)
rg.get_ts_prec(precur_aggr=1)
rg.store_df()

#%%
# path_data = '/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/paper2/output/east/2ts_0ff31_10jun-24aug_lag0-0_ts_from_imports/2020-07-06_11hr_58min_df_data_sstpattern_sstregionssst_dt1_0ff31.h5'
# path_allNH = '/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/paper2/output/east/2ts_0ff31_10jun-24aug_lag0-0_ts_from_imports/2020-07-07_14hr_06min_df_data_sstpattern_sstregionssst_dt1_0ff31.h5'


path_data = rg.path_df_data
from class_fc import fcev

start_time = time()

kwrgs_events = {'event_percentile': 66}

kwrgs_events = kwrgs_events
precur_aggr = 15
use_fold = None
n_boot = 2000
lags_i = np.array([0,15,25])

list_of_fc = [fcev(path_data=path_data, precur_aggr=precur_aggr,
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
                    keys_d=None)]#('SST from corr. map RW', dict(zip(np.arange(10), [['0..1..NorthPacAtl', '0..2..NorthPacAtl']]*10)))),
               # fcev(path_data=path_data, precur_aggr=precur_aggr,
               #      use_fold=use_fold, start_end_TVdate=None,
               #      stat_model= ('logitCV',
               #                  {'Cs':10, #np.logspace(-4,1,10)
               #                  'class_weight':{ 0:1, 1:1},
               #                    'scoring':'neg_brier_score',
               #                    'penalty':'l2',
               #                    'solver':'lbfgs',
               #                    'max_iter':100,
               #                    'kfold':5,
               #                    'seed':1}),
               #      kwrgs_pp={'add_autocorr':False, 'normalize':'datesRV'},
               #      dataset='',
               #      keys_d=('SST from corr. map mx2t', dict(zip(np.arange(10), [['0..1..sst', '0..2..sst']]*10))))]



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


lag_rel = 15
kwrgs = {'wspace':0.16, 'hspace':.25, 'col_wrap':2, 'skip_redundant_title':True,
         'lags_relcurve':[lag_rel], 'fontbase':14, 'figaspect':2}
#kwrgs = {'wspace':0.25, 'col_wrap':3, 'threshold_bin':fc.threshold_pred}
met = ['BSS', 'Rel. Curve']
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

fig = dfplots.valid_figures(dict_merge_all,
                          line_dim=line_dim,
                          group_line_by=group_line_by,
                          **kwrgs)

pathfig_valid = os.path.join(pathexper,'verification_all_met'+ f_format)
fig.savefig(pathfig_valid,
            bbox_inches='tight') # dpi auto 600

fc = list_of_fc[0]
df, fig = fc.plot_feature_importances()
path_feat = pathexper + f'/ifc{1}_logitregul' + f_format
fig.savefig(path_feat, bbox_inches='tight')


fc.dict_sum[0].loc['Precision'].loc['Precision']

fc.dict_sum[0].loc['Accuracy'].loc['Accuracy']