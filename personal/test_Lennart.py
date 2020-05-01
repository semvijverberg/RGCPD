#!/usr/bin/env python
# coding: utf-8

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

import os, inspect, sys


if sys.platform == 'linux':
    import matplotlib as mpl
    mpl.use('Agg')
user_dir = os.path.expanduser('~')
# user_dir = '/mnt/c/Users/lenna/Documents/Studie/2019-2020/Scriptie/RGCPD'

curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-1])
RGCPD_func = os.path.join(main_dir, 'RGCPD')
cluster_func = os.path.join(main_dir, 'clustering/') 
if cluster_func not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(RGCPD_func)
    sys.path.append(cluster_func)
    

path_raw = user_dir + '/surfdrive/Savar/input_raw'    

import numpy as np
# ## Initialize RGCPD class
# args:
# - list_of_name_path
# - start_end_TVdate
# 
#         list_of_name_path : list of name, path tuples. 
#         Convention: first entry should be (name, path) of target variable (TV).
#         list_of_name_path = [('TVname', 'TVpath'), ('prec_name1', 'prec_path1')]
#         
#         TV period : tuple of start- and enddate in format ('mm-dd', 'mm-dd')


from RGCPD import RGCPD
from class_BivariateMI import BivariateMI

#%%


list_of_name_path = [(2, path_raw + '/test_target.nc'),
                     ('test_precur', path_raw + '/test.nc')]


list_for_MI   = [BivariateMI(name='test_precur', func=BivariateMI.corr_map, 
                             kwrgs_func={'alpha':.05, 'FDR_control':True})]


start_end_TVdate = ('3-1', '10-30')
start_end_date = ('1-1', '12-31')

rg = RGCPD(list_of_name_path=list_of_name_path, 
           list_for_MI=list_for_MI,
           start_end_TVdate=start_end_TVdate,
           start_end_date=start_end_date,
           tfreq=10, lags_i=np.array([1]),
           path_outmain=user_dir+'/Savar/output_RGCPD')


selbox = None
anomaly = True

rg.pp_precursors(selbox=selbox, anomaly=anomaly)
#%%
rg.pp_TV(name_ds='ts')

#kwrgs_events={'event_percentile':66}
kwrgs_events=None
rg.traintest(method='random10', kwrgs_events=kwrgs_events)

rg.calc_corr_maps()

rg.cluster_list_MI()
print('hoi')

rg.quick_view_labels(median=False) 

# rg.get_EOFs()

rg.get_ts_prec(precur_aggr=None)

rg.PCMCI_df_data(pc_alpha=None, 
                 tau_max=2,
                 max_combinations=2)

rg.PCMCI_get_links(alpha_level=0.01)

rg.PCMCI_plot_graph(s=1)

rg.quick_view_labels()

rg.plot_maps_sum()