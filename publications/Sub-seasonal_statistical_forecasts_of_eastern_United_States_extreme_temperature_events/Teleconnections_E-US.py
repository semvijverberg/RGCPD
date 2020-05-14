#!/usr/bin/env python
# coding: utf-8

import os, inspect, sys
if sys.platform == 'linux':
    import matplotlib as mpl
    mpl.use('Agg')

user_dir = os.path.expanduser('~')
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-2])
RGCPD_func = os.path.join(main_dir, 'RGCPD')
cluster_func = os.path.join(main_dir, 'clustering/')
if cluster_func not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(RGCPD_func)
    sys.path.append(cluster_func)

import numpy as np




from RGCPD import RGCPD
from RGCPD import BivariateMI
# In[5]:

CPPA_s30  = [('sst_CPPAs30', user_dir + '/surfdrive/output_RGCPD/easternUS/ERA5_CPPA/ERA5_14-05-20_08hr_lag_0_c378f.h5')]
RV = user_dir + '/surfdrive/output_RGCPD/easternUS/tf1_n_clusters5_q95_dendo_c378f.nc'
# RV = user_dir + '/surfdrive/output_RGCPD/1d_ts_38N_275E_90W.nc'

list_of_name_path = [(1 , RV),
                     ('sm2', '/Users/semvijverberg/surfdrive/ERA5/input_raw/sm2_1979-2018_1_12_daily_1.0deg.nc'),
                     ('sm3', '/Users/semvijverberg/surfdrive/ERA5/input_raw/sm3_1979-2018_1_12_daily_1.0deg.nc')]
                      # ('sst', '/Users/semvijverberg/surfdrive/ERA5/input_raw/sst_1979-2018_1_12_daily_1.0deg.nc')]

list_import_ts = CPPA_s30


list_for_MI   = [BivariateMI(name='sm2', func=BivariateMI.corr_map,
                             kwrgs_func={'alpha':.05, 'FDR_control':True},
                             distance_eps=600, min_area_in_degrees2=5),
                 BivariateMI(name='sm3', func=BivariateMI.corr_map,
                              kwrgs_func={'alpha':.05, 'FDR_control':True},
                              distance_eps=600, min_area_in_degrees2=7)]
                  # BivariateMI(name='sst', func=BivariateMI.corr_map,
                  #              kwrgs_func={'alpha':.001, 'FDR_control':True},
                  #              distance_eps=800, min_area_in_degrees2=5)]



start_end_TVdate = ('06-24', '08-22')
start_end_date = ('1-1', '12-31')


rg = RGCPD(list_of_name_path=list_of_name_path,
           list_for_MI=list_for_MI,
           list_import_ts=list_import_ts,
           start_end_TVdate=start_end_TVdate,
           start_end_date=start_end_date,
           tfreq=15, lags_i=np.array([0]),
           path_outmain=user_dir+'/surfdrive/output_RGCPD')


# In[6]:

#selbox = [None, {'sst':[-180,360,-10,90]}]
selbox = None
#anomaly = [True, {'sm1':False, 'sm2':False, 'sm3':False}]
anomaly = [True, {'sm1':False, 'sm2':False, 'sm3':False, 'st2':False}]

rg.pp_precursors(selbox=selbox, anomaly=anomaly)


# ### Post-processing Target Variable

# In[7]:


rg.pp_TV(name_ds='q90tail', detrend=True, anomaly=False)



# In[165]:

#kwrgs_events={'event_percentile':66}
kwrgs_events=None
rg.traintest(method='random10', kwrgs_events=kwrgs_events)



# In[166]:


rg.calc_corr_maps()


# In[167]:


rg.cluster_list_MI()


# In[168]:


rg.quick_view_labels()



# In[ ]:
keys = ['ENSO34', 'PDO', '0..CPPAsv', '0..PEPsv', '0..1..sst', '0..2..sst',
        '0..3..sst', '0..4..sst', '0..5..sst', '0..6..sst', '0..7..sst', '0..8..sst']
rg.get_ts_prec(precur_aggr=1, keys_ext=keys)
rg.store_df()



# In[ ]:





# In[ ]:




