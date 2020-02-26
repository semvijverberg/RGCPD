#!/usr/bin/env python
# coding: utf-8

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

import os, inspect, sys
if sys.platform == 'linux':
    import matplotlib as mpl
    mpl.use('Agg')
user_dir = os.path.expanduser('~')

curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-1])
RGCPD_func = os.path.join(main_dir, 'RGCPD')
cluster_func = os.path.join(main_dir, 'clustering/') 
if cluster_func not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(RGCPD_func)
    sys.path.append(cluster_func)

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

# In[5]:


from RGCPD import RGCPD
from RGCPD import EOF
from RGCPD import BivariateMI

old_CPPA = [('sst_CPPA', '/Users/semvijverberg/surfdrive/MckinRepl/era5_T2mmax_sst_Northern/ran_strat10_s30/data/era5_24-09-19_07hr_lag_0.h5')]
CPPA_s30 = [('sst_CPPAs30', '/Users/semvijverberg/surfdrive/MckinRepl/era5_T2mmax_sst_Northern/Xzkup1_ran_strat10_s30/data/era5_21-01-20_10hr_lag_10_Xzkup1.h5' )]
CPPA_s5  = [('sst_CPPAs5', '/Users/semvijverberg/surfdrive/MckinRepl/era5_T2mmax_sst_Northern/Xzkup1_ran_strat10_s5/data/ERA5_15-02-20_15hr_lag_10_Xzkup1.h5')]

#list_of_name_path = [('t2mmmax',
#                      '/Users/semvijverberg/surfdrive/MckinRepl/RVts/era5_t2mmax_US_1979-2018_averAggljacc0.25d_tf1_n4__to_t2mmax_US_tf1_selclus4_okt19_Xzkup1.npy'),
#                        ('sm1', '/Users/semvijverberg/surfdrive/ERA5/input_raw/sm_1_1979-2018_1_12_daily_1.0deg.nc'),
#                        ('sm2', '/Users/semvijverberg/surfdrive/ERA5/input_raw/sm_2_1979-2018_1_12_daily_1.0deg.nc'),
#                        ('sm3', '/Users/semvijverberg/surfdrive/ERA5/input_raw/sm_3_1979-2018_1_12_daily_1.0deg.nc'),
#                        ('u500', '/Users/semvijverberg/surfdrive/ERA5/input_raw/u500hpa_1979-2018_1_12_daily_2.5deg.nc'),
#                        ('sst', '/Users/semvijverberg/surfdrive/ERA5/input_raw/sst_1979-2018_1_12_daily_1.0deg.nc')]

list_of_name_path = [('t2mmmax',
                      '/Users/semvijverberg/surfdrive/MckinRepl/RVts/era5_t2mmax_US_1979-2018_averAggljacc0.25d_tf1_n4__to_t2mmax_US_tf1_selclus4_okt19_Xzkup1.npy'),
                        # ('sm1', '/Users/semvijverberg/surfdrive/ERA5/input_raw/sm1_1979-2018_1_12_daily_1.0deg.nc'),
                        ('sm2', '/Users/semvijverberg/surfdrive/ERA5/input_raw/sm2_1979-2018_1_12_daily_1.0deg.nc'),                        
                        ('sm3', '/Users/semvijverberg/surfdrive/ERA5/input_raw/sm3_1979-2018_1_12_daily_1.0deg.nc'),     
                        # ('st2', '/Users/semvijverberg/surfdrive/ERA5/input_raw/st_2_1979-2018_1_12_daily_1.0deg.nc'),
                        ('OLR', '/Users/semvijverberg/surfdrive/ERA5/input_raw/OLRtrop_1979-2018_1_12_daily_2.5deg.nc')]

#                        ('u500', '/Users/semvijverberg/surfdrive/ERA5/input_raw/u500hpa_1979-2018_1_12_daily_2.5deg.nc'),
#                        ('v200', '/Users/semvijverberg/surfdrive/ERA5/input_raw/v200hpa_1979-2018_1_12_daily_2.5deg.nc'),                        
#                        ('sst', '/Users/semvijverberg/surfdrive/ERA5/input_raw/sst_1979-2018_1_12_daily_1.0deg.nc'),
#                        ('sm123', '/Users/semvijverberg/surfdrive/ERA5/input_raw/sm_123_1979-2018_1_12_daily_1.0deg.nc')]

import_prec_ts = CPPA_s30

list_for_EOFS = [EOF(name='OLR', neofs=1, selbox=[-180, 360, -15, 30])]
list_for_MI   = [BivariateMI(name='sm2', func=BivariateMI.corr_map, kwrgs_func={'alpha':.05, 'FDF_control':True})]
                            

#list_of_name_path = [('t2mmmax',
#                      '/Users/semvijverberg/surfdrive/MckinRepl/RVts/era5_t2mmax_US_1979-2018_averAggljacc0.25d_tf1_n4__to_t2mmax_US_tf1_selclus4_okt19.npy'),
#                    ('sst', '/Users/semvijverberg/surfdrive/Data_era5/input_raw/sst_1979-2018_1_12_daily_1.0deg.nc'),
#                    ('sm123', '/Users/semvijverberg/surfdrive/Data_era5/input_raw/sm_123_1979-2018_1_12_daily_0.25deg.nc'),
#                    ('v200hpa', '/Users/semvijverberg/surfdrive/Data_era5/input_raw/v200hpa_1979-2018_1_12_daily_2.5deg.nc')]

start_end_TVdate = ('06-24', '08-22')
start_end_TVdate = ('07-06', '08-11')

#start_end_TVdate = ('06-15', '08-31')
start_end_date = ('1-1', '12-31')
kwrgs_corr = {'alpha':1E-3}

rg = RGCPD(list_of_name_path=list_of_name_path, 
           list_for_EOFS=list_for_EOFS,
           list_for_MI=list_for_MI,
           import_prec_ts=import_prec_ts,
           start_end_TVdate=start_end_TVdate,
           start_end_date=start_end_date,
           tfreq=10,
           path_outmain=user_dir+'/surfdrive/output_RGCPD')


# In[6]:

#selbox = [None, {'sst':[-180,360,-10,90]}]
selbox = None
#anomaly = [True, {'sm1':False, 'sm2':False, 'sm3':False}]
anomaly = [True, {'sm1':False, 'sm2':False, 'sm3':False, 'st2':False}]

rg.pp_precursors(selbox=selbox, anomaly=anomaly)


# ### Post-processing Target Variable

# In[7]:


rg.pp_TV()



# In[165]:

#kwrgs_events={'event_percentile':66}
kwrgs_events=None
rg.traintest(method='random10', kwrgs_events=kwrgs_events)

#%%


rg.get_EOFs()



# In[166]:


rg.calc_corr_maps(alpha=1E-2) 


# In[167]:


rg.cluster_regions(distance_eps=700, min_area_in_degrees2=5)


# In[168]:


rg.quick_view_labels() 


# In[169]:


rg.get_ts_prec(precur_aggr=1)


# In[170]:


rg.df_data

rg.store_df()

# In[171]:

rg.get_ts_prec(precur_aggr=None)
rg.PCMCI_df_data(pc_alpha=None, alpha_level=0.1, max_combinations=1)
rg.df_sum

# In[172]:


rg.plot_maps_sum()


# In[173]:


#rg.df_data




# In[ ]:

rg.store_df_PCMCI(add_spatcov=False)



# In[ ]:





# In[ ]:




