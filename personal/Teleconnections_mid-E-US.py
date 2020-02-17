#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import os, inspect, sys
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-1])
RGCPD_func = os.path.join(main_dir, 'RGCPD')
cluster_func = os.path.join(main_dir, 'clustering/') 
if cluster_func not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(RGCPD_func)
    sys.path.append(cluster_func)

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
list_of_name_path = [(3,
                      '/Users/semvijverberg/surfdrive/Data_era5/input_raw/preprocessed/xrclustered_1d_c0f23.nc'),
                    ('sst', '/Users/semvijverberg/surfdrive/Data_era5/input_raw/sst_1979-2018_1_12_daily_2.5deg.nc')]
start_end_TVdate = ('06-24', '08-22')
kwrgs_corr = {'alpha':1E-3}
rg = RGCPD(list_of_name_path=list_of_name_path, start_end_TVdate=start_end_TVdate)


# In[6]:


rg.pp_precursors()


# ### Post-processing Target Variable

# In[7]:


rg.pp_TV()


# In[154]:


get_ipython().run_line_magic('pinfo', 'rg.traintest')


# In[165]:


rg.traintest(method='ran_strat10')


# In[166]:


rg.calc_corr_maps(alpha=1E-3) 


# In[167]:


rg.cluster_regions(distance_eps=800, min_area_in_degrees2=8)


# In[168]:


rg.quick_view_labels() 


# In[169]:


rg.get_ts_prec()


# In[170]:


rg.df_data


# In[171]:


rg.PCMCI_df_data()


# In[172]:


rg.plot_maps_sum()


# In[173]:


rg.df_data


# In[174]:


rg.df_sum


# In[ ]:

rg.store_df_output(add_spatcov=False)



# In[ ]:





# In[ ]:




