#!/usr/bin/env python
# coding: utf-8

# # Forecasting
# Below done with test data, same format as df_data

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import os, inspect, sys
import numpy as np
import pandas as pd
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-1])
python_dir = os.path.join(main_dir, 'RGCPD')
df_ana_dir = os.path.join(main_dir, 'df_analysis/df_analysis/')
if main_dir not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(python_dir)
    sys.path.append(df_ana_dir)


# In[2]:


from func_fc import fcev


# In[3]:


fc = fcev.get_test_data()
fc.df_data


# In[4]:


rename_labels = {'10_1_sst':'Carribean', 
                 '10_7_sst':'east Pacific',
                 '10_4_sst':'mid-Pacific',
                 '10_2_sst':'Great Lakes',
                 '10_3_sst':'Indian Ocean',
                 '10_1_sm123':'local SM',
                 '10_1_z500hpa':'local high press.'}
fc.df_data.rename(columns=rename_labels, inplace=True)
fc.get_TV(kwrgs_events=None)


# Define statmodel:

# In[5]:


logit = ('logit', None)

logitCV = ('logitCV', 
          {'class_weight':{ 0:1, 1:1},
           'scoring':'brier_score_loss',
           'penalty':'l2',
           'solver':'lbfgs'})

GBR_logitCV = ('GBR-logitCV', 
              {'max_depth':3,
               'learning_rate':1E-3,
               'n_estimators' : 750,
               'max_features':'sqrt',
               'subsample' : 0.6,
               'random_state':60} )


# In[6]:


fc.fit_models(stat_model_l=[logitCV], lead_max=45, 
                   keys_d=None, kwrgs_pp={})
y_pred_all, y_pred_c = fc.dict_preds[fc.stat_model_l[0][0]]

# In[7]:


dict_experiments = {}       
fc.perform_validation(n_boot=100, blocksize='auto', 
                              threshold_pred=(2, 'times_clim'))
dict_experiments['test'] = fc.dict_sum


# In[8]:


import valid_plots as dfplots
kwrgs = {'wspace':0.25, 'col_wrap':3, 'threshold_bin':fc.threshold_pred}
met = ['AUC-ROC', 'AUC-PR', 'BSS', 'Rel. Curve', 'Precision', 'Accuracy']
expers = list(dict_experiments.keys())
models   = list(dict_experiments[expers[0]].keys())
line_dim = 'model'


fig = dfplots.valid_figures(dict_experiments, expers=expers, models=models,
                          line_dim=line_dim, 
                          group_line_by=None,  
                          met=met, **kwrgs)


# In[9]:


keys = list(rename_labels.values())
# keys = None
fc.plot_GBR_feature_importances(lag=None, keys=keys)


# In[10]:


import stat_models
keys = tuple(rename_labels.values())
# keys = None
GBR_models_split_lags = fc.dict_models['GBR-logitCV']
stat_models.plot_oneway_partial_dependence(GBR_models_split_lags,
                                          keys=keys,
                                          lags=[0,2,5])


# In[18]:


import stat_models
GBR_models_split_lags = fc.dict_models['GBR-logitCV']
keys = tuple(rename_labels.values())
#plot_pairs = [(keys[2], keys[1])]
df_all = stat_models.plot_twoway_partial_dependence(GBR_models_split_lags, lag_i=2, keys=keys,
                                   plot_pairs=None, min_corrcoeff=0.1)


# In[12]:


from IPython.display import Image
Image(filename=os.path.join(main_dir, "docs/images/pcA_none_ac0.002_at0.05_t2mmax_E-US_vs_sst_labels_mean.png"),
      width=1000, height=200)


# In[13]:


# Soil Moisture labels
Image(filename=os.path.join(main_dir, "docs/images/pcA_none_ac0.002_at0.05_t2mmax_E-US_vs_sm123_labels_mean.png"),
      width=1000, height=400)


# In[14]:


Image(filename=os.path.join(main_dir, "docs/images/pcA_none_ac0.002_at0.05_t2mmax_E-US_vs_z500hpa_labels_mean.png"),
      width=1000, height=200)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




