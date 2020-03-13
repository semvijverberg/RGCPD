#!/usr/bin/env python
# coding: utf-8

# # Clustering

# In[1]:


import os, inspect, sys
import numpy as np
import matplotlib.pyplot as plt
user_dir = os.path.expanduser('~')
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-2])
RGCPD_func = os.path.join(main_dir, 'RGCPD')
cluster_func = os.path.join(main_dir, 'clustering/')
df_ana_func =  os.path.join(main_dir, 'df_analysis/df_analysis/')
if cluster_func not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(RGCPD_func)
    sys.path.append(cluster_func)
    sys.path.append(df_ana_func)


if sys.platform == 'linux':
    import matplotlib as mpl
    mpl.use('Agg')
    root_data = '/scistor/ivm/data_catalogue/reanalysis/ERA5'
else:
    root_data = '/Users/semvijverberg/surfdrive/ERA5'
    
path_outmain = user_dir+'/surfdrive/output_RGCPD/easternUS_EC/'
# In[2]:


import functions_pp
import clustering_spatial as cl
import plot_maps
import df_ana
from RGCPD import RGCPD
list_of_name_path = [('fake', None),
                     ('t2mmmax', root_data + '/input_raw/mx2t_US_1979-2018_1_12_daily_1.0deg.nc')]
rg = RGCPD(list_of_name_path=list_of_name_path,
           path_outmain=path_outmain)
#
#
#
## In[3]:
#
#
#rg.pp_precursors()
#
#
## In[ ]:
#

#rg.list_precur_pp

var_filename = '/Users/semvijverberg/surfdrive/Data_EC/input_pp/tas_2000-2159_1jan_31dec_daily_1.125deg.nc'
LSM = '/Users/semvijverberg/surfdrive/Data_EC/input_raw/mask_North_America_1.125deg.nc'
#%%
import make_country_mask
selbox = (225, 300, 20, 70)
xarray, Country = make_country_mask.create_mask(var_filename, 
                                                kwrgs_load={'selbox':selbox}, 
                                                level='Countries')
mask_US = xarray.values == Country.US
lsm = core_pp.import_ds_lazy(LSM, selbox=selbox)
mask_US = np.logical_and(mask_US, (lsm > .3).values)
xr_mask = xarray.where(mask_US)
xr_mask.values[mask_US]  = 1
xr_mask = cl.mask_latlon(xr_mask, lonmin=237)
xr_mask = cl.mask_latlon(xr_mask, lonmin=238, latmin=39)
xr_mask = cl.mask_latlon(xr_mask, lonmin=239, latmin=38)
xr_mask = cl.mask_latlon(xr_mask, lonmin=240, latmin=36)
plot_maps.plot_labels(xr_mask)



# In[9]:
# =============================================================================
# Clustering co-occurence of anomalies
# =============================================================================
q = [80, 85, 90, 95]
n_clusters = [2,3,4,5,6,7,8]
from time import time
t0 = time()
xrclustered, results = cl.dendogram_clustering(var_filename, mask=xr_mask,
                                               kwrgs_load={'tfreq':1,
                                                           'seldates':('06-15', '08-31'),
                                                           'selbox':selbox},
                                               kwrgs_clust={'q':q,
                                                            'n_clusters':n_clusters,
                                                            'affinity':'jaccard',
                                                            'linkage':'average'})
fig = plot_maps.plot_labels(xrclustered, wspace=.05, hspace=-.2, cbar_vert=.08,
                            row_dim='q', col_dim='n_clusters')
f_name = 'clustering_dendogram_{}'.format(xrclustered.attrs['hash']) + '.pdf'
path_fig = os.path.join(path_outmain, f_name)
plt.savefig(path_fig,
            bbox_inches='tight') # dpi auto 600
print(f'{round(time()-t0, 2)}')

#%%
# =============================================================================
# Clustering correlation Hierarchical Agglomerative Clustering
# =============================================================================
from time import time
t0 = time()
xrclustered, results = cl.correlation_clustering(var_filename, mask=xr_mask,
                                               kwrgs_load={'tfreq':tfreq,
                                                           'seldates':('06-01', '08-31'),
                                                           'selbox':selbox},
                                               clustermethodkey='AgglomerativeClustering',
                                               kwrgs_clust={'n_clusters':n_clusters,
                                                            'affinity':'correlation',
                                                            'linkage':'average'})

plot_maps.plot_labels(xrclustered,  wspace=.05, hspace=-.2, cbar_vert=.08,
                            row_dim='tfreq', col_dim='n_clusters')

f_name = 'clustering_correlation_{}'.format(xrclustered.attrs['hash']) + '.pdf'
path_fig = os.path.join(path_outmain, f_name)
plt.savefig(path_fig,
            bbox_inches='tight') # dpi auto 600
print(f'{round(time()-t0, 2)}')

#%%
# # =============================================================================
# # Clustering OPTICS
# # =============================================================================
# var_filename = rg.list_precur_pp[0][1]
# # mask = [155.0, 230.0, 40.0, 45.0]
# # mask = None
# # mask = '/Users/semvijverberg/surfdrive/Data_era5/input_raw/mask_North_America_0.25deg.nc'
# from time import time ; t0 = time()
# xrclustered, results = cl.correlation_clustering(var_filename, mask=xr_mask,
#                                                kwrgs_load={'tfreq':10,
#                                                            'seldates':('06-01', '08-31'),
#                                                            'selbox':selbox},
#                                                clustermethodkey='OPTICS',
#                                                kwrgs_clust={#'eps':.05,
#                                                             'min_samples':5,
#                                                             'metric':'minkowski',
#                                                              'n_jobs':-1})

# plot_maps.plot_labels(xrclustered)
# print(f'{round(time()-t0, 2)}')


#%%
q_list = [75, 82.5, 90, 95]
for q in q_list:
    for c in n_clusters:    
        q = 5 ; c=6
        xrclust = xrclustered.sel(q=q, n_clusters=c)
        ds = cl.spatial_mean_clusters(var_filename,
                                  xrclust,
                                  selbox=selbox)
        
        ds[f'q{95}'] = cl.percentile_cluster(var_filename, 
                                              xrclust, 
                                              q=95, 
                                              tailmean=False, 
                                              selbox=selbox)

        q_sp = 70
        ds[f'q{q_sp}tail'] = cl.percentile_cluster(var_filename, 
                                              xrclust, 
                                              q=q_sp, 
                                              tailmean=True, 
                                              selbox=selbox)
        q_sp = 90
        ds[f'q{q_sp}tail'] = cl.percentile_cluster(var_filename, 
                                              xrclust, 
                                              q=q_sp, 
                                              tailmean=True, 
                                              selbox=selbox)



        df_clust = functions_pp.xrts_to_df(ds['ts'])
    
        fig = df_ana.loop_df(df_clust, function=df_ana.plot_ac, sharex=False, 
                             colwrap=2, kwrgs={'AUC_cutoff':(14,30), 's':60})
        fig.suptitle('q: {}, n_clusters: {}'.format(q, c), x=.5, y=.97)
        
        df_clust = functions_pp.xrts_to_df(ds[f'q{q}tail'])
    
        fig = df_ana.loop_df(df_clust, function=df_ana.plot_ac, sharex=False, 
                             colwrap=2, kwrgs={'AUC_cutoff':(14,30),'s':60})
        fig.suptitle('tfreq: {}, n_clusters: {}, q{}tail'.format(1, c, q), 
                     x=.5, y=.97)
#%%
t = 15 ; c = 5        
ds = cl.spatial_mean_clusters(var_filename,
                         xrclustered.sel(tfreq=t, n_clusters=c),
                         selbox=selbox)
f_name = 'tf{}_nc{}'.format(int(ds['ts'].tfreq), int(ds['n_clusters'].tfreq))
filepath = os.path.join(path_outmain, f_name)
cl.store_netcdf(ds, filepath=filepath, append_hash='dendo_'+xrclustered.attrs['hash'])

#%%
import pandas as pd
import df_ana
from class_SSA import SSA
keys = ['ts', 'q95', 'q70tail', 'q90tail']
all_ts = []
for key in keys:
    ts = functions_pp.xrts_to_df(ds[key])[1]
    all_ts.append( ts )
merged_ts = pd.concat(all_ts, keys=keys, axis=1)
#merged_ts = merged_ts[['q95','q90tail']]
merged_ts_std = (merged_ts - merged_ts.mean())/merged_ts.std()
merged_ts.iloc[0:5*365].plot()
window = 10
plt.figure()
merged_ts_std.iloc[0:365].plot()

fig = df_ana.loop_df(merged_ts, function=df_ana.plot_ac, sharex=False, 
                             colwrap=2, kwrgs={'AUC_cutoff':(14,30), 's':60})

q_90tail = merged_ts[['q90tail']].resample('W').mean()
q_95 = merged_ts[['q95']].resample('W').mean()
ssa_q90tail = SSA(q_90tail['q90tail'], L=window)
ssa_q95 = SSA(q_95['q95'], L=window)
plt.figure()
ssa_q95.plot_wcorr(max=window)
ssa_q90tail.plot_wcorr(max=window)

q_90tail['F0'] = ssa_q90tail.reconstruct(0).to_frame(name='F0')
q_95['F0'] = ssa_q95.reconstruct(0).to_frame(name='F0')
q_90tail['q90tail']
plt.figure()
q_90tail.plot()
q_95.plot()
plt.figure()
(q_90tail['q90tail'] - q_90tail['F0']).plot()
plt.figure()
(q_95['q95'] - q_95['F0']).plot()
plt.figure()
((q_95['q95'] - q_95['F0']) - (q_90tail['q90tail'] - q_90tail['F0'])).plot()
#%%
fig = df_ana.loop_df(merged_ts, function=df_ana.plot_spectrum, sharey=False, sharex=False, 
                             colwrap=2, kwrgs={})

# In[ ]:





