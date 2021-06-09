#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 09:35:06 2020

@author: semvijverberg
"""

import os, inspect, sys
import numpy as np
import xarray as xr
import pandas as pd
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
    root_data = user_dir+'/surfdrive/Scripts/RGCPD/publications/paper_Raed/data/'
else:
    root_data = user_dir+'/Dropbox/VIDI_Coumou/Paper3_Sem/GDHY_MIRCA2000_Soy/USDA/'

path_outmain = user_dir+'/surfdrive/Scripts/RGCPD/publications/paper_Raed/clustering'

import plot_maps, core_pp, functions_pp
import clustering_spatial as cl

#%% Soy bean GDHY
apply_mask_nonans = True
detrend_via_spatial_mean = False
missing_years = 0

raw_filename = os.path.join(root_data, 'masked_rf_gs_county_grids.nc')
selbox = [253,290,28,50] ; years = list(range(1950, 2020))

ds_raw = core_pp.import_ds_lazy(raw_filename, selbox=selbox)['variable'].rename({'z':'time'})
ds_raw.name = 'Soy_Yield'
ds_raw['time'] = pd.to_datetime([f'{y+1949}-01-01' for y in ds_raw.time.values])
ds_raw = ds_raw.sel(time=core_pp.get_oneyr(ds_raw, *years))

ano = ds_raw - ds_raw.mean(dim='time')
np.isnan(ano).sum(dim='time').plot()

core_pp.detrend_lin_longterm(ano)
_, trend = core_pp.detrend(ano, method='loess', return_trend=True)
#%%
if apply_mask_nonans:
    allways_data_mask = np.isnan(ano).sum(dim='time') <= missing_years
    ano = ano.where(allways_data_mask)
    if missing_years != 0:
        ano = ano.interpolate_na(dim='time', limit=2) # max 2 consecutive NaNs
        allways_data_mask = np.isnan(ano).sum(dim='time') <= 0
        ano = ano.where(allways_data_mask)
# plot_maps.plot_corr_maps(ano, row_dim='time', cbar_vert=.09)
if detrend_via_spatial_mean :
    detrend_spat_mean = ano.mean(dim=('latitude', 'longitude'))
    trend = detrend_spat_mean - core_pp.detrend(detrend_spat_mean)
    ano = ano - trend
else:
    ano = core_pp.detrend_lin_longterm(ano)

var_filename = raw_filename[:-3] + '_pp.nc'
ano.to_netcdf(var_filename)


#%%

# =============================================================================
# Clustering Ward Hierarchical Agglomerative Clustering
# =============================================================================
from time import time
t0 = time()
xrclustered, results = cl.sklearn_clustering(var_filename, mask=np.mean(~np.isnan(ano), axis=0) == 1,
                                               kwrgs_load={'tfreq':None,
                                                           'seldates':None,
                                                           'selbox':None},
                                               clustermethodkey='AgglomerativeClustering',
                                               kwrgs_clust={'n_clusters':[2,3,4,5,6,7],
                                                            'affinity':'euclidean',
                                                            'linkage':['ward', 'average']})

plot_maps.plot_labels(xrclustered,  kwrgs_plot={'wspace':.05, 'hspace':.15, 'cbar_vert':.05,
                      'row_dim':'n_clusters', 'col_dim':'linkage'})

f_name = 'clustering_Hierchical_correlation_{}'.format(xrclustered.attrs['hash']) + '.pdf'
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
xrclustered, results = cl.sklearn_clustering(var_filename, mask=np.mean(~np.isnan(ano), axis=0) == 1,
                                               kwrgs_load={'tfreq':None,
                                                           'seldates':None,
                                                           'selbox':None},
                                               clustermethodkey='AgglomerativeClustering',
                                               kwrgs_clust={'n_clusters':[2,3,4,5,6,7],
                                                            'affinity':'correlation',
                                                            'linkage':'average'})

plot_maps.plot_labels(xrclustered,  kwrgs_plot={'wspace':.05, 'hspace':.15, 'cbar_vert':.05,
                      'row_dim':'n_clusters'})

f_name = 'clustering_Hierchical_correlation_{}'.format(xrclustered.attrs['hash']) + '.pdf'
path_fig = os.path.join(path_outmain, f_name)
plt.savefig(path_fig,
            bbox_inches='tight') # dpi auto 600
print(f'{round(time()-t0, 2)}')

#%%
# =============================================================================
# Clustering K-means Euclidian
# =============================================================================
from time import time
t0 = time()
xrclustered, results = cl.sklearn_clustering(var_filename, mask=np.mean(~np.isnan(ano), axis=0) == 1,
                                               kwrgs_load={'tfreq':None,
                                                           'seldates':None,
                                                           'selbox':None},
                                               clustermethodkey='KMeans',
                                               kwrgs_clust={'n_clusters':[2,3,4,8,10],
                                                            'random_state':[0,1,2]})
                                                            # 'linkage':'average'})

plot_maps.plot_labels(xrclustered,  kwrgs_plot={'wspace':.05, 'hspace':.15, 'cbar_vert':.05,
                            'row_dim':'n_clusters', 'col_dim':'random_state'})

f_name = 'clustering_KMeans_{}'.format(xrclustered.attrs['hash']) + '.pdf'
path_fig = os.path.join(path_outmain, f_name)
plt.savefig(path_fig,
            bbox_inches='tight') # dpi auto 600
print(f'{round(time()-t0, 2)}')

#%%
# =============================================================================
# Clustering co-occurence of anomalies Hierarchical Agglomerative Clustering
# =============================================================================
from time import time

quantiles = [50,55,60,65]
n_clusters = [2,3,4,5,6,7,8]

t0 = time()
xrclustered, results = cl.dendogram_clustering(var_filename, mask=np.mean(~np.isnan(ano), axis=0) == 1,
                                               kwrgs_load={'tfreq':None,
                                                           'seldates':None,
                                                           'selbox':None},
                                               kwrgs_clust={'q':quantiles,
                                                            'n_clusters':n_clusters,
                                                            'affinity':'jaccard',
                                                            'linkage':'average'},
                                               n_cpu=3)

plot_maps.plot_labels(xrclustered,  kwrgs_plot={'wspace':.05, 'hspace':.15, 'cbar_vert':.05,
                      'row_dim':'n_clusters', 'col_dim':'q'})

f_name = 'clustering_Hierchical_dendogram_{}'.format(xrclustered.attrs['hash']) + '.pdf'
path_fig = os.path.join(path_outmain, f_name)
plt.savefig(path_fig,
            bbox_inches='tight') # dpi auto 600
print(f'{round(time()-t0, 2)}')

#%%
# =============================================================================
# Clustering co-occurence of anomalies - Again clustering large cluster
# =============================================================================
from time import time

quantiles = [50,55,60,65]
n_clusters = [2,3,4,5,6,7,8]

t0 = time()
xrclustered, results = cl.dendogram_clustering(var_filename, mask=(xrclustered.sel(n_clusters=8).sel(q=65)==2),
                                               kwrgs_load={'tfreq':None,
                                                           'seldates':None,
                                                           'selbox':None},
                                               kwrgs_clust={'q':quantiles,
                                                            'n_clusters':n_clusters,
                                                            'affinity':'jaccard',
                                                            'linkage':'average'})

plot_maps.plot_labels(xrclustered,  wspace=.05, hspace=.15, cbar_vert=.05,
                            row_dim='n_clusters', col_dim='q')

f_name = 'clustering_Hierchical_dendogram_{}'.format(xrclustered.attrs['hash']) + '.pdf'
path_fig = os.path.join(path_outmain, f_name)
plt.savefig(path_fig,
            bbox_inches='tight') # dpi auto 600
print(f'{round(time()-t0, 2)}')

#%%
years = list(range(1975, 2020))

ds_raw = core_pp.import_ds_lazy(raw_filename, selbox=selbox)['variable'].rename({'z':'time'})
ds_raw.name = 'Soy_Yield'
ds_raw['time'] = pd.to_datetime([f'{y+1949}-01-01' for y in ds_raw.time.values])
ds_raw = ds_raw.sel(time=core_pp.get_oneyr(ds_raw, *years))

linkage = 'ward' ; c =4
ds = cl.spatial_mean_clusters(ds_raw,
                         xrclustered.sel(linkage=linkage, n_clusters=c))
df = ds.ts.to_dataframe().pivot_table(index='time', columns='cluster')['ts']

f_name = 'linkage_{}_nc{}'.format(linkage, int(c))
filepath = os.path.join(path_outmain, f_name)
cl.store_netcdf(ds, filepath=filepath, append_hash='dendo_'+xrclustered.attrs['hash'])

#%% Soy bean USDA

raw_filename = '/Users/semvijverberg/Dropbox/VIDI_Coumou/Paper3_Sem/GDHY_MIRCA2000_Soy/USDA/usda_soy.nc'

selbox = [250,290,28,50]
ds = core_pp.import_ds_lazy(raw_filename, selbox=selbox)['variable'].rename({'z':'time'})
ds.name = 'Soy_Yield'

ds['time'] = pd.to_datetime([f'{y+1949}-01-01' for y in ds.time.values])
ds.attrs['dataset'] = 'USDA'
ds.attrs['planting_months'] = 'May/June'
ds.attrs['harvest_months'] = 'October'

ts = functions_pp.area_weighted(ds).mean(dim=('latitude', 'longitude')) # old, but silly to do area-weighted mean
cl.store_netcdf(ts, filepath=
                '/Users/semvijverberg/Dropbox/VIDI_Coumou/Paper3_Sem/GDHY_MIRCA2000_Soy/USDA/usda_soy_spatial_mean_ts.nc')
#%% Maize yield USDA

raw_filename = os.path.join('/Users/semvijverberg/surfdrive/VU_Amsterdam/GDHY_MIRCA2000_Soy/USDA/usda_maize.nc')

ds = core_pp.import_ds_lazy(raw_filename)['variable'].rename({'z':'time'})
ds.name = 'Maize_Yield'

ds['time'] = pd.to_datetime([f'{y+1949}-01-01' for y in ds.time.values])
ds.attrs['dataset'] = 'USDA'
ds.attrs['planting_months'] = 'May/June'
ds.attrs['harvest_months'] = 'October'

ts = functions_pp.area_weighted(ds).mean(dim=('latitude', 'longitude'))
cl.store_netcdf(ts, filepath=
                os.path.join('/Users/semvijverberg/surfdrive/VU_Amsterdam/GDHY_MIRCA2000_Soy/USDA/usda_maize_spatial_mean_ts.nc'))

#%%

allways_data_mask = np.isnan(ds).mean(dim='time')==0
ts_mask = ds.where(allways_data_mask).mean(dim=('latitude', 'longitude'))
cl.store_netcdf(ts_mask, filepath=
                '/Users/semvijverberg/Dropbox/VIDI_Coumou/Paper3_Sem/GDHY_MIRCA2000_Soy/USDA/usda_soy_spatial_mean_ts_allways_data.nc')
#%%
ano = ds - ds.mean(dim='time')
plot_maps.plot_corr_maps(ano.isel(time=range(0,40,5)), row_dim='time', cbar_vert=.09)


