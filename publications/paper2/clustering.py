#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 15:26:32 2020

@author: semvijverberg
"""

# # Clustering

# In[1]:


import os, inspect, sys
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
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
    root_data = os.path.join(user_dir, 'surfdrive/ERA5/')
else:
    root_data = '/Users/semvijverberg/surfdrive/ERA5'

path_outmain = user_dir+'/surfdrive/output_RGCPD/circulation_US_HW'
# In[2]:


import functions_pp, find_precursors, core_pp
import clustering_spatial as cl
import plot_maps
import df_ana
from RGCPD import RGCPD
from RGCPD import BivariateMI ; import class_BivariateMI
list_of_name_path = [('fake', None),
                     ('t2m', root_data + '/input_raw/t2m_US_1979-2020_1_12_daily_0.25deg.nc')]
rg = RGCPD(list_of_name_path=list_of_name_path,
           path_outmain=path_outmain)



# In[3]:


rg.pp_precursors()


# In[ ]:


rg.list_precur_pp

var_filename = rg.list_precur_pp[0][1]

import pandas as pd
ds = core_pp.import_ds_lazy(var_filename)
ds.sel(time=core_pp.get_subdates(pd.to_datetime(ds.time.values), start_end_date=('06-01', '08-31'))).mean(dim='time').plot()
#%%
import make_country_mask
orography = os.path.join(user_dir, 'surfdrive/ERA5/input_raw/Orography.nc')
selbox = (225, 300, 25, 70)
xarray, Country = make_country_mask.create_mask(var_filename, kwrgs_load={'selbox':selbox}, level='Countries')
# mask_US_CA = np.logical_or(xarray.values == Country.US, xarray.values==Country.CA)
mask_US_CA = xarray.values == Country.US
# xr_mask =  xarray.where(mask_US_CA)
xr_mask = xarray.where(make_country_mask.binary_erosion(mask_US_CA))
# xr_mask =  xarray.where(make_country_mask.binary_erosion(np.nan_to_num(xr_mask)))
xr_mask.values[~np.isnan(xr_mask)] = 1
xr_mask = find_precursors.xrmask_by_latlon(xr_mask, upper_right=(270, 63))
# mask small Western US Island
xr_mask = find_precursors.xrmask_by_latlon(xr_mask, bottom_left=(228, 58))
# add Rocky mask
geo_surf_height = core_pp.import_ds_lazy(orography,
                                  var='z_NON_CDM', selbox=selbox) / 9.81
geo_surf_height = geo_surf_height.drop('time').drop('realization')
plot_maps.plot_corr_maps(geo_surf_height, cmap=plt.cm.Oranges, clevels=np.arange(0, 2600, 500))
mask_Rockies = geo_surf_height < 1500
plot_maps.plot_labels(mask_Rockies)
xr_mask = xr_mask.where(mask_Rockies)

plot_maps.plot_labels(xr_mask)



# In[9]:
# =============================================================================
# Clustering co-occurence of anomalies
# =============================================================================
tfreq = [5, 10, 15, 30]
n_clusters = [2,3,4,5,6,7,8]
from time import time
t0 = time()
xrclustered, results = cl.dendogram_clustering(var_filename, mask=xr_mask,
                                               kwrgs_load={'tfreq':tfreq,
                                                           'seldates':('06-01', '08-31'),
                                                           'start_end_TVdate':('06-01', '08-31'),
                                                           'selbox':selbox},
                                               kwrgs_clust={'q':66,
                                                            'n_clusters':n_clusters,
                                                            'affinity':'jaccard',
                                                            'linkage':'average'})

xrclustered.attrs['hash'] +='_US'
fig = plot_maps.plot_labels(xrclustered,
                            kwrgs_plot={'wspace':.03, 'hspace':-.35,
                                        'cbar_vert':.09,
                                        'row_dim':'n_clusters', 'col_dim':'tfreq'})
f_name = 'clustering_dendogram_{}'.format(xrclustered.attrs['hash']) + '.pdf'
path_fig = os.path.join(rg.path_outmain, f_name)
plt.savefig(path_fig,
            bbox_inches='tight') # dpi auto 600
print(f'{round(time()-t0, 2)}')

#%%
# # =============================================================================
# # Clustering correlation Hierarchical Agglomerative Clustering
# # =============================================================================
# from time import time
# t0 = time()
# xrclustered, results = cl.correlation_clustering(var_filename, mask=xr_mask,
#                                                kwrgs_load={'tfreq':tfreq,
#                                                            'seldates':('06-01', '08-31'),
#                                                            'selbox':selbox},
#                                                clustermethodkey='AgglomerativeClustering',
#                                                kwrgs_clust={'n_clusters':n_clusters,
#                                                             'affinity':'correlation',
#                                                             'linkage':'average'})

# plot_maps.plot_labels(xrclustered,  wspace=.05, hspace=-.2, cbar_vert=.08,
#                             row_dim='tfreq', col_dim='n_clusters')

# f_name = 'clustering_correlation_{}'.format(xrclustered.attrs['hash']) + '.pdf'
# path_fig = os.path.join(rg.path_outmain, f_name)
# plt.savefig(path_fig,
#             bbox_inches='tight') # dpi auto 600
# print(f'{round(time()-t0, 2)}')

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


# t = 15 ; c=3
# xrclust = xrclustered.sel(tfreq=t, n_clusters=c)
# ds = cl.spatial_mean_clusters(var_filename,
#                           xrclust,
#                           selbox=selbox)
# q = 75
# ds[f'q{q}tail'] = cl.percentile_cluster(var_filename,
#                                       xrclust,
#                                       q=q,
#                                       tailmean=True,
#                                       selbox=selbox)
# q = 50
# ds[f'q{q}tail'] = cl.percentile_cluster(var_filename,
#                                       xrclust,
#                                       q=q,
#                                       tailmean=True,
#                                       selbox=selbox)
# q = 25
# ds[f'q{q}tail'] = cl.percentile_cluster(var_filename,
#                                       xrclust,
#                                       q=q,
#                                       tailmean=True,
#                                       selbox=selbox)


# df_clust = functions_pp.xrts_to_df(ds['ts'])

# fig = df_ana.loop_df(df_clust, function=df_ana.plot_ac, sharex=False,
#                      colwrap=2, kwrgs={'AUC_cutoff':(14,30), 's':60})
# fig.suptitle('tfreq: {}, n_clusters: {}'.format(t, c), x=.5, y=.97)

# df_clust = functions_pp.xrts_to_df(ds[f'q{q}tail'])

# fig = df_ana.loop_df(df_clust, function=df_ana.plot_ac, sharex=False,
#                      colwrap=2, kwrgs={'AUC_cutoff':(14,30),'s':60})
# fig.suptitle('tfreq: {}, n_clusters: {}, q{}tail'.format(t, c, q),
#              x=.5, y=.97)
#%%
t = 15 ; c = 3
ds = cl.spatial_mean_clusters(var_filename,
                             xrclustered.sel(tfreq=t, n_clusters=c),
                             selbox=selbox)
ds['xrclusteredall'] = xrclustered
f_name = 'tf{}_nc{}'.format(int(t), int(c))
filepath = os.path.join(rg.path_outmain, f_name)
cl.store_netcdf(ds, filepath=filepath, append_hash='dendo_'+xrclustered.attrs['hash'])

#%% Check spatial correlation within clusters
# filepath = '/Users/semvijverberg/surfdrive/output_RGCPD/circulation_US_HW/tf15_nc3_dendo_0ff31.nc'

list_of_name_path = [(2, filepath),
                     ('mx2t', root_data + '/input_raw/mx2t_US_1979-2018_1_12_daily_0.25deg.nc')]
rg = RGCPD(list_of_name_path=list_of_name_path,
           path_outmain=path_outmain,
           tfreq=15,
           start_end_TVdate=('06-01', '08-31'))
rg.pp_precursors()
rg.pp_TV()
rg.get_clust()
xrclustered = rg.ds['xrclustered']
#%% Get timeseries at specific points within gridcell
import plot_maps
ds = core_pp.import_ds_lazy(rg.list_precur_pp[0][1])
np_array_xy = np.array([[-90, 35], [-90, 42], [-90, 50],
                        [-119,35], [-120,45], [-110,55]])
colors = ["#22223b","#ffbe0b","#fb5607","#ff006e","#8338ec","#3a86ff"][::-1]
scatter = [['all', [np_array_xy, {'s':100, 'zorder':2,
                                  'color':colors[:6]}] ]]
fig = plot_maps.plot_labels(rg.ds['xrclustered'],
                      {'scatter':scatter})

#%%
npts = np.zeros( (np_array_xy.shape[0], ds.time.size) )
for i, xy in enumerate(np_array_xy):
    npts[i] = ds.sel(longitude=(180+(180+xy[0])),
                     latitude=xy[1])

df_ts = pd.DataFrame(npts.T, index=pd.to_datetime(ds.time.values),
                     columns=['90W-42N', '90W-35N', '90W-50N',
                              '120W-35N', '120W-45N', '110W-55N'])

TVpath = os.path.join(user_dir,
                      'surfdrive/Scripts/RGCPD/publications/paper2/data/',
                      'df_ts_paper2_clustercorr_{}.h5'.format(xrclustered.attrs['hash']))

functions_pp.store_hdf_df({'df_ts':df_ts}, file_path=TVpath)
#%%

list_xr = []
for point in df_ts.columns:
    list_of_name_path = [(point, TVpath),
                         ('mx2t', root_data + '/input_raw/mx2t_US_1979-2018_1_12_daily_0.25deg.nc')]
    list_for_MI   = [BivariateMI(name='mx2t', func=class_BivariateMI.corr_map,
                                  alpha=.05, FDR_control=True, lags=np.array([0]))]

    rg = RGCPD(list_of_name_path=list_of_name_path,
               list_for_MI=list_for_MI,
               path_outmain=path_outmain,
               tfreq=15,
               start_end_TVdate=('06-01', '08-31'))
    rg.pp_precursors()
    rg.pp_TV(name_ds=point)
    rg.traintest(False)
    rg.calc_corr_maps()
    precur = rg.list_for_MI[0]
    corr_xr = precur.corr_xr[0,0]
    list_xr.append(corr_xr)
point_corr = xr.concat(list_xr, dim='points')
point_corr['points'] = ('points', list(df_ts.columns))
#%%
subtitles = np.array([point_corr.points]).reshape(-1, 3)
scatter =[[(0,0), [np_array_xy[[0]], {'s':20, 'zorder':2, 'color':"dodgerblue"}] ],
          [(0,1), [np_array_xy[[1]], {'s':20, 'zorder':2, 'color':"dodgerblue"}] ],
           [(0,2), [np_array_xy[[2]], {'s':20, 'zorder':2, 'color':"dodgerblue"}] ],
           [(1,0), [np_array_xy[[3]], {'s':20, 'zorder':2, 'color':"dodgerblue"}] ],
           [(1,1), [np_array_xy[[4]], {'s':20, 'zorder':2, 'color':"dodgerblue"}] ],
           [(1,2), [np_array_xy[[5]], {'s':20, 'zorder':2, 'color':"dodgerblue"}] ]]

plot_maps.plot_corr_maps(point_corr,
                         mask_xr = point_corr['mask'],
                         col_dim='points',
                         aspect=1.5, hspace=.2,
                         subtitles=subtitles,
                         scatter=scatter,
                         col_wrap=3,
                         cbar_vert=-.03,
                         n_xticks=4, n_yticks=4,
                         clevels=np.arange(-1,1.1,.2))
f_name = 'one_point_corr_maps_t2m_{}'.format(xrclustered.attrs['hash'])
filepath = os.path.join(rg.path_outmain, f_name)
plt.savefig(filepath+'.pdf', bbox_inches='tight')
# np_array_xy = np.array([[-100, 40]])
# ax = fig.axes[0]
# ax.scatter(x=np_array_xy[:,0], y=np_array_xy[:,1],
#             color="dodgerblue",
#             s=20,
#             alpha=0.5,
#             transform=ccrs.PlateCarree()) ## Important