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
    
path_outmain = user_dir+'/surfdrive/output_RGCPD/easternUS'
# In[2]:


import functions_pp, core_pp
import clustering_spatial as cl
import plot_maps
import df_ana
from RGCPD import RGCPD
list_of_name_path = [('fake', None),
                     ('mxt2', root_data + '/input_raw/mx2t_US_1979-2018_1_12_daily_0.25deg.nc')]
rg = RGCPD(list_of_name_path=list_of_name_path,
           path_outmain=path_outmain)



# In[3]:


rg.pp_precursors()


# In[ ]:


rg.list_precur_pp

var_filename = rg.list_precur_pp[0][1]

#%%
import make_country_mask

# xarray, Country = make_country_mask.create_mask(var_filename, kwrgs_load={'selbox':selbox}, level='Countries')
# mask_US = (xarray.values == Country.US)
# mask_US = make_country_mask.binary_erosion(mask_US)
# mask_US = make_country_mask.binary_erosion(mask_US)
# mask_US = make_country_mask.binary_opening(mask_US)
# xr_mask = xarray.where(mask_US)
# xr_mask.values[mask_US]  = 1
# xr_mask = cl.mask_latlon(xr_mask, latmax=63, lonmax=270)

selbox = (232, 295, 25, 50)
xr_mask = core_pp.import_ds_lazy('/Users/semvijverberg/surfdrive/Scripts/rasterio/mask_North_America_0.25deg_orig.nc', 
                                  var='lsm', selbox=selbox)
xr_mask.values = make_country_mask.binary_erosion(xr_mask.values)
plot_maps.plot_labels(xr_mask)



# In[9]:
# =============================================================================
# Clustering co-occurence of anomalies
# =============================================================================
q = [80, 85, 90, 95]
n_clusters = [2,3,4,5,6,7,8]
tfreq = 1
from time import time
t0 = time()
xrclustered, results = cl.dendogram_clustering(var_filename, mask=xr_mask,
                                               kwrgs_load={'tfreq':tfreq,
                                                           'seldates':('06-24', '08-22'),
                                                           'selbox':selbox},
                                               kwrgs_clust={'q':q,
                                                            'n_clusters':n_clusters,
                                                            'affinity':'jaccard',
                                                            'linkage':'average'})

# xr_temp = xrclustered.sel(longitude=
#                           np.arange(232., 295., .25)).sel(latitude=np.arange(25, 50, .25)).copy()
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
path_fig = os.path.join(rg.path_outmain, f_name)
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



# for c in n_clusters:  
q = 85 ; c=5  
xrclust = xrclustered.sel(q=q, n_clusters=c)
ds = cl.spatial_mean_clusters(var_filename,
                          xrclust,
                          selbox=selbox)

ds[f'q{95}'] = cl.percentile_cluster(var_filename, 
                                      xrclust, 
                                      q=95, 
                                      tailmean=False, 
                                      selbox=selbox)

q_sp = 50
ds[f'q{q_sp}tail'] = cl.percentile_cluster(var_filename, 
                                      xrclust, 
                                      q=q_sp, 
                                      tailmean=True, 
                                      selbox=selbox)        

q_sp = 65
ds[f'q{q_sp}tail'] = cl.percentile_cluster(var_filename, 
                                      xrclust, 
                                      q=q_sp, 
                                      tailmean=True, 
                                      selbox=selbox)

q_sp = 75
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
#%%


dims = list(ds.coords.keys())
standard_dim = ['latitude', 'longitude', 'time', 'mask', 'cluster']
dims = [d for d in dims if d not in standard_dim]
params = [dims[0], int(ds.coords[dims[0]]), dims[1], int(ds.coords[dims[1]])]
f_name = 'tf{}_{}{}_{}{}'.format(int(tfreq), *params)
filepath = os.path.join(path_outmain, f_name)
cl.store_netcdf(ds, filepath=filepath, append_hash='dendo_'+xrclustered.attrs['hash'])

#%%
# # =============================================================================
# # regrid for quicker validation
# # =============================================================================
# to_grid=1
# xr_regrid = cl.regrid_array(var_filename, to_grid=to_grid)
# cl.store_netcdf(xr_regrid, filepath=None, append_hash=f'{to_grid}d')

# xr_rg_clust = cl.regrid_array(xrclustered, to_grid=to_grid, periodic=False)
# ds = cl.spatial_mean_clusters(var_filename,
#                               xr_rg_clust)
# cl.store_netcdf(ds, filepath=None, append_hash=f'{to_grid}d_' + xrclustered.attrs['hash'])



# In[ ]:





