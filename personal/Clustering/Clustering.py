#!/usr/bin/env python
# coding: utf-8

# # Clustering

# In[1]:


import os, inspect, sys
import numpy as np
user_dir = os.path.expanduser('~')
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-2])
RGCPD_func = os.path.join(main_dir, 'RGCPD')
cluster_func = os.path.join(main_dir, 'clustering/')
if cluster_func not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(RGCPD_func)
    sys.path.append(cluster_func)


if sys.platform == 'linux':
    import matplotlib as mpl
    mpl.use('Agg')
    root_data = '/scistor/ivm/data_catalogue/reanalysis/ERA5'
else:
    root_data = '/Users/semvijverberg/surfdrive/ERA5'
    
path_outmain = user_dir+'/surfdrive/output_RGCPD'
# In[2]:



import clustering_spatial as cl
import plot_maps
from RGCPD import RGCPD
list_of_name_path = [('fake', None),
                     ('t2mmmax', root_data + '/input_raw/mx2t_US_1979-2018_1_12_daily_1.0deg.nc')]
rg = RGCPD(list_of_name_path=list_of_name_path,
           path_outmain=path_outmain)



# In[3]:


rg.pp_precursors()


# In[ ]:


rg.list_precur_pp

var_filename = rg.list_precur_pp[0][1]

#%%
import make_country_mask
selbox = (225, 300, 30, 70)
xarray, Country = make_country_mask.create_mask(var_filename, kwrgs_load={'selbox':selbox}, level='Countries')
mask_US_CA = np.logical_or(xarray.values == Country.US, xarray.values==Country.CA)
xr_mask = xarray.where(make_country_mask.binary_erosion(mask_US_CA))
xr_mask.values[make_country_mask.binary_erosion(mask_US_CA)]  = 1
plot_maps.plot_labels(xr_mask)



# In[9]:
# =============================================================================
# Clustering co-occurence of anomalies
# =============================================================================

from time import time
t0 = time()
xrclustered, results = cl.dendogram_clustering(var_filename, mask=xr_mask,
                                               kwrgs_load={'tfreq':[30],
                                                           'seldates':('06-15', '08-31'),
                                                           'selbox':selbox},
                                               kwrgs_clust={'q':66,
                                                            'n_clusters':[2],
                                                            'affinity':'jaccard',
                                                            'linkage':'average'})
plot_maps.plot_labels(xrclustered, wspace=.05, hspace=-.2, cbar_vert=.08,
                      row_dim='tfreq', col_dim='n_clusters')
print(f'{round(time()-t0, 2)}')

#%%
# =============================================================================
# Clustering correlation Hierarchical Agglomerative Clustering
# =============================================================================
from time import time
t0 = time()
xrclustered, results = cl.correlation_clustering(var_filename, mask=xr_mask,
                                               kwrgs_load={'tfreq':[5,10,15,30],
                                                           'seldates':('06-01', '08-31'),
                                                           'selbox':selbox},
                                               clustermethodkey='AgglomerativeClustering',
                                               kwrgs_clust={'n_clusters':[2,3,6,7,8],
                                                            'affinity':'correlation',
                                                            'linkage':'average'})

fig = plot_maps.plot_labels(xrclustered,  wspace=.05, hspace=-.2, cbar_vert=.08,
                            row_dim='tfreq', col_dim='n_clusters')
path_fig = os.path.join(rg.path_outmain, 'clustering_{}'.format(xrclustered.attrs['hash'])) + '.pdf'
fig.savefig(path_fig,
            bbox_inches='tight') # dpi auto 600
print(f'{round(time()-t0, 2)}')

#%%
# =============================================================================
# Clustering OPTICS
# =============================================================================
var_filename = rg.list_precur_pp[0][1]
# mask = [155.0, 230.0, 40.0, 45.0]
# mask = None
# mask = '/Users/semvijverberg/surfdrive/Data_era5/input_raw/mask_North_America_0.25deg.nc'
from time import time ; t0 = time()
xrclustered, results = cl.correlation_clustering(var_filename, mask=xr_mask,
                                               kwrgs_load={'tfreq':10,
                                                           'seldates':('06-01', '08-31'),
                                                           'selbox':selbox},
                                               clustermethodkey='OPTICS',
                                               kwrgs_clust={#'eps':.05,
                                                            'min_samples':5,
                                                            'metric':'minkowski',
                                                             'n_jobs':-1})

plot_maps.plot_labels(xrclustered)
print(f'{round(time()-t0, 2)}')


#%%
ds = cl.spatial_mean_clusters(var_filename,
                              xrclustered.sel(tfreq=10, n_clusters=6),
                              selbox=selbox)
cl.store_netcdf(ds, filepath=None, append_hash=xrclustered.attrs['hash'])

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





