#!/usr/bin/env python
# coding: utf-8

# # Clustering

# In[1]:


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


# In[2]:



import clustering_spatial as cl
import plot_maps
from RGCPD import RGCPD




# In[3]:


rg.pp_precursors()


# In[ ]:


rg.list_precur_pp


# In[9]:


var_filename = rg.list_precur_pp[0][1]
#mask = [250.0, 255.0, 40.0, 45.0]
mask = '/Users/semvijverberg/surfdrive/Data_era5/input_raw/mask_North_America_0.25deg.nc'
from time import time
t0 = time()

xrclustered, results = cl.dendogram_clustering(var_filename, mask=mask, 
                                               kwrgs_load={'seldates':('06-24', '08-21')},
                                               q=95, kwrgs_clust={'n_clusters':4,
                                                                 'affinity':'jaccard',
                                                                 'linkage':'complete'})
plot_maps.plot_labels(xrclustered)
print(f'{round(time()-t0, 2)}')

ds = cl.spatial_mean_clusters(var_filename, xrclustered)
cl.store_netcdf(ds, filepath=None, append_hash=xrclustered.attrs['hash'])

#%%
# regrid for quicker validation
to_grid=1
xr_regrid = cl.regrid_array(var_filename, to_grid=to_grid)
cl.store_netcdf(xr_regrid, filepath=None, append_hash=f'{to_grid}d')


xr_rg_clust = cl.regrid_array(xrclustered, to_grid=to_grid, periodic=False)
ds = cl.spatial_mean_clusters('/Users/semvijverberg/surfdrive/Data_era5/input_raw/preprocessed/t2mmax_US_1979-2018_1jan_31dec_daily_1deg.nc.nc', 
                              xr_rg_clust)
cl.store_netcdf(ds, filepath=None, append_hash=f'{to_grid}d_' + xrclustered.attrs['hash'])



# In[ ]:
TVpath = '/Users/semvijverberg/surfdrive/Data_era5/input_raw/preprocessed/xrclustered_1d_c0f23.nc'
list_of_name_path = [(3, TVpath)]
rg = RGCPD(list_of_name_path=list_of_name_path)
rg.pp_TV()




