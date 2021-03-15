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

path_outmain = user_dir+'/surfdrive/output_RGCPD/circulation_US_HW/one-point-corr_maps_clusters'
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
region = 'US'
if region == 'US':
    selbox = (230, 300, 25, 50)
    TVpath = '/Users/semvijverberg/surfdrive/output_RGCPD/circulation_US_HW/tf10_nc5_dendo_3e180_US.nc'
    np_array_xy = np.array([[-84, 34], [-96, 40], [-87, 42],
                            [-122,40], [-122,46], [-117,46]])
    t, c = 10, 5

elif region == 'USCA':
    selbox = (225, 300, 25, 60)
    TVpath = '/Users/semvijverberg/surfdrive/output_RGCPD/circulation_US_HW/tf15_nc3_dendo_2f9a0_USCA_1979-2020.nc'
    np_array_xy = np.array([[-100, 33], [-95, 40], [-88, 35], [-83,40],
                            [-118,36], [-120,47], [-126,53], [-120,56]])
    t, c = 30, 7
elif region == 'init':
    selbox = (225, 300, 25, 60)
    TVpath = '/Users/semvijverberg/surfdrive/output_RGCPD/circulation_US_HW/tf15_nc3_dendo_0ff31.nc'
    np_array_xy = np.array([[-98, 35], [-95, 45], [-85, 35], [-83,45],
                            [-118,36], [-120,47], [-120,56], [-106,53]])
    t, c = 15, 3

ds = core_pp.import_ds_lazy(TVpath)
if region != 'init':
    xrclustered = ds['xrclusteredall'].sel(tfreq=t, n_clusters=5)
    xrclustered = xrclustered.where(xrclustered.values!=-9999)
else:
    xrclustered = ds['xrclustered']


size = 100
colors = plt.cm.tab20.colors[:np_array_xy.shape[0]]
scatter = [['all', [np_array_xy, {'s':size, 'zorder':2,
                                  'color':colors,
                                  'edgecolors':'black'}] ]]
regions= list(np.unique(xrclustered)[~np.isnan(np.unique(xrclustered))])
if region == 'USCA':
    dic = {4:3, 3:4}
else:
    dic = {2:3, 3:2}
xrclustered = find_precursors.view_or_replace_labels(xrclustered, regions,
                                                     [int(dic.get(n, n)) for n in regions])
fig = plot_maps.plot_labels(xrclustered,
                      {'scatter':scatter,
                       'zoomregion':selbox})
fig.set_facecolor('white')
fig.axes[0].set_facecolor('white')
f_name = 'scatter_clusters_t2m_{}_{}'.format(xrclustered.attrs['hash'], region)
filepath = os.path.join(rg.path_outmain, f_name)

plt.savefig(filepath+'.pdf', bbox_inches='tight')


#%% Check spatial correlation within clusters

# TVpath = '/Users/semvijverberg/surfdrive/output_RGCPD/circulation_US_HW/tf15_nc3_dendo_0ff31.nc'

# list_of_name_path = [(1, TVpath),
#                      ('mx2t', root_data + '/input_raw/mx2t_US_1979-2018_1_12_daily_0.25deg.nc')]
# rg = RGCPD(list_of_name_path=list_of_name_path,
#            path_outmain=path_outmain,
#            tfreq=15,
#            start_end_TVdate=('06-01', '08-31'))
# rg.pp_precursors()
# rg.pp_TV()
# rg.get_clust()
# xrclustered = rg.ds['xrclustered'].where(xrclustered!=-9999)


#%% Get timeseries at specific points within gridcell
ds_t2m = core_pp.import_ds_lazy(rg.list_of_name_path[1][1], selbox=selbox)
npts = np.zeros( (np_array_xy.shape[0], ds_t2m.time.size) )
for i, xy in enumerate(np_array_xy):
    npts[i] = ds_t2m.sel(longitude=(180+(180+xy[0])),
                         latitude=xy[1])


columns = [f'{abs(c[0])}W-{c[1]}N' for c in np_array_xy]
df_ts = pd.DataFrame(npts.T, index=pd.to_datetime(ds_t2m.time.values),
                     columns=columns)

TVpath = os.path.join(user_dir,
                      'surfdrive/Scripts/RGCPD/publications/paper2/data/',
                      'df_ts_paper2_clustercorr_{}.h5'.format(xrclustered.attrs['hash']))

functions_pp.store_hdf_df({'df_ts':df_ts}, file_path=TVpath)
#%% Calculate corr maps

list_xr = []
for point in df_ts.columns:
    list_of_name_path = [('', TVpath),
                         ('t2m', root_data + '/input_raw/t2m_US_1979-2020_1_12_daily_0.25deg.nc')]
    list_for_MI   = [BivariateMI(name='t2m', func=class_BivariateMI.corr_map,
                                  alpha=.05, FDR_control=True, lags=np.array([0]))]

    rg = RGCPD(list_of_name_path=list_of_name_path,
               list_for_MI=list_for_MI,
               path_outmain=path_outmain,
               tfreq=15,
               start_end_TVdate=('06-01', '08-31'),
               save=False)
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
if region == 'USCA':
    col_wrap = 4
elif region == 'US':
    col_wrap = 3
subtitles = np.array([point_corr.points]).reshape(-1, col_wrap)
# scatter =[[(0,0), [np_array_xy[[0]], {'s':size, 'zorder':2, 'color':colors[0]}] ],
#           [(0,1), [np_array_xy[[1]], {'s':size, 'zorder':2, 'color':colors[1]}] ],
#           [(0,2), [np_array_xy[[2]], {'s':size, 'zorder':2, 'color':colors[2]}] ],
#           [(0,3), [np_array_xy[[2]], {'s':size, 'zorder':2, 'color':colors[3]}] ],
#           [(1,0), [np_array_xy[[3]], {'s':size, 'zorder':2, 'color':colors[4]}] ],
#           [(1,1), [np_array_xy[[4]], {'s':size, 'zorder':2, 'color':colors[5]}] ],
#           [(1,2), [np_array_xy[[5]], {'s':size, 'zorder':2, 'color':colors[6]}] ],
#           [(1,3), [np_array_xy[[5]], {'s':size, 'zorder':2, 'color':colors[7]}] ]]
scatter = None
plot_maps.plot_corr_maps(point_corr,
                         mask_xr = None,
                         col_dim='points',
                         aspect=1.5, hspace=.2,
                         subtitles=subtitles,
                         scatter=scatter,
                         col_wrap=col_wrap,
                         cbar_vert=-.03,
                         x_ticks=np.arange(240, 301, 20),
                         y_ticks=np.arange(0,61,15),
                         clevels=np.arange(-1,1.1,.2),
                         zoomregion=selbox)
f_name = 'one_point_corr_maps_t2m_{}_{}'.format(xrclustered.attrs['hash'], region)
filepath = os.path.join(rg.path_outmain, f_name)
plt.savefig(filepath+'.pdf', bbox_inches='tight')