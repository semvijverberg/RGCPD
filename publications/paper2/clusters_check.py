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
import make_country_mask
import clustering_spatial as cl
import plot_maps
import df_ana
from RGCPD import RGCPD
from RGCPD import BivariateMI ; import class_BivariateMI
list_of_name_path = [('fake', None),
                     ('t2m', root_data + '/input_raw/mx2t_USfft_1979-2020_1_12_daily_0.25deg.nc')]
rg = RGCPD(list_of_name_path=list_of_name_path,
           path_outmain=path_outmain)



# In[3]:


rg.pp_precursors()


# In[ ]:


rg.list_precur_pp

var_filename = rg.list_precur_pp[0][1]
region = 'USCA'
#%%

import pandas as pd
ds = core_pp.import_ds_lazy(var_filename)
ds.sel(time=core_pp.get_subdates(pd.to_datetime(ds.time.values), start_end_date=('06-01', '08-31'))).mean(dim='time').plot()


#%%

if region == 'USCAnew':
    selbox = (230, 300, 25, 70)
    TVpath = os.path.join(path_outmain, 'tf10_nc5_dendo_0cbf8_US.nc')
    np_array_xy = np.array([[-91, 36], [-85, 34], [-81, 38],
                            [-116,36], [-122,41], [-117,46]])
    t, c = 15, 6

elif region == 'USCA':
    selbox = (230, 300, 25, 70)
    TVpath = os.path.join(path_outmain, 'tf10_nc5_dendo_5dbee_USCA.nc')
    t, c = 30, 8
    np_array_xy = np.array([[-100, 34], [-94, 38], [-88, 35], [-83,38],
                            [-113,34], [-120,38], [-120,48], [-124,56]])
    TVpath = os.path.join(path_outmain, 'tf10_nc5_dendo_0a6f6USCA.nc')
    t, c = 10, 5

elif region == 'init':
    selbox = (230, 300, 25, 60)
    TVpath = '/Users/semvijverberg/surfdrive/output_RGCPD/circulation_US_HW/tf15_nc3_dendo_0ff31.nc'
    np_array_xy = np.array([[-98, 35], [-95, 45], [-85, 35], [-84,45],
                            [-118,36], [-120,47], [-120,56], [-106,53]])
    t, c = 15, 3

ds_cl = core_pp.import_ds_lazy(TVpath)
if region != 'init':
    xrclustered = ds_cl['xrclusteredall'].sel(tfreq=t, n_clusters=c)
    xrclustered = xrclustered.where(xrclustered.values!=-9999)
else:
    xrclustered = ds_cl['xrclustered']


size = 100
colors = plt.cm.tab20.colors[:np_array_xy.shape[0]]
scatter = [['all', [np_array_xy, {'s':size, 'zorder':2,
                                  'color':colors,
                                  'edgecolors':'black'}] ]]
# regions= list(np.unique(xrclustered)[~np.isnan(np.unique(xrclustered))])
# if region == 'USCA':
#     dic = {4:3, 3:4}
# else:
#     dic = {2:3, 3:2}
# xrclustered = find_precursors.view_or_replace_labels(xrclustered.copy(), regions,
#                                                      [int(dic.get(n, n)) for n in regions])
if region == 'USCA':
    mask_cl_n = find_precursors.view_or_replace_labels(xrclustered.copy(), [1,5])
    mask_cl_n  = make_country_mask.binary_erosion(~np.isnan(mask_cl_n) )
    mask_cl_s = ~np.isnan(find_precursors.view_or_replace_labels(xrclustered.copy(), [2]))
    mask_cl = ~np.logical_or(mask_cl_n, mask_cl_s)
elif region =='US':
    mask_cl = find_precursors.view_or_replace_labels(xrclustered.copy(), [1,3])
    mask_cl = np.isnan(mask_cl)
elif region == 'init':
    mask_cl_e = find_precursors.view_or_replace_labels(xrclustered.copy(), [3])
    mask_cl_e  = make_country_mask.binary_erosion(~np.isnan(mask_cl_e) )
    mask_cl_w = ~np.isnan(find_precursors.view_or_replace_labels(xrclustered.copy(), [1]))
    mask_cl = ~np.logical_or(mask_cl_w, mask_cl_e)

fig = plot_maps.plot_labels(xrclustered,
                      {'scatter':scatter,
                       'zoomregion':selbox,
                       'mask_xr':mask_cl,
                       'x_ticks':np.arange(240, 300, 20),
                       'y_ticks':np.arange(0,61,10),
                       'add_cfeature':'LAKES'}) # np.isnan(mask_cl)
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
ds_t2m = core_pp.import_ds_lazy(var_filename, selbox=selbox)
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
point_corr.attrs.pop('units')

mask_cl_forcorr = xr.concat([mask_cl]*np_array_xy.shape[0], dim='points')
mask_cl_forcorr['points'] = ('points', list(df_ts.columns))
#%%




if region == 'USCA' or region == 'init':
    col_wrap = 4
    scatter =[[(0,0), [np_array_xy[[0]], {'s':size, 'zorder':2, 'color':colors[0], 'edgecolors':'black'}] ],
              [(0,1), [np_array_xy[[1]], {'s':size, 'zorder':2, 'color':colors[1], 'edgecolors':'black'}] ],
              [(0,2), [np_array_xy[[2]], {'s':size, 'zorder':2, 'color':colors[2], 'edgecolors':'black'}] ],
              [(0,3), [np_array_xy[[3]], {'s':size, 'zorder':2, 'color':colors[3], 'edgecolors':'black'}] ],
              [(1,0), [np_array_xy[[4]], {'s':size, 'zorder':2, 'color':colors[4], 'edgecolors':'black'}] ],
              [(1,1), [np_array_xy[[5]], {'s':size, 'zorder':2, 'color':colors[5], 'edgecolors':'black'}] ],
              [(1,2), [np_array_xy[[6]], {'s':size, 'zorder':2, 'color':colors[6], 'edgecolors':'black'}] ],
              [(1,3), [np_array_xy[[7]], {'s':size, 'zorder':2, 'color':colors[7], 'edgecolors':'black'}] ]]
    hspace=-.33 ;  cbar_vert=.08

elif region == 'US':
    col_wrap = 3
    scatter =[[(0,0), [np_array_xy[[0]], {'s':size, 'zorder':2, 'color':colors[0], 'edgecolors':'black'}] ],
              [(0,1), [np_array_xy[[1]], {'s':size, 'zorder':2, 'color':colors[1], 'edgecolors':'black'}] ],
              [(0,2), [np_array_xy[[2]], {'s':size, 'zorder':2, 'color':colors[2], 'edgecolors':'black'}] ],
              [(1,0), [np_array_xy[[3]], {'s':size, 'zorder':2, 'color':colors[3], 'edgecolors':'black'}] ],
              [(1,1), [np_array_xy[[4]], {'s':size, 'zorder':2, 'color':colors[4], 'edgecolors':'black'}] ],
              [(1,2), [np_array_xy[[5]], {'s':size, 'zorder':2, 'color':colors[5], 'edgecolors':'black'}] ]]
    hspace=-.5 ;  cbar_vert=.14


subtitles = np.array([point_corr.points]).reshape(-1, col_wrap)
# scatter = None
plot_maps.plot_corr_maps(point_corr,
                         mask_xr = mask_cl_forcorr,
                         col_dim='points',
                         aspect=1.5, hspace=hspace,
                         cbar_vert=cbar_vert,
                         subtitles=subtitles,
                         scatter=scatter,
                         col_wrap=col_wrap,
                         x_ticks=np.arange(240, 300, 20),
                         y_ticks=np.arange(0,61,10),
                         clevels=np.arange(-1,1.05,.1),
                         cmap=plt.cm.coolwarm,
                         zoomregion=selbox)
f_name = 'one_point_corr_maps_t2m_{}_{}'.format(xrclustered.attrs['hash'], region)
filepath = os.path.join(rg.path_outmain, f_name)
# plt.savefig(filepath+'.png', bbox_inches='tight', dpi=200)

#%% Plot all clustering results again
fig = plot_maps.plot_labels(ds['xrclusteredall'],
                            kwrgs_plot={'wspace':.03, 'hspace':-.35,
                                        'cbar_vert':.09,
                                        'row_dim':'n_clusters',
                                        'col_dim':'q',
                                        'x_ticks':np.arange(240, 300, 20),
                                        'y_ticks':np.arange(0,61,10)})
f_name = 'clustering_dendogram_{}'.format(xrclustered.attrs['hash']) + '.png'
path_fig = os.path.join(rg.path_outmain, f_name)
fig.savefig(path_fig,
            bbox_inches='tight', dpi=200) # dpi auto 600



#%%
if region != 'init':
    ds_cl_ts = core_pp.get_selbox(ds_cl['xrclusteredall'].sel(tfreq=t, n_clusters=c),
                                  selbox)
    ds_new = cl.spatial_mean_clusters(var_filename,
                                      ds_cl_ts,
                                      selbox=selbox)
    ds_new['xrclusteredall'] = xrclustered
    f_name = 'tf{}_nc{}'.format(int(t), int(c))
    filepath = os.path.join(rg.path_outmain, f_name)
    cl.store_netcdf(ds_new, filepath=filepath, append_hash='dendo_'+xrclustered.attrs['hash'])

    TVpath = filepath + '_' + 'dendo_'+xrclustered.attrs['hash'] + '.nc'
