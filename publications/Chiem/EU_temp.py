#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 16:49:11 2020

@author: semvijverberg
"""

import os, inspect, sys
import numpy as np
from time import time
import cartopy.crs as ccrs ; import matplotlib.pyplot as plt

user_dir = os.path.expanduser('~')
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-2])
assert main_dir.split('/')[-1] == 'RGCPD', 'main dir is not RGCPD dir'
RGCPD_func = os.path.join(main_dir, 'RGCPD')
cluster_func = os.path.join(main_dir, 'clustering/')
fc_dir = os.path.join(main_dir, 'forecasting')
if cluster_func not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(RGCPD_func)
    sys.path.append(cluster_func)
    sys.path.append(fc_dir)

path_raw = user_dir + '/surfdrive/ERA5/input_raw'


from RGCPD import RGCPD
from RGCPD import BivariateMI
import class_BivariateMI
import xarray as xr


if sys.platform == 'linux':
    import matplotlib as mpl
    mpl.use('Agg')

TVpath = '/Users/semvijverberg/surfdrive/Scripts/RGCPD/publications/Chiem/data/1D_series_daily_anom_NWeurope_mean.nc'
path_out_main = os.path.join(main_dir, 'publications/Chiem/output/')
path_data = os.path.join(main_dir, 'publications/Chiem/data/')
cluster_label = ''
name_ds='t2m-mean-anom'



start_end_date = ('1-1', '08-31')
start_end_TVdate = ('06-01', '08-31')
start_end_year = (1979, 2018)


freqs = [10,30]


#%% run RGPD
exper = 'corr'

if exper == 'corr':
    func = class_BivariateMI.corr_map
    kwrgs_func = {}
elif exper == 'parcorrtime':
    func = class_BivariateMI.parcorr_map_time
    kwrgs_func = {}

df = {}
for i, tfreq in enumerate(freqs):

    list_of_name_path = [(cluster_label, TVpath),
                         ('sst', os.path.join(path_raw, 'sst_1979-2018_1_12_daily_1.0deg.nc'))]


    list_for_MI   = [BivariateMI(name='sst', func=func,
                                alpha=.01, FDR_control=True,
                                kwrgs_func=kwrgs_func,
                                distance_eps=800, min_area_in_degrees2=3,
                                calc_ts='region mean', selbox=(-180,360,-65,90),
                                lags=np.array([10]), lag_as_gap=True)]



    rg = RGCPD(list_of_name_path=list_of_name_path,
               list_for_MI=list_for_MI,
               list_import_ts=None,
                start_end_TVdate=start_end_TVdate,
                start_end_date=start_end_date,
                start_end_year=start_end_year,
                tfreq=tfreq,
                path_outmain=path_out_main,
                append_pathsub='_' + name_ds)


    rg.pp_TV(name_ds=name_ds, detrend=True)
    rg.pp_precursors()

    rg.traintest(method='no_train_test_split')
    rg.calc_corr_maps()

    precur = rg.list_for_MI[0]
    #%%
    import matplotlib
    # Optionally set font to Computer Modern to avoid common missing font errors
    matplotlib.rc('font', family='serif', serif='cm10')

    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']

    save = True
    if exper == 'corr' and precur.lag_as_gap==False:
        title = r'$corr(t2m_t, SST_{t-lag})$'
    elif exper == 'corr' and precur.lag_as_gap:
            title = r'$corr(t2m_t, SST_{t_{gap}})$'
    elif exper == 'parcorrtime' and precur.lag_as_gap==False:
        title = r'$parcorr(t2m_t, SST_{t-lag}\ |\ t2m_{t-1},SST_{t-lag-1})$'
    elif exper == 'parcorrtime' and precur.lag_as_gap:
        title = r'$parcorr(t2m_t, SST_{t_{gap}}\ |\ t2m_{t-1},SST_{t_{gap}-1})$'

    lags = rg.list_for_MI[0].lags
    subtitles = np.array([[f'lag {l}' for l in lags]]) #, f'lag 2 (15 day lead)']] )
    kwrgs_plot = {'row_dim':'split', 'col_dim':'lag','aspect':2, 'hspace':0,
                  'wspace':0, 'size':4, 'cbar_vert':-.08,
                  'units':'Corr. Coeff. [-]',
                  'clim':(-.60,.60), 'map_proj':ccrs.PlateCarree(central_longitude=0),
                  'y_ticks':np.arange(-90,91,20),
                  'title':title, 'subtitles':subtitles,
                  'title_fontdict':{'fontsize':16, 'fontweight':'bold'}}
    rg.plot_maps_corr(var='sst', save=save,
                      kwrgs_plot=kwrgs_plot,
                      min_detect_gc=1.0,
                      append_str=f'{exper}_{tfreq}tf_gap{precur.lag_as_gap}')

    df[tfreq] = rg

#%%
import plot_maps;

corr = rg.list_for_MI[0].corr_xr.mean(dim='split').drop('time')
list_xr = [corr.expand_dims('timescale', axis=0) for i in range(len(freqs))]
corr = xr.concat(list_xr, dim = 'timescale')
corr['timescale'] = ('timescale', freqs)

np_data = np.zeros_like(corr.values)
np_mask = np.zeros_like(corr.values)
for i, f in enumerate(freqs):
    rg = df[f]
    vals = rg.list_for_MI[0].corr_xr.mean(dim='split').values
    np_data[i] = vals
    mask = rg.list_for_MI[0].corr_xr.mask.mean(dim='split')
    np_mask[i] = mask

corr.values = np_data
mask = (('timescale', 'lag', 'latitude', 'longitude'), np_mask )
corr.coords['mask'] = mask

kwrgs_plot = {'aspect':2, 'hspace':.3,
                  'wspace':.02, 'size':2, 'cbar_vert':0,
                  'units':'Corr. Coeff. [-]',
                  'clim':(-.60,.60), 'map_proj':ccrs.PlateCarree(central_longitude=0),
                  'y_ticks':np.arange(-90,91,20),
                  'title':title,
                  'title_fontdict':{'fontsize':16, 'fontweight':'bold'}}

if precur.lag_as_gap:
    corr = corr.rename({'lag':'gap'}) ; dim = 'gap'

fig = plot_maps.plot_corr_maps(corr, mask_xr=corr.mask, col_dim='timescale',
                               row_dim=corr.dims[1],
                               **kwrgs_plot)

f_name = 'corr_map_{}_a{}'.format(precur.name,
                                  precur.alpha) + '_' + \
                                  f'{exper}_gap{precur.lag_as_gap}'
fig_path = os.path.join(rg.path_outsub1, f_name)+rg.figext

plt.savefig(fig_path, bbox_inches='tight')

# rg.cluster_list_MI()
# rg.quick_view_labels(save=False)