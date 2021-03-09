#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 12:51:37 2021

@author: semvijverberg
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
user_dir = os.path.expanduser('~')
os.chdir(os.path.join(user_dir,
                      'surfdrive/Scripts/RGCPD/publications/paper_Raed/'))
curr_dir = os.path.join(user_dir, 'surfdrive/Scripts/RGCPD/RGCPD/')
main_dir = '/'.join(curr_dir.split('/')[:-2])
RGCPD_func = os.path.join(main_dir, 'RGCPD')
assert main_dir.split('/')[-1] == 'RGCPD', 'main dir is not RGCPD dir'
cluster_func = os.path.join(main_dir, 'clustering/')
fc_dir = os.path.join(main_dir, 'forecasting')

if cluster_func not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(RGCPD_func)
    sys.path.append(cluster_func)
    sys.path.append(fc_dir)

path_raw = user_dir + '/surfdrive/ERA5/input_raw'


# from RGCPD import RGCPD
from RGCPD import BivariateMI
import find_precursors

sst = BivariateMI('sst')
path = '/Users/semvijverberg/Desktop/cluster/surfdrive/output_paper3/USDA_Soy/random_20/s1'
sst.load_files(path, '8de750')


prec_labels = sst.prec_labels.copy()
corr_xr = sst.corr_xr.copy()

def get_detect_splits(xr_in, min_detect_gc=.5):

    n_splits = xr_in.split.size
    min_d = round(n_splits * (1- min_detect_gc),0)
    # 1 == non-significant, 0 == significant
    mask = (~np.isnan(xr_in)).sum(dim='split') > min_d
    xr_in = xr_in.mean(dim='split')
    if min_detect_gc<.1 or min_detect_gc>1.:
        raise ValueError( 'give value between .1 en 1.0')
    return xr_in.where(mask)

prec_labels = get_detect_splits(prec_labels) ; corr_xr = get_detect_splits(corr_xr)


#%% Plot SST corr maps
import plot_maps
subtitles = np.array([['Lag 1: March-April mean'], ['Lag 2: May-June mean'],
                       ['Lag 3: July-Aug mean'], ['Lag 4: Sep-Oct mean']])
kwrgs_plotcorr_sst = {'row_dim':'lag', 'col_dim':'split','aspect':4,
                      'hspace':.38, 'wspace':-.15, 'size':2, 'cbar_vert':0.055,
                      'map_proj':ccrs.PlateCarree(central_longitude=220),
                       # 'y_ticks':np.arange(25,56,10), 'x_ticks':np.arange(230, 295, 15),
                       'y_ticks':False, 'x_ticks':False,
                       'title':'', 'subtitles':subtitles,
                       'subtitle_fontdict':{'fontsize':30},
                       'clevels':np.arange(-.8,.9,.1),
                       'clabels':np.arange(-.8,.9,.4),
                       'cbar_tick_dict':{'labelsize':25},
                      'title_fontdict':{'fontsize':16, 'fontweight':'bold'}}

f = find_precursors.view_or_replace_labels
# mask_xr = np.isnan(f(prec_labels.copy(), regions=None))
mask_xr = np.isnan(prec_labels.copy())
fig = plot_maps.plot_corr_maps(corr_xr, mask_xr, **kwrgs_plotcorr_sst)
#%%

pacific_region = find_precursors.view_or_replace_labels(prec_labels, [1])
subtitles = np.array([['Lag 1'], ['Lag 2'], ['Lag 3'], ['Lag 4']])
kwrgs = {'row_dim':'lag', 'col_dim':'split', 'size':3, 'aspect':1, 'hspace':-.15,
          'y_ticks':False, 'x_ticks':False, 'subtitles':subtitles,
          'subtitle_fontdict':{'fontsize':30},
          'zoomregion':[150, 260, -10,60]}
f = plot_maps.plot_labels(pacific_region, kwrgs_plot=kwrgs)

for ax in f.axes[:3]:
    ax.background_patch.set_facecolor('white')
    ax.coastlines(color='white',
                  alpha=1,
                  facecolor='white',
                  linewidth=0)
    # whiteoutline subplot
    ax.spines['geo'].set_edgecolor('white')

plt.savefig(os.path.join("/Users/semvijverberg/surfdrive/output_paper3/extra_plots_paper/schematic_pacific.pdf"), facecolor=(1,1,1,0))
