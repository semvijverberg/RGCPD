#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 12:08:45 2019

@author: semvijverberg
"""
import os

import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import itertools
import numpy as np
import xarray as xr
import pandas as pd
# from matplotlib.colors import LinearSegmentedColormap, colors
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from typing import List, Tuple, Union

flatten = lambda l: list(itertools.chain.from_iterable(l))

# =============================================================================
# 2 Plotting functions for correlation maps
# and causal region maps
# =============================================================================



def extend_longitude(data):
    import xarray as xr
    import numpy as np
    plottable = xr.concat([data, data.sel(longitude=data.longitude[:1])], dim='longitude').to_dataset(name="ds")
    plottable["longitude"] = np.linspace(0,360, len(plottable.longitude))
    plottable = plottable.to_array(dim='ds').squeeze(dim='ds').drop('ds')
    return plottable

def plot_corr_maps(corr_xr, mask_xr=None, map_proj=None, row_dim='split',
                   col_dim='lag', clim='relaxed', hspace=-0.6, wspace=0.02,
                   size=2.5, cbar_vert=-0.01, units='units', cmap=None,
                   clevels=None, clabels=None, cticks_center=None,
                   cbar_tick_dict: dict={}, drawbox=None, title=None,
                   title_fontdict: dict=None, subtitles: np.ndarray=None,
                   subtitle_fontdict: dict=None, zoomregion=None,
                   aspect=None, n_xticks=5, n_yticks=3, x_ticks: Union[bool, np.ndarray]=None,
                   y_ticks: Union[bool, np.ndarray]=None, add_cfeature: str=None,
                   textinmap: list=None):

    '''
    zoomregion = tuple(east_lon, west_lon, south_lat, north_lat)
    '''
    #%%
    # default parameters
    # mask_xr=None ; row_dim='split'; col_dim='lag'; clim='relaxed';
    # size=2.5; cbar_vert=-0.01; units='units'; cmap=None; hspace=-0.6;
    # clevels=None; clabels=None; cticks_center=None; map_proj=None ; wspace=.03;
    # drawbox=None; subtitles=None; title=None; lat_labels=True; zoomregion=None
    # aspect=None; n_xticks=5; n_yticks=3; title_fontdict=None; x_ticks=None;
    # y_ticks=None; add_cfeature=None; textinmap=None

    if map_proj is None:
        cen_lon = int(corr_xr.longitude.mean().values)
        map_proj = ccrs.LambertCylindrical(central_longitude=cen_lon)

    if row_dim not in corr_xr.dims:
        corr_xr = corr_xr.expand_dims(row_dim, 0)
        if mask_xr is not None and row_dim not in mask_xr.dims:
            mask_xr = mask_xr.expand_dims(row_dim, 0)
    if col_dim not in corr_xr.dims:
        corr_xr = corr_xr.expand_dims(col_dim, 0)
        if mask_xr is not None and col_dim not in mask_xr.dims:
            mask_xr = mask_xr.expand_dims(col_dim, 0)

    var_n   = corr_xr.name
    rows    = corr_xr[row_dim].values
    cols    = corr_xr[col_dim].values


    rename_dims = {row_dim:'row', col_dim:'col'}
    rename_dims_inv = {'row':row_dim, 'col':col_dim}
    plot_xr = corr_xr.rename(rename_dims)
    if mask_xr is not None:
        plot_mask = mask_xr.rename(rename_dims)
    dim_coords = plot_xr.squeeze().dims
    dim_coords = [d for d in dim_coords if d not in ['latitude', 'longitude']]
    rename_subs = {d:rename_dims_inv[d] for d in dim_coords}

    lat = plot_xr.latitude
    lon = plot_xr.longitude
    zonal_width = abs(lon[-1] - lon[0]).values
    if aspect is None:
        aspect = (lon.size) / lat.size


    g = xr.plot.FacetGrid(plot_xr, col='col', row='row',
                          subplot_kws={'projection': map_proj},
                          sharex=True, sharey=True,
                          aspect=aspect, size=size)
    figheight = g.fig.get_figheight()

    # =============================================================================
    # Coordinate labels
    # =============================================================================
    import cartopy.mpl.ticker as cticker
    g.set_ticks(fontsize='large')
    if x_ticks is None or x_ticks is False: #auto-ticks, if False, will be masked
        longitude_labels = np.linspace(np.min(lon), np.max(lon), n_xticks, dtype=int)
        longitude_labels = np.array(sorted(list(set(np.round(longitude_labels, -1)))))
    else:
        longitude_labels = x_ticks # if x_ticks==False -> no ticklabels
    if y_ticks is None or y_ticks is False: #auto-ticks, if False, will be masked
        latitude_labels = np.linspace(lat.min(), lat.max(), n_yticks, dtype=int)
        latitude_labels = sorted(list(set(np.round(latitude_labels, -1))))
    else:
        latitude_labels = y_ticks # if y_ticks==False -> no ticklabels

    g.fig.subplots_adjust(hspace=hspace, wspace=wspace)

    if clevels is None:
        class MidpointNormalize(mcolors.Normalize):
            def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
                self.midpoint = midpoint
                mcolors.Normalize.__init__(self, vmin, vmax, clip)

            def __call__(self, value, clip=None):
                # I'm ignoring masked values and all kinds of edge cases to make a
                # simple example...
                x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
                return np.ma.masked_array(np.interp(value, x, y))

        if clim == 'relaxed':
            vmin_ = np.nanpercentile(plot_xr, 1) ; vmax_ = np.nanpercentile(plot_xr, 99)
        elif type(clim) == tuple:
            vmin_, vmax_ = clim
        else:
            vmin_ = plot_xr.min()-0.01 ; vmax_ = plot_xr.max()+0.01

        vmin = np.round(float(vmin_),decimals=2) ; vmax = np.round(float(vmax_),decimals=2)
        clevels = np.linspace(-max(abs(vmin),vmax),max(abs(vmin),vmax),17) # choose uneven number for # steps

    else:
        vmin_ = np.nanpercentile(plot_xr, 1) ; vmax_ = np.nanpercentile(plot_xr, 99)
        vmin = np.round(float(vmin_),decimals=2) ; vmax = np.round(float(vmax_),decimals=2)
        clevels=clevels


    if cmap is None:
        cmap = plt.cm.RdBu_r
    else:
        cmap=cmap

    for col, c_label in enumerate(cols):
        xrdatavar = plot_xr.sel(col=c_label)
        dlon = abs(lon[1] - lon[0])
        if abs(lon[-1] - 360) <= dlon and lon[0] < dlon:
            xrdatavar = extend_longitude(xrdatavar)


        for row, r_label in enumerate(rows):
            print(f"\rPlotting Corr maps {var_n}, {row_dim} {r_label}, {col_dim} {c_label}", end="\n")
            plotdata = xrdatavar.sel(row=r_label).rename(rename_subs).squeeze()

            if mask_xr is not None:
                xrmaskvar = plot_mask.sel(col=c_label)
                if abs(lon[-1] - 360) <= (lon[1] - lon[0]) and lon[0]==0:
                    xrmaskvar = extend_longitude(xrmaskvar)
                plotmask = xrmaskvar.sel(row=r_label)

            # if plotdata is already masked (with nans):
            p_nans = int(100*plotdata.values[np.isnan(plotdata.values)].size / plotdata.size)

            if mask_xr is not None:
                # field not completely masked?
                all_masked = (plotmask.values==False).all()
                if all_masked == False:
                    if p_nans != 100:
                        plotmask.plot.contour(ax=g.axes[row,col],
                                              transform=ccrs.PlateCarree(),
                                          # subplot_kws={'projection': map_proj},
                                          colors=['black'],
                                          linewidths=np.round(zonal_width/150, 1)+0.3,
                                          levels=[float(vmin),float(vmax)],
                                          add_colorbar=False)
        #                try:
        #                    im = plotdata.plot.contourf(ax=g.axes[row,col], transform=ccrs.PlateCarree(),
        #                                        center=0,
        #                                         levels=clevels, cmap=cmap,
        #                                         subplot_kws={'projection':map_proj},add_colorbar=False)
        #                except ValueError:
        #                    print('could not draw contourf, shifting to pcolormesh')

            # if no signifcant regions, still plot corr values, but the causal plot must remain empty
            if mask_xr is None or all_masked==False or (all_masked and 'tigr' not in str(c_label)):
                im = plotdata.plot.pcolormesh(ax=g.axes[row,col], transform=ccrs.PlateCarree(),
                                              center=0, levels=clevels,
                                              cmap=cmap,add_colorbar=False)
                                              # subplot_kws={'projection':map_proj})
            elif all_masked and 'tigr' in c_label:
                g.axes[row,col].text(0.5, 0.5, 'No regions significant',
                      horizontalalignment='center', fontsize='x-large',
                      verticalalignment='center', transform=g.axes[row,col].transAxes)
            # =============================================================================
            # Draw (rectangular) box
            # =============================================================================
            if drawbox is not None:
                from shapely.geometry.polygon import LinearRing
                def get_ring(coords):
                    '''tuple in format: west_lon, east_lon, south_lat, north_lat '''
                    west_lon, east_lon, south_lat, north_lat = coords
                    lons_sq = [west_lon, west_lon, east_lon, east_lon]
                    lats_sq = [north_lat, south_lat, south_lat, north_lat]
                    ring = [LinearRing(list(zip(lons_sq , lats_sq )))]
                    return ring
                if isinstance(drawbox[1], tuple):
                    ring = get_ring(drawbox[1])
                elif isinstance(drawbox[1], list):
                    ring = drawbox[1]

                if drawbox[0] == g.axes.size or drawbox[0] == 'all':
                    g.axes[row,col].add_geometries(ring, ccrs.PlateCarree(),
                                                   facecolor='none', edgecolor='green',
                                                   linewidth=2, linestyle='dashed')
                elif type(drawbox[0]) is tuple:
                    row_box, col_box = drawbox[0]
                    if row == row_box and col == col_box:
                        g.axes[row,col].add_geometries(ring, ccrs.PlateCarree(),
                                                       facecolor='none', edgecolor='green',
                                                       linewidth=2, linestyle='dashed')
            # =============================================================================
            # Add text in plot - list([location, list(tuple(lon,lat,text,kwrgs))])
            # =============================================================================
            if textinmap is not None:
                for list_t in textinmap:
                    if list_t[0] == g.axes.size or list_t[0] == 'all':
                        row_text, col_text = row, col
                    if type(list_t[0]) is tuple:
                        row_text, col_text = list_t[0]
                    if type(list_t[1]) is not list:
                        list_t[1] = [list_t[1]]
                    for t in list_t[1]:
                        lontext, lattext, text, kwrgs = t # lon in degrees west-east
                        kwrgs.update(dict(horizontalalignment='center',
                                         transform=ccrs.Geodetic())) # standard settings
                        g.axes[row_text,col_text].text(int(lontext), int(lattext),
                                                       text, **kwrgs)


            # =============================================================================
            # Subtitles
            # =============================================================================
            if subtitles is not None:
                if subtitle_fontdict is None:
                    subtitle_fontdict = dict({'fontsize' : 16})
                g.axes[row,col].set_title(subtitles[row,col], fontdict=subtitle_fontdict,
                                          loc='center')
            # =============================================================================
            # Format coordinate ticks
            # =============================================================================
            if map_proj.proj4_params['proj'] in ['merc', 'eqc', 'cea']:
                ax = g.axes[row,col]
                # x-ticks and labels
                ax.set_xticks(longitude_labels[:], crs=ccrs.PlateCarree())
                if x_ticks is not False:
                    ax.set_xticklabels(longitude_labels[:], fontsize=12)
                    lon_formatter = cticker.LongitudeFormatter()
                    ax.xaxis.set_major_formatter(lon_formatter)
                else:
                    fake_labels = [' ' * len( str(l) ) for l in longitude_labels]
                    g.axes[row,col].set_xticklabels(fake_labels, fontsize=12)
                # y-ticks and labels
                g.axes[row,col].set_yticks(latitude_labels, crs=ccrs.PlateCarree())
                if y_ticks is not False:
                    g.axes[row,col].set_yticklabels(latitude_labels, fontsize=12)
                    lat_formatter = cticker.LatitudeFormatter()
                    g.axes[row,col].yaxis.set_major_formatter(lat_formatter)
                else:
                    fake_labels = [' ' * len( str(l) ) for l in latitude_labels]
                    g.axes[row,col].set_yticklabels(fake_labels, fontsize=12)

                if np.logical_and(y_ticks==False, x_ticks==False)==False:
                    # if no ticks, then also no gridlines
                    g.axes[row,col].grid(linewidth=1, color='black', alpha=0.3, linestyle='--')
                g.axes[row,col].set_ylabel('')
                g.axes[row,col].set_xlabel('')
            g.axes[row,col].coastlines(color='black',
                                          alpha=0.3,
                                          facecolor='grey',
                                          linewidth=2)
            # black outline subplot
            g.axes[row,col].spines['geo'].set_edgecolor('black')


            if corr_xr.name is not None:
                if corr_xr.name[:3] == 'sst':
                    g.axes[row,col].add_feature(cfeature.LAND, facecolor='grey', alpha=0.3)
            if add_cfeature is not None:
                g.axes[row,col].add_feature(cfeature.__dict__[add_cfeature],
                                            facecolor='grey', alpha=0.2)


            if zoomregion is not None:
                g.axes[row,col].set_extent(zoomregion, crs=ccrs.PlateCarree())
            else:
                g.axes[row,col].set_extent([lon[0], lon[-1],
                                       lat[0], lat[-1]], crs=ccrs.PlateCarree())




    # =============================================================================
    # lay out settings FacetGrid and colorbar
    # =============================================================================

    # height colorbor 1/10th of height of subfigure
    height = g.axes[-1,0].get_position().height / 10
    bottom_ysub = (figheight/40)/(rows.size*2) + cbar_vert
    cbar_ax = g.fig.add_axes([0.25, bottom_ysub,
                              0.5, height]) #[left, bottom, width, height]

    if units == 'units' and 'units' in corr_xr.attrs:
        clabel = corr_xr.attrs['units']
    elif units != 'units' and units is not None:
        clabel = units
    else:
        clabel = ''

    if cticks_center is None:
        if clabels is None:
            clabels = clevels[::2]
        plt.colorbar(im, cax=cbar_ax , orientation='horizontal', # norm=norm,
                 label=clabel, ticks=clabels, extend='neither')
    else:
        cbar = plt.colorbar(im, cbar_ax,
                            orientation='horizontal', extend='neither',
                            label=clabel)
        cbar.set_ticks(clevels + 0.5)
        cbar.set_ticklabels(np.array(clevels+1, dtype=int),
                            update_ticks=True)
        cbar.update_ticks()
    cbar_ax.tick_params(**cbar_tick_dict)

    if title is not None:
        if title_fontdict is None:
            title_fontdict = dict({'fontsize'     : 18,
                                   'fontweight'   : 'bold'})
        g.fig.suptitle(title, **title_fontdict)
    # plt.tight_layout(pad=1.1-0.02*rows.size, h_pad=None, w_pad=None, rect=None)

    # print("\n")


    #%%
    return g.fig

def causal_reg_to_xarray(df_links, list_MI):
    #%%
    '''
    Returns Dataset of merged variables, this aligns there coordinates (easy for plots)
    Returns list_ds to keep the original dimensions
    '''

    # ensure list_MI only contains the processed MI
    list_MI = [p for p in list_MI if (hasattr(p, 'prec_labels'))]
    splits = df_links.index.levels[0]

    df_c = df_links.sum(axis=1) >= 1
    # only MI vars:
    var_MI = set()
    for s in splits:
        var_MI.update([i for i in df_c.loc[s].index if '..' in i])

    df_c = df_c.loc[:,var_MI]

    # collect var en region labels
    var = pd.Series([i[1].split('..')[-1] for i in df_c.index],
                    index=df_c.index)
    region_number = pd.Series([int(i[1].split('..')[-2]) for i in df_c.index],
                    index=df_c.index)
    df_c = pd.concat([df_c, var, region_number],
              keys=['C.D.', 'var', 'region_number'], axis=1)


    var_rel_sizes = {i:precur.area_grid.sum() for i,precur in enumerate(list_MI)}
    sorted_sizes = sorted(var_rel_sizes.items(), key=lambda kv: kv[1], reverse=False)
    var_large_to_small = [s[0] for s in sorted_sizes]



    def apply_new_mask(new_mask, label_tig, corr_xr, corr_tig):
        # wghts_splits = np.array(new_mask, dtype=int).sum(0)
        # wghts_splits = wghts_splits / wghts_splits.max()
        label_tig.sel(lag=lag_cor).values[~new_mask] = np.nan
        corr_tig.sel(lag=lag_cor).values[~new_mask] = np.nan
        # Old! 11-11-19 - Tig: apply weights and take mean over splits
        # Tig: no longer applying weights and take mean over splits
        orig_corr_val = corr_tig.sel(lag=lag_cor).values
        corr_tig.sel(lag=lag_cor).values = orig_corr_val

        orig_corr_val = corr_xr.sel(lag=lag_cor).values
        corr_xr.sel(lag=lag_cor).values = orig_corr_val
        return label_tig, corr_xr, corr_tig

    dict_ds = {}
    for idx in var_large_to_small:
        precur = list_MI[idx]
        var = precur.name
        ds_var = xr.Dataset()
        regs_c = df_c.loc[ df_c['var'] == var ].copy()

        label_tig = precur.prec_labels.copy()
        # if show spatcov of var was used: convert all labels to one
        if regs_c.size==0:
            regs_c = df_c.loc[ df_c['var'] == var+'_sp' ].copy()
            if regs_c.index.size <= splits.size:
                # only spatcov is available:
                label_tig = label_tig.where(np.isnan(label_tig), other=1.)
                regs_c['region_number'] += 1
        corr_tig = precur.corr_xr.copy()
        corr_xr  = precur.corr_xr.copy()
        if regs_c.size != 0:
            # if causal regions exist:
            for lag_cor in label_tig.lag.values:

                var_tig = label_tig.sel(lag=lag_cor)

                reg_cd = regs_c[regs_c['C.D.']]



                new_mask = np.zeros( shape=var_tig.shape, dtype=bool)
                for s in splits.values:
                    try:
                        labels = list(reg_cd.loc[s].region_number.values)
                    except:
                        labels = []
                    for l in labels:
                        new_mask[s][var_tig[s].values == l] = True

                out = apply_new_mask(new_mask, label_tig, corr_xr, corr_tig)
                label_tig, corr_xr, corr_tig = out
        else:
            for lag_cor in label_tig.lag.values:
                var_tig = label_tig.sel(lag=lag_cor)
                new_mask = np.zeros( shape=var_tig.shape, dtype=bool)
                out = apply_new_mask(new_mask, label_tig, corr_xr, corr_tig)
                label_tig, corr_xr, corr_tig = out

        ds_var[var+'_corr'] = corr_xr
        ds_var[var+'_corr_tigr'] = corr_tig
        ds_var[var+'_labels'] = precur.prec_labels.copy()
        ds_var[var+'_labels_tigr'] = label_tig.copy()
        dict_ds[var] = ds_var

#    list_ds = [item for k,item in dict_ds.items()]
#    ds = xr.auto_combine(list_ds)

    #%%
    return dict_ds

def plot_labels_vars_splits(dict_ds, df_links, figpath, paramsstr, RV_name,
                            save: bool=False, filetype='.pdf', mean_splits=True,
                            cols: List=['corr', 'C.D.'], kwrgs_plot={}):

    #%%
    # =============================================================================
    print('\nPlotting all fields significant at alpha_level_tig, while conditioning on parents'
          ' that were found in the PC step')
    # =============================================================================

    variables = list(dict_ds.keys())
    lags = ds = dict_ds[variables[0]].lag.values
    for lag in lags:

        for i, var in enumerate(variables):
            ds = dict_ds[var]

            if mean_splits == True:
                f_name = '{}_{}_vs_{}_labels_mean'.format(paramsstr, RV_name, var) + filetype
            else:
                f_name = '{}_{}_vs_{}_labels'.format(paramsstr, RV_name, var) + filetype
            if save:
                filepath = os.path.join(figpath, f_name)
            else:
                filepath  = False
            plot_labels_RGCPD(ds, var, lag, filepath,
                              mean_splits, cols, kwrgs_plot)
    #%%
    return

def plot_labels_RGCPD(ds, var, lag, filepath,
                      mean_splits=True, cols: List=['corr', 'C.D.'],
                      kwrgs_plot={}):
    #%%
    ds_l = ds.sel(lag=lag)
    splits = ds.split
    list_xr = [] ; name = []
    if cols == ['corr','C.D.']:
        columns = ['labels', 'labels_tigr']
        subtitles_l = [[f'{var} region labels', f'{var} regions C.D.']]
        subtitles_r = [[f'robustness {var} corr.', f'robustness {var} C.D.']]
    elif cols == ['corr']:
        columns = ['labels']
        subtitles_l = [[f'{var} region labels']]
        subtitles_r = [[f'robustness {var} corr.']]
    elif cols == ['C.D.']:
        columns = columns = ['labels_tigr']
        subtitles_l = [[f'{var} regions C.D.']]
        subtitles_r = [[f'robustness {var} C.D.']]


#    columns = ['labels']
    robustness_l = []
    if mean_splits == True:
        for c in columns:
            name.append(var+'_'+c)
            robustness = (ds_l[var+'_'+c] > 0).astype('int').sum(dim='split')
            robustness_l.append(robustness)
            wgts_splits = robustness / splits.size
            mask = (wgts_splits > 0.5).astype('bool')
#            prec_labels = ds_l[var+'_'+c].mean(dim='split') # changed plotting labels 23-01-201=20
            # fill all nans with label that was present in one of the splits
            # mean labels
            xr_labels = ds_l[var+'_'+c]
            squeeze_labels = xr_labels.sel(split=0)
            labels = np.zeros_like(squeeze_labels)
            for s in xr_labels.split:
                onesplit = xr_labels.sel(split=s)
                nonanmask = ~np.isnan(onesplit).values
                labels[nonanmask] = onesplit.values[nonanmask]
            squeeze_labels.values = labels
            squeeze_labels = squeeze_labels.where(labels!=0)
            list_xr.append(squeeze_labels.where(mask))
    else:
        for c in columns:
            name.append(var+'_'+c)
            prec_labels = ds_l[var+'_'+c]
            list_xr.append(prec_labels)


    prec_labels = xr.concat(list_xr, dim='lag')
    prec_labels = prec_labels.assign_coords(lag=name)

    # colors of cmap are dived over min to max in n_steps.
    # We need to make sure that the maximum value in all dimensions will be
    # used for each plot (otherwise it assign inconsistent colors)

    kwrgs_labels = _get_kwrgs_labels(prec_labels)
    kwrgs_labels['subtitles'] = np.array(subtitles_l)

    if mean_splits == True:
        kwrgs_labels['cbar_vert'] = -0.1
    else:
        kwrgs_labels['cbar_vert'] = -0.025

    kwrgs_labels.update(kwrgs_plot) # add and overwrite manual kwrgs
    if np.isnan(prec_labels.values).all() == False:

        plot_corr_maps(prec_labels,
                       **kwrgs_labels)
        plt.savefig(filepath, bbox_inches='tight')
        plt.show() ; plt.close()

        if mean_splits == True:
            # plot robustness


            colors = plt.cm.magma_r(np.linspace(0,0.7, 20))
            colors[-1] = plt.cm.magma_r(np.linspace(0.99,1, 1))
            cm = mcolors.LinearSegmentedColormap.from_list('test', colors, N=255)
            clevels = np.linspace(splits.min().values, splits.max().values+1,
                                  splits.max().values+2)
            units = 'No. of times significant [0 ... {}]'.format(splits.size)
            kwrgs_rob = {'clevels':clevels,
                         'cmap':cm,
                         'subtitles':np.array(subtitles_r),
                         'cticks_center':None,
                         'units' : units}
            for key, item in kwrgs_rob.items():
                kwrgs_labels[key] = item
            # kwrgs_labels['cbar_vert'] += .05
            robust = xr.concat(robustness_l, dim='lag')
            robust = robust.assign_coords(lag=name)
            # robust.lag.values = subtitles_r[0]
            robust = robust.where(robust.values != 0.)
            plot_corr_maps(robust-1E-9,
                           **kwrgs_labels)
            f_name = f'robustness_{var}_lag{lag}.' + filepath.split('.')[-1]
            fig_path = '/'.join(filepath.split('/')[:-1])
            plt.savefig(os.path.join(fig_path, f_name), bbox_inches='tight')
            plt.show() ; plt.close()



    else:
        print(f'No significant regions for {var}')
    #%%
    return

def plot_corr_vars_splits(dict_ds, df_sum, figpath, paramsstr, RV_name,
                          save: bool=False, filetype='.pdf', mean_splits=True,
                          cols: List=['corr', 'C.D.'], kwrgs_plot={}):
    #%%
    # =============================================================================
    print('\nPlotting all fields significant at alpha_level_tig, while conditioning on parents'
          ' that were found in the PC step')
    # # =============================================================================
    # df_c = df_sum.loc[ df_sum['causal']==True ]
    # # There can be duplicates because tigramite extract the same region at
    # # different lags, this is not important for plotting the spatial regions.
    # df_c = df_c.drop_duplicates()
    # # remove response variable if the ac is a causal link
    # splits = df_sum.index.levels[0]
    # for s in splits:
    #     try:
    #         # try because can be no causal regions in split s
    #         if RV_name in df_c.loc[s].index:
    #             df_c = df_c.drop((s, RV_name), axis=0)
    #     except:
    #         pass

    variables = list(dict_ds.keys())
    lags = ds = dict_ds[variables[0]].lag.values
    for lag in lags:

        for i, var in enumerate(variables):
            ds = dict_ds[var]

            if mean_splits == True:
                f_name = '{}_{}_vs_{}_tigr_corr_mean'.format(paramsstr, RV_name, var) + filetype
            else:
                f_name = '{}_{}_vs_{}_tigr_corr'.format(paramsstr, RV_name, var) + filetype
            filepath = os.path.join(figpath, f_name)
            plot_corr_regions(ds, var, lag, filepath,
                              mean_splits, cols, kwrgs_plot)
    #%%
    return

def _get_kwrgs_labels(prec_labels):
    if np.isnan(prec_labels.values).all() == False:
        max_N_regs = min(20, int(prec_labels.max() + 0.5))
    else:
        max_N_regs = 20
    label_weak = np.nan_to_num(prec_labels.values) >=  max_N_regs

    prec_labels.values[label_weak] = max_N_regs
    steps = max_N_regs+1
    cmap = plt.cm.tab20
    prec_labels.values = prec_labels.values-0.5
    clevels = np.linspace(0, max_N_regs,steps)


    kwrgs_labels = {'size':3, 'clevels':clevels,
                  'cticks_center':True,
                  'cmap':cmap,
                  'units': None}

    if len(prec_labels.shape) == 2 or prec_labels.shape[0] == 1:
        kwrgs_labels['cbar_vert'] = -0.1

    return kwrgs_labels

def plot_labels(prec_labels,
                kwrgs_plot={}):
    xrlabels = prec_labels.copy()
    xrlabels.values = prec_labels.values - 0.5
    kwrgs_labels = _get_kwrgs_labels(xrlabels)
    kwrgs_labels.update(kwrgs_plot)
    return plot_corr_maps(xrlabels, **kwrgs_labels)

def plot_corr_regions(ds, var, lag, filepath,
                      mean_splits=True, cols: List=['corr','C.D.'],
                      kwrgs_plot={}):
    #%%
    ds_l = ds.sel(lag=lag)
    splits = ds.split
    list_xr = [] ; name = []
    list_xr_m = []
    if cols == ['corr','C.D.']:
        columns = [['corr', 'labels'],['corr_tigr', 'labels_tigr']]
        subtitles = np.array([[f'{var} correlated', f'{var} C.D.']])
    elif cols == ['corr']:
        columns = [['corr', 'labels']]
        subtitles = np.array([[f'{var} correlated']])
    elif cols == ['C.D.']:
        columns = [['corr_tigr', 'labels_tigr']]
        subtitles = np.array([[f'{var} C.D.']])

    if mean_splits == True:
        for c in columns:
            name.append(var+'_'+c[0])
            mask_splits = (ds_l[var+'_'+c[1]] > 0).astype('int')
            wgts_splits = mask_splits.sum(dim='split') / splits.size
            mask = (wgts_splits > 0.5).astype('bool')
            corr_splits = ds_l[var+'_'+c[0]]
            corr_mean = corr_splits.mean(dim='split')
            if all(mask.values.flatten()==False) and c[0] == 'corr':
                # if no regions significant in corr map step:
                # do not mask
                corr_mean = corr_mean
            else:
                corr_mean = corr_mean.where(mask)
            list_xr.append(corr_mean)
            list_xr_m.append(mask)
    else:
        for c in columns:
            name.append(var+'_'+c[0])
            mask = (ds_l[var+'_'+c[1]] > 0).astype('bool')
            corr_splits = ds_l[var+'_'+c[0]]
            list_xr.append(corr_splits.where(mask))
            list_xr_m.append(mask)


    corr_xr = xr.concat(list_xr, dim='lag')
    corr_xr = corr_xr.assign_coords(lag=name)
    corr_xr.name = 'sst_corr_and_tigr'

    mask_xr = xr.concat(list_xr_m, dim='lag')
    mask_xr = mask_xr.assign_coords(lag=name)

    if mean_splits:
        cbar_vert = -0.1
        subtitles = subtitles
    else:
        cbar_vert = -0.01
        subtitles = None

    if np.isnan(corr_xr.values).all() == False:
#        kwrgs = {'cbar_vert':-0.05, 'subtitles':np.array([['Soil Moisture']])}

        kwrgs = {'cbar_vert':cbar_vert, 'subtitles':subtitles,
                 'units':'Corr Coefficient'}
        kwrgs.update(kwrgs_plot) # add and overwrite manual kwrgs
        plot_corr_maps(corr_xr, mask_xr, **kwrgs)

        plt.savefig(filepath, bbox_inches='tight')
        plt.show() ; plt.close()
    #%%
