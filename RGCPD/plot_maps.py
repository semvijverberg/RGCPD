#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 12:08:45 2019

@author: semvijverberg
"""
import os

import matplotlib.pyplot as plt
import functions_RGCPD as rgcpd
import cartopy.feature as cfeature
import itertools
import numpy as np
import xarray as xr

import cartopy.crs as ccrs
import pandas as pd

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
    
def plot_corr_maps(corr_xr, mask_xr, map_proj, row_dim='split',
                   col_dim='lag', clim='relaxed', hspace=-0.6, 
                   size=2.5, cbar_vert=0, units='units'):
    #%%
    import matplotlib.colors as colors

    if row_dim not in corr_xr.dims:
        corr_xr = corr_xr.expand_dims(row_dim, 0) 
        mask_xr = mask_xr.expand_dims(row_dim, 0) 
    elif col_dim not in corr_xr.dims:
        corr_xr = corr_xr.expand_dims(col_dim, 0) 
        mask_xr = mask_xr.expand_dims(col_dim, 0)     
    
    var_n   = corr_xr.name        
    rows    = corr_xr[row_dim].values
    cols    = corr_xr[col_dim].values


    rename_dims = {row_dim:'row', col_dim:'col'}
    rename_dims_inv = {'row':row_dim, 'col':col_dim}
    plot_xr = corr_xr.rename(rename_dims)
    plot_mask = mask_xr.rename(rename_dims)
    dim_coords = plot_mask.squeeze().dims
    dim_coords = [d for d in dim_coords if d not in ['latitude', 'longitude']]
    rename_subs = {d:rename_dims_inv[d] for d in dim_coords}
    
    lat = plot_xr.latitude
    lon = plot_xr.longitude
    zonal_width = abs(lon[-1] - lon[0]).values
    
    
    g = xr.plot.FacetGrid(plot_xr, col='col', row='row', subplot_kws={'projection': map_proj},
                      sharex=True, sharey=True,
                      aspect= (lon.size) / lat.size, size=size)
    figwidth = g.fig.get_figwidth() ; figheight = g.fig.get_figheight()
    
    # =============================================================================
    # Coordinate labels
    # =============================================================================
    import cartopy.mpl.ticker as cticker
    longitude_labels = np.linspace(np.min(lon), np.max(lon), 6, dtype=int)
    longitude_labels = np.array(sorted(list(set(np.round(longitude_labels, -1)))))
    latitude_labels = np.linspace(lat.min(), lat.max(), 4, dtype=int)
    latitude_labels = sorted(list(set(np.round(latitude_labels, -1))))    
    g.set_ticks(max_xticks=5, max_yticks=5, fontsize='large')
    g.set_xlabels(label=[str(el) for el in longitude_labels])
    

    g.fig.subplots_adjust(hspace=hspace)
        
    class MidpointNormalize(colors.Normalize):
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            colors.Normalize.__init__(self, vmin, vmax, clip)

        def __call__(self, value, clip=None):
            # I'm ignoring masked values and all kinds of edge cases to make a
            # simple example...
            x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
            return np.ma.masked_array(np.interp(value, x, y))

    if clim == 'relaxed':
        vmin_ = np.percentile(plot_xr, 1) ; vmax_ = np.percentile(plot_xr, 99)
    elif type(clim) == tuple:
        vmin_, vmax_ = clim
    else:
        vmin_ = plot_xr.min()-0.01 ; vmax_ = plot_xr.max()+0.01
            
    vmin = np.round(float(vmin_),decimals=2) ; vmax = np.round(float(vmax_),decimals=2)
    clevels = np.linspace(-max(abs(vmin),vmax),max(abs(vmin),vmax),17) # choose uneven number for # steps
    norm = MidpointNormalize(midpoint=0, vmin=clevels[0],vmax=clevels[-1])
    cmap = plt.cm.RdBu_r
    for col, c_label in enumerate(cols):
        xrdatavar = plot_xr.sel(col=c_label)
        xrmaskvar = plot_mask.sel(col=c_label)
        if abs(lon[-1] - 360) <= (lon[1] - lon[0]):
            xrdatavar = extend_longitude(xrdatavar)
            xrmaskvar = extend_longitude(xrmaskvar)
            
        
        for row, r_label in enumerate(rows):
            print(f"\rPlotting Corr maps {var_n}, {row_dim} {r_label}, {col_dim} {c_label}", end="")
            plotdata = xrdatavar.sel(row=r_label).rename(rename_subs)
#                                               
            plotmask = xrmaskvar.sel(row=r_label)#.rename(
#                                                {'row':row_dim, 'col':col_dim})
            # if plotdata is already masked (with nans):
            p_nans = int(100*plotdata.values[np.isnan(plotdata.values)].size / plotdata.size)
            # field not completely masked?
            if (plotmask.values==True).all() == False:
                if p_nans < 90:
                    plotmask.plot.contour(ax=g.axes[row,col], transform=ccrs.PlateCarree(),
                                          subplot_kws={'projection': map_proj}, colors=['black'],
                                          linewidths=np.round(zonal_width/150, 1)+0.3, levels=[float(vmin),float(vmax)],
                                          add_colorbar=False) #levels=[float(vmin),float(vmax)],
#                try:
#                    im = plotdata.plot.contourf(ax=g.axes[row,col], transform=ccrs.PlateCarree(),
#                                        center=0,
#                                         levels=clevels, cmap=cmap,
#                                         subplot_kws={'projection':map_proj},add_colorbar=False)
#                except ValueError:
#                    print('could not draw contourf, shifting to pcolormesh')
            im = plotdata.plot.pcolormesh(ax=g.axes[row,col], transform=ccrs.PlateCarree(),
                                    center=0,
                                     levels=clevels, cmap=cmap,
                                     subplot_kws={'projection':map_proj},add_colorbar=False)
            if np.nansum(plotdata.values) == 0.:
                g.axes[row,col].text(0.5, 0.5, 'No regions significant',
                      horizontalalignment='center', fontsize='x-large',
                      verticalalignment='center', transform=g.axes[row,col].transAxes)
            g.axes[row,col].set_extent([lon[0], lon[-1], 
                                       lat[0], lat[-1]], ccrs.PlateCarree())



            # =============================================================================
            # set coordinate ticks
            # =============================================================================          
#            rcParams['axes.titlesize'] = 'xx-large'

            if map_proj.proj4_params['proj'] in ['merc', 'eqc', 'cea']:
                ax = g.axes[row,col]
                ax.set_xticks(longitude_labels[:-1], crs=ccrs.PlateCarree())
                ax.set_xticklabels(longitude_labels[:-1], fontsize=12)
                lon_formatter = cticker.LongitudeFormatter()
                ax.xaxis.set_major_formatter(lon_formatter)
                
                
                g.axes[row,col].set_yticks(latitude_labels, crs=ccrs.PlateCarree())
                g.axes[row,col].set_yticklabels(latitude_labels, fontsize=12)
                lat_formatter = cticker.LatitudeFormatter()
                g.axes[row,col].yaxis.set_major_formatter(lat_formatter)
                g.axes[row,col].grid(linewidth=1, color='black', alpha=0.3, linestyle='--')
                g.axes[row,col].set_ylabel('')
                g.axes[row,col].set_xlabel('')
#            g.axes[row,col].coastlines(color='black', 
#                                          alpha=0.3, 
#                                          facecolor='grey', 
#                                          linewidth=2)
            if corr_xr.name[:3] == 'sst':
                g.axes[row,col].add_feature(cfeature.LAND, facecolor='grey', alpha=0.3)
#            if row == rows.size-1:
#                last_ax = g.axes[row,col]
    # lay out settings

    plt.tight_layout(pad=1.1-0.02*rows.size, h_pad=None, w_pad=None, rect=None)
   
    # height colorbor 1/10th of height of subfigure
    height = g.axes[-1,0].get_position().height / 10
    bottom_ysub = (figheight/40)/(rows.size*2) + cbar_vert
    
#    bottom_ysub = last_ax.get_position(original=False).bounds[1] # bottom
    
    cbar_ax = g.fig.add_axes([0.25, bottom_ysub, 
                              0.5, height]) #[left, bottom, width, height]

    if units == 'units' and 'units' in corr_xr.attrs:
        clabel = corr_xr.attrs['units']    
    elif units != 'units' and units is not None:
        clabel = units
    else:
        clabel = 'Corr Coefficient'

    plt.colorbar(im, cax=cbar_ax , orientation='horizontal', norm=norm,
                 label=clabel, ticks=clevels[::4], extend='neither')
    
    print("\n")

    #%%
    return

def causal_reg_to_xarray(ex, df, outdic_actors):
    #%%    
    '''
    Returns Dataset of merged variables, this aligns there coordinates (easy for plots)
    Returns list_ds to keep the original dimensions
    '''
#    outdic_actors_c = outdic_actors.copy()
    df_c = df.loc[ df['causal']==True ]
    # remove response variable if the ac is a 
    # causal link
    if df.index[0] in df_c.index:
        df_c = df_c.drop(df.index[0])
   
    spatial_vars = outdic_actors.keys()
    var_rel_sizes = {outdic_actors[var].area_grid.sum()/7939E6 : var for var in spatial_vars}
    var_large_to_small = [var_rel_sizes[s] for s in sorted(var_rel_sizes, reverse=True)]
    dict_ds = {}
    for i, var in enumerate(var_large_to_small):
        ds_var = xr.Dataset()
        regs_c = df_c.loc[ df_c['var'] == var ]
        actor = outdic_actors[var]
        label_tig = actor.prec_labels.copy()
        corr_tig = actor.corr_xr.copy()
        corr_xr  = actor.corr_xr.copy()
        for lag_cor in label_tig.lag.values:
            
            var_tig = label_tig.sel(lag=lag_cor)
            for lag_t in np.unique(regs_c['lag_tig']):
                reg_c_l = regs_c.loc[ regs_c['lag_tig'] == lag_t]
                labels = list(reg_c_l.label.values)
                
                new_mask = np.zeros( shape=var_tig.shape, dtype=bool)
                
                for l in labels:
                    new_mask[var_tig.values == l] = True

            wghts_splits = np.array(new_mask, dtype=int).sum(0)
            wghts_splits = wghts_splits / wghts_splits.max()
            label_tig.sel(lag=lag_cor).values[~new_mask] = np.nan
            corr_tig.sel(lag=lag_cor).values[~new_mask] = np.nan
            # Tig: apply weights and take mean over splits
            orig_corr_val = corr_tig.sel(lag=lag_cor).values
            corr_tig.sel(lag=lag_cor).values = orig_corr_val * wghts_splits
            # Corr: apply weights and take mean over splits
            orig_corr_val = corr_xr.sel(lag=lag_cor).values
            corr_xr.sel(lag=lag_cor).values = orig_corr_val * wghts_splits
                              
        ds_var[var+'_corr'] = corr_xr.mean(dim='split') 
        ds_var[var+'_corr_tigr'] = corr_tig.mean(dim='split') 
        ds_var[var+'_labels'] = actor.prec_labels.copy()
        ds_var[var+'_labels_tigr'] = label_tig.copy()    
        dict_ds[var] = ds_var
#    list_ds = [item for k,item in dict_ds.items()]
#    ds = xr.auto_combine(list_ds)


    #%%
    return dict_ds

def plotting_per_variable(dict_ds, df_sum, map_proj, ex):
    #%%
    # =============================================================================
    print('\nPlotting all fields significant at alpha_level_tig, while conditioning on parents'
          ' that were found in the PC step')
    # =============================================================================
    df_c = df_sum.loc[ df_sum['causal']==True ]
    # remove response variable if the ac is a 
    # causal link
    if df_sum.index[0] in df_c.index:
        df_c = df_c.drop(df_sum.index[0])
        
    variables = list(dict_ds.keys())
    lags = ds = dict_ds[variables[0]].lag.values
    for lag in lags:
        
        for i, var in enumerate(variables):
            ds = dict_ds[var]
            
            plot_labels(ds, df_c, var, lag, ex)
            
            plot_corr_regions(ds, df_c, var, lag, map_proj, ex)
    #%%
def plot_labels(ds, df_c, var, lag, ex):
    
    ds_l = ds.sel(lag=lag)
    list_xr = [] ; name = []
    for c in ['labels', 'labels_tigr']:
        name.append(var+'_'+c)
        list_xr.append(ds_l[var+'_'+c])
    for_plt = xr.concat(list_xr, dim='lag')
    for_plt.lag.values = name
    if np.isnan(for_plt.values).all() ==False:
        rgcpd.plot_regs_xarray(for_plt, ex)
        plt.show() ; plt.close()
    else:
        print(f'No significant regions for {var}')

def plot_corr_regions(ds, df_c, var, lag, map_proj, ex):
    #%%    
    ds_l = ds.sel(lag=lag)
    list_xr = [] ; name = []
    list_xr_m = []
    for c in [['corr', 'labels'],['corr_tigr', 'labels_tigr']]:
        name.append(var+'_'+c[0])
        list_xr.append(ds_l[var+'_'+c[0]])
        mask = ds_l[var+'_'+c[1]].sum(dim='split').astype('bool')
        list_xr_m.append(mask)
        
    xrdata = xr.concat(list_xr, dim='lag')
    xrdata.lag.values = name
    xrdata.name = 'sst_corr_and_tigr'
#    xrdata = xrdata.expand_dims('variable', axis=0)
#    xrdata.assign_coords(variable=[var])
    xrmask = xr.concat(list_xr_m, dim='lag')
    xrmask.lag.values = name
#    xrmask.values = ~np.isnan(xrdata.values)
    if np.isnan(xrdata.values).all() == False:
        plot_corr_maps(xrdata, xrmask, map_proj)
        fig_filename = '{}_tigr_corr_{}_vs_{}'.format(ex['params'], ex['RV_name'], var) + ex['file_type2']
        plt.savefig(os.path.join(ex['fig_path'], fig_filename), bbox_inches='tight', dpi=ex['png_dpi'])
        plt.show() ; plt.close()
    #%%