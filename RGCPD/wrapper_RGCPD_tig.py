# -*- coding: utf-8 -*-
#%%

import sys, os, io


from netCDF4 import Dataset
import matplotlib.pyplot as plt
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb
import functions_RGCPD as rgcpd
import itertools
import numpy as np
import xarray as xr
import datetime
import cartopy.crs as ccrs
import pandas as pd
import functions_pp
flatten = lambda l: list(itertools.chain.from_iterable(l))

#%%
def calculate_corr_maps(ex, map_proj):
    #%%
    # =============================================================================
    # Load 'exp' dictionairy with information of pre-processed data (variables, paths, filenames, etcetera..)
    # and add RGCPD/Tigrimate experiment settings
    # =============================================================================
    # Response Variable is what we want to predict
    RV = ex[ex['RV_name']]
    ex['lags'] = [l * ex['tfreq'] for l in range(ex['lag_min'],ex['lag_max']+1)]
    ex['time_cycle'] = RV.dates[RV.dates.year == RV.startyear].size # time-cycle of data. total timesteps in one year
    #=====================================================================================
    # Information on period taken for response-variable, already decided in main_download_and_pp
    #=====================================================================================
    ex['time_range_all'] = [0, RV.dates.size]
    #==================================================================================
    # Start of experiment
    #==================================================================================


    # =============================================================================
    # 2) DEFINE PRECURSOS COMMUNITIES:
    # =============================================================================
    # - calculate and plot pattern correltion for differnt fields
    # - create time-series over these regions
    #=====================================================================================
    outdic_actors = dict()
    class act:
        def __init__(self, name, corr_xr, precur_arr):
            self.name = var
            self.corr_xr = corr_xr
            self.precur_arr = precur_arr
            self.lat_grid = precur_arr.latitude.values
            self.lon_grid = precur_arr.longitude.values
            self.area_grid = rgcpd.get_area(precur_arr)
            self.grid_res = abs(self.lon_grid[1] - self.lon_grid[0])

    allvar = ex['vars'][0] # list of all variable names
    for var in allvar[ex['excludeRV']:]: # loop over all variables
        actor = ex[var]
        #===========================================
        # 3c) Precursor field
        #===========================================
#        ncdf = Dataset(os.path.join(actor.path_pp, actor.filename_pp), 'r')  
        precur_arr, actor = functions_pp.import_ds_timemeanbins(actor, ex)
        precur_arr = rgcpd.convert_longitude(precur_arr, 'only_east') 
        # =============================================================================
        # Calculate correlation
        # =============================================================================
        corr_xr = rgcpd.calc_corr_coeffs_new(precur_arr, RV, ex)
        
        # =============================================================================
        # Convert regions in time series
        # =============================================================================
        actor = act(var, corr_xr, precur_arr)
        actor, ex = rgcpd.cluster_DBSCAN_regions(actor, ex)
        if np.isnan(actor.prec_labels.values).all() == False:
            rgcpd.plot_regs_xarray(actor.prec_labels.copy(), ex)
          
        # Order of regions: strongest to lowest correlation strength
        outdic_actors[var] = actor
        # =============================================================================
        # Plot
        # =============================================================================
        if ex['plotin1fig'] == False:
#            xrdata, xrmask = xrcorr_vars([var], outdic_actors, ex)
            plot_corr_maps(corr_xr, corr_xr['mask'], map_proj)

            fig_filename = '{}_corr_{}_vs_{}'.format(ex['params'], ex['RV_name'], var) + ex['file_type2']
            plt.savefig(os.path.join(ex['fig_path'], fig_filename), bbox_inches='tight', dpi=ex['png_dpi'])
            if ex['showplot'] == False:
                plt.close()


#    if ex['plotin1fig'] == True and ex['showplot'] == True:
#        variables = list(outdic_actors.keys())
##        xrdata, xrmask = xrcorr_vars(variables, outdic_actors, ex)
#        plot_corr_maps(xrdata, xrmask, map_proj)
#        fig_filename = '{}_corr_all'.format(ex['params'], allvar[0], var) + ex['file_type2']
#        plt.savefig(os.path.join(ex['fig_path'], fig_filename), bbox_inches='tight', dpi=ex['png_dpi'])
#        if ex['showplot'] == False:
#            plt.close()
#%%
    return ex, outdic_actors

def get_prec_ts(outdic_actors, ex):
    # tsCorr is total time series (.shape[0]) and .shape[1] are the correlated regions
    # stacked on top of each other (from lag_min to lag_max)
    
    ex['n_tot_regs'] = 0
    allvar = ex['vars'][0] # list of all variable names
    for var in allvar[ex['excludeRV']:]: # loop over all variables
        actor = outdic_actors[var]
        
        if np.isnan(actor.prec_labels.values).all():
            actor.ts_corr = np.array([]), []
            pass
        else:
            actor.ts_corr = rgcpd.spatial_mean_regions(actor, ex)
            outdic_actors[var] = actor
            ex['n_tot_regs'] += max([actor.ts_corr[s].shape[1] for s in range(ex['n_spl'])])
    return outdic_actors
        

def run_PCMCI_CV(ex, outdic_actors, map_proj):
    #%%
    df_splits = np.zeros( (ex['n_spl']) , dtype=object)
    df_data_s   = np.zeros( (ex['n_spl']) , dtype=object)
    for s in range(ex['n_spl']):
        progress = 100 * (s+1) / ex['n_spl']
        print(f"\rProgress causal inference - traintest set {progress}%", end="")
        df_splits[s], df_data_s[s] = run_PCMCI(ex, outdic_actors, s, map_proj)
        
    print("\n")
    
    df_data  = pd.concat(list(df_data_s), keys= range(ex['n_spl']))
    df_sum = pd.concat(list(df_splits), keys= range(ex['n_spl']))


    #%%
    return df_sum, df_data

def store_ts(df_data, df_sum, dict_ds, outdic_actors, ex, add_spatcov=True):
    
    
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    file_name = 'fulldata_{}_{}'.format(ex['params'], today)
    ex['path_data'] = os.path.join(ex['fig_subpath'], file_name+'.h5')
    
    if add_spatcov:
        df_sp_s   = np.zeros( (ex['n_spl']) , dtype=object)
        for s in range(ex['n_spl']):
            df_split = df_data.loc[s]
            df_sp_s[s] = rgcpd.get_spatcovs(dict_ds, df_split, s, outdic_actors, normalize=True)
    
        df_sp = pd.concat(list(df_sp_s), keys= range(ex['n_spl']))
        df_data_to_store = pd.merge(df_data, df_sp, left_index=True, right_index=True)
        df_sum_to_store = rgcpd.add_sp_info(df_sum, df_sp)
    else:
        df_data_to_store = df_data
        df_sum_to_store = df_sum
        
    dict_of_dfs = {'df_data':df_data_to_store, 'df_sum':df_sum_to_store}
    if ex['store_format'] == 'hdf5':
        store_hdf_df(dict_of_dfs, ex['path_data'])
        
    return


def store_hdf_df(dict_of_dfs, file_path):
    import warnings
    import tables
    
    warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)
    with pd.HDFStore(file_path, 'w') as hdf:
        for key, item in  dict_of_dfs.items():
            hdf.put(key, item, format='table', data_columns=True)
        hdf.close()
    return

def run_PCMCI(ex, outdic_actors, s, map_proj):
    #=====================================================================================
    #
    # 4) PCMCI-algorithm
    #
    #=====================================================================================

    # save output
    if ex['SaveTF'] == True:
#        from contextlib import redirect_stdout
        orig_stdout = sys.stdout
        # buffer print statement output to f
        if sys.version[:1] == '3':
            sys.stdout = f = io.StringIO()
        elif sys.version[:1] == '2':
            sys.stdout = f = open(os.path.join(ex['fig_subpath'], 'old.txt'), 'w+')

#%%
    # amount of text printed:
    verbosity = 3

    # alpha level for independence test within the pc procedure (finding parents)
    pc_alpha = ex['pcA_sets'][ex['pcA_set']]
    # alpha level for multiple linear regression model while conditining on parents of
    # parents
    alpha_level = ex['alpha_level_tig']
    print('run tigramite 4, run.pcmci')
    print(('alpha level(s) for independence tests within the pc procedure'
          '(finding parents): {}'.format(pc_alpha)))
    print(('alpha level for multiple linear regression model while conditining on parents of '
          'parents: {}'.format(ex['alpha_level_tig'])))

    # Retrieve traintest info
    traintest = ex['traintest']

    # load Response Variable class
    RV = ex[ex['RV_name']]
    # create list with all actors, these will be merged into the fulldata array
    allvar = ex['vars'][0]
    var_names = [] ; actorlist = [] ; cols = [[RV.name]]
    
    for var in allvar[ex['excludeRV']:]:
        print(var)
        actor = outdic_actors[var]
        if actor.ts_corr[s].size != 0:
            ts_train = actor.ts_corr[s].values
            actorlist.append(ts_train)
            # create array which numbers the regions
            var_idx = allvar.index(var) - ex['excludeRV']
            n_regions = actor.ts_corr[s].shape[1]
            actor.var_info = [[i+1, actor.ts_corr[s].columns[i], var_idx] for i in range(n_regions)]
            # Array of corresponing regions with var_names (first entry is RV)
            var_names = var_names + actor.var_info
        cols.append(list(actor.ts_corr[s].columns))
    var_names.insert(0, RV.name)
#    .iloc[traintest[s]['Prec_train_idx']]
        
    # stack actor time-series together:
    fulldata = np.concatenate(tuple(actorlist), axis = 1)   
    
        
    print(('There are {} regions in total'.format(fulldata.shape[1])))
    # add the full 1D time series of interest as first entry:

    fulldata = np.column_stack((RV.RVfullts, fulldata))
    
    
    df_data = pd.DataFrame(fulldata, columns=flatten(cols), index=actor.ts_corr[s].index)
    RVfull_train = RV.RVfullts.isel(time=traintest[s]['Prec_train_idx'])
    datesfull_train = pd.to_datetime(RVfull_train.time.values)
    data = df_data.loc[datesfull_train].values
    print((data.shape))

    
    # get RV datamask (same shape als data)
    datesRV_train   = pd.to_datetime(traintest[s]['RV_train'].time.values)
    data_mask = [True if d in datesRV_train else False for d in datesfull_train]
    data_mask = np.repeat(data_mask, data.shape[1]).reshape(data.shape)

    # add traintest mask to fulldata
    dates_all = pd.to_datetime(RV.RVfullts.time.values)
    dates_RV  = pd.to_datetime(RV.RV_ts.time.values)
    df_data['TrainIsTrue'] = [True if d in datesfull_train else False for d in dates_all]
    df_data['RV_mask'] = [True if d in dates_RV else False for d in dates_all]


    # ======================================================================================================================
    # tigramite 3
    # ======================================================================================================================

    T, N = data.shape # Time, Regions
    # ======================================================================================================================
    # Initialize dataframe object (needed for tigramite functions)
    # ======================================================================================================================
    dataframe = pp.DataFrame(data=data, mask=data_mask, var_names=var_names)
    # ======================================================================================================================
    # pc algorithm: only parents for selected_variables are calculated
    # ======================================================================================================================
    
    parcorr = ParCorr(significance='analytic',
                      mask_type='y',
                      verbosity=verbosity)
    #==========================================================================
    # multiple testing problem:
    #==========================================================================
    pcmci   = PCMCI(dataframe=dataframe,
                    cond_ind_test=parcorr,
                    selected_variables=None, 
                    verbosity=4)
    
    # selected_variables : list of integers, optional (default: range(N))
    #    Specify to estimate parents only for selected variables. If None is
    #    passed, parents are estimated for all variables.

    # ======================================================================================================================
    #selected_links = dictionary/None
    results = pcmci.run_pcmci(tau_max=ex['tigr_tau_max'], pc_alpha = pc_alpha, tau_min = 0, 
                              max_combinations=ex['max_comb_actors'])

    q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], fdr_method='fdr_bh')

    pcmci.print_significant_links(p_matrix = results['p_matrix'],
                                   q_matrix = q_matrix,
                                   val_matrix = results['val_matrix'],
                                   alpha_level = alpha_level)
    
    # returns all parents, not just causal precursors (of lag>0)
    sig = rgcpd.return_sign_parents(pcmci, pq_matrix=q_matrix,
                                            val_matrix=results['val_matrix'],
                                            alpha_level=alpha_level)


    all_parents = sig['parents']
#    link_matrix = sig['link_matrix']
    
    links_RV = all_parents[0]
    
    df = rgcpd.bookkeeping_precursors(links_RV, var_names)
    #%%

    rgcpd.print_particular_region_new(links_RV, var_names, s, outdic_actors, map_proj, ex)

    
#%%
    if ex['SaveTF'] == True:
        if sys.version[:1] == '3':
            fname = f's{s}_' +ex['params']+'.txt'
            file = io.open(os.path.join(ex['fig_subpath'], fname), mode='w+')
            file.write(f.getvalue())
            file.close()
            f.close()
        elif sys.version[:1] == '2':
            f.close()
        sys.stdout = orig_stdout
        

    return df, df_data

#%%


# =============================================================================
# 2 Plotting functions for correlation maps (xarray_plot_region)
# and causal region maps (plottingfunction)
# =============================================================================

def extend_longitude(data):
    import xarray as xr
    import numpy as np
    plottable = xr.concat([data, data.sel(longitude=data.longitude[:1])], dim='longitude').to_dataset(name="ds")
    plottable["longitude"] = np.linspace(0,360, len(plottable.longitude))
    plottable = plottable.to_array(dim='ds').squeeze(dim='ds').drop('ds')
    return plottable
    
def plot_corr_maps(corr_xr, xrmask, map_proj, kwrgs={'hspace':-0.6}):
    #%%
    import matplotlib.colors as colors

    if 'split' not in corr_xr.dims:
        corr_xr = corr_xr.expand_dims('split', 0) 
        xrmask = xrmask.expand_dims('split', 0) 
        
        
    splits = corr_xr['split'].values
    var_n     = corr_xr.name
    lags      = corr_xr['lag'].values
    lat = corr_xr.latitude
    lon = corr_xr.longitude
    zonal_width = abs(lon[-1] - lon[0]).values
    
    g = xr.plot.FacetGrid(corr_xr, col='lag', row='split', subplot_kws={'projection': map_proj},
                      sharex=True, sharey=True,
                      aspect= (lon.size) / lat.size, size=3)

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
    
    if 'hspace' in kwrgs.keys():
        g.fig.subplots_adjust(hspace=kwrgs['hspace'])
        
    class MidpointNormalize(colors.Normalize):
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            colors.Normalize.__init__(self, vmin, vmax, clip)

        def __call__(self, value, clip=None):
            # I'm ignoring masked values and all kinds of edge cases to make a
            # simple example...
            x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
            return np.ma.masked_array(np.interp(value, x, y))
    vmin = np.round(float(corr_xr.min())-0.01,decimals=2) ; vmax = np.round(float(corr_xr.max())+0.01,decimals=2)
    clevels = np.linspace(-max(abs(vmin),vmax),max(abs(vmin),vmax),17) # choose uneven number for # steps
    norm = MidpointNormalize(midpoint=0, vmin=clevels[0],vmax=clevels[-1])
    cmap = 'coolwarm'
    for col, lag in enumerate(lags):
        xrdatavar = corr_xr.sel(lag=lag)
        xrmaskvar = xrmask.sel(lag=lag)
        if abs(lon[-1] - 360) <= (lon[1] - lon[0]):
            xrdatavar = extend_longitude(xrdatavar)
            xrmaskvar = extend_longitude(xrmaskvar)
            
        
        for row, s in enumerate(splits):
            print(f"\rPlotting Corr maps {var_n}, fold {s}, lag {lag}", end="")
            plotdata = xrdatavar.sel(split=s)
            plotmask = xrmaskvar.sel(split=s)
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
            g.axes[row,col].coastlines()

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
            if row == splits[-1]:
                last_ax = g.axes[row,col]
    # lay out settings

    plt.tight_layout(pad=1.1-0.02*splits.size, h_pad=None, w_pad=None, rect=None)
   
    # height colorbor 1/10th of height of subfigure
    height = g.axes[-1,0].get_position().height / 10
    
    bottom_ysub = last_ax.get_position(original=False).bounds[1] # bottom
    
    cbar_ax = g.fig.add_axes([0.25, bottom_ysub, 
                              0.5, height]) #[left, bottom, width, height]
    
    
    plt.colorbar(im, cax=cbar_ax , orientation='horizontal', norm=norm,
                 label='Corr Coefficient', ticks=clevels[::4], extend='neither')
    
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
   
    variable = [v for v in np.unique(df_c['var']) if v != ex['RV_name']]
    var_rel_sizes = {outdic_actors[var].area_grid.sum()/7939E6 : var for var in variable}
    var_large_to_small = [var_rel_sizes[s] for s in sorted(var_rel_sizes, reverse=True)]
    list_ds = []
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

def plotting_per_variable(dict_ds, df, map_proj, ex):
    #%%
    # =============================================================================
    print('\nPlotting all fields significant at alpha_level_tig, while conditioning on parents'
          ' that were found in the PC step')
    # =============================================================================
    df_c = df.loc[ df['causal']==True ]
    # remove response variable if the ac is a 
    # causal link
    if df.index[0] in df_c.index:
        df_c = df_c.drop(df.index[0])
        
    variables = [v for v in np.unique(df_c['var']) if v != ex['RV_name']]
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
    
def standard_settings_and_tests(ex):
    '''Some boring settings and Perform some test'''

    RV = ex[ex['RV_name']]
    # =============================================================================
    # Test if you're not have a lag that will precede the start date of the year
    # =============================================================================
    # first date of year to be analyzed:
    firstdoy = RV.datesRV.min() - np.timedelta64(ex['tfreq'] * ex['lag_max'], 'D')
    if firstdoy < RV.dates[0] and (RV.dates[0].month,RV.dates[0].day) != (1,1):
        tdelta = RV.datesRV.min() - RV.dates.min()
        ex['lag_max'] = int(tdelta / np.timedelta64(ex['tfreq'], 'D'))
        print(('Changing maximum lag to {}, so that you not skip part of the '
              'year.'.format(ex['lag_max'])))
        
    # Some IO settings
    ex['store_format']   = 'hdf5'
    ex['file_type1'] = ".pdf"
    ex['file_type2'] = ".png"
    ex['png_dpi'] = 400
    ex['SaveTF'] = True # if false, output will be printed in console
    ex['plotin1fig'] = False
    ex['showplot'] = True
    # output paths
    method_str = '_'.join([ex['method'], 's'+ str(ex['seed'])])
    ex['subfolder_exp'] = ex['path_exp_periodmask'] +'_'+ method_str
    ex['path_output'] = os.path.join(ex['subfolder_exp'])
    ex['fig_path'] = os.path.join(ex['subfolder_exp'])
    
    ex['params'] = '{}_ac{}_at{}'.format(ex['pcA_set'], ex['alpha'],
                                                      ex['alpha_level_tig'])
    if os.path.isdir(ex['fig_path']) != True : os.makedirs(ex['fig_path'])
    ex['fig_subpath'] = os.path.join(ex['fig_path'], '{}_subinfo'.format(ex['params']))
    if os.path.isdir(ex['fig_subpath']) != True : os.makedirs(ex['fig_subpath'])

    
    assert RV.startyear == ex['startyear'], ('Make sure the dates '
             'of the RV match with the actors')
    assert ((ex['excludeRV'] == 1) and (ex['importRV_1dts'] == True))==False, ('Are you sure you want '
             'exclude first element of array ex[\'vars\']. You are importing a seperate '
             ' time series, so you probably do not need to skip the first variable in ex[\'vars\'] ')
    # =============================================================================
    # Save Experiment design
    # =============================================================================
    filename_exp_design = os.path.join(ex['fig_subpath'], 'input_dic_{}.npy'.format(ex['params']))
    np.save(filename_exp_design, ex)
    print('\n\t**\n\tOkay, end of Part 2!\n\t**' )

    print('\n**\nBegin summary of main experiment settings\n**\n')
    print('Response variable is {} is correlated vs {}'.format(RV.name,
          ex['vars'][0][:]))
    start_day = '{}-{}'.format(RV.dates[0].day, RV.dates[0].month_name())
    end_day   = '{}-{}'.format(RV.dates[-1].day, RV.dates[-1].month_name())
    print('Part of year investigated: {} - {}'.format(start_day, end_day))
    print('Part of year predicted (RV period): {} '.format(RV.RV_name_range[:-1]))
    print('Temporal resolution: {} {}'.format(ex['tfreq'], ex['input_freq']))
    print('Lags: {} to {}'.format(ex['lag_min'], ex['lag_max']))
    print('Traintest setting: {} seed {}'.format(method_str.split('_')[0], method_str.split('_s')[1]))
    one_year_RV_data = RV.datesRV.where(RV.datesRV.year==RV.startyear).dropna(how='all').values
    print('For example\nPredictant (only one year) is:\n{} at \n{}\n'.format(ex['RV_name'],
          one_year_RV_data))
    print('\tVS\n')
    shift_lag_days = one_year_RV_data - pd.Timedelta(int(ex['lag_min']*ex['tfreq']), unit='d')
    print('Predictor (only one year) is:\n{} at lag {} {}s\n{}\n'.format(
            ex['vars'][0][-1], int(ex['lag_min']*ex['tfreq']), ex['input_freq'][0], 
            shift_lag_days))
    print('\n**\nEnd of summary\n**\n')

    print('\nNext time, you\'re able to redo the experiment by loading in the dict '
          '\'filename_exp_design\'.\n')
    return ex

    
#        kwrgs = dict( {'title' : for_plt.attrs['title'], 'clevels' : 'notdefault', 
#                       'steps' : ex['max_N_regs']+1, 'subtitles': None,
#                       'vmin' : 0, 'vmax' : ex['max_N_regs'], 
#                       'cmap' : cmap, 'column' : 1,
#                       'cbar_vert' : adjust_vert_cbar, 'cbar_hght' : 0.0,
#                       'adj_fig_h' : adj_fig_h, 'adj_fig_w' : 1., 
#                       'hspace' : 0.0, 'wspace' : 0.08, 
#                       'cticks_center' : False, 'title_h' : 0.95} )
#        filename = '{}_{}_vs_{}'.format(ex['params'], ex['RV_name'], for_plt.name) + ex['file_type2']
#        rgcpd.plotting_wrapper(for_plt, ex, filename, kwrgs=kwrgs)
            

