# -*- coding: utf-8 -*-
#%%

import sys, os, io


from netCDF4 import Dataset
import matplotlib.pyplot as plt
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb
import functions_RGCPD as rgcpd
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import pandas as pd
import functions_pp


#%%
def calculate_corr_maps(ex, map_proj):
    #%%
    # =============================================================================
    # Load 'exp' dictionairy with information of pre-processed data (variables, paths, filenames, etcetera..)
    # and add RGCPD/Tigrimate experiment settings
    # =============================================================================
#    ex = np.load(str(filename_exp_design2)).item()
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
        def __init__(self, name, xr_corr, precur_arr):
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
        #%%
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
            xrdata, xrmask = xrcorr_vars([var], outdic_actors, ex)
            plot_corr_maps(xrdata, xrmask, map_proj)

            fig_filename = '{}_corr_{}_vs_{}'.format(ex['params'], ex['RV_name'], var) + ex['file_type2']
            plt.savefig(os.path.join(ex['fig_path'], fig_filename), bbox_inches='tight', dpi=ex['png_dpi'])
            if ex['showplot'] == False:
                plt.close()


    if ex['plotin1fig'] == True and ex['showplot'] == True:
        variables = list(outdic_actors.keys())
        xrdata, xrmask = xrcorr_vars(variables, outdic_actors, ex)
        plot_corr_maps(xrdata, xrmask, map_proj)
        fig_filename = '{}_corr_all'.format(ex['params'], allvar[0], var) + ex['file_type2']
        plt.savefig(os.path.join(ex['fig_path'], fig_filename), bbox_inches='tight', dpi=ex['png_dpi'])
        if ex['showplot'] == False:
            plt.close()
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
            actor.tsCorr, actor.tsCorr_labels = np.array([]), []
            pass
        else:
            actor.tsCorr, actor.tsCorr_labels = rgcpd.spatial_mean_regions(actor, ex)
            outdic_actors[var] = actor
            ex['n_tot_regs'] += actor.tsCorr.shape[1]
    return outdic_actors
        
        

def run_PCMCI(ex, outdic_actors, map_proj):
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




    # load Response Variable class

    RV = ex[ex['RV_name']]
    # create list with all actors, these will be merged into the fulldata array
    allvar = ex['vars'][0]
    var_names = [] ; actorlist = []
    
    for var in allvar[ex['excludeRV']:]:
        print(var)
        actor = outdic_actors[var]
        if actor.tsCorr.size != 0:
            actorlist.append(actor.tsCorr)
            # create array which numbers the regions
            var_idx = allvar.index(var) - ex['excludeRV']
            n_regions = actor.tsCorr.shape[1]
            actor.var_info = [[i+1, actor.tsCorr_labels[i], var_idx] for i in range(n_regions)]
            # Array of corresponing regions with var_names (first entry is RV)
            var_names = var_names + actor.var_info
            
    # stack actor time-series together:
    fulldata = np.concatenate(tuple(actorlist), axis = 1)   
    var_names.insert(0, RV.name)
        
    print(('There are {} regions in total'.format(fulldata.shape[1])))
    # add the full 1D time series of interest as first entry:
    fulldata = np.column_stack((RV.RVfullts, fulldata))
    # save fulldata
    file_name = 'fulldata_{}'.format(ex['params'])#,'.pdf' ])
    fulldata.dump(os.path.join(ex['fig_subpath'], file_name+'.pkl'))

    file_name = 'list_actors_{}'.format(ex['params'])

    # ======================================================================================================================
    # tigramite 3
    # ======================================================================================================================
    data = fulldata
    print((data.shape))
    # RV mask False for period that I want to analyse
    idx_start_RVperiod = int(np.where(RV.dates == RV.datesRV.min())[0])
    data_mask = np.ones(data.shape, dtype='bool') # true for actor months, false for RV months
    steps_in_oneyr = len(RV.datesRV[RV.datesRV.year == ex['startyear']])
    for i in range(steps_in_oneyr): # total timesteps RV period, 12 is
        data_mask[idx_start_RVperiod+i:: ex['time_cycle'],:] = False
    T, N = data.shape # Time, Regions
    # ======================================================================================================================
    # Initialize dataframe object (needed for tigramite functions)
    # ======================================================================================================================
    dataframe = pp.DataFrame(data=data, mask=data_mask, var_names=var_names)
    # Create 'time axis' and variable names
#    datatime = np.arange(len(data))
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
    

        
    def return_sign_parents(pc_class, pq_matrix, val_matrix,
                                alpha_level=0.05):
          # Initialize the return value
        all_parents = dict()
        for j in pc_class.selected_variables:
            # Get the good links
            good_links = np.argwhere(pq_matrix[:, j, :] <= alpha_level)
            # Build a dictionary from these links to their values
            links = {(i, -tau): np.abs(val_matrix[i, j, abs(tau) ])
                     for i, tau in good_links}
            # Sort by value
            all_parents[j] = sorted(links, key=links.get, reverse=True)
        # Return the significant parents
        return {'parents': all_parents,
                'link_matrix': pq_matrix <= alpha_level}
    
    
    sig = return_sign_parents(pcmci, pq_matrix=q_matrix,
                                            val_matrix=results['val_matrix'],
                                            alpha_level=alpha_level)


    all_parents = sig['parents']
    link_matrix = sig['link_matrix']
    
#    """
#    what's this?
#    med = LinearMediation(dataframe=dataframe,
#                use_mask =True,
#                mask_type ='y',
#                data_transform = None)   # False or None for sklearn.preprocessing.StandardScaler =
#
#    med.fit_model(all_parents=all_parents, tau_max= tau_max)
#    """

    # parents of index of interest:
    # parents_neighbors = all_parents, estimates, iterations
    links_RV = all_parents[0]
    #%%

    n_parents = len(links_RV)
    lags = list(np.arange(ex['lag_min'], ex['lag_max']+1E-9, dtype=int))
    for i in range(n_parents):
        tigr_lag = links_RV[i][1] #-1 There was a minus, but is it really correct?
        index_in_fulldata = links_RV[i][0]
        print("\n\nunique_label_format: \n\'lag\'_\'regionlabel\'_\'var\'")
        if index_in_fulldata>0:
            uniq_label = var_names[index_in_fulldata][1]
            var_name = uniq_label.split('_')[-1]
            according_varname = uniq_label
            according_number = int(uniq_label.split('_')[1])
#            according_var_idx = ex['vars'][0].index(var_name)
            corr_lag = int(uniq_label.split('_')[0])
            l_idx = lags.index(corr_lag)
            print('index in fulldata {}: region: {} at lag {}'.format(
                    index_in_fulldata, uniq_label, tigr_lag))
            # *********************************************************
            # print and save only significant regions
            # *********************************************************
            according_fullname = '{} at lag {} - ts_index_{}'.format(according_varname,
                                  tigr_lag, index_in_fulldata)
            


            actor = outdic_actors[var_name]
            prec_labels_actor = actor.prec_labels
            rgcpd.print_particular_region_new(ex, according_number, l_idx, prec_labels_actor,
                                          map_proj, according_fullname)
            fig_file = '{}{}'.format(according_fullname, ex['file_type2'])

            plt.savefig(os.path.join(ex['fig_subpath'], fig_file), dpi=ex['png_dpi'])
#            plt.show()
            plt.close()

            print('                                        ')
            # *********************************************************
            # save data
            # *********************************************************
            according_fullname = str(according_number) + according_varname
            name = ''.join([str(index_in_fulldata),'_',uniq_label])
            print((fulldata[:,index_in_fulldata].size))
            print(name)
        else :
            print('Index itself is also causal parent -> skipped')
            print('*******************              ***************************')
#%%
    if ex['SaveTF'] == True:
        if sys.version[:1] == '3':
            file = io.open(os.path.join(ex['fig_subpath'], ex['params']+'.txt'), mode='w+')
            file.write(f.getvalue())
            file.close()
            f.close()
        elif sys.version[:1] == '2':
            f.close()
        sys.stdout = orig_stdout
#%%
    # create dataframe output:
    # var_names = [reg_number, str({lag_corr}_{reg_number}_{var_name}), lag_tigr]
    var_names_ = var_names.copy()
    index = [n[1] for n in var_names_[1:]] ; index.insert(0, var_names_[0])
    link_names = [var_names_[l[0]][1] if l[0] !=0 else var_names_[l[0]] for l in links_RV]
    
    
    # check if two lags of same region are tigr significant
    idx_tigr = [l[0] for l in links_RV] ; 
    for r in np.unique(idx_tigr):
        if idx_tigr.count(r) != 1:
            double = var_names_[r][1]
            idx = int(np.argwhere(np.array(index)==double)[0])
            # append each double to index for the dataframe
            for i in range(idx_tigr.count(r)-1):
                index.insert(idx+i, double)
                d = len(index) - len(var_names_)
                var_names_.insert(idx+i+1, var_names_[idx+1-d])
                
    

#    mask_causal = np.zeros( (len(index) + doubles), dtype=bool)
#    for n in link_names:
#        np.argwhere(index==n)
    l = [n[1].split('_')[-1] for n in var_names_[1:]]
    l.insert(0, var_names_[0])
    var = np.array(l)
    mask_causal = np.array([True if i in link_names else False for i in index])
    lag_corr_map = np.array([int(n[1][0]) for n in var_names_[1:]]) ; 
    lag_corr_map = np.insert(lag_corr_map, 0, 0)
    region_number = np.array([int(n[0]) for n in var_names_[1:]])
    region_number = np.insert(region_number, 0, 0)
    lag_tigr_map = {str(links_RV[i][1])+'..'+link_names[i]:links_RV[i][1] for i in range(len(link_names))}
    sorted(lag_tigr_map, reverse=False)
    lag_tigr_ = index.copy() ; track_idx = list(range(len(index)))
    for k in lag_tigr_map.keys():
        l = int(k.split('..')[0])
        var_temp = k.split('..')[1]
        idx = lag_tigr_.index(var_temp)
        track_idx.remove(idx)
        lag_tigr_[idx] = l
    for i in track_idx:
        lag_tigr_[i] = np.nan
    lag_tigr_ = np.array(lag_tigr_)
    mask_causal = ~np.isnan(lag_tigr_)
    


    
#    print(var.shape, lag_corr_map.shape, region_number.shape, mask_causal.shape, lag_caus.shape)


    data = np.concatenate([var[None,:], lag_corr_map[None,:], region_number[None,:],
                            mask_causal[None,:], lag_tigr_[None,:]], axis=0)
    df = pd.DataFrame(data=data.T, index=index, 
                      columns=['var', 'lag_corr_map', 'region_number', 'causal', 'lag_causal'])
    df['causal'] = df['causal'] == 'True'
    df = df.astype({'var':str, 'lag_corr_map':int,
                               'region_number':int, 'causal':bool, 'lag_causal':float})
    #%%                      
    print(df)
    return df
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


def xrcorr_vars(print_vars, outdic_actors, ex):
    #%%
    import cartopy.crs as ccrs
    import numpy as np
    import xarray as xr
    outd = outdic_actors
    list_Corr = []
    list_mask = []
    if print_vars == 'all':
        variables = list(outd.keys())[:]
    else:
        variables = print_vars
    
    for var in variables:
        lags = list(range(ex['lag_min'], ex['lag_max']+1))
        lags = ['{} ({} {})'.format(l, l*ex['tfreq'], ex['input_freq'][:1]) for l in lags]
        lat = outd[var].lat_grid
        lon = outd[var].lon_grid
        list_Corr.append(outd[var].Corr_Coeff.data[None,:,:].reshape(lat.size,lon.size,len(lags)))
        list_mask.append(outd[var].Corr_Coeff.mask[None,:,:].reshape(lat.size,lon.size,len(lags)))
        Corr_regvar = np.array(list_Corr)
        mask_regvar = np.array(list_mask)
    
        xrdata = xr.DataArray(data=Corr_regvar, coords=[variables, lat, lon, lags],
                            dims=['variable','latitude','longitude','lag'], name='Corr Coeff')
        xrmask = xr.DataArray(data=mask_regvar, coords=[variables, lat, lon, lags],
                            dims=['variable','latitude','longitude','lag'], name='Corr Coeff')
    #%%
    return xrdata, xrmask
    
def plot_corr_maps(xrdata, xrmask, map_proj):
    #%%
    import matplotlib.colors as colors


    variables = xrdata['variable'].values
    lags      = xrdata['lag'].values
    lat = xrdata.latitude
    lon = xrdata.longitude
    zonal_width = abs(lon[-1] - lon[0]).values
    
    g = xr.plot.FacetGrid(xrdata, col='variable', row='lag', subplot_kws={'projection': map_proj},
                      aspect= (lon.size) / lat.size, size=3)
    
    figheight = g.fig.get_figheight()
    class MidpointNormalize(colors.Normalize):
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            colors.Normalize.__init__(self, vmin, vmax, clip)

        def __call__(self, value, clip=None):
            # I'm ignoring masked values and all kinds of edge cases to make a
            # simple example...
            x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
            return np.ma.masked_array(np.interp(value, x, y))
    vmin = np.round(float(xrdata.min())-0.01,decimals=2) ; vmax = np.round(float(xrdata.max())+0.01,decimals=2)
    clevels = np.linspace(-max(abs(vmin),vmax),max(abs(vmin),vmax),17) # choose uneven number for # steps
    norm = MidpointNormalize(midpoint=0, vmin=clevels[0],vmax=clevels[-1])
    cmap = 'coolwarm'
    for col, var in enumerate(variables):
        xrdatavar = xrdata.sel(variable=var)
        xrmaskvar = xrmask.sel(variable=var)
        if abs(lon[-1] - 360) <= (lon[1] - lon[0]):
            xrdatavar = extend_longitude(xrdatavar)
            xrmaskvar = extend_longitude(xrmaskvar)
            
        for row, lag in enumerate(lags):
            print('Plotting Corr maps {}, lag {}'.format(var, lag))
            plotdata = xrdatavar.sel(lag=lag)
            plotmask = xrmaskvar.sel(lag=lag)
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

    plt.tight_layout()
    g.axes[row,col].get_position()
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    cbar_ax = g.fig.add_axes([0.25, 0.0, 0.5, figheight/150]) #[left, bottom, width, height]
    plt.colorbar(im, cax=cbar_ax , orientation='horizontal', norm=norm,
                 label='Corr Coefficient', ticks=clevels[::4], extend='neither')
    #%%
    return

def causal_reg_to_xarray(ex, df, outdic_actors):
    #%%    
#    outdic_actors_c = outdic_actors.copy()
    df_c = df.loc[ df['causal']==True ]
    # remove response variable if the ac is a 
    # causal link
    if df.index[0] in df_c.index:
        df_c = df_c.drop(df.index[0])
    allvar = ex['vars'][0]
    precursor_fields = allvar[ex['excludeRV']:]
    lags = list(range(ex['lag_min'],ex['lag_max']+1))
    
    
    ds = xr.Dataset()
    for var in precursor_fields:
        actor = outdic_actors[var]
        lats = actor.lat_grid
        lons = actor.lon_grid
        corr_coeff = actor.Corr_Coeff.reshape(len(lags), lats.size, lons.size)
        xr_corr = actor.prec_labels.copy()
        xr_corr.values = corr_coeff
        ds[var+'_labels'] = actor.prec_labels
        ds[var+'_corr'] = xr_corr
        ds[var+'_labels_tigr'] = actor.prec_labels.copy()
        ds[var+'_corr_tigr'] = xr_corr.copy()
    
    for i, var in enumerate(np.unique(df_c['var'])):
        regs_c = df_c.loc[ df_c['var'] == var ]
        for lag_cor in ds[var+'_labels_tigr'].lag.values:
            var_tig = ds[var+'_labels_tigr'].sel(lag=lag_cor)
            for lag_t in np.unique(regs_c['lag_causal']):
                reg_c_l = regs_c.loc[ regs_c['lag_causal'] == lag_t]
                labels = reg_c_l.region_number.values
                
                new_mask = np.zeros( shape=var_tig.shape, dtype=bool)
                
                for l in labels:
                    new_mask[var_tig.values == l] = True
            
            ds[var+'_labels_tigr'].sel(lag=lag_cor).values[~new_mask] = np.nan
            ds[var+'_corr_tigr'].sel(lag=lag_cor).values[~new_mask] = np.nan
    #%%
    return ds

def plotting_per_variable(ds, df, map_proj, ex):
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
    lags = ds.lag.values
    for lag in lags:
        
        for i, var in enumerate(np.unique(df_c['var'])):
            
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
    
    ds_l = ds.sel(lag=lag)
    list_xr = [] ; name = []
    for c in ['corr', 'corr_tigr']:
        name.append(var+'_'+c)
        list_xr.append(ds_l[var+'_'+c])
    xrdata = xr.concat(list_xr, dim='lag')
    xrdata.lag.values = name
    xrdata.name = 'sst_corr_and_tigr'
    xrdata = xrdata.expand_dims('variable', axis=0)
    xrdata.assign_coords(variable=[var])
    xrmask = xrdata.copy()
    xrmask.values = ~np.isnan(xrdata.values)
    if np.isnan(xrdata.values).all() == False:
        plot_corr_maps(xrdata, xrmask, map_proj)
        fig_filename = '{}_tigr_corr_{}_vs_{}'.format(ex['params'], ex['RV_name'], var) + ex['file_type2']
        plt.savefig(os.path.join(ex['fig_path'], fig_filename), bbox_inches='tight', dpi=ex['png_dpi'])
        plt.show() ; plt.close()

    
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

    # Some output settings
    ex['file_type1'] = ".pdf"
    ex['file_type2'] = ".png"
    ex['png_dpi'] = 400
    ex['SaveTF'] = True # if false, output will be printed in console
    ex['plotin1fig'] = False
    ex['showplot'] = True
    # output paths
    ex['path_output'] = os.path.join(ex['path_exp_periodmask'])
    ex['fig_path'] = os.path.join(ex['path_exp_periodmask'])
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
    filename_exp_design2 = os.path.join(ex['fig_subpath'], 'input_dic_{}.npy'.format(ex['params']))
    np.save(filename_exp_design2, ex)
    print('\n\t**\n\tOkay, end of Part 2!\n\t**' )

    print('\n**\nBegin summary of main experiment settings\n**\n')
    print('Response variable is {} is correlated vs {}'.format(ex['vars'][0][0],
          ex['vars'][0][1:]))
    start_day = '{}-{}'.format(RV.dates[0].day, RV.dates[0].month_name())
    end_day   = '{}-{}'.format(RV.dates[-1].day, RV.dates[-1].month_name())
    print('Part of year investigated: {} - {}'.format(start_day, end_day))
    print('Part of year predicted (RV period): {} '.format(RV.RV_name_range[:-1]))
    print('Temporal resolution: {} {}'.format(ex['tfreq'], ex['input_freq']))
    print('Lags: {} to {}'.format(ex['lag_min'], ex['lag_max']))
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
          '\'filename_exp_design2\'.\n')
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
            
    #%%

#    def finalfigure(xrdata, all_regions_deladj, file_name):
#        g = xr.plot.FacetGrid(xrdata, col='names_col', row='names_row', subplot_kws={'projection': map_proj},
#                          aspect= (lon.size) / lat.size, size=3)
#        figwidth = g.fig.get_figwidth() ; figheight = g.fig.get_figheight()
#        if xrdata.max() >= 2:
#            cmap = plt.cm.Dark2
#            clevels = np.arange(int(xrdata.min()), int(xrdata.max())+1E-9, 1)
#        else:
#            cmap = plt.cm.Greens
#            clevels = [0., 0.95, 1.0]
#
#
#        for row in xrdata.names_row.values:
#            rowidx = list(xrdata.names_row.values).index(row)
#            plotrow = xrdata.sel(names_row=row)
#            for col in xrdata.names_col.values:
#                colidx = list(xrdata.names_col.values).index(col)
#
#
#                plotdata = extend_longitude(plotrow.sel(names_col=names_col[colidx]))
#                if np.sum(plotdata) == 0.:
#                    g.axes[rowidx,colidx].text(0.5, 0.5, 'No regions significant',
#                                  horizontalalignment='center', fontsize='x-large',
#                                  verticalalignment='center', transform=g.axes[rowidx,colidx].transAxes)
#                elif np.sum(plotdata) > 0.:
#                    im = plotdata.plot.contourf(ax=g.axes[rowidx,colidx], transform=ccrs.PlateCarree(),
#                                                    cmap=cmap, levels=clevels,
#                                                    subplot_kws={'projection':map_proj},
#                                                    add_colorbar=False)
#                    plotdata = extend_longitude(plotrow.sel(names_col=names_col[1]))
#                    if np.sum(plotdata) != 0.:
#                        contourmask = np.array(np.nan_to_num(plotdata.where(plotdata > 0.)), dtype=int)
#                        contourmask[contourmask!=0] = 2
#                        plotdata.data = contourmask
#                        plotdata.plot.contour(ax=g.axes[rowidx,colidx], transform=ccrs.PlateCarree(),
#                                                            colors=['black'], levels=[0,1,2],
#                                                            subplot_kws={'projection':map_proj},
#                                                            add_colorbar=False)
#                g.axes[rowidx,colidx].set_extent([lon[0], lon[-1], 
#                                       lat[0], lat[-1]], ccrs.PlateCarree())
#            g.axes[rowidx,0].text(-figwidth/100, 0.5, row,
#                      horizontalalignment='center', fontsize='x-large',
#                      verticalalignment='center', transform=g.axes[rowidx,0].transAxes)
#        for ax in g.axes.flat:
#            ax.coastlines(color='grey', alpha=0.3)
#            ax.set_title('')
#        g.axes[0,1].set_title(names_col[1] + '\nat alpha={} with '
#                      'pc_alpha(s)={}'.format(ex['alpha_level_tig']  , ex['pcA_sets'][ex['pcA_set']]), fontsize='x-large')
#        g.axes[0,0].set_title(names_col[0] + '\nat Corr p-value={}'.format(ex['alpha']),
#                      fontsize='x-large')
##        g.axes[rowidx,0].text(0.5, figwidth/100, 'Black contours are not significant after MCI',
##                      horizontalalignment='center', fontsize='x-large',
##                      verticalalignment='center', transform=g.axes[rowidx,0].transAxes)
#        if ex['plotin1fig'] == False and xrdata.sum() != 0:
#            cbar_ax = g.fig.add_axes([0.25, (figheight/25)/len(g.row_names),
#                                      0.5, (figheight/150)/len(g.row_names)])
#            plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
##        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#        plt.subplots_adjust(wspace=0.1, hspace=-0.3)
#        g.fig.savefig(os.path.join(ex['fig_path'], file_name + ex['file_type2']),dpi=250)
#        if ex['showplot'] == False:
#            plt.close()
#        return
#
#
#    if ex['plotin1fig'] == True:
#        names_row = [i +' all lags' for i in prec_names]
#        xrdata = xr.DataArray(data=array, coords=[names_col, names_row, lat, lon],
#                            dims=['names_col','names_row','latitude','longitude'], name='Corr Coeff')
#        xrdata.data = np.nan_to_num(xrdata.data)
#        xrdata.data[xrdata.data > 0.5] = 1.
#        all_regions_deladj = all_regions_del
#        file_name = '{}_tigout_all'.format(ex['params'])
#        finalfigure(xrdata, all_regions_deladj, file_name)
#    if ex['plotin1fig'] == False:
#        if ex['input_freq'] == 'daily'  : dt = 'days'
#        if ex['input_freq'] == 'monthly': dt = 'months'
#        for var in allvar[ex['excludeRV']:]:
#            varidx = allvar.index(var) - ex['excludeRV']
#            onevar_array = array[:,varidx,:,:,:].copy()
#            names_row = []
#            for lag in lags:
#                names_row.append(var + '\n-{} {}'.format(lag * ex['tfreq'], dt))
#
#            xrdata = xr.DataArray(data=onevar_array, coords=[names_col, names_row, lat, lon],
#                                dims=['names_col','names_row','latitude','longitude'], name='Corr Coeff')
#
#            all_regions_deladj = all_regions_del[varidx]
#            file_name = '{}_tigout_{}'.format(ex['params'], var)
#            finalfigure(xrdata, all_regions_deladj, file_name)
# #%%
#    return


