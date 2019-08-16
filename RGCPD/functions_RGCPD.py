# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
matplotlib.rcParams['backend'] = "Qt4Agg"
from pylab import *
import matplotlib.pyplot as plt
import wrapper_RGCPD_tig
import functions_pp
import scipy
import pandas as pd
from statsmodels.sandbox.stats import multicomp
import xarray as xr
import cartopy.crs as ccrs
import itertools
flatten = lambda l: list(itertools.chain.from_iterable(l))
def get_oneyr(datetime):
        return datetime.where(datetime.year==datetime.year[0]).dropna()
from dateutil.relativedelta import relativedelta as date_dt


def convert_longitude(data, to_format='only_east'):
    import numpy as np
    import xarray as xr
    if to_format == 'west_east':
        lon_above = data.longitude[np.where(data.longitude > 180)[0]]
        lon_normal = data.longitude[np.where(data.longitude <= 180)[0]]
        # roll all values to the right for len(lon_above amount of steps)
        data = data.roll(longitude=len(lon_above))
        # adapt longitude values above 180 to negative values
        substract = lambda x, y: (x - y)
        lon_above = xr.apply_ufunc(substract, lon_above, 360)
        if lon_normal.size != 0:
            if lon_normal[0] == 0.:
                convert_lon = xr.concat([lon_above, lon_normal], dim='longitude')
            
            else:
                convert_lon = xr.concat([lon_normal, lon_above], dim='longitude')
        else:
            convert_lon = lon_above

    elif to_format == 'only_east':
        lon_above = data.longitude[np.where(data.longitude >= 0)[0]]
        lon_below = data.longitude[np.where(data.longitude < 0)[0]]
        lon_below += 360
        data = data.roll(longitude=len(lon_below))
        convert_lon = xr.concat([lon_above, lon_below], dim='longitude')
    data['longitude'] = convert_lon
    return data
	
	
def corr_new(D, di):
	"""
	This function calculates the correlation coefficent r  and the the pvalue p for each grid-point of field D with the response-variable di
	"""
	x = np.ma.zeros(D.shape[1])
	corr_di_D = np.ma.array(data = x, mask =False)	
	sig_di_D = np.array(x)
	
	for i in range(D.shape[1]):
		r, p = scipy.stats.pearsonr(di,D[:,i])
		corr_di_D[i]= r 
		sig_di_D[i]= p
									
	return corr_di_D, sig_di_D

	
def calc_corr_coeffs_new(precur_arr, RV, ex):
    #%%
#    v = ncdf ; V = array ; RV.RV_ts = ts of RV, time_range_all = index range of whole ts
    """
    This function calculates the correlation maps for precur_arr for different lags. 
    Field significance is applied to test for correltion.
    This function uses the following variables (in the ex dictionary)
    prec_arr: array
    time_range_all: a list containing the start and the end index, e.g. [0, time_cycle*n_years]
    lag_steps: number of lags
    time_cycle: time cycyle of dataset, =12 for monthly data...
    RV_period: indices that matches the response variable time series
    alpha: significance level

    """
    lag_steps = ex['lag_max'] - ex['lag_min'] +1
    tfreq = ex['tfreq']
    lags = np.arange(ex['lag_min']*tfreq, ex['lag_max']*tfreq+1E-9, tfreq, dtype=int)
    assert lag_steps >= 0, ('Maximum lag is larger then minimum lag, not allowed')
		    

    traintest, ex = functions_pp.rand_traintest_years(RV.RV_ts, precur_arr, ex)
    # make new xarray to store results
    xrcorr = precur_arr.isel(time=0).drop('time').copy()
    # add lags
    list_xr = [xrcorr.expand_dims('lag', axis=0) for i in range(lag_steps)]
    xrcorr = xr.concat(list_xr, dim = 'lag')
    xrcorr['lags'] = ('lag', lags)
    # add train test split     
    list_xr = [xrcorr.expand_dims('split', axis=0) for i in range(ex['n_conv'])]
    xrcorr = xr.concat(list_xr, dim = 'split')           
    xrcorr['split'] = ('split', range(ex['n_conv']))
    
    print('\n{} - calculating correlation maps'.format(precur_arr.name))
    np_data = np.zeros_like(xrcorr.values)
    np_mask = np.zeros_like(xrcorr.values)
    def corr_single_split(RV_ts, precur, RV_period, ex):
        
        lag_steps = ex['lag_max'] - ex['lag_min'] +1
        lat = precur_arr.latitude.values
        lon = precur_arr.longitude.values
        
        z = np.zeros((lat.size*lon.size,lag_steps))
        Corr_Coeff = np.ma.array(z, mask=z)
        # reshape
        sat = np.reshape(precur_arr.values, (precur_arr.shape[0],-1))
        
    	
        for i in range(lag_steps):
    
            lag = ex['lag_min'] + i
            months_indices_lagged = [r - lag for r in RV_period]	
            sat_winter = sat[months_indices_lagged]
    		
    		# correlation map and pvalue at each grid-point:
            corr_di_sat, sig_di_sat = corr_new(sat_winter, RV_ts)
    		
            if ex['FDR_control'] == True:
    			# test for Field significance and mask unsignificant values			
    			# FDR control:
                adjusted_pvalues = multicomp.multipletests(sig_di_sat, method='fdr_bh')			
                ad_p = adjusted_pvalues[1]
    			
                corr_di_sat.mask[ad_p> ex['alpha']] = True
    
            else:
                corr_di_sat.mask[sig_di_sat> ex['alpha']] = True
    			
    			
            Corr_Coeff[:,i] = corr_di_sat[:]
                
        Corr_Coeff = np.ma.array(data = Corr_Coeff[:,:], mask = Corr_Coeff.mask[:,:])
        Corr_Coeff = Corr_Coeff.reshape(lat.size,lon.size,len(lags)).swapaxes(2,1).swapaxes(1,0)
        return Corr_Coeff

    for s in xrcorr.split.values:
        progress = 100 * (s+1) / ex['n_conv']
        print(f"\rProgress traintest set {progress}%", end="") 
        # =============================================================================
        # Split train test methods ['random'k'fold', 'leave_'k'_out', ', 'no_train_test_split']        
        # =============================================================================
        RV_ts = traintest[s]['RV_train']
        precur = precur_arr.isel(time=traintest[s]['Prec_train_idx'])
        dates_RV  = pd.to_datetime(RV_ts.time.values)
        dates_all = pd.to_datetime(precur.time.values)
        string_RV = list(dates_RV.strftime('%Y-%m-%d'))
        string_full = list(dates_all.strftime('%Y-%m-%d'))
        RV_period = [string_full.index(date) for date in string_full if date in string_RV]
        
        ma_data = corr_single_split(RV_ts, precur, RV_period, ex)
        np_data[s] = ma_data.data
        np_mask[s] = ma_data.mask
    xrcorr.values = np_data
    mask = (('split', 'lag', 'latitude', 'longitude'), np_mask )
    xrcorr.coords['mask'] = mask
    #%%
    return xrcorr
	
def calc_corr_coeffs(precur_arr, RV, ex):
    #%%
#    v = ncdf ; V = array ; RV.RV_ts = ts of RV, time_range_all = index range of whole ts
    """
    This function calculates the correlation maps for precur_arr for different lags. 
    Field significance is applied to test for correltion.
    This function uses the following variables (in the ex dictionary)
    prec_arr: array
    time_range_all: a list containing the start and the end index, e.g. [0, time_cycle*n_years]
    lag_steps: number of lags
    time_cycle: time cycyle of dataset, =12 for monthly data...
    RV_period: indices that matches the response variable time series
    alpha: significance level

    """
    lag_steps = ex['lag_max'] - ex['lag_min'] +1
    
    assert lag_steps >= 0, ('Maximum lag is larger then minimum lag, not allowed')
		    
    lat = precur_arr.latitude.values
    lon = precur_arr.longitude.values
    

    z = np.zeros((lat.size*lon.size,lag_steps))
    Corr_Coeff = np.ma.array(z, mask=z)
	
    # reshape
    sat = np.reshape(precur_arr.values, (precur_arr.shape[0],-1))
    print('\n{} - calculating correlation maps'.format(precur_arr.name))
	
	
    for i in range(lag_steps):

        lag = ex['lag_min'] + i
        months_indices_lagged = [r - lag for r in ex['RV_period']]
       
#        if ex['time_match_RV'] == True:
#            months_indices_lagged = [r - lag for r in ex['RV_period']]
#        else:
#            # recreate RV_period of precursor to match the correct indices           
#            start_prec = pd.Timestamp(RV.RVfullts[ex['RV_period'][0]].time.values) - date_dt(months=lag * ex['tfreq'])
#            start_ind = int(np.where(pd.to_datetime(precur_arr.time.values) == start_prec)[0])
#            new_n_oneyr  = get_oneyr(pd.to_datetime(precur_arr.time.values)).size
#            RV_period = [ (yr*new_n_oneyr)-start_ind for yr in range(1,int(ex['n_yrs'])+1)]
#            months_indices_lagged = [r - (lag) for r in RV_period]

            
#            precur_arr.time.values[months_indices_lagged]
            # only winter months 		
        sat_winter = sat[months_indices_lagged]
		
		# correlation map and pvalue at each grid-point:
        corr_di_sat, sig_di_sat = corr_new(sat_winter, RV.RV_ts)
		
        if ex['FDR_control'] == True:
				
			# test for Field significance and mask unsignificant values			
			# FDR control:
            adjusted_pvalues = multicomp.multipletests(sig_di_sat, method='fdr_bh')			
            ad_p = adjusted_pvalues[1]
			
            corr_di_sat.mask[ad_p> ex['alpha']] = True

        else:
            corr_di_sat.mask[sig_di_sat> ex['alpha']] = True
			
			
        Corr_Coeff[:,i] = corr_di_sat[:]
            
    Corr_Coeff = np.ma.array(data = Corr_Coeff[:,:], mask = Corr_Coeff.mask[:,:])
	#%%
    return Corr_Coeff

def get_area(ds):
    longitude = ds.longitude
    latitude = ds.latitude
    
    Erad = 6.371e6 # [m] Earth radius
#    global_surface = 510064471909788
    # Semiconstants
    gridcell = np.abs(longitude[1] - longitude[0]).values # [degrees] grid cell size
    
    # new area size calculation:
    lat_n_bound = np.minimum(90.0 , latitude + 0.5*gridcell)
    lat_s_bound = np.maximum(-90.0 , latitude - 0.5*gridcell)
    
    A_gridcell = np.zeros([len(latitude),1])
    A_gridcell[:,0] = (np.pi/180.0)*Erad**2 * abs( np.sin(lat_s_bound*np.pi/180.0) - np.sin(lat_n_bound*np.pi/180.0) ) * gridcell
    A_gridcell2D = np.tile(A_gridcell,[1,len(longitude)])
#    A_mean = np.mean(A_gridcell2D)
    return A_gridcell2D


def quick_plot_mask(actor):
    mask  = actor.Corr_Coeff.mask
    lats    = actor.lat_grid
    lons    = actor.lon_grid
    plt.figure(figsize=(10,15)) ; 
    plt.imshow(np.reshape(mask, (lats.size, lons.size))) 
    return

def cluster_DBSCAN_regions(actor, ex):
    #%%
    """
	Calculates the time-series of the actors based on the correlation coefficients and plots the according regions. 
	Only caluclates regions with significant correlation coefficients
	"""
    from sklearn import cluster
    from sklearn import metrics
    from haversine import haversine
    import xarray as xr
    
#    var = 'sst'
#    actor = outdic_actors[var]
    Corr_Coeff  = actor.Corr_Coeff
    lats    = actor.lat_grid
    lons    = actor.lon_grid
    area_grid   = actor.area_grid/ 1E6 # in km2

    aver_area_km2 = 7939     # np.mean(actor.area_grid) with latitude 0-90 / 1E6
    wght_area = area_grid / aver_area_km2
    ex['min_area_km2'] = ex['min_area_in_degrees2'] * 111.131 * ex['min_area_in_degrees2'] * 78.85
    min_area = ex['min_area_km2'] / aver_area_km2
    ex['min_area_samples'] = min_area
    

    lags = np.arange(ex['lag_min'], ex['lag_max']+1E-9)
    lag_steps = lags.size

    Number_regions_per_lag = np.zeros(lag_steps)
    ex['n_tot_regs'] = 0
    
    
    def mask_sig_to_cluster(mask_and_data, lons, lats, wght_area, ex):
        mask_sig_1d = mask_and_data.mask[:,:]==False
        data = mask_and_data.data
    
        np_dbregs   = np.zeros( (lons.size*lats.size, lag_steps), dtype=int )
        labels_sign_lag = []
        label_start = 0
        
        for sign in [-1, 1]:
            mask = mask_sig_1d.copy()
            mask[np.sign(data) != sign] = False
            n_gc_sig_sign = mask[mask==True].size
            labels_for_lag = np.zeros( (lag_steps, n_gc_sig_sign), dtype=bool)
            meshgrid = np.meshgrid(lons.data, lats.data)
            mask_sig = np.reshape(mask, (lats.size, lons.size, lag_steps))
            sign_coords = [] ; count=0
            weights_core_samples = []
            for l in range(lag_steps):
                sign_c = meshgrid[0][ mask_sig[:,:,l] ], meshgrid[1][ mask_sig[:,:,l] ]
                n_sign_c_lag = len(sign_c[0])
                labels_for_lag[l][count:count+n_sign_c_lag] = True
                count += n_sign_c_lag
                sign_coords.append( [(sign_c[1][i], sign_c[0][i]-180) for i in range(sign_c[0].size)] )
                
                weights_core_samples.append(wght_area[mask_sig[:,:,l]].reshape(-1))
                
            sign_coords = flatten(sign_coords)
            if len(sign_coords) != 0:
                weights_core_samples = flatten(weights_core_samples)
                # calculate distance between sign coords accross all lags to keep labels 
                # more consistent when clustering
                distance = metrics.pairwise_distances(sign_coords, metric=haversine)
                dbresult = cluster.DBSCAN(eps=ex['distance_eps'], min_samples=ex['min_area_samples'], 
                                          metric='precomputed').fit(distance, 
                                          sample_weight=weights_core_samples)
                labels = dbresult.labels_ + 1
                # all labels == -1 (now 0) are seen as noise:
                labels[labels==0] = -label_start
                individual_labels = labels + label_start
                [labels_sign_lag.append((l, sign)) for l in np.unique(individual_labels) if l != 0]

                for l in range(lag_steps):
                    mask_sig_lag = mask[:,l]==True
                    np_dbregs[:,l][mask_sig_lag] = individual_labels[labels_for_lag[l]]
                label_start = int(np_dbregs[mask].max())
            else:
                pass
            np_regs = np.reshape(np_dbregs, (lats.size, lons.size, lag_steps))
            np_regs = np_regs.swapaxes(0,-1).swapaxes(1,2)
        return np_regs, labels_sign_lag


    prec_labels_np = np.zeros( (lag_steps, lats.size, lons.size) )
#    labels_sign = np.zeros( (lag_steps), dtype=list )
    mask_and_data = Corr_Coeff.copy()
#    if 
    # prec_labels_np: shape [lags, lats, lons]
    prec_labels_np, labels_sign_lag = mask_sig_to_cluster(mask_and_data, lons, lats, wght_area, ex)
    
    
    if np.nansum(prec_labels_np) == 0. and mask_and_data.mask.all()==False:
        print('\nSome significantly correlating gridcells found, but too randomly located and '
              'interpreted as noise by DBSCAN, make distance_eps lower '
              'to relax contrain.\n')
        prec_labels_ord = prec_labels_np
    if mask_and_data.mask.all()==True:
        print('\nNo significantly correlating gridcells found.\n')
        prec_labels_ord = prec_labels_np
    else:
        # order regions on corr strength
        # based on median of upper 25 percentile
        corr_strength = {}
        totalsize_lag0 = area_grid[prec_labels_np[0]!=0].mean() / 1E5
        for l_idx, lag in enumerate(lags):
            # check if region is higher lag is actually too small to be a cluster:
            prec_field = prec_labels_np[l_idx,:,:]
            
            for i, reg in enumerate(np.unique(prec_field)[1:]):
                are = area_grid.copy()
                are[prec_field!=reg]=0
                area_prec_reg = are.sum()/1E5
                if area_prec_reg < ex['min_area_km2']/1E5:
                    print(reg, area_prec_reg, ex['min_area_km2']/1E5, 'not exceeding area')
                    prec_field[prec_field==reg] = 0
                if area_prec_reg >= ex['min_area_km2']/1E5:
                    Corr_value = mask_and_data.data[prec_field.reshape(-1)==reg, l_idx]
                    weight_by_size = (area_prec_reg / totalsize_lag0)**0.1
                    Corr_value.sort()
                    strength   = abs(np.median(Corr_value[int(0.75*Corr_value.size):]))
                    Corr_strength = np.round(strength * weight_by_size, 10)
                    corr_strength[Corr_strength + l_idx*1E-5] = '{}_{}'.format(l_idx,reg)
            Number_regions_per_lag[l_idx] = np.unique(prec_field)[1:].size
            prec_labels_np[l_idx,:,:] = prec_field
        
        
        # Reorder - strongest correlation region is number 1, etc... ,
        strongest = sorted(corr_strength.keys())[::-1]
        actor.corr_strength = corr_strength
        reassign = {} ; key_dupl = [] ; new_reg = 0 
        order_str_actor = {}
        for i, key in enumerate(strongest):
            old_lag_reg = corr_strength[key]
            old_reg = int(old_lag_reg.split('_')[-1])
            if old_reg not in key_dupl:
                new_reg += 1
                reassign[old_reg] = new_reg
            key_dupl.append( old_reg )
            new_lag_reg = old_lag_reg.split('_')[0] +'_'+ str(reassign[old_reg])
            order_str_actor[new_lag_reg] = i+1
        
        
        actor.order_str_actor = order_str_actor
        prec_labels_ord = np.zeros(prec_labels_np.shape, dtype=int)
        for i, reg in enumerate(reassign.keys()):   
            prec_labels_ord[prec_labels_np == reg] = reassign[reg]
    #        print('reg {}, to {}'.format(reg, reassign[reg]))
#                

    
    
    actor.n_regions_lag = Number_regions_per_lag
    ex['n_tot_regs']    += int(np.sum(Number_regions_per_lag))
    
    lags = list(range(ex['lag_min'], ex['lag_max']+1))
    lags_coord = ['{} ({} {})'.format(l, l*ex['tfreq'], ex['input_freq'][:1]) for l in lags]
    prec_labels = xr.DataArray(data=prec_labels_ord, coords=[lags_coord, lats, lons], 
                          dims=['lag','latitude','longitude'], 
                          name='{}_labels_init'.format(actor.name), 
                          attrs={'units':'Precursor regions [ordered for Corr strength]'})
    prec_labels = prec_labels.where(prec_labels_ord!=0.)
    prec_labels.attrs['title'] = prec_labels.name
    actor.prec_labels = prec_labels
    #%%
    return actor, ex 
    
    
def plot_regs_xarray(for_plt, ex):
        
    ex['max_N_regs'] = min(20, int(for_plt.max() + 0.5))
    label_weak = np.nan_to_num(for_plt.values) >=  ex['max_N_regs']
    for_plt.values[label_weak] = ex['max_N_regs']


    adjust_vert_cbar = 0.0 ; adj_fig_h = 1.4
    
        
    cmap = plt.cm.tab20
    for_plt.values = for_plt.values-0.5
#    if np.unique(for_plt.values[~np.isnan(for_plt.values)]).size == 1:
#        for_plt[0,0,0] = 0
    kwrgs = dict( {'title' : for_plt.attrs['title'], 'clevels' : 'notdefault', 
                   'steps' : ex['max_N_regs']+1, 'subtitles': None,
                   'vmin' : 0, 'vmax' : ex['max_N_regs'], 
                   'cmap' : cmap, 'column' : 1,
                   'cbar_vert' : adjust_vert_cbar, 'cbar_hght' : 0.0,
                   'adj_fig_h' : adj_fig_h, 'adj_fig_w' : 1., 
                   'hspace' : 0.0, 'wspace' : 0.08, 
                   'cticks_center' : False, 'title_h' : 0.95} )
    filename = '{}_{}_vs_{}'.format(ex['params'], ex['RV_name'], for_plt.name) + ex['file_type2']
    plotting_wrapper(for_plt, ex, filename, kwrgs=kwrgs)

    #%%


def spatial_mean_regions(actor, ex):
    #%%
    
    #    var = 'sst'
#    actor = outdic_actors[var]
    var             = actor.name
    Corr_Coeff      = actor.Corr_Coeff
    lats            = actor.lat_grid
    lons            = actor.lon_grid
    lags = np.arange(ex['lag_min'], ex['lag_max']+1E-9, dtype=int)
    lag_steps = lags.size
    n_time          = actor.precur_arr.time.size
    order_str       = actor.order_str_actor
    actbox = np.reshape(actor.precur_arr.values, (n_time, 
                  lats.size*lons.size))  

#    print(actor.prec_labels.squeeze().values[~np.isnan(actor.prec_labels.squeeze().values)])
    
    
    ts_list = np.zeros( (lag_steps), dtype=list )
    track_names = []
    for l_idx, lag in enumerate(lags):
        prec_labels_lag = actor.prec_labels.isel(lag=l_idx).values
        
        if prec_labels_lag.shape == actbox[0].shape:
            Regions = prec_labels_lag
        elif prec_labels_lag.shape == (lats.shape[0], lons.shape[0]):
            Regions = np.reshape(prec_labels_lag, (prec_labels_lag.size))
        
        
        regions_for_ts = [int(key.split('_')[1]) for key in order_str.keys() if int(key.split('_')[0]) ==l_idx]
        
        # get lonlat array of area for taking spatial means 
        lons_gph, lats_gph = np.meshgrid(lons, lats)
        cos_box = np.cos(np.deg2rad(lats_gph))
        cos_box_array = np.repeat(cos_box[None,:], actbox.shape[0], 0)
        cos_box_array = np.reshape(cos_box_array, (cos_box_array.shape[0], -1))
        
    
        # this array will be the time series for each feature
        ts_regions_lag_i = np.zeros((actbox.shape[0], len(regions_for_ts)))
        
        # track sign of eacht region
        sign_ts_regions = np.zeros( len(regions_for_ts) )
        
        
        # calculate area-weighted mean over features
        for r in regions_for_ts:
            track_names.append(f'{lag}_{r}_{var}')
            idx = regions_for_ts.index(r)
            # start with empty lonlat array
            B = np.zeros(Regions.shape)
            # Mask everything except region of interest
            B[Regions == r] = 1	
    #        # Calculates how values inside region vary over time, wgts vs anomaly
    #        wgts_ano = meanbox[B==1] / meanbox[B==1].max()
    #        ts_regions_lag_i[:,idx] = np.nanmean(actbox[:,B==1] * cos_box_array[:,B==1] * wgts_ano, axis =1)
            # Calculates how values inside region vary over time
            ts_regions_lag_i[:,idx] = np.nanmean(actbox[:,B==1] * cos_box_array[:,B==1], axis =1)
            # get sign of region
            sign_ts_regions[idx] = np.sign(np.mean(Corr_Coeff.data[B==1]))
        ts_list[l_idx] = ts_regions_lag_i
    tsCorr = np.concatenate(tuple(ts_list), axis = 1)
    
        
    #%%
    return tsCorr, track_names


def print_particular_region_new(ex, number_region, corr_lag, prec_labels, map_proj, title):
#    (number_region, Corr_Coeff_lag_i, latitudes, longitudes, map_proj, title)=(according_number, Corr_precursor[:, :], actor.lat_grid, actor.lon_grid, map_proj, according_fullname) 
    #%%
    # check if only one lag is tested:

#    lag_steps = prec_labels.lag.size

#    latitudes = actor.lat_grid
#    longitudes = actor.lon_grid
    

    for_plt = prec_labels.where(prec_labels.values==number_region).isel(lag=corr_lag)

    map_proj = map_proj
    fig = plt.figure(figsize=(6, 4))
    ax = plt.axes(projection=map_proj)
    im = for_plt.plot.pcolormesh(ax=ax, cmap=plt.cm.BuPu,
                     transform=ccrs.PlateCarree(), add_colorbar=False)
    plt.colorbar(im, ax=ax , orientation='horizontal')
    ax.coastlines(color='grey', alpha=0.3)
    ax.set_title(title)
        
#%%
    return fig

		
def plotting_wrapper(plotarr, ex, filename=None,  kwrgs=None):
    import os

    try:
        folder_name = os.path.join(ex['figpathbase'], ex['exp_folder'])
    except:
        folder_name = ex['fig_path']
        
    if os.path.isdir(folder_name) != True : 
        os.makedirs(folder_name)

    if kwrgs == None:
        kwrgs = dict( {'title' : plotarr.name, 'clevels' : 'notdefault', 'steps':17,
                        'vmin' : -3*plotarr.std().values, 'vmax' : 3*plotarr.std().values, 
                       'cmap' : plt.cm.RdBu_r, 'column' : 1, 'subtitles' : None} )
    else:
        kwrgs = kwrgs
        kwrgs['title'] = plotarr.attrs['title']
        
    if filename != None:
        file_name = os.path.join(folder_name, filename)
        kwrgs['savefig'] = True
    else:
        kwrgs['savefig'] = False
        file_name = 'Users/semvijverberg/Downloads/test.png'
    finalfigure(plotarr, file_name, kwrgs)
    

def finalfigure(xrdata, file_name, kwrgs):
    #%%
    import cartopy.feature as cfeature
    from shapely.geometry.polygon import LinearRing
    import cartopy.mpl.ticker as cticker
    import matplotlib as mpl
    
    map_proj = ccrs.PlateCarree(central_longitude=220)  
    lons = xrdata.longitude.values
    lats = xrdata.latitude.values
    strvars = [' {} '.format(var) for var in list(xrdata.dims)]
    var = [var for var in strvars if var not in ' longitude latitude '][0] 
    var = var.replace(' ', '')
    g = xr.plot.FacetGrid(xrdata, col=var, col_wrap=kwrgs['column'], sharex=True,
                      sharey=True, subplot_kws={'projection': map_proj},
                      aspect= (xrdata.longitude.size) / xrdata.latitude.size, size=3.5)
    figwidth = g.fig.get_figwidth() ; figheight = g.fig.get_figheight()

    lon_tick = xrdata.longitude.values
    dg = abs(lon_tick[1] - lon_tick[0])
    periodic = (np.arange(0, 360, dg).size - lon_tick.size) < 1
    
    longitude_labels = np.linspace(np.min(lon_tick), np.max(lon_tick), 6, dtype=int)
    longitude_labels = np.array(sorted(list(set(np.round(longitude_labels, -1)))))

#    longitude_labels = np.concatenate([ longitude_labels, [longitude_labels[-1]], [180]])
#    longitude_labels = [-150,  -70,    0,   70,  140, 140]
    latitude_labels = np.linspace(xrdata.latitude.min(), xrdata.latitude.max(), 4, dtype=int)
    latitude_labels = sorted(list(set(np.round(latitude_labels, -1))))
    
    g.set_ticks(max_xticks=5, max_yticks=5, fontsize='small')
    g.set_xlabels(label=[str(el) for el in longitude_labels])

    
    if kwrgs['clevels'] == 'default':
        vmin = np.round(float(xrdata.min())-0.01,decimals=2) ; vmax = np.round(float(xrdata.max())+0.01,decimals=2)
        clevels = np.linspace(-max(abs(vmin),vmax),max(abs(vmin),vmax),17) # choose uneven number for # steps
    else:
        vmin=kwrgs['vmin']
        vmax=kwrgs['vmax']
        
        clevels = np.linspace(vmin, vmax,kwrgs['steps'])

    cmap = kwrgs['cmap']
    
    n_plots = xrdata[var].size
    for n_ax in np.arange(0,n_plots):
        ax = g.axes.flatten()[n_ax]
#        print(n_ax)
        if periodic == True:
            plotdata = wrapper_RGCPD_tig.extend_longitude(xrdata[n_ax])
        else:
            plotdata = xrdata[n_ax].squeeze()
        im = plotdata.plot.pcolormesh(ax=ax, cmap=cmap,
                               transform=ccrs.PlateCarree(),
                               subplot_kws={'projection': map_proj},
                                levels=clevels, add_colorbar=False)
        ax.coastlines(color='black', alpha=0.3, facecolor='grey')
        ax.add_feature(cfeature.LAND, facecolor='grey', alpha=0.1)
        
        ax.set_extent([lons[0], lons[-1], lats[0], lats[-1]], ccrs.PlateCarree())
        if kwrgs['subtitles'] == None:
            pass
        else:
            fontdict = dict({'fontsize'     : 18,
                             'fontweight'   : 'bold'})
            ax.set_title(kwrgs['subtitles'][n_ax], fontdict=fontdict, loc='center')
        
        if 'drawbox' in kwrgs.keys():
            lons_sq = [-215, -215, -130, -130] #[-215, -215, -125, -125] #[-215, -215, -130, -130] 
            lats_sq = [50, 20, 20, 50]
            ring = LinearRing(list(zip(lons_sq , lats_sq )))
            ax.add_geometries([ring], ccrs.PlateCarree(), facecolor='none', edgecolor='green',
                              linewidth=3.5)
        
        if 'ax_text' in kwrgs.keys():
            ax.text(0.0, 1.01, kwrgs['ax_text'][n_ax],
            verticalalignment='bottom', horizontalalignment='left',
            transform=ax.transAxes,
            color='black', fontsize=15)
            
        if map_proj.proj4_params['proj'] in ['merc', 'eqc']:
#            print(True)
            ax.set_xticks(longitude_labels[:-1], crs=ccrs.PlateCarree())
            ax.set_xticklabels(longitude_labels[:-1], fontsize=12)
            lon_formatter = cticker.LongitudeFormatter()
            ax.xaxis.set_major_formatter(lon_formatter)
            
            ax.set_yticks(latitude_labels, crs=ccrs.PlateCarree())
            ax.set_yticklabels(latitude_labels, fontsize=12)
            lat_formatter = cticker.LatitudeFormatter()
            ax.yaxis.set_major_formatter(lat_formatter)
            ax.grid(linewidth=1, color='black', alpha=0.3, linestyle='--')
            ax.set_xlabel('')
            ax.set_ylabel('')
            
            
        else:
            pass
        
    if 'title_h' in kwrgs.keys():
        title_height = kwrgs['title_h']
    else:
        title_height = 0.98
    g.fig.text(0.5, title_height, kwrgs['title'], fontsize=20,
               fontweight='heavy', transform=g.fig.transFigure,
               horizontalalignment='center',verticalalignment='top')
    
    if 'adj_fig_h' in kwrgs.keys():
        g.fig.set_figheight(figheight*kwrgs['adj_fig_h'], forward=True)
    if 'adj_fig_w' in kwrgs.keys():
        g.fig.set_figwidth(figwidth*kwrgs['adj_fig_w'], forward=True)

    if 'cbar_vert' in kwrgs.keys():
        cbar_vert = (figheight/40)/(n_plots*2) + kwrgs['cbar_vert']
    else:
        cbar_vert = (figheight/40)/(n_plots*2)
    if 'cbar_hght' in kwrgs.keys():
        cbar_hght = (figheight/40)/(n_plots*2) + kwrgs['cbar_hght']
    else:
        cbar_hght = (figheight/40)/(n_plots*2)
    if 'wspace' in kwrgs.keys():
        g.fig.subplots_adjust(wspace=kwrgs['wspace'])
    if 'hspace' in kwrgs.keys():
        g.fig.subplots_adjust(hspace=kwrgs['hspace'])
    if 'extend' in kwrgs.keys():
        extend = kwrgs['extend'][0]
    else:
        extend = 'neither'

    # new cbar positioning
    y0 = ax.figbox.bounds[1]
    cbar_ax = g.fig.add_axes([0.25, y0-0.05+kwrgs['cbar_vert'], 
                                  0.5, cbar_hght], label='cbar')

    if 'clim' in kwrgs.keys(): #adjust the range of colors shown in cbar
        cnorm = np.linspace(kwrgs['clim'][0],kwrgs['clim'][1],11)
        vmin = kwrgs['clim'][0]
    else:
        cnorm = clevels

    norm = mpl.colors.BoundaryNorm(boundaries=cnorm, ncolors=256)
#    cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, orientation='horizontal', 
#                 extend=extend, ticks=cnorm, norm=norm)

    cbar = plt.colorbar(im, cbar_ax, cmap=cmap, orientation='horizontal', 
                 extend=extend, norm=norm)

    if 'cticks_center' in kwrgs.keys():
        cbar = plt.colorbar(im, cbar_ax, cmap=cmap, orientation='horizontal', 
                 extend=extend, norm=norm)
        cbar.set_ticks(clevels + 0.5)
        ticklabels = np.array(clevels+1, dtype=int)
        cbar.set_ticklabels(ticklabels, update_ticks=True)
        cbar.update_ticks()
    
    if 'extend' in kwrgs.keys():
        if kwrgs['extend'][0] == 'min':
            cbar.cmap.set_under(cbar.to_rgba(kwrgs['vmin']))
    cbar.set_label(xrdata.attrs['units'], fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    #%%
    if kwrgs['savefig'] != False:
        g.fig.savefig(file_name ,dpi=250, frameon=True)
    
    return



