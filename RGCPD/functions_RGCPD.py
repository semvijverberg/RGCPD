# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib
matplotlib.rcParams['backend'] = "Qt4Agg"
from pylab import *
import matplotlib.pyplot as plt
import wrapper_RGCPD_tig
import functions_pp
import func_fc
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

def func_dates_min_lag(dates, lag):
    dates_min_lag = pd.to_datetime(dates.values) - pd.Timedelta(int(lag), unit='d')
    ### exlude leap days from dates_train_min_lag ###


    # ensure that everything before the leap day is shifted one day back in time
    # years with leapdays now have a day less, thus everything before
    # the leapday should be extended back in time by 1 day.
    mask_lpyrfeb = np.logical_and(dates_min_lag.month == 2,
                                         dates_min_lag.is_leap_year
                                         )
    mask_lpyrjan = np.logical_and(dates_min_lag.month == 1,
                                         dates_min_lag.is_leap_year
                                         )
    mask_ = np.logical_or(mask_lpyrfeb, mask_lpyrjan)
    new_dates = np.array(dates_min_lag)
    new_dates[mask_] = dates_min_lag[mask_] - pd.Timedelta(1, unit='d')
    dates_min_lag = pd.to_datetime(new_dates)
    # to be able to select date in pandas dataframe
    dates_min_lag_str = [d.strftime('%Y-%m-%d %H:%M:%S') for d in dates_min_lag]
    return dates_min_lag_str, dates_min_lag

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
    n_lags = len(ex['lags_i'])
    tfreq = ex['tfreq']
    lags = ex['lags']
    assert n_lags >= 0, ('Maximum lag is larger then minimum lag, not allowed')


    traintest, ex = functions_pp.rand_traintest_years(RV, precur_arr, ex)
    # make new xarray to store results
    xrcorr = precur_arr.isel(time=0).drop('time').copy()
    # add lags
    list_xr = [xrcorr.expand_dims('lag', axis=0) for i in range(n_lags)]
    xrcorr = xr.concat(list_xr, dim = 'lag')
    xrcorr['lag'] = ('lag', lags)
    # add train test split
    list_xr = [xrcorr.expand_dims('split', axis=0) for i in range(ex['n_spl'])]
    xrcorr = xr.concat(list_xr, dim = 'split')
    xrcorr['split'] = ('split', range(ex['n_spl']))

    print('\n{} - calculating correlation maps'.format(precur_arr.name))
    np_data = np.zeros_like(xrcorr.values)
    np_mask = np.zeros_like(xrcorr.values)
    def corr_single_split(RV_ts, precur, RV_period, ex):

        lat = precur_arr.latitude.values
        lon = precur_arr.longitude.values

        z = np.zeros((lat.size*lon.size,len(ex['lags']) ) )
        Corr_Coeff = np.ma.array(z, mask=z)
        # reshape
#        sat = np.reshape(precur_arr.values, (precur_arr.shape[0],-1))

        dates_RV = pd.to_datetime(RV_ts.time.values)
        for i, lag in enumerate(ex['lags']):

            dates_lag = func_dates_min_lag(dates_RV, lag)[1]
            prec_lag = precur_arr.sel(time=dates_lag)
            prec_lag = np.reshape(prec_lag.values, (prec_lag.shape[0],-1))


    		# correlation map and pvalue at each grid-point:
            corr_di_sat, sig_di_sat = corr_new(prec_lag, RV_ts)

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
        progress = 100 * (s+1) / ex['n_spl']
        # =============================================================================
        # Split train test methods ['random'k'fold', 'leave_'k'_out', ', 'no_train_test_split']
        # =============================================================================
        RV_ts = traintest[s]['RV_train']
        precur = precur_arr.isel(time=traintest[s]['Prec_train_idx'])
        dates_RV  = pd.to_datetime(RV_ts.time.values)
        n = dates_RV.size ; r = int(100*n/RV.dates_RV.size )
        print(f"\rProgress traintest set {progress}%, trainsize=({n}dp, {r}%)", end="")
        dates_all = pd.to_datetime(precur.time.values)
        string_RV = list(dates_RV.strftime('%Y-%m-%d'))
        string_full = list(dates_all.strftime('%Y-%m-%d'))
        RV_period = [string_full.index(date) for date in string_full if date in string_RV]

        ma_data = corr_single_split(RV_ts, precur, RV_period, ex)
        np_data[s] = ma_data.data
        np_mask[s] = ma_data.mask
    print("\n")
    xrcorr.values = np_data
    mask = (('split', 'lag', 'latitude', 'longitude'), np_mask )
    xrcorr.coords['mask'] = mask
    #%%
    return xrcorr

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

def cluster_DBSCAN_regions(actor, ex):
    #%%
    """
	Calculates the time-series of the actors based on the correlation coefficients and plots the according regions.
	Only caluclates regions with significant correlation coefficients
	"""
    import xarray as xr

#    var = 'sst'
#    actor = outdic_actors[var]
    corr_xr  = actor.corr_xr
    n_spl  = corr_xr.coords['split'].size
    lats    = corr_xr.latitude
    lons    = corr_xr.longitude
    area_grid   = actor.area_grid/ 1E6 # in km2

    aver_area_km2 = 7939     # np.mean(actor.area_grid) with latitude 0-90 / 1E6
    wght_area = area_grid / aver_area_km2
    ex['min_area_km2'] = ex['min_area_in_degrees2'] * 111.131 * ex['min_area_in_degrees2'] * 78.85
    min_area = ex['min_area_km2'] / aver_area_km2
    ex['min_area_samples'] = min_area


    lags = ex['lags']
    n_lags = lags.size


    ex['n_tot_regs'] = 0


    prec_labels_np = np.zeros( (n_spl, n_lags, lats.size, lons.size), dtype=int )
    labels_sign_lag = np.zeros( (n_spl), dtype=list)
    mask_and_data = corr_xr.copy()

    # group regions per split (no information leak train test)
    if ex['group_split'] == 'seperate':
        for s in range(n_spl):
            progress = 100 * (s+1) / ex['n_spl']
            print(f"\rProgress traintest set {progress}%", end="")
            mask_and_data_s = corr_xr.sel(split=s)
            grouping_split = mask_sig_to_cluster(mask_and_data_s, wght_area, ex)
            prec_labels_np[s] = grouping_split[0]
            labels_sign_lag[s] = grouping_split[1]
    # group regions regions the same accross splits
    elif ex['group_split'] == 'together':

        mask = abs(mask_and_data.mask -1)
        mask_all = mask.sum(dim='split') / mask.sum(dim='split')
        m_d_together = mask_and_data.isel(split=0).copy()
        m_d_together['mask'] = abs(mask_all.fillna(0) -1)
        m_d_together.values = np.sign(mask_and_data).mean(dim='split')
        grouping_split = mask_sig_to_cluster(m_d_together, wght_area, ex)

        for s in range(n_spl):
            m_all = grouping_split[0].copy()
            mask_split = corr_xr.sel(split=s).mask
            m_all[mask_split.astype('bool').values] = 0
            labs = np.unique(mask_split==0)[1:]
            l_sign = grouping_split[1]
            labs_s = [t for t in l_sign if t[0] in labs]
            prec_labels_np[s] = m_all
            labels_sign_lag[s] = labs_s


    if np.nansum(prec_labels_np) == 0. and mask_and_data.mask.all()==False:
        print('\nSome significantly correlating gridcells found, but too randomly located and '
              'interpreted as noise by DBSCAN, make distance_eps lower '
              'to relax contrain.\n')
        prec_labels_ord = prec_labels_np
    if mask_and_data.mask.all()==True:
        print('\nNo significantly correlating gridcells found.\n')
        prec_labels_ord = prec_labels_np
    else:
        prec_labels_ord = np.zeros_like(prec_labels_np)
        if ex['group_split'] == 'seperate':
            for s in range(n_spl):
                prec_labels_s = prec_labels_np[s]
                corr_vals     = corr_xr.sel(split=s).values
                reassign = reorder_strength(prec_labels_s, corr_vals, area_grid, ex)
                prec_labels_ord[s] = relabel(prec_labels_s, reassign)
        elif ex['group_split'] == 'together':
            # order based on mean corr_value:
            corr_vals = corr_xr.mean(dim='split').values
            prec_label_s = grouping_split[0].copy()
            reassign = reorder_strength(prec_label_s, corr_vals, area_grid, ex)
            for s in range(n_spl):
                prec_labels_s = prec_labels_np[s]
                prec_labels_ord[s] = relabel(prec_labels_s, reassign)



    ex['n_tot_regs']    += int(np.unique(prec_labels_ord)[1:].size)


#    lags_coord = ['{} ({} {})'.format(l, l*tfreq, ex['input_freq'][:1]) for l in lags]
    prec_labels = xr.DataArray(data=prec_labels_ord, coords=[range(n_spl), lags, lats, lons],
                          dims=['split', 'lag','latitude','longitude'],
                          name='{}_labels_init'.format(actor.name),
                          attrs={'units':'Precursor regions [ordered for Corr strength]'})
    prec_labels = prec_labels.where(prec_labels_ord!=0.)
    prec_labels.attrs['title'] = prec_labels.name
    actor.prec_labels = prec_labels
    #%%
    return actor, ex

def mask_sig_to_cluster(mask_and_data_s, wght_area, ex):
    from sklearn import cluster
    from sklearn import metrics
    from haversine import haversine
    mask_sig_1d = mask_and_data_s.mask.astype('bool').values == False
    data = mask_and_data_s.data
    lons = mask_and_data_s.longitude.values
    lats = mask_and_data_s.latitude.values
    n_lags = mask_and_data_s.lag.size

    np_dbregs   = np.zeros( (n_lags, lats.size, lons.size), dtype=int )
    labels_sign_lag = []
    label_start = 0

    for sign in [-1, 1]:
        mask = mask_sig_1d.copy()
        mask[np.sign(data) != sign] = False
        n_gc_sig_sign = mask[mask==True].size
        labels_for_lag = np.zeros( (n_lags, n_gc_sig_sign), dtype=bool)
        meshgrid = np.meshgrid(lons.data, lats.data)
        mask_sig = np.reshape(mask, (n_lags, lats.size, lons.size))
        sign_coords = [] ; count=0
        weights_core_samples = []
        for l in range(n_lags):
            sign_c = meshgrid[0][ mask_sig[l,:,:] ], meshgrid[1][ mask_sig[l,:,:] ]
            n_sign_c_lag = len(sign_c[0])
            labels_for_lag[l][count:count+n_sign_c_lag] = True
            count += n_sign_c_lag
            # shape sign_coords = [(lats, lons)]
            sign_coords.append( [(sign_c[1][i], sign_c[0][i]-180) for i in range(sign_c[0].size)] )
            weights_core_samples.append(wght_area[mask_sig[l,:,:]].reshape(-1))

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

            for l in range(n_lags):
                mask_sig_lag = mask[l,:,:]==True
                np_dbregs[l,:,:][mask_sig_lag] = individual_labels[labels_for_lag[l]]
            label_start = int(np_dbregs[mask].max())
        else:
            pass
        np_regs = np.array(np_dbregs, dtype='int')
    return np_regs, labels_sign_lag

def reorder_strength(prec_labels_s, corr_vals, area_grid, ex):
    #%%
    # order regions on corr strength
    # based on median of upper 25 percentile

    Number_regions_per_lag = np.zeros(len(ex['lags']))
    corr_strength = {}
    totalsize_lag0 = area_grid[prec_labels_s[0]!=0].mean() / 1E5
    for l_idx, lag in enumerate(ex['lags']):
        # check if region is higher lag is actually too small to be a cluster:
        prec_field = prec_labels_s[l_idx,:,:]

        for i, reg in enumerate(np.unique(prec_field)[1:]):
            are = area_grid.copy()
            are[prec_field!=reg]=0
            area_prec_reg = are.sum()/1E5
            if area_prec_reg < ex['min_area_km2']/1E5:
                print(reg, area_prec_reg, ex['min_area_km2']/1E5, 'not exceeding area')
                prec_field[prec_field==reg] = 0
            if area_prec_reg >= ex['min_area_km2']/1E5:
                Corr_value = corr_vals[l_idx, prec_field==reg]
                weight_by_size = (area_prec_reg / totalsize_lag0)**0.1
                Corr_value.sort()
                strength   = abs(np.median(Corr_value[int(0.75*Corr_value.size):]))
                Corr_strength = np.round(strength * weight_by_size, 10)
                corr_strength[Corr_strength + l_idx*1E-5] = '{}_{}'.format(l_idx,reg)
        Number_regions_per_lag[l_idx] = np.unique(prec_field)[1:].size
        prec_labels_s[l_idx,:,:] = prec_field


    # Reorder - strongest correlation region is number 1, etc... ,
    strongest = sorted(corr_strength.keys())[::-1]
#    actor.corr_strength = corr_strength
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
    return reassign

#    actor.order_str_actor = order_str_actor
    #%%
    return reassign

def relabel(prec_labels_s, reassign):
    prec_labels_ord = np.zeros(prec_labels_s.shape, dtype=int)
    for i, reg in enumerate(reassign.keys()):
        prec_labels_ord[prec_labels_s == reg] = reassign[reg]
    return prec_labels_ord


def spatial_mean_regions(actor, ex):
    #%%

    var             = actor.name
    corr_xr         = actor.corr_xr
    prec_labels     = actor.prec_labels
    n_spl           = corr_xr.split.size
    lags = ex['lags']

    actbox = actor.precur_arr.values
    ts_corr = np.zeros( (n_spl), dtype=object)

    for s in range(n_spl):
        corr = corr_xr.isel(split=0)
        labels = prec_labels.isel(split=0)

        ts_list = np.zeros( (lags.size), dtype=list )
        track_names = []
        for l_idx, lag in enumerate(lags):
            labels_lag = labels.isel(lag=l_idx).values

            regions_for_ts = list(np.unique(labels_lag[~np.isnan(labels_lag)]))
            a_wghts = actor.area_grid / actor.area_grid.mean()

            # this array will be the time series for each feature
            ts_regions_lag_i = np.zeros((actbox.shape[0], len(regions_for_ts)))

            # track sign of eacht region
            sign_ts_regions = np.zeros( len(regions_for_ts) )


            # calculate area-weighted mean over features
            for r in regions_for_ts:
                track_names.append(f'{lag}_{int(r)}_{var}')
                idx = regions_for_ts.index(r)
                # start with empty lonlat array
                B = np.zeros(labels_lag.shape)
                # Mask everything except region of interest
                B[labels_lag == r] = 1
        #        # Calculates how values inside region vary over time, wgts vs anomaly
        #        wgts_ano = meanbox[B==1] / meanbox[B==1].max()
        #        ts_regions_lag_i[:,idx] = np.nanmean(actbox[:,B==1] * cos_box_array[:,B==1] * wgts_ano, axis =1)
                # Calculates how values inside region vary over time
                ts_regions_lag_i[:,idx] = np.nanmean(actbox[:,B==1] * a_wghts[B==1], axis =1)
                # get sign of region
                sign_ts_regions[idx] = np.sign(np.mean(corr.isel(lag=l_idx).values[B==1]))
            ts_list[l_idx] = ts_regions_lag_i

        tsCorr = np.concatenate(tuple(ts_list), axis = 1)
        df_tscorr = pd.DataFrame(tsCorr, index=pd.to_datetime(actor.precur_arr.time.values),
                                 columns=track_names)
        df_tscorr.name = str(s)
        ts_corr[s] = df_tscorr
    #%%
    return ts_corr


def get_spatcovs(dict_ds, df_split, s, outdic_actors, normalize=True):
    #%%

    lag = 0
    TrainIsTrue = df_split['TrainIsTrue']
    times = df_split.index
#    n_time = times.size
    options = ['_spatcov', '_spatcov_caus']
    columns = []
    for var in outdic_actors.keys():
        for select in options:
            columns.append(var+select)


    data = np.zeros( (len(columns), times.size) )
    df_sp_s = pd.DataFrame(data.T, index=times, columns=columns)
    dates_train = TrainIsTrue[TrainIsTrue.values].index
#    dates_test  = TrainIsTrue[TrainIsTrue.values==False].index
    for var, actor in outdic_actors.items():
        ds = dict_ds[var]
        for i, select in enumerate(['_labels', '_labels_tigr']):
            # spat_cov over test years using corr fields from training

            full_timeserie = actor.precur_arr
            corr_vals = ds[var + "_corr"][0]
            mask = ds[var + select].sel(split=s).isel(lag=lag)
            pattern = corr_vals.where(~np.isnan(mask))
            if np.isnan(pattern.values).all():
                # no regions of this variable and split
                pass
            else:
                if normalize == True:
                    spatcov_full = calc_spatcov(full_timeserie, pattern)
                    mean = spatcov_full.sel(time=dates_train).mean(dim='time')
                    std = spatcov_full.sel(time=dates_train).std(dim='time')
                    spatcov_test = ((spatcov_full - mean) / std)
                elif normalize == False:
                    spatcov_test = calc_spatcov(full_timeserie, pattern)
                pd_sp = pd.Series(spatcov_test.values, index=times)
                col = options[i]
                df_sp_s[var + col] = pd_sp
    for i, bckgrnd in enumerate(['tcov', 'caus']):
        cols = [col for col in df_sp_s.columns if col[-4:] == bckgrnd]
        key = options[i]
        df_sp_s['all'+key] = df_sp_s[cols].mean(axis=1)
    #%%
    return df_sp_s


def add_sp_info(df_sum, df_sp):
    #%%
    splits = df_sum.index.levels[0]
    cols = df_sum.loc[0].columns
    df_sum_new = np.zeros( (splits.size), dtype=object)
    for s in splits:

        df_sum_s = df_sum.loc[s]
        df_sp_s = df_sp.loc[s]
        keys = list(df_sp_s.columns)
        data = np.zeros( (len(keys), len(cols)) )
        data[:,:] = 999
        data[:,-2] = [True if k[-4:]=='caus' else False for k in keys]
        data[:,-1] = np.nan
        df_append = pd.DataFrame(data=data, index=keys, columns = cols)
        df_merged = pd.concat([df_sum_s, df_append])
        df_merged = df_merged.astype({'label':int, 'lag_corr':int,
                               'region_number':int, 'var':str,
                               'causal':bool, 'lag_tig':float})
        df_sum_new[s] = df_merged

    return pd.concat(list(df_sum_new), keys= range(splits.size))

def calc_spatcov(full_timeserie, pattern):
#%%
#    full_timeserie = var_train_reg
#    pattern = ds_Sem['pattern_CPPA'].sel(lag=lag)


#    # trying to matching dimensions
#    if pattern.shape != full_timeserie[0].shape:
#        try:
#            full_timeserie = full_timeserie.sel(latitude=pattern.latitude)
#        except:
#            pattern = pattern.sel(latitude=full_timeserie.latitude)


    mask = np.ma.make_mask(np.isnan(pattern.values)==False)

    n_time = full_timeserie.time.size
    n_space = pattern.size


#    mask_pattern = np.tile(mask_pattern, (n_time,1))
    # select only gridcells where there is not a nan
    full_ts = np.nan_to_num(np.reshape( full_timeserie.values, (n_time, n_space) ))
    pattern = np.nan_to_num(np.reshape( pattern.values, (n_space) ))

    mask_pattern = np.reshape( mask, (n_space) )
    full_ts = full_ts[:,mask_pattern]
    pattern = pattern[mask_pattern]

#    crosscorr = np.zeros( (n_time) )
    spatcov   = np.zeros( (n_time) )
#    covself   = np.zeros( (n_time) )
#    corrself  = np.zeros( (n_time) )
    for t in range(n_time):
        # Corr(X,Y) = cov(X,Y) / ( std(X)*std(Y) )
        # cov(X,Y) = E( (x_i - mu_x) * (y_i - mu_y) )
#        crosscorr[t] = np.correlate(full_ts[t], pattern)
        M = np.stack( (full_ts[t], pattern) )
        spatcov[t] = np.cov(M)[0,1] #/ (np.sqrt(np.cov(M)[0,0]) * np.sqrt(np.cov(M)[1,1]))
#        sqrt( Var(X) ) = sigma_x = std(X)
#        spatcov[t] = np.cov(M)[0,1] / (np.std(full_ts[t]) * np.std(pattern))
#        covself[t] = np.mean( (full_ts[t] - np.mean(full_ts[t])) * (pattern - np.mean(pattern)) )
#        corrself[t] = covself[t] / (np.std(full_ts[t]) * np.std(pattern))
    dates_test = full_timeserie.time
#    corrself = xr.DataArray(corrself, coords=[dates_test.values], dims=['time'])

#    # standardize
#    corrself -= corrself.mean(dim='time', skipna=True)

    # cov xarray
    spatcov = xr.DataArray(spatcov, coords=[dates_test.values], dims=['time'])
#%%
    return spatcov

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

def bookkeeping_precursors(links_RV, var_names):
    #%%
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

    l = [n[1].split('_')[-1] for n in var_names_[1:]]
    l.insert(0, var_names_[0])
    var = np.array(l)
    mask_causal = np.array([True if i in link_names else False for i in index])
    lag_corr_map = np.array([int(n[1][0]) for n in var_names_[1:]]) ;
    lag_corr_map = np.insert(lag_corr_map, 0, 0)
    region_number = np.array([int(n[0]) for n in var_names_[1:]])
    region_number = np.insert(region_number, 0, 0)
    label = np.array([int(n[1].split('_')[1]) for n in var_names_[1:]])
    label = np.insert(label, 0, 0)
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

    data = np.concatenate([label[None,:],  lag_corr_map[None,:], region_number[None,:], var[None,:],
                            mask_causal[None,:], lag_tigr_[None,:]], axis=0)
    df = pd.DataFrame(data=data.T, index=index,
                      columns=['label', 'lag_corr', 'region_number', 'var', 'causal', 'lag_tig'])
    df['causal'] = df['causal'] == 'True'
    df = df.astype({'label':int, 'lag_corr':int,
                               'region_number':int, 'var':str, 'causal':bool, 'lag_tig':float})
    #%%
    print("\n\n")
    print(df)
    return df

def print_particular_region_new(links_RV, var_names, s, outdic_actors, map_proj, ex):

    #%%
    n_parents = len(links_RV)

    for i in range(n_parents):
        tigr_lag = links_RV[i][1] #-1 There was a minus, but is it really correct?
        index_in_fulldata = links_RV[i][0]
        print("\n\nunique_label_format: \n\'lag\'_\'regionlabel\'_\'var\'")
        if index_in_fulldata>0:
            uniq_label = var_names[index_in_fulldata][1]
            var_name = uniq_label.split('_')[-1]
            according_varname = uniq_label
            according_number = int(float(uniq_label.split('_')[1]))
#            according_var_idx = ex['vars'][0].index(var_name)
            corr_lag = int(uniq_label.split('_')[0])
            print('index in fulldata {}: region: {} at lag {}'.format(
                    index_in_fulldata, uniq_label, tigr_lag))
            # *********************************************************
            # print and save only significant regions
            # *********************************************************
            according_fullname = '{} at lag {} - ts_index_{}'.format(according_varname,
                                  tigr_lag, index_in_fulldata)



            actor = outdic_actors[var_name]
            prec_labels = actor.prec_labels.sel(split=s)

            for_plt = prec_labels.where(prec_labels.values==according_number).isel(lag=corr_lag)

            map_proj = map_proj
            plt.figure(figsize=(6, 4))
            ax = plt.axes(projection=map_proj)
            im = for_plt.plot.pcolormesh(ax=ax, cmap=plt.cm.BuPu,
                             transform=ccrs.PlateCarree(), add_colorbar=False)
            plt.colorbar(im, ax=ax , orientation='horizontal')
            ax.coastlines(color='grey', alpha=0.3)
            ax.set_title(according_fullname)
            fig_file = 's{}_{}{}'.format(s, according_fullname, ex['file_type2'])

            plt.savefig(os.path.join(ex['fig_subpath'], fig_file), dpi=ex['png_dpi'])
#            plt.show()
            plt.close()
            # =============================================================================
            # Print to text file
            # =============================================================================
            print('                                        ')
            # *********************************************************
            # save data
            # *********************************************************
            according_fullname = str(according_number) + according_varname
            name = ''.join([str(index_in_fulldata),'_',uniq_label])

#            print((fulldata[:,index_in_fulldata].size))
            print(name)
        else :
            print('Index itself is also causal parent -> skipped')
            print('*******************              ***************************')

#%%
    return

def plot_regs_xarray(for_plt, ex):
    #%%
    ex['max_N_regs'] = min(20, int(for_plt.max() + 0.5))
    label_weak = np.nan_to_num(for_plt.values) >=  ex['max_N_regs']
    for_plt.values[label_weak] = ex['max_N_regs']


    adjust_vert_cbar = 0.0 ; adj_fig_h = 1.0


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
                   'hspace' : 0.2, 'wspace' : 0.08,
                   'cticks_center' : False, 'title_h' : 1.01} )
    filename = '{}_{}_vs_{}'.format(ex['params'], ex['RV_name'], for_plt.name) + ex['file_type2']

    for l in for_plt.lag.values:
        plotting_wrapper(for_plt.sel(lag=l), ex, filename, kwrgs=kwrgs)
    #%%
    return

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
    plt.tight_layout()


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
        cbar_vert = 0 + kwrgs['cbar_vert']
    else:
        cbar_vert = 0
    if 'cbar_hght' in kwrgs.keys():
        # height colorbor 1/10th of height of subfigure
        cbar_h = g.axes[-1,-1].get_position().height / 10
        cbar_hght = cbar_h + kwrgs['cbar_hght']

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
    cbar_ax = g.fig.add_axes([0.25, -y0 + 0.1*y0,
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
        g.fig.savefig(file_name ,dpi=250, bbox_inches='tight')
    #%%
    return



