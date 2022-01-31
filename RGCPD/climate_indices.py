#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 09:25:11 2019

@author: semvijverberg
"""
import os
import find_precursors
from time import time
import numpy as np
import xarray as xr
import pandas as pd
import functions_pp
import plot_maps
from concurrent.futures import ProcessPoolExecutor
import core_pp


# get indices
def ENSO_34(filepath, df_splits=None, get_ENSO_states: bool=True):
    #%%
#    file_path = '/Users/semvijverberg/surfdrive/Data_era5/input_raw/sst_1979-2018_1_12_daily_2.5deg.nc'
    '''
    See http://www.cgd.ucar.edu/staff/cdeser/docs/deser.sstvariability.annrevmarsci10.pdf
    selbox has format of (lon_min, lon_max, lat_min, lat_max)
    '''

    # if df_splits is None:
    #     seldates = None
    # else:
    #     seldates = df_splits.loc[0].index

#    {'la_min':-5, # select domain in degrees east
#     'la_max':5,
#     'lo_min':-170,
#     'lo_max':-120},

    kwrgs_pp = {'selbox' :  (190, 240, -5, 5),
                'format_lon': 'only_east',
                'seldates': None}


    ds = core_pp.import_ds_lazy(filepath, **kwrgs_pp)
    dates = pd.to_datetime(ds.time.values)
    data = functions_pp.area_weighted(ds).mean(dim=('latitude', 'longitude'))
    df_ENSO = pd.DataFrame(data=data.values,
                           index=dates, columns=['ENSO34'])
    if df_splits is not None:
        splits = df_splits.index.levels[0]
        df_ENSO = pd.concat([df_ENSO]*splits.size, axis=0, keys=splits)

    if get_ENSO_states:
        '''
        From Anderson 2017 - Life cycles of agriculturally relevant ENSO
        teleconnections in North and South America.
        http://doi.wiley.com/10.1002/joc.4916
        mean boreal wintertime (October, November, December) SST anomaly amplitude
        in the Niño 3.4 region exceeded 1 of 2 standard deviation.
        '''
        if hasattr(df_ENSO.index, 'levels'):
            df_ENSO_s = df_ENSO.loc[0]
        else:
            df_ENSO_s = df_ENSO
        dates = df_ENSO_s.index
        df_3monthmean = df_ENSO_s.rolling(3, center=True, min_periods=1).mean()
        std_ENSO = df_3monthmean.std()
        OND, groups = core_pp.get_subdates(dates,
                                           start_end_date=('10-01', '12-31'),
                                           returngroups=True)
        OND_ENSO = df_3monthmean.loc[OND].groupby(groups).mean()
        nino_yrs = OND_ENSO[OND_ENSO>df_3monthmean.mean()+std_ENSO][:].dropna().index #+ 1
        nina_yrs = OND_ENSO[OND_ENSO<df_3monthmean.mean()-std_ENSO][:].dropna().index #+ 1
        neutral = [y for y in OND_ENSO.index if y not in core_pp.flatten([nina_yrs, nino_yrs])]
        states = {}
        for i, d in enumerate(dates):
            if d.year in nina_yrs:
                states[d.year] = -1
            if d.year in neutral:
                states[d.year] = 0
            if d.year in nino_yrs:
                states[d.year] = 1

        cycle_list = []
        for s,v in [('EN', 1), ('LN',-1)]:
            ENSO_cycle = {d.year:0 for d in dates}
            for i, year in enumerate(np.unique(dates.year)):
                # d = dates[1]
                # if states[year] == v:
                #     s = 'EN'
                # elif states[year] == -1:
                #     s = 'LN'
                if states[year] == v:
                    ENSO_cycle[year] = f'{s}0'
                    if year-1 in dates.year and states[year-1] != v:
                        ENSO_cycle[year-1] = f'{s}-1'
                    if year+1 in dates.year and states[year+1] != v:
                        ENSO_cycle[year+1] = f'{s}+1'
            cycle_list.append(ENSO_cycle)

        time_index = pd.to_datetime([f'{y}-01-01' for y in states.keys()])
        df_state = pd.concat([pd.Series(states), pd.Series(cycle_list[0]),
                              pd.Series(cycle_list[1])],
                       axis=1, keys=['state', 'EN_cycle', 'LN_cycle'])
        df_state.index = time_index

        if hasattr(df_ENSO.index, 'levels'): # copy to other traintest splits
            df_state = pd.concat([df_state]*splits.size, keys=splits)

        composites = np.zeros(3, dtype=object)
        for i, yrs in enumerate([nina_yrs, neutral, nino_yrs]):
            composite = [d for d in dates if d.year in yrs]
            composites[i] = ds.sel(time=composite).mean(dim='time')
        composites = xr.concat(composites, dim='state')
        composites['state'] = ['Nina', 'Neutral', 'Nino']

        plot_maps.plot_corr_maps(composites, row_dim='state', hspace=0.5)
        out = df_ENSO, [np.array(nina_yrs),
                         np.array(neutral),
                         np.array(nino_yrs)], df_state
    else:
        out = df_ENSO
    #%%
    return out

    #%%

def PDO_single_split(s, ds_monthly, ds, df_splits):

    splits = df_splits.index.levels[0]
    progress = 100 * (s+1) / splits.size
    dates_train_origtime = df_splits.loc[s]['TrainIsTrue'][df_splits.loc[s]['TrainIsTrue']].index
    dates_test_origtime  = df_splits.loc[s]['TrainIsTrue'][~df_splits.loc[s]['TrainIsTrue']].index

    n = dates_train_origtime.size ; r = int(100*n/df_splits.loc[s].index.size )
    print(f"\rProgress PDO traintest set {progress}%, trainsize=({n}dp, {r}%)", end="")

    # convert Train test year from original time to monthly
    train_yrs = np.unique(dates_train_origtime.year)
    dates_monthly = pd.to_datetime(ds_monthly.time.values)
    dates_all_train = pd.to_datetime([d for d in dates_monthly if d.year in train_yrs])

    PDO_pattern, solver, adjust_sign = get_PDO(ds_monthly.sel(time=dates_all_train))
    data_train = find_precursors.calc_spatcov(ds.sel(time=dates_train_origtime).values,
                                              PDO_pattern.values)
    df_train = pd.DataFrame(data=data_train, index=dates_train_origtime, columns=['PDO'])
    if splits.size > 1:
        data_test = find_precursors.calc_spatcov(ds.sel(time=dates_test_origtime).values,
                                                 PDO_pattern.values)
        df_test = pd.DataFrame(data=data_test, index=dates_test_origtime, columns=['PDO'])
        df = pd.concat([df_test, df_train]).sort_index()
    else:
        df = df_train
    return (df, PDO_pattern)

def PDO(filepath
        , df_splits=None, n_jobs=1):
    #%%
    '''
    PDO is calculated based upon all data points in the training years,
    Subsequently, the PDO pattern is projection on the sst.sel(time=dates_train)
    to enable retrieving the PDO timeseries on a subset on the year.
    It is similarly also projected on the dates_test
    From https://climatedataguide.ucar.edu/climate-data/pacific-decadal-oscillation-pdo-definition-and-indices
    See http://www.cgd.ucar.edu/staff/cdeser/docs/deser.sstvariability.annrevmarsci10.pdf

    selbox has format of (lon_min, lon_max, lat_min, lat_max)
    '''
    t0 = time()
#    old format selbox
#    {'la_min':20, # select domain in degrees east
#     'la_max':70,
#     'lo_min':115,
#     'lo_max':250},

    kwrgs_pp = {'selbox' :  (115, 250, 20, 70),
                'format_lon': 'only_east'}

    ds = core_pp.import_ds_lazy(filepath, **kwrgs_pp)
    ds_monthly = ds.resample(time='M',restore_coord_dims=False).mean(dim='time', skipna=True)
    # ds_global = core_pp.import_ds_lazy(filepath)
    # ds.mean(dim=('latitude','longitude')) # global mean SST anomaly each timestep

    if df_splits is None:
        print('No train-test split')
        iterables = [np.array([0]),pd.to_datetime(ds.time.values)]
        df_splits = pd.DataFrame(data=np.ones(ds.time.size),
                                 index=pd.MultiIndex.from_product(iterables),
                                 columns=['TrainIsTrue'], dtype=bool)
    splits = df_splits.index.levels[0]
    data = np.zeros( (splits.size, ds.latitude.size, ds.longitude.size) )
    PDO_patterns = xr.DataArray(data,
                                coords=[splits, ds.latitude.values, ds.longitude.values],
                                dims = ['split', 'latitude', 'longitude'])


    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
            futures = [pool.submit(PDO_single_split, s, ds_monthly, ds, df_splits) for s in range(splits.size)]
            results = [future.result() for future in futures]
    else:
        results = [PDO_single_split(s, ds_monthly, ds, df_splits) for s in range(splits.size)]


    list_PDO_ts = [r[0] for r in results]

    time_ = time() - t0
    print(time_/60)

    for s in splits:
        PDO_patterns[s] = results[s][1]

    df_PDO = pd.concat(list_PDO_ts, axis=0, keys=splits)
    # merge df_splits
    df_PDO = df_PDO.merge(df_splits, left_index=True, right_index=True)
    if splits.size > 1:
        # train test splits should not be equal
        assert float((df_PDO.loc[1]['PDO'] - df_PDO.loc[0]['PDO']).mean()) != 0, (
                    'something went wrong with train test splits')
    #%%
    return df_PDO, PDO_patterns




def get_PDO(sst_Pacific):
    #%%
    from eofs.xarray import Eof
#    PDO   = functions_pp.find_region(sst, region='PDO')[0]
    coslat = np.cos(np.deg2rad(sst_Pacific.coords['latitude'].values)).clip(0., 1.)
    area_weights = np.tile(coslat[..., np.newaxis],(1,sst_Pacific.longitude.size))
    area_weights = area_weights / area_weights.mean()
    solver = Eof(sst_Pacific, area_weights)
    # Retrieve the leading EOF, expressed as the correlation between the leading
    # PC time series and the input SST anomalies at each grid point, and the
    # leading PC time series itself.
    eof1 = solver.eofsAsCovariance(neofs=1).squeeze()
    PDO_warmblob = eof1.sel(latitude=slice(30,40)).sel(longitude=slice(180,200)).mean() # flip sign oef pattern and ts
    assert ~np.isnan(PDO_warmblob), 'Could not verify sign of PDO, check lat lon axis'
    init_sign = np.sign(PDO_warmblob)
    if init_sign != -1.:
        adjust_sign = -1
    else:
        adjust_sign = 1
    eof1 *= adjust_sign
    return eof1, solver, adjust_sign

def progressBar(current, total, barLength = 20):
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))

    print('Progress: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')

def PNA_z500(filepath_z):
    '''
    From Liu et al. 2015: Recent contrasting winter temperature changes over
    North America linked to enhanced positive Pacific‐North American pattern.

    https://onlinelibrary.wiley.com/doi/abs/10.1002/2015GL065656

    PNA = z1 - z2 + z3 - z4
    z1 = Z (15 - 25N, 180 - 140W)
    z2 = Z (40 - 50N, 180 - 140W)
    z3 = Z (45 - 60N, 125 - 105W)
    z4 = Z (25 - 35N, 90 - 70W)

    Parameters
    ----------
    filepath : TYPE
        filepath to SST Netcdf4.

    Returns
    -------
    PNA.

    '''
    load = core_pp.import_ds_lazy
    progressBar(1, 4)
    z1 = functions_pp.area_weighted(load(filepath_z,
                                         **{'selbox' :  (180, 220, 15, 25)}))
    progressBar(2, 4)
    z2 = functions_pp.area_weighted(load(filepath_z,
                                         **{'selbox' :  (180, 220, 40, 50)}))
    z3 = functions_pp.area_weighted(load(filepath_z,
                                         **{'selbox' :  (235, 255, 45, 60)}))
    progressBar(3, 4)
    z4 = functions_pp.area_weighted(load(filepath_z,
                                         **{'selbox' :  (270, 290, 25, 35)}))
    progressBar(4, 4)
    PNA = z1.mean(dim=('latitude', 'longitude')) - z2.mean(dim=('latitude', 'longitude')) \
        + z3.mean(dim=('latitude', 'longitude')) - z4.mean(dim=('latitude', 'longitude'))
    return PNA.to_dataframe(name='PNA')
