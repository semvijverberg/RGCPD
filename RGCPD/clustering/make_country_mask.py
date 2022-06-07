#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 11:26:17 2020

@author: semvijverberg
"""


from .. import core_pp
from .enums import Continents, Country, US_States


import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapefile
from shapely import geometry
# use this to manually adapt the cluster a bit:
from scipy.ndimage import (binary_closing, binary_dilation, binary_erosion,
                           binary_fill_holes, binary_opening)



DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))


def country_mask(coordinates):

    shapefile_path = DATA_ROOT + '/ne_50m_admin_0_countries/ne_50m_admin_0_countries.shp'

    countries = shapefile.Reader(shapefile_path)

    iso_a2_index = [field[0] for field in countries.fields[1:]].index("ISO_A2")
    # name in english
    adm0_name_en = [field[0] for field in countries.fields[:]].index("NAME")

    shapes = [geometry.shape(shape) for shape in countries.shapes()]
    records = [record[iso_a2_index] for record in countries.records()]
    fullname = [record[adm0_name_en] for record in countries.records()]

    country_codes = np.empty(len(coordinates), int)
    names = []
    for index, coordinate in enumerate(coordinates):
        if len(coordinates) > 1000:
            print(f"\rCreating Country Mask: "
                  f"{index+1:7d}/{len(coordinates):7d} "
                  f"({float(index)/float(len(coordinates)):3.1%})", end="")

        point = geometry.Point(coordinate)

        for shape, iso_a2 in zip(shapes, records):
            if point.within(shape):
                country_codes[index] = Country[iso_a2] if iso_a2 in Country.__members__ else -1
                if iso_a2 not in names and iso_a2 in Country.__members__:
                    names.append(iso_a2)
                    names.append(fullname[records.index(iso_a2)])
                break
            else:
                country_codes[index] = -1

    names = np.reshape(names, (-1,2))
    # countries found in mask
    abbrev_in_mask = [k for k in Country._member_names_ if k in names[:,0]]
    country_code = [[k, Country.__getitem__(k).value] for k in abbrev_in_mask]
    names = np.concatenate((np.array(([[dict(names)[a[0]]] for a in country_code])),
                            np.array(country_code)),
                            axis=1)
    df_names = pd.DataFrame(names,
                             columns=['name', 'abbrev', 'label'],
                             index=np.array(country_code)[:,1])
    if len(coordinates) > 1000:
        print()

    return country_codes, df_names

def US_State_mask(coordinates):

    shapefile_path = os.path.join(DATA_ROOT,
                          'ne_50m_admin_1_states_provinces/ne_50m_admin_1_states_provinces.shp')

    Province = shapefile.Reader(shapefile_path)

    adm0_sr_index = [field[0] for field in Province.fields[:]].index("abbrev")
    # name in english
    adm0_name = [field[0] for field in Province.fields[:]].index("adm0_sr")
    code_local = [field[0] for field in Province.fields[:]].index("code_local")

    shapes = [geometry.shape(shape) for shape in Province.shapes()]
    records = [record[adm0_sr_index] for record in Province.records()]
    fullname = [record[adm0_name] for record in Province.records()]
    abbrev = [record[code_local] for record in Province.records()]

    # collect name, abbrev, and mask label
    country_codes = np.empty(len(coordinates), int)
    names = []
    for index, coordinate in enumerate(coordinates):
        if len(coordinates) > 1000:
            print(f"\rCreating Country Mask: "
                  f"{index+1:7d}/{len(coordinates):7d} "
                  f"({float(index)/float(len(coordinates)):3.1%})", end="")

        point = geometry.Point(coordinate)

        for shape, adm0_sr in zip(shapes, records):
            if point.within(shape):
                country_codes[index] = US_States[adm0_sr] if adm0_sr in US_States.__members__ else -1
                if adm0_sr not in names and adm0_sr in US_States.__members__:
                    names.append(adm0_sr)
                    names.append(fullname[abbrev.index('US.'+adm0_sr)])
                break
            else:
                country_codes[index] = -1

    names = np.reshape(names, (-1,2))
    # Gridpoints in US States (not all states may be in domain)
    abbrev_in_mask = [k for k in US_States._member_names_ if k in names[:,0]]
    country_code = [[k, US_States.__getitem__(k).value] for k in abbrev_in_mask]
    names = np.concatenate((np.array(([[dict(names)[a[0]]] for a in country_code])),
                            np.array(country_code) ),
                            axis=1)
    df_names = pd.DataFrame(names,
                             columns=['name', 'abbrev', 'label'],
                             index=np.array(country_code)[:,1])

    if len(coordinates) > 1000:
        print() # print enter

    return country_codes, df_names

def Continent_mask(coordinates):

    shapefile_path = os.path.join(DATA_ROOT,
                          'ne_50m_admin_0_countries/ne_50m_admin_0_countries.shp')

    countries = shapefile.Reader(shapefile_path)

    Continent_index = [field[0] for field in countries.fields[1:]].index("CONTINENT")

    shapes = [geometry.shape(shape) for shape in countries.shapes()]
    records = [record[Continent_index] for record in countries.records()]

    country_codes = np.empty(len(coordinates), int)

    for index, coordinate in enumerate(coordinates):
        if len(coordinates) > 1000:
            print(f"\rCreating Continent Mask: "
                  f"{index+1:7d}/{len(coordinates):7d} "
                  f"({float(index)/float(len(coordinates)):3.1%})", end="")

        point = geometry.Point(coordinate)

        for shape, continent in zip(shapes, records):
            continent = continent.replace(' ','_')
            if point.within(shape):
                country_codes[index] = Continents[continent] if continent in Continents.__members__ else -1
                break
            else:
                country_codes[index] = -1

    if len(coordinates) > 1000:
        print()

    names = [[k, Continents.__getitem__(k).value] for k in Continents.__members__]
    df_names = pd.DataFrame(np.array(names)[:],
                            index=np.array(names)[:,0],
                            columns=['name', 'label'])
    return country_codes, df_names

def era_coordinate_grid(ds):
    # Get Latitudes and Longitudes from ERA .nc file
    latitudes = ds.latitude.values
    longitudes = ds.longitude.values
    # Create Coordinate Grid
    coordinates = np.empty((len(latitudes), len(longitudes), 2), np.float32)
    coordinates[..., 0] = longitudes.reshape(1, -1)
    coordinates[..., 1] = latitudes.reshape(-1, 1)

    return coordinates


def create_mask(da, level='Continents', path=None):
    '''
    Parameters
    ----------
    da: xr.DataArray
        DataArray with shape (time, lat, lon) or (lat, lon)
    Level : TYPE, optional
        Countries, Continents or US_States.
        The default is 'Continents'.
    path : str, optional
        path to store mask. If None, stored in ~/Downloads folder.
        The default is None.

    Returns
    -------
    mask : xr.DataArray, mask with labels for each Country/Continent/US_state.
    df_names : pd.DataFrame with columns ['name', 'abbrev', 'label']
    '''

    if path is None:
        folder_file = get_download_path()

    mask_dir = os.path.join(folder_file, 'masks')
    if os.path.isdir(mask_dir) != True : os.makedirs(mask_dir)
    mask_file = os.path.join(mask_dir, level)


    lo_min, lo_max = float(da.longitude.min()), float(da.longitude.max())
    la_min, la_max = float(da.latitude.min()), float(da.latitude.max())

    domainstr = '_lats[{}_{}]_lons[{}_{}]'.format(int(la_min), int(la_max),
                                                  int(lo_min), int(lo_max))
    mask_file = mask_file + domainstr


    # check input shape of da. Should be (time, lat, lon) or (lat, lon)
    if 'time' in da.dims:
        mask_xr = da.isel(time=0).copy().drop('time')
    else:
        assert len(da.shape) == 2, ('No time dimension and array.shape'
                        ' is not equal to 2.')
        mask_xr  = da.copy()

    if level == 'Continents':
        abbrev_class = Continents
    elif level == 'Countries':
        abbrev_class = Country
    elif level == 'US_States':
        abbrev_class = US_States
    if os.path.exists(mask_file+'.nc'):
        return core_pp.import_ds_lazy(mask_file+'.nc'), abbrev_class



    # Load Coordinates and Normalize to ShapeFile Coordinates
    coordinates = era_coordinate_grid(da)
    coordinates[..., 0][coordinates[..., 0] > 180] -= 360

    # Take Center of Grid Cell as Coordinate
    coordinates[..., 0] += (coordinates[0, 1, 0] - coordinates[0, 0, 0]) / 2
    coordinates[..., 1] += (coordinates[1, 0, 1] - coordinates[0, 0, 1]) / 2


    # Create Mask
    if level == 'Continents':
        mask, df_names = Continent_mask(coordinates.reshape(-1, 2))
    elif level == 'Countries':
        mask, df_names = country_mask(coordinates.reshape(-1, 2))
    elif level == 'US_States':
        mask, df_names = US_State_mask(coordinates.reshape(-1, 2))
    mask = mask.reshape(coordinates.shape[:2])



    # np.save(mask_file+'.npy', mask)

    mask_xr.name = 'country_mask'
    for index, row in df_names.iterrows():
        mask_xr.attrs[row['name']] = row['label']
    mask_xr.values = mask
    mask_xr = mask_xr.astype(int)
    mask_xr.to_netcdf(mask_file + '.nc', mode='w')

    return mask_xr, df_names


def plot_earth(view="EARTH"):

    # Create Big Figure
    plt.rcParams['figure.figsize'] = [18, 8]

    # create Projection and Map Elements
    projection = ccrs.PlateCarree()
    ax = plt.axes(projection=projection)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.STATES)
    ax.add_feature(cfeature.LAKES, color="white")
    ax.add_feature(cfeature.OCEAN, color="white")
    ax.add_feature(cfeature.LAND, color="lightgray")

    if view == "US":
        ax.set_xlim(-130, -65)
        ax.set_ylim(25, 50)
    elif view == "EAST US":
        ax.set_xlim(-105, -65)
        ax.set_ylim(25, 50)
    elif view == "EARTH":
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)

    return projection

def get_download_path():
    """Returns the default downloads path for linux or windows"""
    if os.name == 'nt':
        import winreg
        sub_key = r'SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders'
        downloads_guid = '{374DE290-123F-4565-9164-39C4925E467B}'
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, sub_key) as key:
            location = winreg.QueryValueEx(key, downloads_guid)[0]
        return location
    else:
        return os.path.join(os.path.expanduser('~'), 'Downloads')


#if __name__ == '__main__':
#    era_country_mask('/Users/bram/Documents/Computational Science/Thesis/heatwave/data/ERA/t2m_1979-2017_1_12_daily_2.5deg.nc')
