#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 11:26:17 2020

@author: semvijverberg
"""



import numpy as np

from shapely import geometry
import shapefile
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
# import xarray as xr
import os
import core_pp
from enums import Country, Continents, US_States

# use this to manually adapt the cluster a bit:
from scipy.ndimage import binary_closing, binary_dilation, binary_erosion, binary_fill_holes, binary_opening

DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))


def country_mask(coordinates):

    shapefile_path = DATA_ROOT + '/ne_50m_admin_0_countries/ne_50m_admin_0_countries.shp'

    countries = shapefile.Reader(shapefile_path)

    iso_a2_index = [field[0] for field in countries.fields[1:]].index("ISO_A2")

    shapes = [geometry.shape(shape) for shape in countries.shapes()]
    records = [record[iso_a2_index] for record in countries.records()]

    country_codes = np.empty(len(coordinates), np.int)

    for index, coordinate in enumerate(coordinates):
        if len(coordinates) > 1000:
            print(f"\rCreating Country Mask: "
                  f"{index+1:7d}/{len(coordinates):7d} "
                  f"({float(index)/float(len(coordinates)):3.1%})", end="")

        point = geometry.Point(coordinate)

        for shape, iso_a2 in zip(shapes, records):
            if point.within(shape):
                country_codes[index] = Country[iso_a2] if iso_a2 in Country.__members__ else -1
                break
            else:
                country_codes[index] = -1

    if len(coordinates) > 1000:
        print()

    return country_codes

def US_State_mask(coordinates):

    shapefile_path = DATA_ROOT + '/ne_50m_admin_1_states_provinces/ne_50m_admin_1_states_provinces.shp'

    Province = shapefile.Reader(shapefile_path)

    adm0_sr_index = [field[0] for field in Province.fields[:]].index("abbrev")

    shapes = [geometry.shape(shape) for shape in Province.shapes()]
    records = [record[adm0_sr_index] for record in Province.records()]

    country_codes = np.empty(len(coordinates), int)

    for index, coordinate in enumerate(coordinates):
        if len(coordinates) > 1000:
            print(f"\rCreating Country Mask: "
                  f"{index+1:7d}/{len(coordinates):7d} "
                  f"({float(index)/float(len(coordinates)):3.1%})", end="")

        point = geometry.Point(coordinate)

        for shape, adm0_sr in zip(shapes, records):
            if point.within(shape):
                country_codes[index] = US_States[adm0_sr] if adm0_sr in US_States.__members__ else -1
                break
            else:
                country_codes[index] = -1

    if len(coordinates) > 1000:
        print()

    return country_codes

def Continent_mask(coordinates):

    shapefile_path = DATA_ROOT + 'ne_50m_admin_0_countries/ne_50m_admin_0_countries.shp'

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

    return country_codes

def era_coordinate_grid(ds):
    # Get Latitudes and Longitudes from ERA .nc file
    latitudes = ds.latitude.values
    longitudes = ds.longitude.values
    # Create Coordinate Grid
    coordinates = np.empty((len(latitudes), len(longitudes), 2), np.float32)
    coordinates[..., 0] = longitudes.reshape(1, -1)
    coordinates[..., 1] = latitudes.reshape(-1, 1)

    return coordinates


def create_mask(path, kwrgs_load={}, level='Continents'):
    '''
    Parameters
    ----------
    path: str
        full path to netcdf file for which you want to create a mask
    kwrgs_load : Dict, optional
        See kwargs core_pp.import_ds_lazy?
    Level : TYPE, optional
        Countries or Continents The default is 'Continents'.

    Returns
    -------
    xr.DataArray
        mask with labels for each country.

    '''
    f_name = os.path.splitext(path)[0].split('/')[-1]
    folder_file = '/'.join(os.path.splitext(path)[0].split('/')[:-1])
    mask_dir = os.path.join(folder_file, 'masks')
    if os.path.isdir(mask_dir) != True : os.makedirs(mask_dir)
    mask_file = os.path.join(mask_dir, f_name + '_' + level)
    if 'selbox' in kwrgs_load.keys():
        lo_min, lo_max, la_min, la_max = kwrgs_load['selbox']
        domainstr = '_lats[{}_{}]_lons[{}_{}]'.format(int(la_min), int(la_max),
                                      int(lo_min), int(lo_max))
        mask_file = mask_file + domainstr
    if level == 'Continents':
        abbrev_class = Continents
    elif level == 'Countries':
        abbrev_class = Country
    elif level == 'US_States':
        abbrev_class = US_States
    if os.path.exists(mask_file+'.nc'):
        return core_pp.import_ds_lazy(mask_file+'.nc'), abbrev_class


    ds = core_pp.import_ds_lazy(path, **kwrgs_load)

    # Load Coordinates and Normalize to ShapeFile Coordinates
    coordinates = era_coordinate_grid(ds)
    coordinates[..., 0][coordinates[..., 0] > 180] -= 360

    # Take Center of Grid Cell as Coordinate
    coordinates[..., 0] += (coordinates[0, 1, 0] - coordinates[0, 0, 0]) / 2
    coordinates[..., 1] += (coordinates[1, 0, 1] - coordinates[0, 0, 1]) / 2

    # Create Mask
    if level == 'Continents':
        mask = Continent_mask(coordinates.reshape(-1, 2)).reshape(coordinates.shape[:2])
    elif level == 'Countries':
        mask = country_mask(coordinates.reshape(-1, 2)).reshape(coordinates.shape[:2])
    elif level == 'US_States':
        mask = US_State_mask(coordinates.reshape(-1, 2)).reshape(coordinates.shape[:2])


    country_code = [{k : abbrev_class.__getitem__(k).value} for k in abbrev_class._member_names_]
    # np.save(mask_file+'.npy', mask)
    if 'time' in ds.dims:
        mask_xr = ds.isel(time=0).copy().drop('time')
    else:
        mask_xr  = ds.copy()
    mask_xr.name = 'country_mask'
    for dic in country_code:
        key, value = list(dic.items())[0]
        mask_xr.attrs[key] = value
#    mask_xr.attrs = {'country_code': country_code}
    mask_xr.values = mask
    mask_xr = mask_xr.astype(int)
    mask_xr.to_netcdf(mask_file + '.nc', mode='w')

    return mask_xr, abbrev_class


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


#if __name__ == '__main__':
#    era_country_mask('/Users/bram/Documents/Computational Science/Thesis/heatwave/data/ERA/t2m_1979-2017_1_12_daily_2.5deg.nc')
