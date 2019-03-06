#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 15:42:46 2019

@author: semvijverberg
"""
import os
import numpy as np
import pandas as pd
output_folder = '/Users/semvijverberg/Downloads'

base_path = "/Users/semvijverberg/surfdrive/Data_ERAint/"
exp_folder = ''
path_raw = os.path.join(base_path, 'input_raw')
path_pp  = os.path.join(base_path, 'input_pp')

def Variable(self, ex):
    self.startyear = ex['startyear']
    self.endyear = ex['endyear']
    self.startmonth = 1
    self.endmonth = 12
    self.grid = ['{}'.format(ex['grid_res']),'{}'.format(ex['grid_res'])]
    self.dataset = ex['dataset']
    self.base_path = ex['base_path']
    self.path_raw = ex['path_raw']
    self.path_pp = ex['path_pp']
    return self

class Var_ECMWF_download():
    """Levtypes: \n surface  :   sfc \n model level  :   ml (1 to 137) \n pressure levels (1000, 850.. etc)
    :   pl \n isentropic level    :   pt
    \n
    Monthly Streams:
    Monthly mean of daily mean  :   moda
    Monthly mean of analysis timesteps (synoptic monthly means)  :   mnth
    Daily Streams:
    Operational (for surface)   :   oper
    """
    
    
    def __init__(self, ex, idx):

#        from datetime import datetime, timedelta
#        import pandas as pd
#        import calendar
#        import os
        vclass = Variable(self, ex)
        # shared information of ECMWF downloaded variables
        # variables specific information
        if ex['vars'][2][idx] == 'sfc':
            vclass.dataset = '{}'.format('reanalysis-era5-single-levels')
        vclass.name = ex['vars'][0][idx]
        vclass.var_cf_code = ex['vars'][1][idx]
        vclass.levtype = ex['vars'][2][idx]
        vclass.lvllist = ex['vars'][3][idx]
        vclass.stream = 'oper'
            
        vclass.years    = [str(yr) for yr in np.arange(ex['startyear'], 
                                               ex['endyear']+1E-9, dtype=int)]
        vclass.months   = [str(yr) for yr in np.arange(ex['firstmonth'], 
                                               ex['lastmonth']+1E-9, dtype=int)] 
        vclass.days     = [str(yr) for yr in np.arange(1, 31+1E-9, dtype=int)] 
        

        vclass.time = list(ex['time'].strftime('%H:%M'))
        vclass.filename = '{}_{}-{}_{}_{}_{}_{}deg'.format(vclass.name, 
                           vclass.startyear, vclass.endyear, vclass.startmonth, 
                           vclass.endmonth, 'daily', ex['grid_res']).replace(' ', '_')
        vclass.format = '{}'.format(ex['format'])
        print(('\n\t**\n\t{} {}-{} on {} grid\n\t**\n'.format(vclass.name, 
               vclass.startyear, vclass.endyear, vclass.grid)))
        
        


ex = dict(
     {'dataset'     :       'ERA-5',
     'grid_res'     :       0.5,
     'startyear'    :       1979, # download startyear
     'endyear'      :       2018, # download endyear
     'firstmonth'   :       1,
     'lastmonth'    :       12,
     'time'         :       pd.DatetimeIndex(start='00:00', end='23:00', 
                                freq=(pd.Timedelta(6, unit='h'))),
     'format'       :       'netcdf',
     'base_path'    :       base_path,
     'path_raw'     :       path_raw,
     'path_pp'     :        path_pp}
     )

ex['vars'] = [['2m_temperature'],['34.128'],['sfc'],['0']]

vclass = Var_ECMWF_download(ex, idx=0)


dict_retrieval =  ('\'{}\',\n\t{{\n'.format(vclass.dataset)+
                     '\t\'variable\'         :   \'{}\',\n'.format(vclass.name)+
                     '\t\'product_type\'     :   \'reanalysis\',\n'
                     '\t\'year\'         :     {},\n'.format(vclass.years)+
                     '\t\'month\'        :     {},\n'.format(vclass.months)+
                     '\t\'day\'          :     {},\n'.format(vclass.days)+
                     '\t\'grid\'         :     {},\n'.format(vclass.grid)+
                     '\t\'time\'         :     {},\n'.format(vclass.time)+
                     '\t\'format\'       :     \'{}\',\n'.format(vclass.format)+
                     '\t}'
                     )
print(dict_retrieval)
            

#
## write output in textfile
text_lines = ['import cdstoolbox as ct\n',
              '@ct.application(title=\"download variable\")',
              '@ct.output.dataarray()',
              'def download_and_pp():',
              '\tdata = ct.catalogue.retrieve({})'.format(dict_retrieval),
              '\tdata = ct.climate.daily_mean(data)',
              #'\tdetrend = ct.stats.detrend(data_daily)',
              '\tanom = ct.climate.anomaly(data)',
              '\treturn anom'
              ]
#
txtfile = os.path.join(output_folder, 'ERA5_retrieval.txt')

with open(txtfile, "w") as text_file:
#    max_key_len = max([len(i) for i in line])
    for line in text_lines:
#        key_len = len(line)
#        expand = max_key_len - key_len
#        key_exp = key + ' ' * expand
        printline = line
        print(printline)
        print(printline, file=text_file)



#%%
def detrend_anom_ncdf3D(xarray):
    #%%
#    filename = '/Users/semvijverberg/surfdrive/MckinRepl/SST/sst_daily_1982_2017_mcKbox.nc'
#    filename = os.path.join(ex['path_raw'], 'tmpfiles', 'tmprmtime.nc')
#    outfile = '/Users/semvijverberg/surfdrive/MckinRepl/SST/sst_daily_1982_2017_mcKbox_detrend.nc'
#    encoding= { 'anom' : {'dtype': 'int16', 'scale_factor': 0.001, '_FillValue': -999}}
#    encoding = None
    import xarray as xr
    import pandas as pd
    import numpy as np
    from netCDF4 import num2date
#    filename = os.path.join(ex['path_pp'], 'sst_1979-2017_2mar_31aug_dt-1days_2.5deg.nc')
    ncdf = xr.open_dataset(filename, decode_cf=True, decode_coords=True, decode_times=False)
    variables = list(ncdf.variables.keys())
    strvars = [' {} '.format(var) for var in variables]
    var = [var for var in strvars if var not in ' time time_bnds longitude latitude lev lon lat '][0] 
    var = var.replace(' ', '')
    ds = ncdf[var]

    if 'latitude' and 'longitude' not in ds.dims:
        ds = ds.rename({'lat':'latitude', 
                   'lon':'longitude'})
    
#    marray = np.squeeze(ncdf.to_array(name=var))
    numtime = ds['time']
    dates = num2date(numtime, units=numtime.units, calendar=numtime.attrs['calendar'])
    if numtime.attrs['calendar'] != 'gregorian':
        dates = [d.strftime('%Y-%m-%d') for d in dates]
    dates = pd.to_datetime(dates)
    stepsyr = dates.where(dates.year == dates.year[0]).dropna(how='all')
    
    ds['time'] = dates
    
    def _detrendfunc2d(arr_oneday):
        from scipy import signal
        no_nans = np.nan_to_num(arr_oneday)
        detrended = signal.detrend(no_nans, axis=0, type='linear')
        nan_true = np.isnan(arr_oneday)
        detrended[nan_true] = np.nan
        return detrended
    
    
    def detrendfunc2d(arr_oneday):
        return xr.apply_ufunc(_detrendfunc2d, arr_oneday, 
                              dask='parallelized',
                              output_dtypes=[float])
#        return xr.apply_ufunc(_detrendfunc2d, arr_oneday.compute(), 
#                              dask='parallelized',
#                              output_dtypes=[float])
                
    output = np.empty( (ds.time.size,  ds.latitude.size, ds.longitude.size), dtype='float32' )
    output[:] = np.nan
    for i in range(stepsyr.size):
        sd =(dates[i::stepsyr.size])
        arr_oneday = ds.sel(time=sd)#.chunk({'latitude': 100, 'longitude': 100})
        output[i::stepsyr.size] = detrendfunc2d(arr_oneday) 
    

    output = xr.DataArray(output, name=var, dims=ds.dims, coords=ds.coords)
    
    # copy original attributes to xarray
    output.attrs = ds.attrs
    # ensure mask 
    output = output.where(output.values != 0.).fillna(-9999)
    encoding = ( {var : {'_FillValue': -9999}} )
    mask =  (('latitude', 'longitude'), (output.values[0] != -9999) )
    output.coords['mask'] = mask
#    xarray_plot(output[0])
    
    # save netcdf
    output.to_netcdf(outfile, mode='w', encoding=encoding)
#    diff = output - abs(marray)
#    diff.to_netcdf(filename.replace('.nc', 'diff.nc'))
    #%%
    return 

        
#%%