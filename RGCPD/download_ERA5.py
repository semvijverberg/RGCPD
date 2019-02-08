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