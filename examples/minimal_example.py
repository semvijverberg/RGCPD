#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 14:41:05 2018

@author: semvijverberg
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 17:48:31 2018

@author: semvijverberg
"""
import time
start_time = time.time()
import inspect, os, sys
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
script_dir = os.path.join(curr_dir, '..', 'RGCPD')
# To link modules in RGCPD folder to this script
os.chdir(script_dir)
sys.path.append(script_dir)
script_dir = os.getcwd()
if sys.version[:1] == '3':
    from importlib import reload as rel
import numpy as np
import pandas as pd
import functions_pp
import cartopy.crs as ccrs
retrieve_ERA_i_field = functions_pp.retrieve_ERA_i_field
copy_stdout = sys.stdout

# *****************************************************************************
# *****************************************************************************
# Part 1 Downloading (opt), preprocessing(opt), choosing general experiment settings
# *****************************************************************************
# *****************************************************************************
# We will be discriminating between actors and the Response Variable (what you want 
# to predict). 
# You can choose to download ncdfs through ECMWF MARS, or you can give your own 
# ncdfs.
# It is also possible to insert own 1D time serie for both actors and the RV.

# this will be your basepath, all raw_input and output will stored in subfolder 
# which will be made when running the code
#base_path = "/Users/semvijverberg/surfdrive/Scripts/RGCPD/tests/Data_ERAint/"
base_path = "/Users/semvijverberg/surfdrive/Scripts/RGCPD/examples/Data_ERAint"
exp_folder = ''
path_raw = os.path.join(base_path, 'input_raw')
path_pp  = os.path.join(base_path, 'input_pp')
if os.path.isdir(path_raw) == False : os.makedirs(path_raw) 
if os.path.isdir(path_pp) == False: os.makedirs(path_pp)

# *****************************************************************************
# Step 1 Create dictionary and variable class (and optionally download ncdfs)
# *****************************************************************************
# The dictionary is used as a container with all information for the experiment
# The dic is saved at intermediate steps, so you can continue the experiment 
# from these break points. It also stored as a log in the final output.
ex = dict(
     {'dataset'     :       'ERA-i',
     'grid_res'     :       2.5,
     'startyear'    :       2010, # download startyear
     'endyear'      :       2015, # download endyear
     'base_path'    :       base_path,
     'path_raw'     :       path_raw,
     'path_pp'      :       path_pp,
     'vars'         :       [[]]} # initialize ex['vars']
     )

# =============================================================================
# In minimal example, option 2 and 3 are illustrated
# =============================================================================

# Option 2:
import_precursor_ncdf  = True
# Option 3:
import_RV_ncdf = True



# 22222222222222222222222222222222222222222222222222222222222222222222222222222
# Import ncdf lonlat fields to be precursors.
# 22222222222222222222222222222222222222222222222222222222222222222222222222222
# Must have same period, daily data and on same grid
ex['precursor_ncdf'] = [['sst', 'sst_{}-{}_1_12_daily_'
                          '{}deg.nc'.format(ex['startyear'], ex['endyear'],
                           ex['grid_res'])]]
  
    
# 33333333333333333333333333333333333333333333333333333333333333333333333333333
# Import ncdf field to be Response Variable.
# 33333333333333333333333333333333333333333333333333333333333333333333333333333

ex['RVnc_name'] =  ['t2mmax', ('t2mmax_{}-{}_1_12_daily_'
                          '{}deg.nc'.format(ex['startyear'], ex['endyear'],
                           ex['grid_res']))]


ex['excludeRV'] = 0 # if 0, then corr fields of RV_1dts calculated vs. RV netcdf


# =============================================================================
# Make a class for each variable, this class contains variable specific information,
# needed to download and post process the data. Along the way, information is added 
# to class based on decisions that you make. 
# =============================================================================

if import_RV_ncdf == True:
    RV_name = ex['RVnc_name'][0]
    ex['vars'][0].insert(0, RV_name)
    var_class = functions_pp.Var_import_RV_netcdf(ex) 
    ex[ex['RVnc_name'][0]] = var_class
    print(('inserted own netcdf as Response Variable {}\n'.format(RV_name)))
    
if import_precursor_ncdf == True:
    for idx in np.arange(len(ex['precursor_ncdf'])):       
        var_class = functions_pp.Var_import_precursor_netcdf(ex, idx) 
        ex[var_class.name] = var_class


# =============================================================================
#  If no time series is imported, it will take the first of ex['vars] as the
#  Response Variable
# =============================================================================
RV_name = ex['vars'][0][0]
RV_actor_names = "_".join(ex['vars'][0])
# if import RVts == False, then a spatial mask is used for the RV
ex['spatial_mask_naming'] = 'averAggljacc_tf14_n8'
ex['spatial_mask_file'] = os.path.join(ex['path_pp'], 'RVts2.5', 
                      't2mmax_1979-2017_1jun-24aug_averAggljacc_tf14_n8'+'.npy')
    
    
# =============================================================================
# General Temporal Settings, frequency, lags, part of year investigated
# =============================================================================
# Information needed to pre-process, 
# Select temporal frequency in days:
ex['tfreq'] = 30

# choose lags to test
lag_min = int(np.timedelta64(2, 'W') / np.timedelta64(ex['tfreq'], 'D')) 
ex['lag_min'] = max(1, lag_min)
ex['lag_max'] = ex['lag_min'] + 0
# s(elect)startdate and enddate create the period of year you want to investigate:
ex['sstartdate'] = '{}-3-1 09:00:00'.format(ex['startyear'])
ex['senddate']   = '{}-08-31 09:00:00'.format(ex['startyear'])

ex['exp_pp'] = '{}_m{}-{}_dt{}'.format(RV_actor_names, 
                    ex['sstartdate'].split('-')[1], ex['senddate'].split('-')[1], ex['tfreq'])

ex['path_exp'] = os.path.join(base_path, exp_folder, ex['exp_pp'])
if os.path.isdir(ex['path_exp']) == False : os.makedirs(ex['path_exp'])

# =============================================================================
# Preprocess data (this function uses cdo/nco and requires execution rights of
# the created bash script)
# =============================================================================
# First time: Read Docstring by typing 'functions_pp.preprocessing_ncdf?' in console
# Solve permission error by giving bash script execution right, read Docstring

functions_pp.perform_post_processing(ex)

# *****************************************************************************  
# *****************************************************************************
# Step 3 Settings for Response Variable (RV) 
# *****************************************************************************  
# *****************************************************************************
class RV_seperateclass:
    def __init__(self):
        self.name = None
        self.RVfullts = None
        self.RVts = None
        
RV = RV_seperateclass()
# =============================================================================
# 3.1 Select RV period (which period of the year you want to predict)
# =============================================================================
# If you don't have your own timeseries yet, then we assume you want to make
# one using the first variable listed in ex['vars']. 

RV_months = [6,7,8]
RV, ex, RV_name_range = functions_pp.RV_spatial_temporal_mask(ex, RV, False, RV_months)
ex[ex['RV_name']] = RV
    
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
    
# create this subfolder in ex['path_exp'] for RV_period and spatial mask    
ex['path_exp_periodmask'] =  ex['path_exp_periodmask'] + '_lag{}-{}'.format(
                                                ex['lag_min'], ex['lag_max'])
                            

if os.path.isdir(ex['path_exp_periodmask']) != True : os.makedirs(ex['path_exp_periodmask'])
filename_exp_design1 = os.path.join(ex['path_exp_periodmask'], 'input_dic_part_1.npy')


print('\n\t**\n\tOkay, end of Part 1!\n\t**' )
print('\nNext time, you can choose to start with part 2 by loading in '
      'part 1 settings from dictionary \'filename_exp_design1\'\n')
np.save(filename_exp_design1, ex)
#%% 
# *****************************************************************************
# *****************************************************************************
# Part 2 Configure RGCPD/Tigramite settings
# *****************************************************************************
# *****************************************************************************
ex = np.load(filename_exp_design1, encoding='latin1').item()
ex['alpha'] = 0.05 # set significnace level for correlation maps
ex['alpha_fdr'] = 2*ex['alpha'] # conservative significance level
ex['FDR_control'] = False # Do you want to use the conservative alpha_fdr or normal alpha?
# If your pp data is not a full year, there is Maximum meaningful lag given by: 
#ex['lag_max'] = dates[dates.year == 1979].size - ex['RV_oneyr'].size
ex['alpha_level_tig'] = 0.2 # Alpha level for final regression analysis by Tigrimate
ex['pcA_sets'] = dict({   # dict of sets of pc_alpha values
      'pcA_set1a' : [ 0.05], # 0.05 0.01 
      'pcA_set1b' : [ 0.01], # 0.05 0.01 
      'pcA_set1c' : [ 0.1], # 0.05 0.01 
      'pcA_set2'  : [ 0.2, 0.1, 0.05, 0.01, 0.001], # set2
      'pcA_set3'  : [ 0.1, 0.05, 0.01], # set3
      'pcA_set4'  : [ 0.5, 0.4, 0.3, 0.2, 0.1], # set4
      'pcA_none'  : None # default
      })
ex['pcA_set'] = 'pcA_set1a' 
ex['la_min'] = -89 # select domain of correlation analysis
ex['la_max'] = 89
ex['lo_min'] = -180
ex['lo_max'] = 360
# Some output settings
ex['file_type1'] = ".pdf"
ex['file_type2'] = ".png" 
ex['SaveTF'] = True # if false, output will be printed in console
ex['plotin1fig'] = False 
ex['showplot'] = True
central_lon_plots = 240
map_proj = ccrs.LambertCylindrical(central_longitude=central_lon_plots)
# output paths
ex['path_output'] = os.path.join(ex['path_exp_periodmask'])
ex['fig_path'] = os.path.join(ex['path_exp_periodmask'])
ex['params'] = '{}_ac{}_at{}'.format(ex['pcA_set'], ex['alpha'],
                                                  ex['alpha_level_tig'])
if os.path.isdir(ex['fig_path']) != True : os.makedirs(ex['fig_path'])
ex['fig_subpath'] = os.path.join(ex['fig_path'], '{}_subinfo'.format(ex['params']))
if os.path.isdir(ex['fig_subpath']) != True : os.makedirs(ex['fig_subpath'])                                  
# =============================================================================
# Save Experiment design
# =============================================================================
assert RV.startyear == ex['startyear'], ('Make sure the dates '
         'of the RV match with the actors')
#assert ((ex['excludeRV'] == 1) and (importRV_1dts == True))==False, ('Are you sure you want '
#         'exclude first element of array ex[\'vars\'] since you are importing a seperate '
#         ' time series ') 
         

filename_exp_design2 = os.path.join(ex['fig_subpath'], 'input_dic_{}.npy'.format(ex['params']))
np.save(filename_exp_design2, ex)

print('\n\t**\n\tOkay, end of Part 2!\n\t**' )

print('\n**\nBegin summary of main experiment settings\n**\n')
print('Response variable is {} is correlated vs {}'.format(ex['vars'][0][0],
      ex['vars'][0][1:]))
start_day = '{}-{}'.format(int(ex['adjstartdate'][8:10]), int(ex['adjstartdate'][5:7]))
end_day   = '{}-{}'.format(int(ex['senddate'][8:10]), int(ex['senddate'][5:7]))
print('Part of year investigated: {} - {}'.format(start_day, end_day))
print('Part of year predicted (RV period): {} '.format(RV_name_range[:-1]))
print('Temporal resolution: {} days'.format(ex['tfreq']))
print('Lags: {} to {}'.format(ex['lag_min'], ex['lag_max']))
one_year_RV_data = RV.datesRV.where(RV.datesRV.year==RV.startyear).dropna(how='all').values
print('For example\nPredictant (only one year) is:\n{} at \n{}\n'.format(RV_name,
      one_year_RV_data))
print('\tVS\n')
shift_lag_days = one_year_RV_data - pd.Timedelta(int(ex['lag_min']*ex['tfreq']), unit='d')
print('Predictor (only one year) is:\n{} at lag {} days\n{}\n'.format(
        ex['vars'][0][-1], int(ex['lag_min']*ex['tfreq']), shift_lag_days))
print('\n**\nEnd of summary\n**\n')

print('\nNext time, you\'re able to redo the experiment by loading in the dict '
      '\'filename_exp_design2\'.\n')
#%%
# *****************************************************************************
# *****************************************************************************
# Part 3 Start your experiment by running RGCPD python script with settings
# *****************************************************************************
# *****************************************************************************
import wrapper_RGCPD_tig
# =============================================================================
# Find precursor fields (potential precursors)
# =============================================================================
ex, outdic_actors = wrapper_RGCPD_tig.calculate_corr_maps(ex, map_proj)
print('\n\nThe plots below make to scientific sense, since they are based on 5 years '
      'of data.\n\n')
#%% 
# =============================================================================
# Run tigramite to extract causal precursors
# =============================================================================
parents_RV, var_names = wrapper_RGCPD_tig.run_PCMCI(ex, outdic_actors, map_proj)
#%%
# =============================================================================
# Plot final results
# =============================================================================
wrapper_RGCPD_tig.plottingfunction(ex, parents_RV, var_names, outdic_actors, map_proj)
print("--- {:.2} minute(s) ---".format((time.time() - start_time)/60))

