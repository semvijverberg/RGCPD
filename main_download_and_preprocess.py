#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 17:48:31 2018

@author: semvijverberg
"""
import time
start_time = time.time()
import os, sys
os.chdir('/Users/semvijverberg/Surfdrive/Scripts/Tigramite')
script_dir = os.getcwd()
if sys.version[:1] == '3':
    from importlib import reload as rel
import numpy as np
import pandas as pd
import functions_pp
import matplotlib.pyplot as plt
from datetime import datetime
import xarray as xr
import cartopy.crs as ccrs
import pickle
retrieve_ERA_i_field = functions_pp.retrieve_ERA_i_field
import_array = functions_pp.import_array
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
base_path = "/Users/semvijverberg/surfdrive/Data_ERAint/"
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
     'startyear'    :       1979,
     'endyear'      :       2017,
     'base_path'    :       base_path,
     'path_raw'     :       path_raw,
     'path_pp'     :       path_pp}
     )


# True if you want to download ncdfs (in ex['vars']) through ECMWF MARS,  
# only analytical fields 
ECMWFdownload = True
# True if you have your own Response Variable time serie you want to insert
importRVts = False 
    
# own ncdfs must have same period, daily data and on same grid
ex['own_actor_nc_names'] = [[]]
#ex['own_RV_nc_name'] = []
ex['own_RV_nc_name'] =  ['t2mmax', ('t2mmax_1979-2017_1_12_daily_'
                          '{}deg.nc'.format(ex['grid_res']))]
ex['excludeRV'] = 1 # if 0, then corr fields are calculated vs. first of ex['vars'] 
# =============================================================================
# Info to download ncdf from ECMWF, atm only analytical fields (no forecasts)
# =============================================================================
# You need the ecmwf-api-client package for this option.
if ECMWFdownload == True:
    # See http://apps.ecmwf.int/datasets/. 
#    ex['vars']      =       [['t2m'],['167.128'],['sfc'],[0]]
#    ex['vars']      =       [['sst', 'z', 'u'],['34.128','129.128','131.128'],
#                             ['sfc', 'pl', 'pl'],[0, '500', '500']]
    ex['vars']      =       [['sst'],['34.128'],['sfc'],['0']]
#    ex['vars']      =       [['u'],['131.128'],['pl'],['500']]
#    ex['vars']      =       [['z'], ['129.128'],['pl'], ['500']]
#    ex['vars']      =       [['t2m', 'sst', 'u', 't100'],
#                            ['167.128', '34.128', '131.128', '130.128'],
#                            ['sfc', 'sfc', 'pl', 'pl'],[0, 0, '500', '100']]
#    ex['vars']     =   [
#                        ['t2m', 'u'],              # ['name_RV','name_actor', ...]
#                        ['167.128', '131.128'],    # ECMWF param ids
#                        ['sfc', 'pl'],             # Levtypes
#                        [0, 200],                  # Vertical levels
#                        ]
elif ECMWFdownload == False:
    ex['vars']      =       [[]]
# =============================================================================
# Note, ex['vars'] is expanded if you have own ncdfs, the first index will 
# always be the Response Variable, unless you set importRVts = True
# =============================================================================
# =============================================================================
# Make a class for each variable, this class contains variable specific information,
# needed to download and post process the data. Along the way, information is added 
# to class based on decisions that you make. 
# =============================================================================
if ECMWFdownload == True:
    for idx in range(len(ex['vars'][0]))[:]:
        # class for ECMWF downloads
        var_class = functions_pp.Variable(ex, idx, ECMWFdownload) 
        ex[ex['vars'][0][idx]] = var_class
# =============================================================================
# Downloading data from Era-interim?  
# =============================================================================
if ECMWFdownload == True:
    for var in ex['vars'][0]:
        var_class = ex[var]
        retrieve_ERA_i_field(var_class)
        
if len(ex['own_actor_nc_names'][0]) != 0:
    print(ex['own_actor_nc_names'][0][0])
    for idx in range(len(ex['own_actor_nc_names'])):
        ECMWFdownload = False
        var_class = functions_pp.Variable(ex, idx, ECMWFdownload) 
        ex[ex['vars'][0][idx]] = var_class
if len(ex['own_RV_nc_name']) != 0 and importRVts == False:
    ECMWFdownload = False
    RV_name = ex['own_RV_nc_name'][0]
    ex['vars'][0].insert(0, RV_name)
    var_class = functions_pp.Variable(ex, idx, ECMWFdownload) 
    ex[ex['own_RV_nc_name'][0]] = var_class
    print(('inserted own netcdf as Response Variable {}\n'.format(RV_name)))
# =============================================================================
# Now we have collected all info on what variables will be analyzed, based on
# downloading, own netcdfs / importing RV time serie.
# =============================================================================
if importRVts == True:
    RV_name = 'tmax_EUS'
    RV_actor_names = RV_name + '_' + "_".join(ex['vars'][0])
    ex['RVts_filename'] = 't2mmax_1Jun-24Aug_compAggljacc_tf14_n9.npy'
    
elif importRVts == False:
    # if no time series is imported, it will take the first of ex['vars] as the
    # Response Variable
    RV_name = ex['vars'][0][0]
    RV_actor_names = "_".join(ex['vars'][0])
    # if import RVts == False, then a spatial mask is used for the RV
    ex['maskname'] = 'aver_tf14_n6'
    ex['path_masks'] = os.path.join(ex['path_pp'], 'RVts2.5', 
                          't2mmax_11jun-30aug_averAggljacc_tf14_n6'+'.npy')
    
    
# =============================================================================
# General Temporal Settings, frequency, lags, part of year investigated
# =============================================================================
# Information needed to pre-process, 
# Select temporal frequency:
ex['tfreqlist'] = [7]# [1,2,4,7,14,21,35]
for freq in ex['tfreqlist']:
    ex['tfreq'] = freq
    # choose lags to test
    lag_min = int(np.timedelta64(2, 'W') / np.timedelta64(ex['tfreq'], 'D')) 
    ex['lag_min'] = max(1, lag_min)
    ex['lag_max'] = ex['lag_min'] + 2
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
    for var in ex['vars'][0]:
        var_class = ex[var]
        outfile, datesstr, var_class = functions_pp.datestr_for_preproc(var_class, ex)
#        var_class, ex = functions_pp.preprocessing_ncdf(outfile, datesstr, var_class, ex)
        if os.path.isfile(outfile) == True: 
            print('looks like you already have done the pre-processing,\n'
                  'to save time let\'s not do it twice..')
            # but we will update the dates stored in var_class:
            var_class, ex = functions_pp.update_dates(var_class, ex)
            pass
        else:    
            var_class, ex = functions_pp.preprocessing_ncdf(outfile, datesstr, var_class, ex)
    
    
    # *****************************************************************************  
    # *****************************************************************************
    # Step 3 Settings for Response Variable (RV) 
    # *****************************************************************************  
    # *****************************************************************************
            
    # =============================================================================
    # 3.1 Select RV period (which period of the year you want to predict)
    # =============================================================================
    if importRVts == True:
        RV_name = 'RV_imp'
        dicRV = np.load(os.path.join(ex['path_pp'], 'RVts2.5', ex['RVts_filename'])).item()
    #    dicRV = pickle.load( open(os.path.join(ex['path_pp'],ex['RVts_filename']+'.pkl'), "rb") ) 
        
        class RV_seperateclass:
            RVfullts = dicRV['RVfullts']
            dates = pd.to_datetime(dicRV['RVfullts'].time.values)
        RV = RV_seperateclass()
        RV.startyear = RV.dates.year[0]
        RV.endyear = RV.dates.year[-1]
        if RV.startyear != ex['startyear']:
            print('make sure the dates of the RV match with the actors')
    
    elif importRVts == False:
        RV_name = ex['vars'][0][0]
        # RV should always be the first variable of the vars list in ex
        RV = ex[RV_name]
        RVarray, RV = functions_pp.import_array(RV)
        
    one_year = RV.dates.where(RV.dates.year == RV.startyear+1).dropna()
    months = [6,7,8] # Selecting the timesteps of 14 day mean ts that fall in juli and august
    RV_period = []
    for mon in months:
        # append the indices of each year corresponding to your RV period
        RV_period.insert(-1, np.where(RV.dates.month == mon)[0] )
    RV_period = [x for sublist in RV_period for x in sublist]
    RV_period.sort()
    ex['RV_period'] = RV_period
    RV.datesRV = RV.dates[RV_period]
    months = dict( {1:'jan',2:'feb',3:'mar',4:'apr',5:'may',6:'jun',
                    7:'jul',8:'aug',9:'sep',10:'okt',11:'nov',12:'dec' } )
    RV_name_range = '{}{}-{}{}_'.format(RV.datesRV[0].day, months[RV.datesRV.month[0]], 
                     RV.datesRV[-1].day, months[RV.datesRV.month[-1]] )
    
    # =============================================================================
    # 3.2 Select spatial mask to create 1D timeseries (e.g. a SREX region)
    # =============================================================================

    if importRVts == True:
        i = len(RV_name_range)
        ex['path_exp_periodmask'] = os.path.join(ex['path_exp'], RV_name_range + 
                                      ex['RVts_filename'][i:])
                                              
    elif importRVts == False:
        ex['path_exp_periodmask'] = os.path.join(ex['path_exp'], RV_name_range + 
                                      ex['maskname'] )
    # If you don't have your own timeseries yet, then we assume you want to make
    # one using the first variable listed in ex['vars']. You can 
    # load a spatial mask here and use it to create your
    # full timeseries (of length equal to actor time series)  
                                                    
    try:
        mask_dic = np.load(ex['path_masks'], encoding='latin1').item()
        RV_array = mask_dic['RV_array']
    #        nor_lon = mask.longitude
    #        US_mask = mask.roll(longitude=2)
    #        mask['longitude'] = nor_lon
        functions_pp.xarray_plot(RV_array)
    except IOError as e:
        print('\n\n**\nSpatial mask not found.\n')
    #              'Place your spatial mask in folder: \n{}\n'
    #              'and rerun this section.\n**'.format(ex['path_pp'], 'grids'))
        raise(e)
    RVarray.coords['mask'] = RV_array.mask
    RV.RVfullts = RVarray.where(RVarray.mask==False).mean(dim=['latitude','longitude']).squeeze()
    
    RV.RV_ts = RV.RVfullts[ex['RV_period']] # extract specific months of MT index 
    # Store added information in RV class to the exp dictionary
    ex['RV_name'] = RV_name
    ex[RV_name] = RV
    
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
    ex = np.load(filename_exp_design1).item()
    ex['alpha'] = 0.01 # set significnace level for correlation maps
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
    filename_exp_design2 = os.path.join(ex['fig_subpath'], 'input_dic_{}.npy'.format(ex['params']))
    np.save(filename_exp_design2, ex)
    print('\n\t**\n\tOkay, end of Part 2!\n\t**' )
    print('\nNext time, you\'re able to redo the experiment by loading in the dict '
          '\'filename_exp_design2\'.\n')
    #%%
    # *****************************************************************************
    # *****************************************************************************
    # Part 3 Start your experiment by running RGCPD python script with settings
    # *****************************************************************************
    # *****************************************************************************
    import main_RGCPD_tig3
    # =============================================================================
    # Find precursor fields (potential precursors)
    # =============================================================================
    ex, outdic_actors = main_RGCPD_tig3.calculate_corr_maps(filename_exp_design2, map_proj)
    #%% 
    # =============================================================================
    # Run tigramite to extract causal precursors
    # =============================================================================
    parents_RV, var_names = main_RGCPD_tig3.run_PCMCI(ex, outdic_actors, map_proj)
    #%%
    # =============================================================================
    # Plot final results
    # =============================================================================
    main_RGCPD_tig3.plottingfunction(ex, parents_RV, var_names, outdic_actors, map_proj)
    print("--- {:.2} minutes ---".format((time.time() - start_time)/60))
    #%%
    
    
    #
    ## save to github
    #import os
    #import subprocess
    #runfile = os.path.join(script_dir, 'saving_repository_to_Github.sh')
    #subprocess.call(runfile)
    #
    #
    #
    #
    ## depricated
    ##%%
    #idx = 0
    #temperature = ( Variable(name='t2m', dataset='ERA-i', var_cf_code=ex['vars'][1][idx], 
    #                   levtype=ex['vars'][2][idx], lvllist=ex['vars'][3][idx], startyear=ex['startyear'], 
    #                   endyear=ex['endyear'], startmonth=ex['dstartmonth'], endmonth=ex['dendmonth'], 
    #                   grid=ex['grid_res'], stream='oper') )
    ## Download variable to input_raw
    #retrieve_ERA_i_field(temperature)
    ## preprocess variable to input_pp_exp'expnumber'
    ##functions_pp.preprocessing_ncdf(temperature, ex['grid_res'], ex['tfreq'], ex['exp'])
    ##marray, temperature = functions_pp.import_array(temperature, path='pp')
    ##marray
    ##%%
    #idx = 1
    #sst = ( Variable(name='sst', dataset='ERA-i', var_cf_code=ex['vars'][1][idx], 
    #                   levtype=ex['vars'][2][idx], lvllist=ex['vars'][3][idx], startyear=ex['startyear'], 
    #                   endyear=ex['endyear'], startmonth=ex['dstartmonth'], endmonth=ex['dendmonth'], 
    #                   grid=ex['grid_res'], stream='oper') )
    ## Download variable
    #retrieve_ERA_i_field(sst)
    #functions_pp.preprocessing_ncdf(sst, ex['grid_res'], ex['tfreq'], ex['exp'])
    #
    #
    ##%%
    #
    ## =============================================================================
    ## Simple example of cdo commands within python by calling bash script
    ## =============================================================================
    #import functions_pp
    #infile = os.path.join(var_class.base_path, 'input_raw', var_class.filename)
    #outfile = os.path.join(var_class.base_path, 'input_pp', 'output.nc')
    ##tmp = os.path.join(temperature.base_path, 'input_raw', temperature.filename)
    #args1 = 'cdo timmean {} {}'.format(infile, outfile)
    ##args2 = 'cdo setreftime,1900-01-01,0,1h -setcalendar,gregorian {} {} '.format(infile, outfile)
    #args = [args1]
    #
    #functions_pp.kornshell_with_input(args)
    # =============================================================================
    # How to run python script from python script:
    # =============================================================================
    #import subprocess
    #script_path = os.path.join(script_dir, 'main_RGCPD_tig3.py')
    #bash_and_args = ['python', script_path]
    #p = subprocess.Popen(bash_and_args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    #out = p.communicate(filename_exp_design2)
    #print(out[0])