# -*- coding: utf-8 -*-
import numpy
import matplotlib
matplotlib.rcParams['backend'] = "Qt4Agg"
from pylab import *
import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap, shiftgrid, cm
from netCDF4 import Dataset
from netcdftime import utime
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
import pandas
from pandas import DataFrame
import scipy
from scipy import signal
from datetime import datetime 
import datetime
from matplotlib.patches import Polygon
from matplotlib import gridspec
import seaborn as sns
from statsmodels.sandbox.stats import multicomp
import xarray as xr
import cartopy.crs as ccrs




def extract_data(d, D, index_range, ex):	
	"""
	Extracts the array of variable d for indices index_range over the domain box
	d: netcdf elements
	D: the data array
	index_range: a list containing the start and the end index, e.g. [0, time_cycle*n_years]
	"""
	
	
	index_0 = index_range[0]
	index_n = index_range[1]

	if 'latitude' in list(d.variables.keys()):
	  lat = d.variables['latitude'][:]
	else:
	  lat = d.variables['lat'][:]
	if 'longitude' in list(d.variables.keys()):
	  lon = d.variables['longitude'][:]
	else:
	  lon = d.variables['lon'][:]

	if lon.min() < 0: 
	  lon[ lon < 0] = 360 + lon[ lon < 0]


#	time = d.variables['time'][:]
#	unit = d.variables['time'].units
#	u = utime(unit)
#	date = u.num2date(time[:])
	D = D[index_0: index_n, :, :]
	D = D[:,(lat>=ex['la_min']) & (lat<=ex['la_max']),:]
	D = D[:,:,(lon>=ex['lo_min']) & (lon<=ex['lo_max'])]
	
	return D

	
	
def plot_basemap_options(m):

	''' specifies basemap options for basemap m'''
	
	# Basemap options:
	m.drawcoastlines(color='gray', linewidth=0.35)
	#m.drawcountries(color='gray', linewidth=0.2)
	m.drawmapboundary(fill_color='white', color='gray')
	#m.drawmapboundary(color='gray')
	#m.fillcontinents(color='white',lake_color='white')
	m.drawmeridians(numpy.arange(0, 360, 30), color='lightgray')
	m.drawparallels(numpy.arange(-90, 90, 30), color='lightgray')


	
	

# This functions merges sets which contain at least on common element. It was taken from:
# http://stackoverflow.com/questions/9110837/python-simple-list-merging-based-on-intersections

def merge_neighbors(lsts):
  sets = [set(lst) for lst in lsts if lst]
  merged = 1
  while merged:
    merged = 0
    results = []
    while sets:
      common, rest = sets[0], sets[1:]
      sets = []
      for x in rest:
        if x.isdisjoint(common):
          sets.append(x)
        else:
          merged = 1
          common |= x
      results.append(common)
    sets = results
  return sets
	
def corr_new(D, di):
	"""
	This function calculates the correlation coefficent r  and the the pvalue p for each grid-point of field D with the response-variable di
	"""
	x = numpy.ma.zeros(D.shape[1])
	corr_di_D = numpy.ma.array(data = x, mask =False)	
	sig_di_D = numpy.array(x)
	
	for i in range(D.shape[1]):
		r, p = scipy.stats.pearsonr(di,D[:,i])
		corr_di_D[i]= r 
		sig_di_D[i]= p
									
	return corr_di_D, sig_di_D

	
def calc_corr_coeffs_new(v, V, RVts, time_range_indices, ex):
#    v = ncdf ; V = array ; time_range_indices = RV.RV_ts
    """
    This function calculates the correlation maps for fied V for different lags. Field significance is applied to test for correltion.
    v: netcdf element
    V: array
    box: list of form [la_min, la_max, lo_min, lo_max]
    time_range_indices: a list containing the start and the end index, e.g. [0, time_cycle*n_years]
    lag_steps: number of lags
    time_cycle: time cycyle of dataset, =12 for monthly data...
    RV_period: indices that matches the response variable time series
    alpha: significance level

    """
    lag_steps = ex['lag_max'] - ex['lag_min'] +1
		
    d = v
	
    if 'latitude' in list(d.variables.keys()):
        lat = d.variables['latitude'][:]
    else:
        lat = d.variables['lat'][:]
    if 'longitude' in list(d.variables.keys()):
        lon = d.variables['longitude'][:]
    else:
        lon = d.variables['lon'][:]

    if lon.min() < 0: 
        lon[lon < 0] = 360 + lon[lon < 0]
	
    lat_grid = lat[(lat>=ex['la_min']) & (lat<=ex['la_max'])]
    lon_grid = lon[(lon>=ex['lo_min']) & (lon<=ex['lo_max'])]
	
    la = lat_grid.shape[0]
    lo = lon_grid.shape[0]
	
    lons, lats = numpy.meshgrid(lon_grid,lat_grid)

    A1 = numpy.zeros((la,lo))
    z = numpy.zeros((la*lo,lag_steps))
    Corr_Coeff = numpy.ma.array(z, mask=z)
	
	
    # extract data	
    sat = extract_data(v, V, time_range_indices, ex)	
    # reshape
    sat = np.reshape(sat, (sat.shape[0],-1))
    
    allkeysncdf = list(d.variables.keys())
    dimensionkeys = ['time', 'lat', 'lon', 'latitude', 'longitude']
    var = [keync for keync in allkeysncdf if keync not in dimensionkeys][0]  
    print(('calculating correlation maps for {}'.format(var)))
	
	
    for i in range(lag_steps):

        lag = ex['lag_min'] + i
		
        print(('lag', lag))
        months_indices_lagged = [r - lag for r in ex['RV_period']]
		
        # only winter months 		
        sat_winter = sat[months_indices_lagged]
		
		# correlation map and pvalue at each grid-point:
        corr_di_sat, sig_di_sat = corr_new(sat_winter, RVts)
		
        if ex['FDR_control'] == True:
				
			# test for Field significance and mask unsignificant values			
			# FDR control:
            adjusted_pvalues = multicomp.multipletests(sig_di_sat, method='fdr_bh')			
            ad_p = adjusted_pvalues[1]
			
            corr_di_sat.mask[ad_p> ex['alpha']] = True

        else:
            corr_di_sat.mask[sig_di_sat> ex['alpha']] = True
			
			
            Corr_Coeff[:,i] = corr_di_sat[:]
	
    return Corr_Coeff, lat_grid, lon_grid
	

def plot_corr_coeffs(Corr_Coeff, m, lag_min, lat_grid, lon_grid, title='Corr Maps for different time lags', Corr_mask=False):	
	'''
	This function plots the differnt corr coeffs on map m. the variable title must be a string. If mask==True, only significant values are shown.
	'''
	
	print('plotting correlation maps...')
	n_rows = Corr_Coeff.shape[1]
	fig = plt.figure(figsize=(4, 2*n_rows))
	#fig.subplots_adjust(left=None, bottom = None, right=None, top=0.3, wspace=0.1, hspace= 0.1)

	plt.suptitle(title, fontsize = 14)

	if Corr_Coeff.count()==0:
		vmin = -0.99 
		vmax = 0.99
	
	else:
		vmin = Corr_Coeff.min()
		vmax = Corr_Coeff.max()
		
	maxabs = max([np.abs(vmin), vmax]) + 0.01
	levels = np.linspace(- maxabs, maxabs , 13) 
	levels = [round(elem,2) for elem in levels]
	
	#gs1 = gridspec.GridSpec(2, Corr_Coeff.shape[1]/2) 
	for i in range(Corr_Coeff.shape[1]):
		plt.subplot(Corr_Coeff.shape[1], 1, i+1)
		lag = lag_min +i
		plt.title('lag = -' + str(lag), fontsize =12)
		
		
		corr_di_sat = numpy.ma.array(data = Corr_Coeff[:,i], mask = Corr_Coeff.mask[:,i])
		
		la = lat_grid.shape[0]
		lo = lon_grid.shape[0]
		
		# lons_ext = np.zeros((lon_grid.shape[0]+1))
		# lons_ext[:-1] = lon_grid
		# lons_ext[-1] = 360

		# lons, lats = np.meshgrid(lons_ext,lat_grid)
		lons, lats = np.meshgrid(lon_grid,lat_grid)
		
		
		# reshape for plotting
		corr_di_sat = numpy.reshape(corr_di_sat, (la, lo))
		corr_di_sat_significance = numpy.zeros(corr_di_sat.shape)
		corr_di_sat_significance[corr_di_sat.mask==False]=1				
		
		# # make new dimension for plotting
		# B = np.zeros((corr_di_sat.shape[0], corr_di_sat.shape[1]+1))
		# B[:, :-1] = corr_di_sat
		# B[:, -1] = corr_di_sat[:, 0]
	
		# D = np.zeros((corr_di_sat_significance.shape[0], corr_di_sat_significance.shape[1]+1))
		# D[:, :-1] = corr_di_sat_significance
		# D[:, -1] = corr_di_sat_significance[:, 0]	
		

		# if (Corr_mask==True) | (numpy.sum(corr_di_sat_significance)==0):
		if (Corr_mask==True):
			# plotting otions:
			im = m.contourf(lons,lats, corr_di_sat, vmin = vmin, vmax = vmax, latlon=True, levels = levels, cmap="RdBu_r")
			# m.colorbar(location="bottom")
			plot_basemap_options(m)


		elif (numpy.sum(corr_di_sat_significance)==0):
			im = m.contourf(lons,lats, corr_di_sat.data, vmin = vmin, vmax = vmax, latlon=True, levels = levels, cmap="RdBu_r")
			# m.colorbar(location="bottom")
			plot_basemap_options(m)
		
		else:				
			
			plot_basemap_options(m)		
			im = m.contourf(lons,lats, corr_di_sat.data, vmin = vmin, vmax = vmax, latlon=True, levels = levels, cmap="RdBu_r")				
			m.contour(lons,lats, corr_di_sat_significance, latlon = True, linewidths=0.2,colors='k')

			#m.colorbar(location="bottom")
			#m.scatter(lons,lats,corr_di_sat_significance,alpha=0.7,latlon=True, color="k")
	
	
	# vertical colorbar
	# cax2 = fig.add_axes([0.92, 0.3, 0.013, 0.4])
	# cb = fig.colorbar(im, cax=cax2, orientation='vertical')
	# cb.outline.set_linewidth(.1)
	# cb.ax.tick_params(labelsize = 7)
	
	
	cax2 = fig.add_axes([0.25, 0.07, 0.5, 0.014])
	cb = fig.colorbar(im, cax=cax2, orientation='horizontal')
	cb.outline.set_linewidth(.1)
	cb.ax.tick_params(labelsize = 7)
	
	#fig.tight_layout(rect=[0, 0.03, 1, 0.93])
	return fig




def define_regions_and_rank_new(Corr_Coeff, lat_grid, lon_grid):
	'''
	takes Corr Coeffs and defines regions by strength

	return A: the matrix whichs entries correspond to region. 1 = strongest, 2 = second strongest...
	'''
	print('extracting causal precursor regions ...\n')

	
	# initialize arrays:
	# A final return array 
	A = numpy.ma.copy(Corr_Coeff)
	#========================================
	# STEP 1: mask nodes which were never significantly correlatated to index (= count=0)
	#========================================
	
	#========================================
	# STEP 2: define neighbors for everey node which passed Step 1
	#========================================

	indices_not_masked = numpy.where(A.mask==False)[0].tolist()

	lo = lon_grid.shape[0]
	la = lat_grid.shape[0]
	
	# create list of potential neighbors:
	N_pot=[[] for i in range(A.shape[0])]

	#=====================
	# Criteria 1: must bei geographical neighbors:
	#=====================
	for i in indices_not_masked:
		n = []	

		col_i= i%lo
		row_i = i//lo

		# knoten links oben
		if i==0:
			n= n+[lo-1, i+1, lo ]

		# knoten rechts oben	
		elif i== lo-1:
			n= n+[i-1, 0, i+lo]

		# knoten links unten
		elif i==(la-1)*lo:
			n= n+ [i+lo-1, i+1, i-lo]

		# knoten rechts unten
		elif i == la*lo-1:
			n= n+ [i-1, i-lo+1, i-lo]

		# erste zeile
		elif i<lo:
			n= n+[i-1, i+1, i+lo]
	
		# letzte zeile:
		elif i>la*lo-1:
			n= n+[i-1, i+1, i-lo]
	
		# erste spalte
		elif col_i==0:
			n= n+[i+lo-1, i+1, i-lo, i+lo]
	
		# letzt spalte
		elif col_i ==lo-1:
			n= n+[i-1, i-lo+1, i-lo, i+lo]
	
		# nichts davon
		else:
			n = n+[i-1, i+1, i-lo, i+lo]
	
	#=====================
	# Criteria 2: must be all at least once be significanlty correlated 
	#=====================	
		m =[]
		for j in n:
			if j in indices_not_masked:
					m = m+[j]
		
		# now m contains the potential neighbors of gridpoint i

	
	#=====================	
	# Criteria 3: sign must be the same for each step 
	#=====================				
		l=[]
	
		cc_i = A.data[i]
		cc_i_sign = numpy.sign(cc_i)
		
	
		for k in m:
			cc_k = A.data[k]
			cc_k_sign = numpy.sign(cc_k)
		

			if cc_i_sign *cc_k_sign == 1:
				l = l +[k]

			else:
				l = l
			
		if len(l)==0:
			l =[]
			A.mask[i]=True	
			
		else: l = l +[i]	
		
		
		N_pot[i]=N_pot[i]+ l	



	#========================================	
	# STEP 3: merge overlapping set of neighbors
	#========================================
	
	Regions = merge_neighbors(N_pot)
	
	#========================================
	# STEP 4: assign a value to each region
	#========================================
	

	# 2) combine 1A+1B 
	B = numpy.abs(A)
	
	# 3) calculate the area size of each region	
	
	Area =  [[] for i in range(len(Regions))]
	
	for i in range(len(Regions)):
		indices = numpy.array(list(Regions[i]))
		indices_lat_position = indices//lo
		lat_nodes = lat_grid[indices_lat_position[:]]
		cos_nodes = numpy.cos(numpy.deg2rad(lat_nodes))		
		
		area_i = [numpy.sum(cos_nodes)]
		Area[i]= Area[i]+area_i
	
	#---------------------------------------
	# OPTIONAL: Exclude regions which only consist of less than n nodes
	# 3a)
	#---------------------------------------	
	
	R=[]
	Ar=[]
	for i in range(len(Regions)):
		if len(Regions[i])>=5:
			R.append(Regions[i])
			Ar.append(Area[i])
	
	Regions = R
	Area = Ar	
	
	
	
	# 4) calcualte region value:
	
	C = numpy.zeros(len(Regions))
	
	Area = numpy.array(Area)
	for i in range(len(Regions)):
		C[i]=Area[i]*numpy.mean(B[list(Regions[i])])


	
	
	# mask out those nodes which didnot fullfill the neighborhood criterias
	A.mask[A==0] = True	
		
		
	#========================================
	# STEP 5: rank regions by region value
	#========================================
	
	# rank indices of Regions starting with strongest:
	sorted_region_strength = numpy.argsort(C)[::-1]
	
	# give ranking number
	# 1 = strongest..
	# 2 = second strongest
	
	for i in range(len(Regions)):
		j = list(sorted_region_strength)[i]
		A[list(Regions[j])]=i+1
		
	return A	
	
	

	

def calc_actor_ts_and_plot(Corr_Coeff, actbox, ex, lat_grid, lon_grid, var):
    """
	Calculates the time-series of the actors based on the correlation coefficients and plots the according regions. 
	Only caluclates regions with significant correlation coefficients
	"""
    if Corr_Coeff.ndim == 1:
        lag_steps = 1
        n_rows = 1
    else:
        lag_steps = Corr_Coeff.shape[1]
        n_rows = Corr_Coeff.shape[1]

	
    la_gph = lat_grid.shape[0]
    lo_gph = lon_grid.shape[0]
    lons_gph, lats_gph = numpy.meshgrid(lon_grid, lat_grid)

    cos_box_gph = numpy.cos(numpy.deg2rad(lats_gph))
    cos_box_gph_array = np.repeat(cos_box_gph[None,:], actbox.shape[0], 0)
    cos_box_gph_array = np.reshape(cos_box_gph_array, (cos_box_gph_array.shape[0], -1))


    Actors_ts_GPH = [[] for i in range(lag_steps)]
	
	#test = [len(a) for a in Actors_ts_GPH]
	#print test
	

	
#	fig_GPH = plt.figure(figsize=(4, 2*n_rows))
#	plt.suptitle(title, fontsize=14)	
	#cmap_regions = 'Paired' 
	#cmap_regions = plt.get_cmap('Paired')

    cmap_regions = matplotlib.colors.ListedColormap(sns.color_palette("Set2"))
    cmap_regions.set_bad('w')
	
    colors = ["faded green"]
    color = sns.xkcd_palette(colors)

    Number_regions_per_lag = np.zeros(lag_steps)
    x = 0
	# vmax = 50
    for i in range(lag_steps):
		
        if Corr_Coeff.ndim ==1:
            Regions_lag_i = define_regions_and_rank_new(Corr_Coeff, lat_grid, lon_grid)
		
        else:
            Regions_lag_i = define_regions_and_rank_new(Corr_Coeff[:,i], lat_grid, lon_grid)
		
		
        if Regions_lag_i.max()> 0:
            n_regions_lag_i = int(Regions_lag_i.max())
            print(('{} regions detected for lag {}, variable {}'.format(n_regions_lag_i, ex['lag_min']+i,var)))
            x_reg = numpy.max(Regions_lag_i)
			
#            levels = numpy.arange(x, x + x_reg +1)+.5
            A_r = numpy.reshape(Regions_lag_i, (la_gph, lo_gph))
            A_r + x
            
#            A_number_region = numpy.zeros(A_r.shape)
#            A_number_region[A_r == number_region]=1
            
#            xr_A_num_reg = xr.DataArray(data=A_r, coords=[lat_grid, lon_grid], dims=('latitude','longitude'))
#            map_proj = map_proj
#            plt.figure(figsize=(6, 4))
#            ax = plt.axes(projection=map_proj)
#            im = xr_A_num_reg.plot.pcolormesh(ax=ax, cmap=plt.cm.BuPu,
#                             transform=ccrs.PlateCarree(), add_colorbar=True)
#            plt.colorbar(im, ax=ax , orientation='horizontal')
#            ax.coastlines(color='grey', alpha=0.3)
#            ax.set_title('lag = -' + str(lag), fontsize=12)
            
#			plt.subplot(lag_steps,1, i+1)
#			lag = ex['lag_min'] +i
#			plt.title('lag = -' + str(lag), fontsize=12)
			
#			plot_basemap_options(m)		
			#m.contourf(lons_gph,lats_gph, A_r, levels, latlon = True, cmap = cmap_regions)
			
			# all in one color:
#			m.contourf(lons_gph,lats_gph, A_r, levels, latlon = True, colors = color, vmin = 1, vmax = n_regions_lag_i)

			
			# if colors should be different for each subplot:
			#m.contourf(lons_gph,lats_gph, A_r, levels, latlon = True, cmap = cmap_regions, vmin = 1, cmax = vmax)
			
			
			#m.colorbar(location="bottom", ticks = numpy.arange(x+1, x+ x_reg+1))

            x = A_r.max() 

			# this array will be the time series for each region
            ts_regions_lag_i = np.zeros((actbox.shape[0], n_regions_lag_i))
				
            for j in range(n_regions_lag_i):
                B = np.zeros(Regions_lag_i.shape)
                B[Regions_lag_i == j+1] = 1	
                ts_regions_lag_i[:,j] = np.mean(actbox[:, B == 1] * cos_box_gph_array[:, B == 1], axis =1)

            Actors_ts_GPH[i] = ts_regions_lag_i
		
        else:
            print(('no regions detected for lag ', ex['lag_min'] + i))	
            Actors_ts_GPH[i] = np.array([])
            n_regions_lag_i = 0
		
        Number_regions_per_lag[i] = n_regions_lag_i
		

    if np.sum(Number_regions_per_lag) ==0:
        print('no regions detected at all')
        Actors_GPH = np.array([])
	
    else:
        print((np.sum(Number_regions_per_lag), ' regions detected in total'))
		
		# check for whcih lag the first regions are detected
        d = 0
		
        while (Actors_ts_GPH[d].shape[0]==0) & (d < lag_steps):
            d = d+1
            print(d)
		
		# make one array out of it:
        Actors_GPH = Actors_ts_GPH[d]
		
        for i in range(d+1, len(Actors_ts_GPH)):
            if Actors_ts_GPH[i].shape[0]>0:		
				
                Actors_GPH = np.concatenate((Actors_GPH, Actors_ts_GPH[i]), axis = 1)		
		
			# if Actors_ts_GPH[i].shape[0]==0:
				# print i+1
				
			# else:
				# Actors_GPH = np.concatenate((Actors_GPH, Actors_ts_GPH[i]), axis = 1)
		

    return Actors_GPH, Number_regions_per_lag#, fig_GPH
	

	
def print_particular_region(number_region, Corr_Coeff_lag_i, actor, map_proj, title):
#    (number_region, Corr_Coeff_lag_i, latitudes, longitudes, map_proj, title)=(according_number, Corr_precursor[:, :], actor.lat_grid, actor.lon_grid, map_proj, according_fullname) 
    #%%
    # check if only one lag is tested:
    if Corr_Coeff_lag_i.ndim == 1:
        lag_steps = 1

    else:
        lag_steps = Corr_Coeff_lag_i.shape[1]

    latitudes = actor.lat_grid
    longitudes = actor.lon_grid
    
    x = 0
    for i in range(lag_steps):
	
        if Corr_Coeff_lag_i.ndim == 1:
            Regions_lag_i = define_regions_and_rank_new(Corr_Coeff_lag_i, latitudes, longitudes)
		
        else:	
            Regions_lag_i = define_regions_and_rank_new(Corr_Coeff_lag_i[:,i], latitudes, longitudes)
		
        if Regions_lag_i.count()==0:
            n_regions_lag_i = 0
		
        else:	
            n_regions_lag_i = int(Regions_lag_i.max())
            x_reg = numpy.max(Regions_lag_i)	
            levels = numpy.arange(x, x + x_reg +1)+.5

		
            A_r = numpy.reshape(Regions_lag_i, (latitudes.size, longitudes.size))
            A_r = A_r + x			
            x = A_r.max() 
            print(x)
		
		
        if (x >= number_region) & (x>0):
					
            A_number_region = numpy.zeros(A_r.shape)
            A_number_region[A_r == number_region]=1
            xr_A_num_reg = xr.DataArray(data=A_number_region, coords=[latitudes, longitudes], dims=('latitude','longitude'))
            map_proj = map_proj
            plt.figure(figsize=(6, 4))
            ax = plt.axes(projection=map_proj)
            im = xr_A_num_reg.plot.pcolormesh(ax=ax, cmap=plt.cm.BuPu,
                             transform=ccrs.PlateCarree(), add_colorbar=False)
            plt.colorbar(im, ax=ax , orientation='horizontal')
            ax.coastlines(color='grey', alpha=0.3)
            ax.set_title(title)
            
            break
#%%
    return

	
		
	



