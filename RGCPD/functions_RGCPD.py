# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
matplotlib.rcParams['backend'] = "Qt4Agg"
from pylab import *
import matplotlib.pyplot as plt
import wrapper_RGCPD_tig
#from mpl_toolkits.basemap import Basemap, shiftgrid, cm
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

def extract_data(d, D, ex):	
	"""
	Extracts the array of variable d for indices index_range over the domain box
	d: netcdf elements
	D: the data array
	index_range: a list containing the start and the end index, e.g. [0, time_cycle*n_years]
	"""
	
	
	index_0 = ex['time_range_all'][0]
	index_n = ex['time_range_all'][1]

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
	m.drawmeridians(np.arange(0, 360, 30), color='lightgray')
	m.drawparallels(np.arange(-90, 90, 30), color='lightgray')


	
	

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
	x = np.ma.zeros(D.shape[1])
	corr_di_D = np.ma.array(data = x, mask =False)	
	sig_di_D = np.array(x)
	
	for i in range(D.shape[1]):
		r, p = scipy.stats.pearsonr(di,D[:,i])
		corr_di_D[i]= r 
		sig_di_D[i]= p
									
	return corr_di_D, sig_di_D

	
def calc_corr_coeffs_new(ncdf, precur_arr, RV, ex):
    #%%
#    v = ncdf ; V = array ; RV.RV_ts = ts of RV, time_range_all = index range of whole ts
    """
    This function calculates the correlation maps for fied V for different lags. Field significance is applied to test for correltion.
    This function uses the following variables (in the ex dictionary)
    ncdf: netcdf element
    prec_arr: array
    box: list of form [la_min, la_max, lo_min, lo_max]
    time_range_all: a list containing the start and the end index, e.g. [0, time_cycle*n_years]
    lag_steps: number of lags
    time_cycle: time cycyle of dataset, =12 for monthly data...
    RV_period: indices that matches the response variable time series
    alpha: significance level

    """
    lag_steps = ex['lag_max'] - ex['lag_min'] +1
    
    assert lag_steps >= 0, ('Maximum lag is larger then minimum lag, not allowed')
		
    d = ncdf
	
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
	
    lons, lats = np.meshgrid(lon_grid,lat_grid)

#    A1 = np.zeros((la,lo))
    z = np.zeros((la*lo,lag_steps))
    Corr_Coeff = np.ma.array(z, mask=z)
	
	


    # reshape
    sat = np.reshape(precur_arr.values, (precur_arr.shape[0],-1))
    
    allkeysncdf = list(d.variables.keys())
    dimensionkeys = ['time', 'lat', 'lon', 'latitude', 'longitude', 'mask', 'levels']
    var = [keync for keync in allkeysncdf if keync not in dimensionkeys][0]  
    print('\ncalculating correlation maps for {}'.format(var))
	
	
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
		
		
		corr_di_sat = np.ma.array(data = Corr_Coeff[:,i], mask = Corr_Coeff.mask[:,i])
		
		la = lat_grid.shape[0]
		lo = lon_grid.shape[0]
		
		# lons_ext = np.zeros((lon_grid.shape[0]+1))
		# lons_ext[:-1] = lon_grid
		# lons_ext[-1] = 360

		# lons, lats = np.meshgrid(lons_ext,lat_grid)
		lons, lats = np.meshgrid(lon_grid,lat_grid)
		
		
		# reshape for plotting
		corr_di_sat = np.reshape(corr_di_sat, (la, lo))
		corr_di_sat_significance = np.zeros(corr_di_sat.shape)
		corr_di_sat_significance[corr_di_sat.mask==False]=1				
		
		# # make new dimension for plotting
		# B = np.zeros((corr_di_sat.shape[0], corr_di_sat.shape[1]+1))
		# B[:, :-1] = corr_di_sat
		# B[:, -1] = corr_di_sat[:, 0]
	
		# D = np.zeros((corr_di_sat_significance.shape[0], corr_di_sat_significance.shape[1]+1))
		# D[:, :-1] = corr_di_sat_significance
		# D[:, -1] = corr_di_sat_significance[:, 0]	
		

		# if (Corr_mask==True) | (np.sum(corr_di_sat_significance)==0):
		if (Corr_mask==True):
			# plotting otions:
			im = m.contourf(lons,lats, corr_di_sat, vmin = vmin, vmax = vmax, latlon=True, levels = levels, cmap="RdBu_r")
			# m.colorbar(location="bottom")
			plot_basemap_options(m)


		elif (np.sum(corr_di_sat_significance)==0):
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


def define_regions_and_rank_new(Corr_Coeff, lat_grid, lon_grid, ex):
    #%%
    '''
	takes Corr Coeffs and defines regions by strength

	return A: the matrix whichs entries correspond to region. 1 = strongest, 2 = second strongest...
    '''
#    print('extracting features ...\n')

	
	# initialize arrays:
	# A final return array 
    A = np.ma.copy(Corr_Coeff)
#    A = np.ma.zeros(Corr_Coeff.shape)
	#========================================
	# STEP 1: mask nodes which were never significantly correlatated to index (= count=0)
	#========================================
	
	#========================================
	# STEP 2: define neighbors for everey node which passed Step 1
	#========================================

    indices_not_masked = np.where(A.mask==False)[0].tolist()

    lo = lon_grid.shape[0]
    la = lat_grid.shape[0]
	
	# create list of potential neighbors:
    N_pot=[[] for i in range(A.shape[0])]

	#=====================
	# Criteria 1: must bei geographical neighbors:
    n_between = ex['prec_reg_max_d']
	#=====================
    for i in indices_not_masked:
        neighb = []
        def find_neighboors(i, lo):
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
            return n
        
        for t in range(n_between+1):
            direct_n = find_neighboors(i, lo)
            if t == 0:
                neighb.append(direct_n)
            if t == 1:
                for n in direct_n:
                    ind_n = find_neighboors(n, lo)
                    neighb.append(ind_n)
        n = list(set(flatten(neighb)))
        if i in n:
            n.remove(i)
        
	
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
        cc_i_sign = np.sign(cc_i)
		
	
        for k in m:
            cc_k = A.data[k]
            cc_k_sign = np.sign(cc_k)
		

            if cc_i_sign *cc_k_sign == 1:
                l = l +[k]

            else:
                l = l
			
            if len(l)==0:
                l =[]
                A.mask[i]=True	
    			
            elif i not in l: 
                l = l + [i]	
		
		
            N_pot[i]=N_pot[i] + l	



	#========================================	
	# STEP 3: merge overlapping set of neighbors
	#========================================
    Regions = merge_neighbors(N_pot)
	
	#========================================
	# STEP 4: assign a value to each region
	#========================================
	

	# 2) combine 1A+1B 
    B = np.abs(A)
	
	# 3) calculate the area size of each region	
	
    Area =  [[] for i in range(len(Regions))]
	
    for i in range(len(Regions)):
        indices = np.array(list(Regions[i]))
        indices_lat_position = indices//lo
        lat_nodes = lat_grid[indices_lat_position[:]]
        cos_nodes = np.cos(np.deg2rad(lat_nodes))		
		
        area_i = [np.sum(cos_nodes)]
        Area[i]= Area[i]+area_i
	
	#---------------------------------------
	# OPTIONAL: Exclude regions which only consist of less than n nodes
	# 3a)
	#---------------------------------------	
	
    # keep only regions which are larger then the mean size of the regions
    if ex['min_n_gc'] == 'mean':
        n_nodes = int(np.mean([len(r) for r in Regions]))
    else:
        n_nodes = ex['min_n_gc']
    
    R=[]
    Ar=[]
    for i in range(len(Regions)):
        if len(Regions[i])>=n_nodes:
            R.append(Regions[i])
            Ar.append(Area[i])
	
    Regions = R
    Area = Ar	
	
	
	
	# 4) calcualte region value:
	
    C = np.zeros(len(Regions))
	
    Area = np.array(Area)
    for i in range(len(Regions)):
        C[i]=Area[i]*np.mean(B[list(Regions[i])])


	
	
#	 mask out those nodes which didnot fullfill the neighborhood criterias
#    A.mask[A==0] = True	
		
		
	#========================================
	# STEP 5: rank regions by region value
	#========================================
	
	# rank indices of Regions starting with strongest:
    sorted_region_strength = np.argsort(C)[::-1]
	
	# give ranking number
	# 1 = strongest..
	# 2 = second strongest
    
    # create clean array
    Regions_lag_i = np.zeros(A.data.shape)
    for i in range(len(Regions)):
        j = list(sorted_region_strength)[i]
        Regions_lag_i[list(Regions[j])]=i+1
    
    Regions_lag_i = np.array(Regions_lag_i, dtype=int)
    Regions_lag_i = np.ma.masked_where(Regions_lag_i==0, Regions_lag_i)
    #%%
    return Regions_lag_i
	

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
            np_regs = np.reshape(np_dbregs, (lats.size, lons.size, lag_steps))
            np_regs = np_regs.swapaxes(0,-1).swapaxes(1,2)
        return np_regs, labels_sign_lag


    prec_labels_np = np.zeros( (lag_steps, lats.size, lons.size) )
#    labels_sign = np.zeros( (lag_steps), dtype=list )
    mask_and_data = Corr_Coeff.copy()
    # prec_labels_np: shape [lags, lats, lons]
    prec_labels_np, labels_sign_lag = mask_sig_to_cluster(mask_and_data, lons, lats, wght_area, ex)
    
    

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

    
#    actor.labels_sign   = labels_sign
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
    
    
    ex['max_N_regs'] = min(20, int(prec_labels.max() + 0.5))
    label_weak = np.nan_to_num(prec_labels.values) >=  ex['max_N_regs']
    prec_labels.values[label_weak] = ex['max_N_regs']
    
    
    if lag_steps >= 2:
        adjust_vert_cbar = 0.0; adj_fig_h=1.4
    elif lag_steps < 2:
        adjust_vert_cbar = 0.1 ; adj_fig_h = 1.4
    
        
    cmap = plt.cm.tab20
    for_plt = prec_labels.copy()
    for_plt.values = for_plt.values-0.5
    kwrgs = dict( {'title' : for_plt.attrs['title'], 'clevels' : 'notdefault', 
                   'steps' : ex['max_N_regs']+1, 'subtitles': None,
                   'vmin' : 0, 'vmax' : ex['max_N_regs'], 
                   'cmap' : cmap, 'column' : 1,
                   'cbar_vert' : adjust_vert_cbar, 'cbar_hght' : 0.0,
                   'adj_fig_h' : adj_fig_h, 'adj_fig_w' : 1., 
                   'hspace' : 0.0, 'wspace' : 0.08, 
                   'cticks_center' : False, 'title_h' : 0.95} )
    filename = '{}_labels_init_{}_vs_{}'.format(ex['params'], ex['RV_name'], actor.name) + ex['file_type2']
    plotting_wrapper(for_plt, ex, filename, kwrgs=kwrgs)

#    for_plt.where(for_plt.values==3)[0].plot()

#    if np.sum(Number_regions_per_lag) != 0:
#        assert np.where(np.isnan(tsCorr))[1].size < 0.5*tsCorr[:,0].size, ('more '
#                       'then 10% nans found, i.e. {} out of {} datapoints'.format(
#                               np.where(np.isnan(tsCorr))[1].size), tsCorr.size)
#        while np.where(np.isnan(tsCorr))[1].size != 0:
#            nans = np.where(np.isnan(tsCorr))
#            print('{} nans were found in timeseries of regions out of {} datapoints'.format(
#                    nans[1].size, tsCorr.size))
#            tsCorr[nans[0],nans[1]] = tsCorr[nans[0]-1,nans[1]]
#            print('taking value of previous timestep')
    #%%
    return actor, ex

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

def calc_actor_ts_and_plot(Corr_Coeff, actbox, ex, lat_grid, lon_grid, var):
    #%%
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
    lons_gph, lats_gph = np.meshgrid(lon_grid, lat_grid)

    cos_box_gph = np.cos(np.deg2rad(lats_gph))
    cos_box_gph_array = np.repeat(cos_box_gph[None,:], actbox.shape[0], 0)
    cos_box_gph_array = np.reshape(cos_box_gph_array, (cos_box_gph_array.shape[0], -1))


    Actors_ts_GPH = [[] for i in range(lag_steps)]
	
	#test = [len(a) for a in Actors_ts_GPH]
	#print test


    Number_regions_per_lag = np.zeros(lag_steps)
    x = 0
    
    
    for i in range(lag_steps):
		
        if Corr_Coeff.ndim ==1:
            Regions_lag_i = define_regions_and_rank_new(Corr_Coeff, lat_grid, lon_grid, ex)
		
        else:
            Regions_lag_i = define_regions_and_rank_new(Corr_Coeff[:,i], lat_grid, lon_grid, ex)
		
        
        
        if Regions_lag_i.max()> 0:
            n_regions_lag_i = int(Regions_lag_i.max())
            print(('{} regions detected for lag {}, variable {}'.format(n_regions_lag_i, ex['lag_min']+i,var)))
            x_reg = np.max(Regions_lag_i)
			
#            levels = np.arange(x, x + x_reg +1)+.5
            A_r = np.reshape(Regions_lag_i, (la_gph, lo_gph))
            A_r + x
            
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
        tsCorr = np.array([])
	
    else:
        print('{} regions detected in total\n'.format(
                        np.sum(Number_regions_per_lag)))
		
		# check for whcih lag the first regions are detected
        d = 0
		
        while (Actors_ts_GPH[d].shape[0]==0) & (d < lag_steps):
            d = d+1
            print(d)
		
		# make one array out of it:
        tsCorr = Actors_ts_GPH[d]
		
        for i in range(d+1, len(Actors_ts_GPH)):
            if Actors_ts_GPH[i].shape[0]>0:		
				
                tsCorr = np.concatenate((tsCorr, Actors_ts_GPH[i]), axis = 1)		
		
			# if Actors_ts_GPH[i].shape[0]==0:
				# print i+1
				
			# else:
				# tsCorr = np.concatenate((tsCorr, Actors_ts_GPH[i]), axis = 1)
    if np.sum(Number_regions_per_lag) != 0:
        assert np.where(np.isnan(tsCorr))[1].size < 0.5*tsCorr[:,0].size, ('more '
                       'then 10% nans found, i.e. {} out of {} datapoints'.format(
                               np.where(np.isnan(tsCorr))[1].size), tsCorr.size)
        while np.where(np.isnan(tsCorr))[1].size != 0:
            nans = np.where(np.isnan(tsCorr))
            print('{} nans were found in timeseries of regions out of {} datapoints'.format(
                    nans[1].size, tsCorr.size))
            tsCorr[nans[0],nans[1]] = tsCorr[nans[0]-1,nans[1]]
            print('taking value of previous timestep')
    #%%
    return tsCorr, Number_regions_per_lag#, fig_GPH
	


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


def print_particular_region(ex, number_region, Corr_Coeff_lag_i, actor, map_proj, title):
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
            Regions_lag_i = define_regions_and_rank_new(Corr_Coeff_lag_i, 
                                                        latitudes, longitudes, ex)
		
        else:	
            Regions_lag_i = define_regions_and_rank_new(Corr_Coeff_lag_i[:,i], 
                                                        latitudes, longitudes, ex)
		
        if np.max(Regions_lag_i.data)==0:
            n_regions_lag_i = 0
		
        else:	
            n_regions_lag_i = int(np.max(Regions_lag_i.data))
#            x_reg = np.max(Regions_lag_i)	
#            levels = np.arange(x, x + x_reg +1)+.5

		
            A_r = np.reshape(Regions_lag_i, (latitudes.size, longitudes.size))
            A_r = A_r + x			
            x = A_r.max() 
            print(x)
		
		
        if (x >= number_region) & (x>0):
					
            A_number_region = np.zeros(A_r.shape)
            A_number_region[A_r == number_region]=1
            xr_A_num_reg = xr.DataArray(data=A_number_region, coords=[latitudes, longitudes], 
                                        dims=('latitude','longitude'))
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

    cbar_ax = g.fig.add_axes([0.25, cbar_vert, 
                                  0.5, cbar_hght], label='cbar')

    if 'clim' in kwrgs.keys(): #adjust the range of colors shown in cbar
        cnorm = np.linspace(kwrgs['clim'][0],kwrgs['clim'][1],11)
        vmin = kwrgs['clim'][0]
    else:
        cnorm = clevels

    norm = mpl.colors.BoundaryNorm(boundaries=cnorm, ncolors=256)
    cbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, orientation='horizontal', 
                 extend=extend, ticks=cnorm, norm=norm)
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



