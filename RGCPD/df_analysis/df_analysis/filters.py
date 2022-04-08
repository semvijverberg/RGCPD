import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import xarray as xr


def lowpass(ts, period):
    """ lowpass filter
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
    ts     .. time series array with axis 0 as time axis
    period .. cutoff period
    """
    N  = 2         # Filter order
    Wn = 1/period  # Cutoff frequency
    B, A = signal.butter(N, Wn, output='ba')
    filtered = signal.filtfilt(B, A, ts, axis=(0), padlen=N-1, padtype='constant')
    
    if type(ts)==xr.core.dataarray.DataArray:
        ts_new = ts.copy()
        ts_new.values = filtered
    else:
        ts_new = filtered
    
    return ts_new


def highpass(ts, period):
    """ highpass filter
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
    ts     .. time series array with axis 0 as time axis
    period .. cutoff period
    """
    N  = 2         # Filter order
    Wn = 1/period  # Cutoff frequency
    B, A = signal.butter(N, Wn, output='ba', btype='highpass')
    filtered = signal.filtfilt(B, A, ts, axis=(0), padlen=N-1, padtype='constant')
    
    if type(ts)==xr.core.dataarray.DataArray:
        ts_new = ts.copy()
        ts_new.values = filtered
    else:
        ts_new = filtered
    
    return ts_new


def bandpass(ts, periods):
    """ highpass filter
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
    ts     .. time series array with axis 0 as time axis
    period .. cutoff period
    """
    N  = 2         # Filter order
    Wn = 1/periods  # Cutoff frequencies
    B, A = signal.butter(N, Wn, output='ba', btype='bandpass')
    filtered = signal.filtfilt(B, A, ts, axis=(0), padlen=N-1, padtype='constant')
    
    if type(ts)==xr.core.dataarray.DataArray:
        ts_new = ts.copy()
        ts_new.values = filtered
    else:
        ts_new = filtered
    
    return ts_new


def chebychev(ts, period):
    N, rp = 6, 1  # N=6 was used in Henley et al. (2015), rp is low to minimize ripples
    Wn = 1/period
    B, A = signal.cheby1(N, rp, Wn)
    filtered = signal.filtfilt(B, A, ts, axis=(0), padlen=N-1, padtype='constant')
    
    if type(ts)==xr.core.dataarray.DataArray:
        ts_new = ts.copy()
        ts_new.values = filtered
    else:
        ts_new = filtered
    
    return ts_new


def notch(ts, period):
    """ single frequency filter 
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.iirnotch.html
    period .. 1/f to filter out
    """
    w0 = 1/period
    Q  = 1.#.2# found to work fairly well for deseasonalizing
    B, A = signal.iirnotch(w0, Q)
    filtered = signal.filtfilt(B, A, ts, axis=(0), padlen=12)
    
    if type(ts)==xr.core.dataarray.DataArray:
        ts_new = ts.copy()
        ts_new.values = filtered
    else:
        ts_new = filtered
    
    return ts_new


def deseasonalize_notch(ts):
    ts_ds = lowpass(lowpass(notch(ts, 12), 12),12)
    return ts_ds


def deseasonalize(ts):
    """ removing monthly mean climatology """
    dt_year = (ts.time[12]-ts.time[0]).values  #
    ts_ds = ts.copy()
    yrly = ts.groupby_bins('time', np.arange(0,len(ts.time)+1,12)/12*dt_year, right=False).mean(dim='time').rename({'time_bins':'time'})
    for j in range(12):
        m = ts.isel(time=slice(j,len(ts)+1,12))
        ts_ds[j::12,:,:] -= (m-yrly.assign_coords(time=m.time)).mean(dim='time')
    return ts_ds



def Lanczos_weights(window, cutoff):
    """Calculate weights for a low pass Lanczos filter.
    window: int     The length of the filter window.
    cutoff: float   The cutoff frequency in inverse time steps.
    """
    order = ((window - 1) // 2 ) + 1
    nwts = 2 * order + 1
    w = np.zeros([nwts])
    n = nwts // 2
    w[n] = 2 * cutoff
    k = np.arange(1., n)
    sigma = np.sin(np.pi * k / n) * n / (np.pi * k)
    firstfactor = np.sin(2. * np.pi * cutoff * k) / (np.pi * k)
    w[n-1:0:-1] = firstfactor * sigma
    w[n+1:-1] = firstfactor * sigma
    weights = xr.DataArray(w[1:-1], dims=['window'])
    return weights


def Lanczos(da, window=121, cutoff=1/120):
    from xr_regression import xr_lintrend
    """ Lanczos lowpass filter with reflective boundary """
    lpw = Lanczos_weights(window=window, cutoff=cutoff)
    da_trend = xr_lintrend(da)
    da_dt = da - da_trend
    da_dt_inv = da_dt.reindex(time=da_dt.time[::-1])
    extended = xr.concat([da_dt_inv, da_dt, da_dt_inv], 'time')
    extended.time.values = np.arange(len(extended.time))
    pos = int(len(extended.time.values)/3)
    da_filtered = extended.rolling(time=window, center=True).construct('window').dot(lpw)[pos:2*pos]/lpw.sum()
    da_filtered.time.values = da.time
    da_filtered = da_filtered + da_trend
#     print(da_filtered)
    return da_filtered
