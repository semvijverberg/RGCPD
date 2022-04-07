"""
contains 3 classes:
"""

import matplotlib.pyplot as plt
import mtspec
import numpy as np
import scipy as sp
import scipy.stats as stats
import xarray as xr
from filters import chebychev, lowpass
from statsmodels.stats.weightstats import DescrStatsW
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_process import ArmaProcess

# from ba_analysis_dataarrays import AnalyzeDataArray

    
    
class AnalyzeTimeSeries():
    """ functions to analyze single 1D xr time series  
    > spectrum
    > AR(1) fitting
    > MC spectral (uncertainty) estimates
    
    """
    
    def __init__(self, ts):
        """ time series analysis
        ts .. (np.ndarray) regularly sampled time series data
        """
        self.ts = ts
        # assert type(self.ts)==np.ndarray
        assert 'time' in ts.coords
        self.len = len(self.ts)
        
        
    def rolling_trends(self, window=11):
        """ returns trends over a rolling window """
        assert type(window)==int
        da = xr.DataArray(data=np.full((self.len), np.nan),
                          coords={'time': self.ts.time}, dims=('time'))
        for t in range(self.len-window):
            da[int(np.floor(window/2))+t] = np.polyfit(np.arange(window), self.ts[t:t+window], 1)[0]
        return da
    
    
    def periodogram(self, d=1., data=None):
        """ naive absolute squared Fourier components 

        input:
        ts .. time series
        d  .. sampling period
        """
        if data is None:  data = np.array(self.ts)
        assert type(data)==np.ndarray
        freq = np.fft.rfftfreq(len(data))
        Pxx = 2*np.abs(np.fft.rfft(data))**2/len(data)
        return freq, Pxx
    

    def Welch(self, d=1, window='hann', data=None):
        """ Welch spectrum """
        if data is None:  data = np.array(self.ts)
        assert type(data)==np.ndarray
        return sp.signal.welch(data, window=window)
    

    def mtspectrum(self, data=None, d=1., tb=4, nt=4):
        """ multi-taper spectrum 

        input:
        ts .. time series
        d  .. sampling period
        tb .. time bounds (bandwidth)
        nt .. number of tapers
        """
        if data is None:  data = np.array(self.ts)
        assert type(data)==np.ndarray
        spec, freq, jackknife, _, _ = mtspec.mtspec(
                    data=data, delta=d, time_bandwidth=tb,
                    number_of_tapers=nt, statistics=True)
        return freq, spec
    
    
    def spectrum(self, data=None, filter_type=None, filter_cutoff=None):
        """ multitaper spectrum """
        if data is None:  data = np.array(self.ts)
        assert type(data)==np.ndarray
        if filter_type is not None:  data = self.filter_timeseries(data, filter_type, filter_cutoff)
        spec, freq, jackknife, _, _ = mtspec.mtspec(
                data=data, delta=1., time_bandwidth=4,
                number_of_tapers=5, statistics=True)
        return (spec, freq, jackknife)
    
    
    def filter_timeseries(self, data, filter_type, filter_cutoff):
        assert filter_type in ['lowpass', 'chebychev']
        assert type(filter_cutoff)==int
        assert filter_cutoff>1
        n = int(filter_cutoff/2)+1  # datapoints to remove from either end due to filter edge effects
        if filter_type=='lowpass':      data = lowpass(data, filter_cutoff)[n:-n]
        elif filter_type=='chebychev':  data = chebychev(data, filter_cutoff)[n:-n]
        return data

    
    def autocorrelation(self, data=None, n=1):
        """ calculates the first n lag autocorrelation coefficient """
        if data is None:
            assert n<self.len
            data = self.ts
        
        n += 1  # zeroth element is lag-0 autocorrelation = 1
        acs = np.ones((n))
        for i in np.arange(1,n):
            acs[i] = np.corrcoef(data[:-i]-data[:-i].mean(), data[i:]-data[i:].mean())[0,1]
        return acs
    
        
    def mc_ar1_ARMA(self, phi, std, n, N=1000):
        """ Monte-Carlo AR(1) processes

        input:
        phi .. (estimated) lag-1 autocorrelation
        std .. (estimated) standard deviation of noise
        n   .. length of original time series
        N   .. number of MC simulations 
        """
        AR_object = ArmaProcess(np.array([1, -phi]), np.array([1]), nobs=n)
        mc = AR_object.generate_sample(nsample=(N,n), scale=std, axis=1, burnin=1000)
        return mc
    
    
        
    def mc_ar1_spectrum(self, data=None, N=1000, filter_type=None, filter_cutoff=None, spectrum='mtm'):
        """ calculates the Monte-Carlo spectrum
        with 1, 2.5, 5, 95, 97.5, 99 percentiles

        input:
        x             .. time series
        spectrum      .. spectral density estimation function
        N             .. number of MC simulations
        filter_type   ..
        filter_cutoff ..
        spectrum      .. 'mtm': multi-taper method, 'per': periodogram, 'Welch'

        output:
        mc_spectrum   .. 
        """
        if data is None:  data = np.array(self.ts)
        assert type(data)==np.ndarray
        AM = ARMA(endog=data, order=(1,0)).fit()
        phi, std = AM.arparams[0], np.sqrt(AM.sigma2)
        mc = self.mc_ar1_ARMA(phi=phi, std=std, n=len(data), N=N)

        if filter_type is not None:
            assert filter_type in ['lowpass', 'chebychev']
            assert type(filter_cutoff)==int
            assert filter_cutoff>1
            n = int(filter_cutoff/2)+1  # datapoints to remove from either end due to filter edge effects
            mc = mc[:,n:-n]
            if filter_type=='lowpass':      mc = lowpass(mc.T, filter_cutoff).T
            elif filter_type=='chebychev':  mc = chebychev(mc.T, filter_cutoff).T

        if spectrum=='mtm':     freq, _ = self.mtspectrum()
        elif spectrum=='per':   freq, _ = self.periodogram()
        elif spectrum=='Welch': freq, _ = self.Welch()
                
        mc_spectra = np.zeros((N, len(freq)))
#         mc_spectra = np.zeros((N, int(len(mc[0,:])/2)+1))#int(self.len/2)+1))

        for i in range(N):
            if spectrum=='mtm': freq, mc_spectra[i,:] = self.mtspectrum(data=mc[i,:])
            elif spectrum=='per': freq, mc_spectra[i,:] = self.periodogram(data=mc[i,:])
            elif spectrum=='Welch': freq, mc_spectra[i,:] = self.Welch(data=mc[i,:])
        mc_spectrum = {}
        mc_spectrum['median'] = np.median(mc_spectra, axis=0)
        mc_spectrum['freq'] = freq
        for p in [1,2.5,5,95,97.5,99]:
            mc_spectrum[str(p)] = np.percentile(mc_spectra, p, axis=0)
        return mc_spectrum
    
    
    @staticmethod
    def test_homoscedasticity(X, Y):
        X1, Y1 = X, Y
        if len(X)>150:  X1 = X[-150:]
        if len(Y)>150:  Y1 = Y[-150:]
        print(f'{stats.levene(X, Y)[1]:4.2e}, {stats.levene(X1, Y1)[1]:4.2e}')
        
    
    def plot_spectrum(self, ax=None):
        spec, freq, _ = self.spectrum()
        if ax is None:  ax = plt.gca()
        l, = ax.loglog(freq, spec)
        return l
        
        
    def plot_spectrum_ar1(self, data=None):
        """ plots spectrum of single time series + AR(1) spectrum
        AR(1) spectrum includes uncertainties from MC simulation
        for the filtered SST indices, the AR(1) process is first fitted to the annual data
        """
        self.load_raw_indices()
        tsa = TimeSeriesAnalysis(self.all_raw_indices[run].values)
        ft, fc = 'lowpass', 13
        spectrum = tsa.spectrum(filter_type=ft, filter_cutoff=fc)
        mc_spectrum = tsa.mc_ar1_spectrum(filter_type=ft, filter_cutoff=fc)
        
        fig, ax = plt.subplots(1, 1, figsize=(8,5))
        ax.tick_params(labelsize=14)
        ax.set_yscale('log')        
        L2 = ax.fill_between(mc_spectrum[1,:], mc_spectrum[2,:],  mc_spectrum[3,:],
                        color='C1', alpha=.3, label='5-95% C.I.')
        L1, = ax.plot(mc_spectrum[1,:], mc_spectrum[0,:], c='C1', label=f'MC AR(1)')     
        L4 = ax.fill_between(spectrum[1], spectrum[2][:, 0], spectrum[2][:, 1],
                       color='C0', alpha=0.3, label=f'{run.upper()} jackknife estimator')
        L3, = ax.plot(spectrum[1], spectrum[0], c='C0', label=f'{run.upper()} spectrum')
        leg1 = plt.legend(handles=[L1, L2], fontsize=14, frameon=False, loc=3)
        ax.legend(handles=[L3,L4], fontsize=14, frameon=False, loc=1)
        ax.add_artist(leg1)
        ymax = 1e1
        if any([mc_spectrum[3,:].max()>ymax, spectrum[2][:, 1].max()>ymax]):
            ymax = max([mc_spectrum[3,:].max(), spectrum[2][:, 1].max()])
        ax.set_ylim((1E-6, ymax))
        ax.set_xlim((0, 0.1))
        ax.set_xlabel(r'Frequency [yr$^{-1}$]', fontsize=14)
        ax.set_ylabel(f'{self.index} Power Spectral Density', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{path_results}/SST/{self.index}_AR1_spectrum_{run}')
        