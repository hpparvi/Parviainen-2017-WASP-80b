import math as mt
import numpy as np

from numpy import array, ones, inf

from pytransit import MandelAgol as MA
from ldtk import LDPSetCreator
from exotk.priors import UP, NP, PriorSet
from exotk.utils.orbits import as_from_rhop
from exotk.utils.likelihood import ll_normal_es
from core import *

class LPFunction(object):
    def __init__(self, time, flux, airmass, nthreads=2, filters=None):
        self.tm = MA(interpolate=True, klims=(0.16,0.19), nthr=nthreads, nk=512) 
        self.nthr = nthreads

        self.time     = array(time)       # Mid-exposure times
        self.flux_o   = array(flux)       # Observed fluxes
        self.airmass  = array(airmass)    # Airmass
        self.npt      = self.flux_o.size  # Number of exposures
        self.npb      = 1

        sc = LDPSetCreator(teff=(5650,35),logg=(4.58,0.015),z=(-0.19,0.08), filters=filters or [pb_filter_bb])
        self.lds = sc.create_profiles(500)
        self.lds.resample_linear_z()
        
        # In lpf_ww.py:  for bb wn analysis White noise std with an upper limit of 20e-4 gives good results since there isn't much noise for bb, but for narrowband analysis, especially for the shorter wavelength passbands, a bigger white noise std is required and since all the fit_color codes use lpf_ww, we have changed the white noise std here when running the nb analysis. However, for bb 50e-4 could be changed to 20e-4, but it doesnt really affect the bb wn parameter results.
        
        self.priors = [NP(  56895.443,   0.01,   'tc'),  ##  0  - Transit centre
                       NP(  3.068,   1e-6,     'p'),  ##  1  - Period
                       UP(  3.0,   5.0,   'rho'),  ##  2  - Stellar density
                       UP(  0.0,   .99,     'b'),  ##  3  - Impact parameter
                       UP( .16**2, 0.19**2,    'k2'),  ##  4  - planet-star area ratio
                       UP(   1e-4,  50e-4,     'e'),  ##  5  - White noise std
                       NP(    0.99,   0.01,     'c'),  ##  6  - Baseline constant
                       NP(    0.0,   0.01,     'x', lims=(-0.1,0.1)),  ##  7  - Residual extinction coefficient
                       UP(   -1.0,    1.0,     'u'),  ##  8  - limb darkening u
                       UP(   -1.0,    1.0,     'v')]  ##  9  - limb darkening v
        self.ps = PriorSet(self.priors)
        
        
    def compute_baseline(self, pv):
        return pv[6]*np.exp(pv[7]*self.airmass)
    
    
    def compute_transit(self, pv):
        _a  = as_from_rhop(pv[2], pv[1])
        _i  = mt.acos(pv[3]/_a)
        _k  = mt.sqrt(pv[4]) 
        return self.tm.evaluate(self.time, _k, pv[8:10], pv[0], pv[1], _a, _i)
    
    
    def compute_lc_model(self, pv):
        return self.compute_baseline(pv)*self.compute_transit(pv)


    def log_posterior(self, pv):
        if any(pv < self.ps.pmins) or any(pv>self.ps.pmaxs):
            return -inf

        lnlike_ld = self.lds.lnlike_qd(pv[8:10])
        lnlike_lc = ll_normal_es(self.flux_o, self.compute_lc_model(pv), pv[5]) 
        return self.ps.c_log_prior(pv) + lnlike_lc + lnlike_ld


    def __call__(self, pv):
        return self.log_posterior(pv)