import math as mt
import numpy as np

from numpy import array, ones, zeros, ones_like, inf, newaxis, sqrt

from pytransit import MandelAgol as MA
from ldtk import LDPSetCreator
from exotk.priors import UP, NP, PriorSet
from exotk.utils.orbits import as_from_rhop
from exotk.utils.likelihood import ll_normal_es
from core import *

class LPFunction(object):
    def __init__(self, time, flux, airmass, nthreads=2):
        self.tm = MA(interpolate=True, klims=(0.16,0.19), nthr=nthreads, nk=512) 
        self.nthr = nthreads

        self.time     = array(time)
        self.flux_o   = array(flux)
        self.airmass  = array(airmass)
        self.npt      = self.flux_o.shape[0]
        self.npb      = self.flux_o.shape[1]
        self._w_bl    = ones_like(flux)
        self._w_ld    = zeros(2*self.npb)
        self._w_pv    = zeros(4+6*self.npb)
        
        sc = LDPSetCreator(teff=(5650,35),logg=(4.58,0.015),z=(-0.19,0.08), filters=pb_filters_nb)
        self.lds = sc.create_profiles(500)
        self.lds.resample_linear_z()
        
        self.priors = [NP(  56895.443,   0.1,   'tc'),  ##  0  - Transit centre
                       NP(  3.068,   1e-6,     'p'),  ##  1  - Period
                       UP(  3.0,   5.0,   'rho'),  ##  2  - Stellar density
                       UP(  0.0,   .99,     'b')]  ##  3  - Impact parameter
        
        for ipb in range(self.npb):
            self.priors.extend([
                       UP( .16**2, 0.19**2, 'k2_{:02d}'.format(ipb)),  ##  4 + 6*i  - planet-star area ratio
                       UP(   1e-4,  50e-4,   'e_{:02d}'.format(ipb)),  ##  5 + 6*i  - White noise std
                       NP(    1.0,   0.01,   'c_{:02d}'.format(ipb)),  ##  6 + 6*i  - Baseline constant
                       NP(    0.0,   0.01,  'x_{:02d}'.format(ipb)),  ##  7 + 6*i  - Residual extinction coefficient
                       UP(   -1.0,    1.0,  'u_{:02d}'.format(ipb)),  ##  8 + 6*i  - limb darkening u
                       UP(   -1.0,    1.0,  'v_{:02d}'.format(ipb))]) ##  9 + 6*i  - limb darkening v
        self.ps = PriorSet(self.priors)
        
        
    def compute_baseline(self, pv):
        return pv[6::6]*np.exp(pv[7::6,]*self.airmass[:,newaxis])
    
    
    def compute_transit(self, pv):
        _a  = as_from_rhop(pv[2], pv[1])
        _i  = mt.acos(pv[3]/_a)
        _k  = np.sqrt(pv[4::6])
        self._w_ld[0::2] = pv[8::6]
        self._w_ld[1::2] = pv[9::6]                 
        return self.tm.evaluate(self.time, _k, self._w_ld, pv[0], pv[1], _a, _i)
    
    
    def compute_lc_model(self, pv):
        return self.compute_baseline(pv)*self.compute_transit(pv)


    def log_posterior(self, pv):
        if any(pv < self.ps.pmins) or any(pv>self.ps.pmaxs):
            return -inf
        
        flux_m = self.compute_lc_model(pv)
        lnlike_lc = sum([ll_normal_es(self.flux_o[:,ipb], flux_m[:,ipb], pv[5+6*ipb]) for ipb in range(self.npb)])
        lnlike_ld = self.lds.lnlike_qd(self._w_ld)                
        return self.ps.c_log_prior(pv) + lnlike_lc + lnlike_ld    

    def __call__(self, pv):
        return self.log_posterior(pv)
