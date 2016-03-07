from __future__ import division
import math as mt
import numpy as np
import pandas as pd

from os.path import join
from numpy import zeros, unique, sqrt, isfinite, concatenate, inf, s_
from copy import  copy, deepcopy

from ldtk import LDPSetCreator, BoxcarFilter, TabulatedFilter
from pytransit import MandelAgol as MA
from pytransit.orbits_f import orbits as of
from exotk.priors import PriorSet, UP, NP, JP
from exotk.utils.orbits import as_from_rhop
from exotk.utils.likelihood import ll_normal_es

TZERO = 2456000

TC = 125.417523 # Zero epoch
P  =   3.067861 # Orbital period

def map_ldc(q1,q2):
    a,b = sqrt(q1), 2.*q2
    return a*b, a*(1.-b)

def map_uv_to_qq(u,v):
    return (u+v)**2, u/(2*(u+v))

class LogPosteriorFunction(object):
    def __init__(self, passbands, constant_k=True, noise='white', use_ldtk=False):
        self.passbands = passbands
        self.unique_pbs = unique(self.passbands)
        self.gpbids = [self.unique_pbs.searchsorted(pb) for pb in self.passbands]
        self.lpbids = copy(self.gpbids)
        self.cpassbands = pd.Series(pd.Categorical(self.passbands, categories='g r i z J H K'.split()))
        self.lcorder = self.cpassbands.sort(inplace=False).index.values

        self.use_ldtk = use_ldtk
        self.constant_k = constant_k
        self.noise = noise
        
        self.nlc = nlc = len(self.passbands)
        self.npb = npb = len(self.unique_pbs)
        self._wrk_ld   = zeros([self.nlc,2])
        
        ## Basic parameters
        ## ----------------
        self.priors = [NP(    TC,   5e-3,   'tc'), ##  0  - Transit centre
                       NP(     P,   3e-4,    'p'), ##  1  - Period
                       UP(  3.50,   4.50,  'rho'), ##  2  - Stellar density
                       UP(  0.00,   0.99,    'b')] ##  3  - Impact parameter
        
        ## Area ratio
        ## ----------
        self._sk2 = len(self.priors)
        if constant_k:
            self.priors.append( UP(0.16**2, 0.18**2, 'k2')) ##  4  - planet-star area ratio
        else:
            self.priors.extend([UP(0.16**2, 0.18**2, 'k2_%i'%ipb) 
                                for ipb in range(self.npb)]) ##  4  - planet-star area ratio
            
        ## Limb darkening
        ## --------------
        self._sq1 = len(self.priors)
        self._sq2 = self._sq1+1
        for ipb in range(self.npb):
            self.priors.extend([UP(0, 1, 'q1_%i'%ipb),      ##  sq1 + 2*ipb -- limb darkening q1
                                UP(0, 1, 'q2_%i'%ipb)])     ##  sq2 + 2*ipb -- limb darkening q2
            
        ## Baseline constant
        ## -----------------
        self._sbl = len(self.priors)
        self.priors.extend([NP(1, 1e-4, 'bl_%i'%ilc) 
                            for ilc in range(self.nlc)]) ##  sbl + ilc -- Baseline constant
        
        ## White noise
        ## -----------
        self._swn = len(self.priors)
        if noise == 'white':
            self.priors.extend([UP(1e-4, 1e-2, 'e_%i'%ilc) 
                                for ilc in range(self.nlc)]) ##  sqn + ilc -- Average white noise
        
        self.ps = PriorSet(self.priors)
        self.set_pv_indices()
    
    
        ## Limb darkening with LDTk
        ## ------------------------
        if use_ldtk:
            dff = pd.read_hdf('data/external_lcs.h5', 'transmission')
            self.filters = []
            for pb in self.unique_pbs:
                self.filters.append(TabulatedFilter(pb, dff.index.values, tm=dff[pb].values))
            self.sc = LDPSetCreator([4150,100], [4.6,0.2], [-0.14,0.16], self.filters)
            self.lp = self.sc.create_profiles(2000)
            self.lp.set_uncertainty_multiplier(2)


    def set_pv_indices(self, sbl=None, swn=None):
        if self.constant_k:
            self.ik2 = [self._sk2]
        else:
            self.ik2 = [self._sk2+pbid for pbid in self.gpbids]
                        
        self.iq1 = [self._sq1+pbid*2 for pbid in self.gpbids]
        self.iq2 = [self._sq2+pbid*2 for pbid in self.gpbids]
        
        sbl = sbl if sbl is not None else self._sbl
        self.ibl = [sbl+ilc for ilc in range(self.nlc)]
        
        if self.noise == 'white':
            swn = swn if swn is not None else self._swn
            self.iwn = [swn+ilc for ilc in range(self.nlc)]
        else:
            self.iwn = []


    def setup_gp(self):
        raise NotImplementedError
        
        
    def lnprior(self, pv):
        if any(pv < self.ps.pmins) or any(pv>self.ps.pmaxs):
            return -inf
        else:
            return self.ps.c_log_prior(pv)
    

    def lnlikelihood(self, pv):
        raise NotImplementedError
    
    def lnlikelihood_wn(self, pv):
        raise NotImplementedError


    def lnlikelihood_rn(self, pv):
        raise NotImplementedError

    
    def lnlikelihood_ld(self, pv):
        if self.use_ldtk:
            uv = zeros([self.npb,2])
            q1 = pv[self._sq1:self._sq1+2*self.npb:2]
            q2 = pv[self._sq2:self._sq2+2*self.npb:2]
            a,b = sqrt(q1), 2*q2
            uv[:,0] = a*b
            uv[:,1] = a*(1.-b)
            return self.lp.lnlike_qd(uv)
        else:
            return 0.0


    def lnposterior(self, pv):
        lnp = self.lnprior(pv)
        if not isfinite(lnp):
            return lnp
        else:
            return lnp + self.lnlikelihood_ld(pv) + self.lnlikelihood(pv)
        

        
class LPF(LogPosteriorFunction):
    def __init__(self, times, fluxes, passbands, constant_k=True, noise='white', use_ldtk=False, nthreads=4):
        """Log-posterior function for a single dataset.
        """
        super(LPF, self).__init__(passbands, constant_k, noise, use_ldtk)
        self.tm = MA(interpolate=False, nthr=nthreads) 
        self.nt = nthreads
        self.times  = times
        self.fluxes = fluxes
        self._wrk_k = zeros(self.nlc)
        self.fmasks = [isfinite(f) for f in fluxes]

        self.mtimes  = [t[m] for t,m in zip(self.times, self.fmasks)]
        self.mfluxes = [f[m] for f,m in zip(self.fluxes, self.fmasks)]

        ## Noise model setup
        ## -----------------
        if noise == 'white':
            self.lnlikelihood = self.lnlikelihood_wn
        elif noise == 'red':
            self.setup_gp()
            self.lnlikelihood = self.lnlikelihood_rn
        else:
            raise NotImplementedError('Bad noise model')


    def compute_baseline(self, pv):
        return pv[self.ibl]
        
   
    def compute_transit(self, pv):
        _a  = as_from_rhop(pv[2], pv[1]) 
        _i  = mt.acos(pv[3]/_a) 
        self._wrk_k[:]  = sqrt(pv[self.ik2])

        a,b = sqrt(pv[self.iq1]), 2.*pv[self.iq2]
        self._wrk_ld[:,0] = a*b
        self._wrk_ld[:,1] = a*(1.-b)

        flux_m = []
        for time,k,ldc in zip(self.mtimes,self._wrk_k,self._wrk_ld): 
            z = of.z_circular(time, pv[0], pv[1], _a, _i, self.nt) 
            flux_m.append(self.tm(z, k, ldc))
        return flux_m
       

    def compute_lc_model(self,pv):
        return [b*m for b,m in zip(self.compute_baseline(pv),self.compute_transit(pv))]
    
        
    def lnlikelihood_wn(self, pv):
        fluxes_m = self.compute_lc_model(pv)
        return sum([ll_normal_es(fo, fm, wn) for fo,fm,wn in zip(self.mfluxes, fluxes_m, pv[self.iwn])]) 


        
class CLPF(LogPosteriorFunction):
    def __init__(self, lpfs, constant_k=True, noise='white', use_ldtk=False):
        """Log-posterior function combining multiple datasets.
        """
        super(CLPF, self).__init__(concatenate([lpf.passbands for lpf in lpfs]), constant_k, noise, use_ldtk)
        self.lpfs = lpfs

        sbl = self._sbl
        swn = self._swn if self.noise == 'white' else None
        for lpf in self.lpfs:
            lpf.gpbids = [self.unique_pbs.searchsorted(pb) for pb in lpf.passbands]
            lpf.set_pv_indices(sbl,swn)
            sbl += lpf.nlc
            if self.noise == 'white':
                swn += lpf.nlc
            
    def lnposterior(self, pv):
        lnp = self.lnprior(pv)
        if not isfinite(lnp):
            return lnp
        else:
            return lnp + self.lnlikelihood_ld(pv) + sum([lpf.lnlikelihood(pv) for lpf in self.lpfs])

    @property
    def times(self):
        return np.sum([lpf.times for lpf in self.lpfs])

    @property
    def fluxes(self):
        return np.sum([lpf.fluxes for lpf in self.lpfs])

    def compute_baseline(self, pv):
        return np.sum([lpf.compute_baseline(pv) for lpf in self.lpfs])

    def compute_transit(self, pv):
        return np.sum([lpf.compute_transit(pv) for lpf in self.lpfs])

    def compute_lc_model(self, pv):
        return np.sum([lpf.compute_lc_model(pv) for lpf in self.lpfs])
