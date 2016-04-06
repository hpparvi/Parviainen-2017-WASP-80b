from __future__ import division
import math as mt
import numpy as np
import pyfits as pf
import seaborn as sb
import pandas as pd

from os.path import join
from IPython.display import HTML, clear_output

from george import GP, HODLRSolver
from george.kernels import ExpKernel, ExpSquaredKernel

from pytransit.orbits_f import orbits as of
from pytransit import MandelAgol as MA

from ldtk import LDPSetCreator, BoxcarFilter, TabulatedFilter

from exotk.de import DiffEvol
from exotk.priors import PriorSet, UP, NP, JP
from exotk.utils.orbits import as_from_rhop
from exotk.utils.likelihood import ll_normal_es
from exotk.utils.misc import fold
from scipy.optimize import fmin, fmin_powell

from numpy import sqrt, zeros, inf, s_, log
from emcee import EnsembleSampler
cp = sb.color_palette()

np.random.seed(0)

TZERO = 2456000

TC = 125.417523 # Zero epoch
P  =   3.067861 # Orbital period

def map_ldc(q1,q2):
    a,b = sqrt(a), 2.*q2
    return a*b, a*(1.-b)

class FGP(object):
    def __init__(self):
        self.kernel = ( 1*ExpKernel(0.1,ndim=4,dim=0)
                       +1*ExpSquaredKernel(0.1,ndim=4,dim=1)*ExpSquaredKernel(0.1,ndim=4,dim=2)
                       +1*ExpSquaredKernel(0.1,ndim=4,dim=3))
        self.gp = GP(self.kernel)
        self._pv = zeros(8)

    def compute(self, pv, inputs):
        self._pv[0] = (10**pv[0])**2
        self._pv[1] = 1./pv[1]
        self._pv[2] = (10**pv[2])**2
        self._pv[3] = 1./pv[3]
        self._pv[4] = 1./pv[4]
        self._pv[5] = (10**pv[5])**2
        self._pv[6] = 1./pv[6]
        self._pv[7] = 10**pv[7]
        self.gp.kernel[:] = log(self._pv[:7])
        self.gp.compute(inputs, self._pv[7])

    def lnlikelihood(self, y):
        return self.gp.lnlikelihood(y)


class LPFunction(object):
    def __init__(self, nthreads=2):
        self.tm = MA(interpolate=False, nthr=nthreads) 
        self.nt = nthreads

        ## Load data
        ## ---------
        f = pd.HDFStore('data/external_lcs.h5','r')
        hnames = [k for k in f.keys() if 'fukui2014/lc' in k]
        dfs = [f[hname] for hname in hnames]
        f.close()

        self.otmasks   = [df.oe_mask.values for df in dfs]
        self.times     = [df.index.values-TZERO for df in dfs]
        self.mtimes    = [t-t.mean() for t in self.times]
        self.fluxes    = [df.flux.values for df in dfs]
        self.airmasses = [df.airmass.values for df in dfs]
        self.dxs       = [df.dx for df in dfs]
        self.dys       = [df.dy for df in dfs]
        self.npt       = [f.size for f in self.fluxes]
        self.passbands = [n.split('/')[3] for n in hnames]
        self.unique_pb = np.unique(self.passbands)
        self.pbids = [self.unique_pb.searchsorted(pb) for pb in self.passbands]
        self.gp_inputs = [np.transpose([t,dx,dy,am]) for t,dx,dy,am in zip(self.times,self.dxs,self.dys,self.airmasses)]

        self.nlc       = len(self.fluxes)
        self.npb       = len(self.unique_pb)
        self._wrk_ld   = zeros([self.npb,2])

        self.lc_names  = ['_'.join(n.split('/')[3:]) for n in hnames]

        
        ## Limb darkening with LDTk
        ## ------------------------
        dft = pd.read_hdf('data/external_lcs.h5','/transmission/fukui2014')
        mapping = dict(G='MITSuME_g', H='IRSF_H', I='MITSuME_Ic', J='IRSF_J', K='IRSF_Ks', R='MITSuME_Rc')
        #passbands = map(lambda s:mapping[s.split('/')[3]], self.unique_pb)
        passbands = [mapping[pb] for pb in self.unique_pb]
        #passbands = map(lambda s:mapping[s], self.unique_pb)

        filters = [TabulatedFilter(pb, dft.index.values, dft[pb].values) for pb in passbands]
        self.sc = LDPSetCreator([4150,100], [4.6,0.2], [-0.14,0.16], filters)
        self.lp = self.sc.create_profiles(2000)
        self.lp.set_uncertainty_multiplier(2)

        ## Red noise with Gaussian processes
        ## ---------------------------------
        self.hps = pd.read_hdf('data/external_lcs.h5', '/fukui2014/gp_hyperparameters')
        self.gps = [FGP() for pv in self.hps.values]
        [gp.compute(pv,gpin) for gp,pv,gpin in zip(self.gps,self.hps.values,self.gp_inputs)]
        
        ## Priors
        ## ------
        self.define_priors()
     

    def define_priors(self):
        nlc, npb = self.nlc, self.npb
        
        self.priors = [NP(    TC,   5e-3,   'tc'), ##  0  - Transit centre
                       NP(     P,   3e-4,    'p'), ##  1  - Period
                       UP(  3.50,   4.50,  'rho'), ##  2  - Stellar density
                       UP(  0.00,   0.99,    'b'), ##  3  - Impact parameter
                       UP(0.16**2, 0.18**2, 'k2')] ##  4  - planet-star area ratio
    
        lds = len(self.priors)
        for ipb in range(self.npb):
            self.priors.extend([UP(0, 1, 'q1_%i'%ipb),      ##  lds     + 2*ipb -- limb darkening q1
                                UP(0, 1, 'q2_%i'%ipb)])     ##  lds + 1 + 2*ipb -- limb darkening q2
                                
        self.ps = PriorSet(self.priors)
        
        self.sk2 = s_[4:5]
        self.swn = None
        self.sbl = None
        self.sq1 = s_[lds   : lds   +2*npb:2]
        self.sq2 = s_[lds+1 : lds+1 +2*npb:2]

     
    def compute_transit(self, pv):
        _a  = as_from_rhop(pv[2], pv[1]) 
        _i  = mt.acos(pv[3]/_a) 
        _k  = sqrt(pv[self.sk2])

        a,b = sqrt(pv[self.sq1]), 2.*pv[self.sq2]
        self._wrk_ld[:,0] = a*b
        self._wrk_ld[:,1] = a*(1.-b)

        flux_m = []
        for ipb, (ilc, time) in zip(self.pbids, enumerate(self.times)): 
            z = of.z_circular(time, pv[0], pv[1], _a, _i, self.nt) 
            flux_m.append(self.tm(z, _k, self._wrk_ld[ipb]))
        return flux_m

    
    def compute_lc_model(self, pv):
        return self.compute_transit(pv)


    def log_likelihood(self, pv):
        flux_m = self.compute_lc_model(pv)
        return sum([gp.lnlikelihood(fo-fm) for gp,fo,fm in zip(self.gps,self.fluxes,flux_m)])


    def __call__(self, pv):
        """Log posterior density"""
        if any(pv < self.ps.pmins) or any(pv>self.ps.pmaxs):
            return -inf

        log_prior_ld = 0 #sum(self.lp.lnlike_qd(self._wrk_ld[ipb], flt=ipb) for ipb in self.pbids)        
        return self.ps.c_log_prior(pv) + log_prior_ld + self.log_likelihood(pv)
