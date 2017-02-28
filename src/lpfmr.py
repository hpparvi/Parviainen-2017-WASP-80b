import sys
from copy import copy
from itertools import chain
from numpy import *

from scipy.signal import medfilt as MF
from scipy.stats import scoreatpercentile as sap
from numpy.random import normal

from .core import *
from .lpf import *
from .extcore import *
from .lpfmd import LPFMD

class LPFMR(LPFMD):
    def __init__(self, passband, lctype='target', use_ldtk=False, n_threads=1, noise='white', pipeline='hp'):
        assert passband in ['bb','nb','K','Na']
        super().__init__(passband, lctype, use_ldtk, n_threads, noise, pipeline)

        self.fluxes = [f/fm for f,fm in zip(self.fluxes, self.fluxes_m)]
        self.priors = self.priors[:self._srp]
        del(self._srp)

        ## Baseline
        ## --------
        self._sbl = len(self.priors)
        for ilc in range(self.nlc):
            self.priors.append(UP( 0.0, 2.0, 'bcn_%i'%ilc)) ##  sbl + ilc -- Baseline constant
            self.priors.append(UP(-1.0, 1.0, 'btl_%i'%ilc)) ##  sbl + ilc -- Linear time trend
            self.priors.append(UP(-1.0, 1.0, 'bal_%i'%ilc)) ##  sbl + ilc -- Linear airmass trend

        ## White noise
        ## -----------
        self._swn = len(self.priors)
        self.priors.extend([UP(3e-4, 4e-3, 'e_%i'%ilc) 
                            for ilc in range(self.nlc)]) ##  sqn + ilc -- Average white noise

        self.ps = PriorSet(self.priors)
        self.set_pv_indices()

        if self.noise == 'red':
            for ilc,i in enumerate(self.iwn):
                self.priors[i] = NP(7e-4, 2e-4, 'e_%i'%ilc, lims=(0,1))
        
        self.ps = PriorSet(self.priors)
        if self.noise == 'red':
            self.setup_gp()

            
    def set_pv_indices(self, sbl=None, swn=None):
        self.ik2 = [self._sk2+pbid for pbid in self.gpbids]
                        
        self.iq1 = [self._sq1+pbid*2 for pbid in self.gpbids]
        self.iq2 = [self._sq2+pbid*2 for pbid in self.gpbids]
        self.uq1 = np.unique(self.iq1)
        self.uq2 = np.unique(self.iq2)
        
        sbl = sbl if sbl is not None else self._sbl
        self.ibcn = [sbl+3*ilc   for ilc in range(self.nlc)]
        self.ibtl = [sbl+3*ilc+1 for ilc in range(self.nlc)]
        self.ibal = [sbl+3*ilc+2 for ilc in range(self.nlc)]
        
        swn = swn if swn is not None else self._swn
        self.iwn = [swn+ilc for ilc in range(self.nlc)]


    def lnposterior(self, pv):
        _k = sqrt(pv[self.ik2]).mean()
        return super().lnposterior(pv) + self.prior_kw.log(_k)


    def compute_lc_model(self,pv):
        bl1,bl2 = self.compute_baseline(pv)
        tr1,tr2 = self.compute_transit(pv)
        self._wrk_lc[0][:] = bl1*tr1/tr1.mean(0)
        self._wrk_lc[1][:] = bl2*tr2/tr2.mean(0)
        return self._wrk_lc


    def compute_baseline(self, pv):
        bl1 = pv[self.ibcn[:self.npb]][:,newaxis] + pv[self.ibtl[:self.npb]][:,newaxis] * self.ctimes[0] + pv[self.ibal[:self.npb]][:,newaxis] * self.airmass[0]
        bl2 = pv[self.ibcn[self.npb:]][:,newaxis] + pv[self.ibtl[self.npb:]][:,newaxis] * self.ctimes[self.npb] + pv[self.ibal[self.npb:]][:,newaxis] * self.airmass[self.npb]
        return bl1, bl2


    def fit_baseline(self, pvpop):
        def baseline(pv, ilc):
            return pv[0] + pv[1]*self.ctimes[ilc] + pv[2]*self.airmass[ilc] 

        pvt = pvpop.copy()
        for i in range(self.nlc):
            m = ~self.otmasks[i]
            pv0 = fmin(lambda pv: ((self.fluxes[i][m]-baseline(pv,i)[m])**2).sum(), 
                       [1,0,0], disp=False, ftol=1e-9, xtol=1e-9)
            pvt[:,self.ibcn[i]] = normal(pv0[0], 0.001,            size=pvt.shape[0])
            pvt[:,self.ibtl[i]] = normal(pv0[1], 0.01*abs(pv0[1]), size=pvt.shape[0])
            pvt[:,self.ibal[i]] = normal(pv0[2], 0.01*abs(pv0[2]), size=pvt.shape[0])
        return pvt
