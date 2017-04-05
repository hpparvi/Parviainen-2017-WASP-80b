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
    """ Ln posterior function implementing the divide-by-white model.
    """
    def __init__(self, passband, lctype='target', use_ldtk=False, n_threads=1, noise='white', pipeline='gc'):
        assert passband in ['bb','nb','K','Na','pr']
        super().__init__(passband, lctype, use_ldtk, n_threads, noise, pipeline)

        # DATA
        # ====
        self.fluxes = [f/fm for f,fm in zip(self.fluxes, self.fluxes_m)]

        # PRIORS
        # ======
        self.priors = self.priors[:self._srp]
        del(self._srp)

        # Orbital parameter priors
        # ------------------------
        # We'll constrain the orbital parameters to the estimates obtained from the GP-model runs.
        # This is necessary since our DW approach doesn't constrain the orbital parameters well.
        #
        self.priors[0] = NP(125.417392,  8e-5,   'tc')  #  0  - Transit centre
        self.priors[1] = NP(3.06785541,  4e-7,    'p')  #  1  - Period
        self.priors[2] = NP(4.17300000,  3e-2,  'rho')  #  2  - Stellar density
        self.priors[3] = NP(0.15200000,  2e-2,    'b')  #  3  - Impact parameter

        # Baseline priors
        # ---------------
        # We'll use a linear model a + b * time + c * airmass to model the baseline, and assume that rest of
        # the systematics are wavelength-independent and removed by the DW approach.
        #
        self._sbl = len(self.priors)
        for ilc in range(self.nlc):
            self.priors.append(UP( 0.0, 2.0, 'bcn_%i'%ilc)) #  sbl + ilc -- Baseline constant
            self.priors.append(UP(-1.0, 1.0, 'btl_%i'%ilc)) #  sbl + ilc -- Linear time trend
            self.priors.append(UP(-1.0, 1.0, 'bal_%i'%ilc)) #  sbl + ilc -- Linear airmass trend

        # White noise
        # -----------
        # As with the direct-model, the average white noise is treated as a free parameter.
        self._swn = len(self.priors)
        self.priors.extend([UP(3e-4, 4e-3, 'e_%i'%ilc) for ilc in range(self.nlc)])

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
        _k = median(sqrt(pv[self.ik2]))
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
        from numpy.linalg import lstsq
        pvt = pvpop.copy()
        for i in range(self.nlc):
            X = array([ones_like(self.ctimes[i]), self.ctimes[i], self.airmass[i]])
            pv = lstsq(X.T, self.fluxes[i])[0]
            pvt[:, self.ibcn[i]] = normal(pv[0], 0.001, size=pvt.shape[0])
            pvt[:, self.ibtl[i]] = normal(pv[1], 0.01 * abs(pv[1]), size=pvt.shape[0])
            pvt[:, self.ibal[i]] = normal(pv[2], 0.01 * abs(pv[2]), size=pvt.shape[0])
        return pvt
