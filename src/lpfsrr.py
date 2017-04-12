import sys
from copy import copy
from itertools import chain
from numpy import *

from scipy.signal import medfilt as MF
from scipy.stats import scoreatpercentile as sap
from numpy.random import normal
from statsmodels.robust import mad

from george.kernels import ConstantKernel, Matern32Kernel

from .core import *
from .lpf import *
from .extcore import *
from .lpfsd import LPFSD

class LPFSRR(LPFSD):
    def __init__(self, passband, lctype='relative', use_ldtk=False, n_threads=1, night=2, pipeline='gc'):
        super().__init__(passband, lctype, use_ldtk, n_threads, night, pipeline)

        self.fluxes = asarray(self.fluxes)
        self.fluxes_m = self.fluxes.mean(0)
        self.fluxes /= self.fluxes_m
        self.wn_estimates = array([sqrt(2) * mad(diff(f)) for f in self.fluxes])

        self.priors = self.priors[:self._sgp]

        # Update the priors using the external data modelling
        # ---------------------------------------------------
        self.priors[0] = NP(125.417380,  8e-5,   'tc')  #  0  - Transit centre
        self.priors[1] = NP(3.06785547,  4e-7,    'p')  #  1  - Period
        self.priors[2] = NP(4.17200000,  3e-2,  'rho')  #  2  - Stellar density
        self.priors[3] = NP(0.16100000,  2e-2,    'b')  #  3  - Impact parameter

        # Change the GP priors slightly
        # -----------------------------
        self.priors.extend([UP(-5, -1, 'log10_tm_amplitude'),
                            UP(-5,  5, 'log10_tm_inv_scale'),
                            UP(-5, -1, 'log10_am_amplitude'),
                            UP(-5,  2, 'log10_am_inv_scale'),
                            UP(-5, -1, 'log10_ra_amplitude'),
                            UP(-5,  2, 'log10_ra_inv_scale')])
        self.ps = PriorSet(self.priors)

        self.prior_kw = NP(0.1707, 3.2e-4, 'kw', lims=(0.16,0.18))


    def set_pv_indices(self, sbl=None, swn=None):
        self.ik2 = [self._sk2 + pbid for pbid in self.gpbids]

        self.iq1 = [self._sq1 + pbid * 2 for pbid in self.gpbids]
        self.iq2 = [self._sq2 + pbid * 2 for pbid in self.gpbids]
        self.uq1 = np.unique(self.iq1)
        self.uq2 = np.unique(self.iq2)

        sbl = sbl if sbl is not None else self._sbl
        self.ibcn = [sbl + ilc for ilc in range(self.nlc)]

        if hasattr(self, '_sgp'):
            self.slgp = s_[self._sgp:self._sgp + 6]


    def setup_gp(self):
        self.gp_inputs = array([self.times[0], self.airmass, self.rotang]).T
        self.kernel = (ConstantKernel(1e-3 ** 2, ndim=3, axes=0) * Matern32Kernel(.01, ndim=3, axes=0)
                     + ConstantKernel(1e-3 ** 2, ndim=3, axes=1) * ExpSquaredKernel(.01, ndim=3, axes=1)
                     + ConstantKernel(1e-3 ** 2, ndim=3, axes=2) * ExpSquaredKernel(.01, ndim=3, axes=2))
        self.gp = GP(self.kernel)
        self.gp.compute(self.gp_inputs, yerr=5e-4)


    def map_to_gp(self, pv):
        log10_to_ln = 1. / log10(e)
        gpp = zeros(6)
        gpp[0] = 2 * pv[0] * log10_to_ln
        gpp[1] =    -pv[1] * log10_to_ln
        gpp[2] = 2 * pv[2] * log10_to_ln
        gpp[3] =    -pv[3] * log10_to_ln
        gpp[4] = 2 * pv[4] * log10_to_ln
        gpp[5] =    -pv[5] * log10_to_ln
        return gpp


    def lnposterior(self, pv):
        _k = median(sqrt(pv[self.ik2]))
        return super().lnposterior(pv) + self.prior_kw.log(_k)


    def compute_lc_model(self, pv, copy=False):
        bl = self.compute_baseline(pv)
        tr = self.compute_transit(pv)
        self._wrk_lc[:] = bl[:,np.newaxis]*tr/tr.mean(0)
        return self._wrk_lc if not copy else self._wrk_lc.copy()
