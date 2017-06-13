import sys
from copy import copy
from itertools import chain
from numpy import *

from scipy.signal import medfilt as MF
from scipy.stats import scoreatpercentile as sap
from numpy.random import normal, seed
from statsmodels.robust import mad

from george.kernels import ConstantKernel, Matern32Kernel, DotProductKernel

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
        self.priors.extend([UP(-5, -1, 'log10_constant'),
                            UP(-5, -1, 'log10_ra_amplitude'),
                            UP(-5,  3, 'log10_ra_inv_scale')])
        self.ps = PriorSet(self.priors)

        self.prior_kw = NP(0.1707, 3.2e-4, 'kw', lims=(0.16,0.18))

        # Use mock data
        # -------------
        if passband == 'nb_mock' and type(self) == LPFSRR:
            self.create_mock_nb_dataset()


    def set_pv_indices(self, sbl=None, swn=None):
        self.ik2 = [self._sk2 + pbid for pbid in self.gpbids]

        self.iq1 = [self._sq1 + pbid * 2 for pbid in self.gpbids]
        self.iq2 = [self._sq2 + pbid * 2 for pbid in self.gpbids]
        self.uq1 = np.unique(self.iq1)
        self.uq2 = np.unique(self.iq2)

        sbl = sbl if sbl is not None else self._sbl
        self.ibcn = [sbl + ilc for ilc in range(self.nlc)]

        if hasattr(self, '_sgp'):
            self.slgp = s_[self._sgp:self._sgp + 3]


    def setup_gp(self):
        self.gp_inputs = array([self.airmass, self.rotang]).T
        self.kernel = (ConstantKernel(1e-3 ** 2, ndim=2, axes=0) + DotProductKernel(ndim=2, axes=0)
                     + ConstantKernel(1e-3 ** 2, ndim=2, axes=1) * ExpSquaredKernel(.01, ndim=2, axes=1))
        self.gp = GP(self.kernel)
        self.gp.compute(self.gp_inputs, yerr=5e-4)


    def map_to_gp(self, pv):
        log10_to_ln = 1. / log10(e)
        gpp = zeros(3)
        gpp[0] = 2 * pv[0] * log10_to_ln
        gpp[1] = 2 * pv[1] * log10_to_ln
        gpp[2] =    -pv[2] * log10_to_ln
        return gpp


    def lnposterior(self, pv):
        _k = median(sqrt(pv[self.ik2]))
        return super().lnposterior(pv) + self.prior_kw.log(_k)


    def compute_lc_model(self, pv, copy=False):
        bl = self.compute_baseline(pv)
        tr = self.compute_transit(pv)
        self._wrk_lc[:] = bl[:,np.newaxis]*tr/tr.mean(0)
        return self._wrk_lc if not copy else self._wrk_lc.copy()


    def create_mock_nb_dataset(self):
        tc, p, rho, b = 125.417380, 3.06785547, 4.17200000, 0.161

        ks = np.full(self.npb, 0.171)
        ks[1::3] = 0.170
        ks[2::3] = 0.172
        ks[[7, 13]] = 0.173

        q1 = array([0.581, 0.582, 0.590, 0.567, 0.541, 0.528, 0.492, 0.490,
                    0.461, 0.440, 0.419, 0.382, 0.380, 0.368, 0.344, 0.328,
                    0.320, 0.308, 0.301, 0.292])

        q2 = array([0.465, 0.461, 0.446, 0.442, 0.425, 0.427, 0.414, 0.409,
                    0.422, 0.402, 0.391, 0.381, 0.379, 0.373, 0.369, 0.365,
                    0.362, 0.360, 0.360, 0.358])

        seed(0)
        cam = normal(0, 0.03, self.nlc)
        ctm = normal(0, 0.08, self.nlc)

        seed(0)
        pv = self.ps.generate_pv_population(1)[0]

        pv[:4] = tc, p, rho, b
        pv[self.ik2] = ks ** 2
        pv[self.iq1] = q1
        pv[self.iq2] = q2
        pv[self._sbl:] = 1.

        fms = self.compute_transit(pv).copy()

        for i, fm in enumerate(fms):
            fm[:] += (normal(0, self.wn_estimates[i], fm.size)
                      + cam[i] * (self.airmass - self.airmass.mean())
                      + ctm[i] * (self.times[0] - self.times[0].mean()))

        self.fluxes = asarray(fms)
        self.fluxes_m = self.fluxes.mean(0)
        self.fluxes /= self.fluxes_m
        self.wn_estimates = array([sqrt(2) * mad(diff(f)) for f in self.fluxes])
