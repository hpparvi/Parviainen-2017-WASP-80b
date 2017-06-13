import sys
from copy import copy
from itertools import chain
from numpy import *
from copy import deepcopy

from scipy.signal import medfilt as MF
from scipy.stats import scoreatpercentile as sap
from numpy.random import normal

from .core import *
from .lpf import *
from .extcore import *
from .lpf import CLPF
from .lpfsr import LPFSR


class LPFMR(CLPF):
    def __init__(self, passband, lctype='relative', use_ldtk=False, n_threads=1, pipeline='gc'):
        self.lpf1 = l1 = LPFSR(passband, lctype, use_ldtk=False, pipeline=pipeline, night=1, n_threads=n_threads)
        self.lpf2 = l2 = LPFSR(passband, lctype, use_ldtk=False, pipeline=pipeline, night=2, n_threads=n_threads)

        sbl = l1._sbl
        swn = l1._swn
        super().__init__((l1, l2), constant_k=False, noise='red', use_ldtk=False)

        for p in self.lpf1.priors[sbl:]:
            p.name += '_1'

        for p in self.lpf2.priors[sbl:]:
            p.name += '_2'

        priors = deepcopy(self.lpf1.priors[:sbl])
        sbl1 = len(priors)
        priors.extend(self.lpf1.priors[sbl:swn])
        sbl2 = len(priors)
        priors.extend(self.lpf2.priors[sbl:swn])
        swn1 = len(priors)
        priors.extend(self.lpf1.priors[swn:])
        swn2 = len(priors)
        priors.extend(self.lpf2.priors[swn:])

        self.priors = priors
        self.ps = PriorSet(priors)

        l1.set_pv_indices(sbl1, swn1)
        l2.set_pv_indices(sbl2, swn2)

        self._sbl = sbl1
        self._swn = swn1

        self.filters = l1.filters
        self.passband = l1.passband
        self.prior_kw = l1.prior_kw

        self.use_ldtk = use_ldtk
        if use_ldtk:
            self.sc = LDPSetCreator([4150, 100], [4.6, 0.2], [-0.14, 0.16], self.filters)
            self.lp = self.sc.create_profiles(2000)
            self.lp.resample_linear_z()
            self.lp.set_uncertainty_multiplier(2)


    @property
    def times(self):
        return [lpf.times for lpf in self.lpfs]


    @property
    def fluxes(self):
        return [lpf.fluxes for lpf in self.lpfs]


    def lnposterior(self, pv):
        _k = median(sqrt(pv[self.ik2]))
        return super().lnposterior(pv) + self.prior_kw.log(_k)


    def compute_baseline(self, pv):
        return [lpf.compute_baseline(pv) for lpf in self.lpfs]


    def compute_transit(self, pv):
        return [lpf.compute_transit(pv) for lpf in self.lpfs]


    def compute_lc_model(self, pv):
        return [lpf.compute_lc_model(pv) for lpf in self.lpfs]

    def fit_baseline(self, pvpop):
        pvpop = self.lpf1.fit_baseline(pvpop)
        pvpop = self.lpf2.fit_baseline(pvpop)
        return pvpop

    def fit_ldc(self, pvpop, emul=1.):
        pvt = pvpop.copy()
        uv, uve = self.lp.coeffs_qd()
        us = array([normal(um, emul * ue, size=pvt.shape[0]) for um, ue in zip(uv[:, 0], uve[:, 0])]).T
        vs = array([normal(vm, emul * ve, size=pvt.shape[0]) for vm, ve in zip(uv[:, 1], uve[:, 1])]).T
        q1s, q2s = map_uv_to_qq(us, vs)
        pvt[:, self.uq1] = q1s
        pvt[:, self.uq2] = q2s
        return pvt
#