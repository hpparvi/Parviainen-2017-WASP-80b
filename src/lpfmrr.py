from numpy import s_, array, median, sqrt
from numpy.random import normal
from exotk.priors import PriorSet, UP, NP

from .core import *
from .extcore import map_uv_to_qq
from .lpf import CLPF
from .lpfsrr import LPFSRR

class LPFMRR(CLPF):
    def __init__(self, passband, lctype='relative', use_ldtk=False, n_threads=1, pipeline='gc'):
        self.lpf1 = l1 = LPFSRR(passband, lctype, use_ldtk=False, pipeline=pipeline, night=1, n_threads=n_threads)
        self.lpf2 = l2 = LPFSRR(passband, lctype, use_ldtk=False, pipeline=pipeline, night=2, n_threads=n_threads)
        super().__init__((l1,l2), constant_k=False, noise='red', use_ldtk=False)
        l1._sgp = l2._sgp = self.ps.ndim
        l1.slgp = l2.slgp = s_[l1._sgp:l1._sgp+4]
        self.priors[:7] = l1.priors[:7]
        self.priors.extend(l1.priors[-4:])
        self.ps = PriorSet(self.priors)

        self.filters = l1.filters
        self.passband = l1.passband
        self.prior_kw = l1.prior_kw

        self.use_ldtk = use_ldtk
        if use_ldtk:
            self.sc = LDPSetCreator([4150,100], [4.6,0.2], [-0.14,0.16], self.filters)
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

    def fit_ldc(self, pvpop, emul=1.):
        pvt = pvpop.copy()
        uv, uve = self.lp.coeffs_qd()
        us = array([normal(um, emul*ue, size=pvt.shape[0]) for um,ue in zip(uv[:,0],uve[:,0])]).T
        vs = array([normal(vm, emul*ve, size=pvt.shape[0]) for vm,ve in zip(uv[:,1],uve[:,1])]).T
        q1s, q2s = map_uv_to_qq(us, vs)
        pvt[:, self.uq1] = q1s
        pvt[:, self.uq2] = q2s
        return pvt
