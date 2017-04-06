from numpy import s_
from exotk.priors import PriorSet, UP, NP

from .lpf import CLPF
from .lpfsd import LPFSD

class LPFMD(CLPF):
    def __init__(self, passband, lctype='relative', use_ldtk=False, n_threads=1, pipeline='gc'):
        self.lpf1 = l1 = LPFSD(passband, lctype, use_ldtk=False, pipeline=pipeline, night=1, n_threads=n_threads)
        self.lpf2 = l2 = LPFSD(passband, lctype, use_ldtk=False, pipeline=pipeline, night=2, n_threads=n_threads)
        super().__init__((l1,l2), constant_k=False, noise='red', use_ldtk=use_ldtk)
        l1._sgp = l2._sgp = self.ps.ndim
        l1.slgp = l2.slgp = s_[l1._sgp:l1._sgp+4]
        self.priors[:7] = l1.priors[:7]
        self.priors.extend(l1.priors[-4:])
        self.ps = PriorSet(self.priors)

    @property
    def times(self):
        return [lpf.times for lpf in self.lpfs]

    @property
    def fluxes(self):
        return [lpf.fluxes for lpf in self.lpfs]

    def compute_baseline(self, pv):
        return [lpf.compute_baseline(pv) for lpf in self.lpfs]

    def compute_transit(self, pv):
        return [lpf.compute_transit(pv) for lpf in self.lpfs]

    def compute_lc_model(self, pv):
        return [lpf.compute_lc_model(pv) for lpf in self.lpfs]
