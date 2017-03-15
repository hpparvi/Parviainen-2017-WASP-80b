from numpy import *

from .core import *
from .lpf import *
from .extcore import *
from .lpfmd import LPFMD

class LPFMD4(LPFMD):
    def __init__(self, passband, lctype='target', use_ldtk=False, n_threads=1, noise='white', pipeline='hp'):
        super().__init__(passband, lctype, use_ldtk, n_threads, noise, pipeline)
        self.priors[0] = NP(855.568, 0.002, 'tc1')
        self.priors[1] = NP(895.450, 0.002, 'tc2')
        self.priors.append(UP(1.0, 1.3, 'kf', 'mean_c_factor'))
        self.ps = PriorSet(self.priors)
        self.set_pv_indices()

    def compute_transit(self, pv):
        _a  = as_from_rhop(pv[2], 3.0678593)
        _i  = mt.acos(pv[3]/_a) 
        _k = sqrt(pv[self.ik2]).mean()
        kf = pv[self.ik2]/_k**2

        a,b = sqrt(pv[self.iq1]), 2.*pv[self.iq2]
        self._wrk_ld[:,0] = a*b
        self._wrk_ld[:,1] = a*(1.-b)

        z1 = of.z_circular(self.times[0], pv[0], 3.0678593, _a, _i, self.nt)
        z2 = of.z_circular(self.times[self.npb], pv[1], 3.0678593, _a, _i, self.nt)

        f1 = self.tm(z1, _k, self._wrk_ld[:self.npb])
        f2 = self.tm(z2, _k, self._wrk_ld[self.npb:])

        return (kf[self.npb:]*(f1-1.)+1.).T, (pv[-1]*kf[:self.npb]*(f2-1.)+1.).T