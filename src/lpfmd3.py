from numpy import *

from .core import *
from .lpf import *
from .extcore import *
from .lpfmd import LPFMD

from exotk.utils.eclipses import Planck

def spotty_k(wl, k0, Tstar, f, DeltaT):
    """
    Modified radius ratio due to unocculted spots.

    Parameters

      wl     : wavelength [m]
      k0     : true geometric radius ratio
      Tstar  : effective stellar temperature [K]
      f      : spot filling factor
      DeltaT : spot temperature difference [K]
    """
    Tspot = Tstar - DeltaT
    fratio = Planck(Tspot, wl) / Planck(Tstar, wl)
    contrast = 1 - fratio

    return k0 * sqrt(1 / (1 - f * contrast))

class LPFMD3(LPFMD):
    def __init__(self, passband, lctype='target', use_ldtk=False, n_threads=1, noise='white', pipeline='hp'):
        super().__init__(passband, lctype, use_ldtk, n_threads, noise, pipeline)
        self.pbc = 1e-9*array(pb_centers[self.passband])

        self.priors.append(UP(0.0, 0.4, 'sf1', 'spot_filling_factor_1'))
        self.priors.append(UP(0.0, 0.4, 'sf2', 'spot_filling_factor_2'))
        self.ps = PriorSet(self.priors)
        self.set_pv_indices()

    def compute_transit(self, pv):
        _a  = as_from_rhop(pv[2], pv[1]) 
        _i  = mt.acos(pv[3]/_a) 
        _k = sqrt(pv[self.ik2]).mean()
        kf = pv[self.ik2]/_k**2

        a,b = sqrt(pv[self.iq1]), 2.*pv[self.iq2]
        self._wrk_ld[:,0] = a*b
        self._wrk_ld[:,1] = a*(1.-b)

        z1 = of.z_circular(self.times[0], pv[0], pv[1], _a, _i, self.nt) 
        z2 = of.z_circular(self.times[self.npb], pv[0], pv[1], _a, _i, self.nt) 

        f1 = self.tm(z1, _k, self._wrk_ld[:self.npb])
        f2 = self.tm(z2, _k, self._wrk_ld[self.npb:])

        sf1 = spotty_k(self.pbc, 1., TSTAR, pv[-2], 1200.)
        sf2 = spotty_k(self.pbc, 1., TSTAR, pv[-1], 1200.)

        return (sf1*kf[self.npb:]*(f1-1.)+1.).T, (sf2**kf[:self.npb]*(f2-1.)+1.).T


    def lnposterior(self, pv):
        _k = median(sqrt(pv[self.ik2]))
        return super().lnposterior(pv) + self.prior_kw.log(_k)