import sys
from copy import copy
from numpy import *

from scipy.signal import medfilt as MF
from numpy.random import normal
from statsmodels.robust import mad

from .core import *
from .lpf import *
from .extcore import *
from .lpfsd import LPFSD
from george.kernels import ConstantKernel, Matern32Kernel

class LPFSR(LPFSD):
    def __init__(self, passband, lctype='target', use_ldtk=False, n_threads=1, night=2, pipeline='gc'):
        super().__init__(passband, lctype, use_ldtk, n_threads, night, pipeline)

        self.fluxes = asarray(self.fluxes)
        self.fluxes_m = self.fluxes.mean(0)
        self.fluxes /= self.fluxes_m
        self.wn_estimates = array([sqrt(2) * mad(diff(f)) for f in self.fluxes])


        # Setup priors
        # ------------
        # System parameters
        # -----------------
        self.priors = [NP(125.417380,  8e-5,   'tc'),  #  0  - Transit centre
                       NP(3.06785547,  4e-7,    'p'),  #  1  - Period
                       NP(4.17200000,  3e-2,  'rho'),  #  2  - Stellar density
                       NP(0.16100000,  2e-2,    'b')]  #  3  - Impact parameter

        # Area ratio
        # ----------
        self._sk2 = len(self.priors)
        self.priors.extend([UP(0.165**2, 0.195**2, 'k2_%s'%pb)
            for pb in self.unique_pbs]) ##  4  - planet-star area ratio
            
        # Limb darkening
        # --------------
        self._sq1 = len(self.priors)
        self._sq2 = self._sq1+1
        for ipb in range(self.npb):
            self.priors.extend([UP(0, 1, 'q1_%i'%ipb),      ##  sq1 + 2*ipb -- limb darkening q1
                                UP(0, 1, 'q2_%i'%ipb)])     ##  sq2 + 2*ipb -- limb darkening q2

        ## Baseline
        ## --------
        self._sbl = len(self.priors)
        self._nbl = 2
        for ilc in range(self.nlc):
            self.priors.append(UP( 0.0, 2.0, 'bcn_%i'%ilc))  #  sbl + ilc -- Baseline constant
            self.priors.append(UP(-1.0, 1.0, 'bal_%i'%ilc))  #  sbl + ilc -- Linear airmass trend

        # Gaussian Process hyperparameters
        # --------------------------------
        self._sgp = len(self.priors)
        self.priors.extend([UP(-5, -1, 'log10_tm_amplitude'),
                            NP(-5,  3, 'log10_tm_inv_scale', lims=[-5, 3]),
                            UP(-5, -1, 'log10_ra_amplitude'),
                            NP(-5,  2, 'log10_ra_inv_scale', lims=[-5, 3])])
        self.ps = PriorSet(self.priors)
        self.set_pv_indices()

        self.prior_kw = NP(0.1707, 3.2e-4, 'kw', lims=(0.16,0.18))
        
        ## Limb darkening with LDTk
        ## ------------------------
        if use_ldtk:
            self.sc = LDPSetCreator([4150,100], [4.6,0.2], [-0.14,0.16], self.filters)
            self.lp = self.sc.create_profiles(2000)
            self.lp.resample_linear_z()
            self.lp.set_uncertainty_multiplier(2)
            
            
    def set_pv_indices(self, sbl=None, swn=None):
        self.ik2 = [self._sk2+pbid for pbid in self.gpbids]
                        
        self.iq1 = [self._sq1+pbid*2 for pbid in self.gpbids]
        self.iq2 = [self._sq2+pbid*2 for pbid in self.gpbids]
        self.uq1 = np.unique(self.iq1)
        self.uq2 = np.unique(self.iq2)

        sbl = sbl if sbl is not None else self._sbl
        self.ibcn = [sbl + 2 * ilc     for ilc in range(self.nlc)]
        self.ibal = [sbl + 2 * ilc + 1 for ilc in range(self.nlc)]

        swn = swn if swn is not None else self._swn
        self.iwn = [swn+ilc for ilc in range(self.nlc)]

        if hasattr(self, '_sgp'):
            self.slgp = s_[self._sgp:self._sgp+4]


    def setup_gp(self):
        self.gp_inputs = array([self.times[0], self.rotang]).T
        self.kernel = (ConstantKernel(1e-3**2, ndim=2, axes=0) * Matern32Kernel(.01, ndim=2, axes=0)
                     + ConstantKernel(1e-3**2, ndim=2, axes=1) * ExpSquaredKernel(.01, ndim=2, axes=1))
        self.gp = GP(self.kernel)
        self.gp.compute(self.gp_inputs, yerr=5e-4)


    def map_to_gp(self, pv):
        log10_to_ln = 1. / log10(e)
        gpp = zeros(4)
        gpp[0] = 2 * pv[0] * log10_to_ln
        gpp[1] =    -pv[1] * log10_to_ln
        gpp[2] = 2 * pv[2] * log10_to_ln
        gpp[3] =    -pv[3] * log10_to_ln
        return gpp


    def lnposterior(self, pv):
        _k = median(sqrt(pv[self.ik2]))
        return super().lnposterior(pv) + self.prior_kw.log(_k)


    def compute_lc_model(self, pv, copy=False):
        bl = self.compute_baseline(pv)
        tr = self.compute_transit(pv)
        self._wrk_lc[:] = bl*tr/tr.mean(0)
        return self._wrk_lc if not copy else self._wrk_lc.copy()


    def compute_baseline(self, pv):
        bl = ( pv[self.ibcn][:,newaxis]
             + pv[self.ibal][:,newaxis] * self.airmass)
        return bl


    def fit_baseline(self, pvpop):
        from numpy.linalg import lstsq
        pvt = pvpop.copy()
        X = array([ones(self.npt), self.airmass])
        for i in range(self.nlc):
            pv = lstsq(X.T, self.fluxes[i])[0]
            pvt[:, self.ibcn[i]] = normal(pv[0], 0.001, size=pvt.shape[0])
            pvt[:, self.ibal[i]] = normal(pv[1], 0.01 * abs(pv[1]), size=pvt.shape[0])
        return pvt


    def fit_ldc(self, pvpop, emul=1.):
        pvt = pvpop.copy()
        uv, uve = self.lp.coeffs_qd()
        us = array([normal(um, emul*ue, size=pvt.shape[0]) for um,ue in zip(uv[:,0],uve[:,0])]).T
        vs = array([normal(vm, emul*ve, size=pvt.shape[0]) for vm,ve in zip(uv[:,1],uve[:,1])]).T
        q1s, q2s = map_uv_to_qq(us, vs)
        pvt[:, self.uq1] = q1s
        pvt[:, self.uq2] = q2s
        return pvt
