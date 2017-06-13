import sys
from copy import copy
from numpy import *

from scipy.signal import medfilt as MF
from numpy.random import normal, seed
from statsmodels.robust import mad

from .core import *
from .lpf import *
from .extcore import *
from .lpfsd import LPFSD
from george.kernels import ConstantKernel, Matern32Kernel

class LPFSR(LPFSD):
    def __init__(self, passband, lctype='target', use_ldtk=False, n_threads=1, night=2, pipeline='gc'):
        super().__init__(passband, lctype, use_ldtk, n_threads, night, pipeline)
        self.lnlikelihood = self.lnlikelihood_wn
        self.noise = 'white'

        self.fluxes = asarray(self.fluxes)
        self.fluxes_m = self.fluxes.mean(0)
        self.fluxes /= self.fluxes_m
        self.wn_estimates = array([sqrt(2) * mad(diff(f)) for f in self.fluxes])
        self.times = self.times[0]
        self.ctimes  = self.times-self.times.mean()


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
        self._nbl = 3
        for ilc in range(self.nlc):
            self.priors.append(UP( 0.0, 2.0, 'bcn_%i'%ilc))  #  sbl + ilc -- Baseline constant
            self.priors.append(UP(-1.0, 1.0, 'btl_%i'%ilc))  #  sbl + ilc -- Linear time trend
            self.priors.append(UP(-1.0, 1.0, 'bal_%i'%ilc))  #  sbl + ilc -- Linear airmass trend

        ## White noise
        ## -----------
        self._swn = len(self.priors)
        self.priors.extend([UP(3e-4, 4e-3, 'e_%i'%ilc)
                            for ilc in range(self.nlc)]) ##  sqn + ilc -- Average white noise

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
            
        # Use mock data
        # -------------
        if passband == 'nb_mock' and type(self) == LPFSR:
            self.create_mock_nb_dataset()


    def set_pv_indices(self, sbl=None, swn=None):
        self.ik2 = [self._sk2+pbid for pbid in self.gpbids]
                        
        self.iq1 = [self._sq1+pbid*2 for pbid in self.gpbids]
        self.iq2 = [self._sq2+pbid*2 for pbid in self.gpbids]
        self.uq1 = np.unique(self.iq1)
        self.uq2 = np.unique(self.iq2)

        sbl = sbl if sbl is not None else self._sbl
        self.ibcn = [sbl + 3 * ilc     for ilc in range(self.nlc)]
        self.ibtl = [sbl + 3 * ilc + 1 for ilc in range(self.nlc)]
        self.ibal = [sbl + 3 * ilc + 2 for ilc in range(self.nlc)]

        swn = swn if swn is not None else self._swn
        self.iwn = [swn+ilc for ilc in range(self.nlc)]


    def setup_gp(self):
        pass

    def map_to_gp(self, pv):
        raise NotImplementedError


    def lnposterior(self, pv):
        _k = median(sqrt(pv[self.ik2]))
        return super().lnposterior(pv) + self.prior_kw.log(_k)


    def compute_transit(self, pv):
        _a = as_from_rhop(pv[2], pv[1])
        _i = mt.acos(pv[3] / _a)
        _k = sqrt(pv[self.ik2]).mean()
        kf = pv[self.ik2] / _k ** 2

        a, b = sqrt(pv[self.iq1]), 2. * pv[self.iq2]
        self._wrk_ld[:, 0] = a * b
        self._wrk_ld[:, 1] = a * (1. - b)

        z = of.z_circular(self.times, pv[0], pv[1], _a, _i, self.nt)
        f = self.tm(z, _k, self._wrk_ld)
        return (kf * (f - 1.) + 1.).T


    def compute_lc_model(self, pv, copy=False):
        bl = self.compute_baseline(pv)
        tr = self.compute_transit(pv)
        self._wrk_lc[:] = bl*tr/tr.mean(0)
        return self._wrk_lc if not copy else self._wrk_lc.copy()


    def compute_baseline(self, pv):
        bl = ( pv[self.ibcn][:,newaxis]
             + pv[self.ibtl][:,newaxis] * self.ctimes
             + pv[self.ibal][:,newaxis] * self.airmass)
        return bl


    def lnlikelihood_wn(self, pv):
        fluxes_m = self.compute_lc_model(pv)
        return sum([ll_normal_es(fo, fm, wn) for fo,fm,wn in zip(self.fluxes, fluxes_m, pv[self.iwn])])


    def fit_baseline(self, pvpop):
        from numpy.linalg import lstsq
        pvt = pvpop.copy()
        X = array([ones(self.npt), self.ctimes, self.airmass])
        for i in range(self.nlc):
            pv = lstsq(X.T, self.fluxes[i])[0]
            pvt[:, self.ibcn[i]] = normal(pv[0], 0.001, size=pvt.shape[0])
            pvt[:, self.ibtl[i]] = normal(pv[1], 0.01 * abs(pv[1]), size=pvt.shape[0])
            pvt[:, self.ibal[i]] = normal(pv[2], 0.01 * abs(pv[2]), size=pvt.shape[0])
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