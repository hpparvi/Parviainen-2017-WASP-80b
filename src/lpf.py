from __future__ import division

import math as mt
from copy import copy

from exotk.priors import PriorSet, UP, NP
from exotk.utils.likelihood import ll_normal_es
from exotk.utils.orbits import as_from_rhop
from numpy import zeros, sqrt, isfinite, inf
from pytransit import MandelAgol as MA
from pytransit.orbits_f import orbits as of

from .core import *


def map_qq_to_uv(q1, q2):
    a, b = sqrt(q1), 2. * q2
    return a * b, a * (1. - b)


def map_uv_to_qq(u, v):
    return (u + v) ** 2, u / (2 * (u + v))


class LogPosteriorFunction(object):
    def __init__(self, passbands, constant_k=True, noise='white', use_ldtk=False, **kwargs):
        self.passbands = pd.Categorical(passbands, categories=c_passbands, ordered=True)
        self.lcorder = self.passbands.argsort()
        self.unique_pbs = self.passbands.order().unique()
        self.passbands = pd.Categorical(self.passbands, categories=self.unique_pbs.__array__(), ordered=True)
        self.gpbids = self.passbands.labels.copy()
        self.lpbids = copy(self.gpbids)

        self.use_ldtk = use_ldtk
        self.constant_k = constant_k
        self.noise = noise

        self.nlc = nlc = len(self.passbands)
        self.npb = npb = len(self.unique_pbs)
        self._wrk_ld = zeros([self.nlc, 2])

        ldf_path = kwargs.get('ldf_path', 'data/external_lcs.h5')

        # Basic parameters
        # ----------------
        self.priors = [NP(TC, 5e-3, 'tc'),  ##  0  - Transit centre
                       NP(P, 3e-4, 'p'),  ##  1  - Period
                       UP(3.50, 4.50, 'rho'),  ##  2  - Stellar density
                       UP(0.00, 0.99, 'b')]  ##  3  - Impact parameter

        # Area ratio
        # ----------
        self._sk2 = len(self.priors)
        if constant_k:
            self.priors.append(UP(0.160 ** 2, 0.185 ** 2, 'k2_w'))  ##  4  - planet-star area ratio
        else:
            self.priors.extend([UP(0.160 ** 2, 0.185 ** 2, 'k2_%s' % pb)
                                for pb in self.unique_pbs])  ##  4  - planet-star area ratio

        # Limb darkening
        # --------------
        self._sq1 = len(self.priors)
        self._sq2 = self._sq1 + 1
        for pb in self.unique_pbs:
            self.priors.extend([UP(0, 1, 'q1_%s' % pb),  ##  sq1 + 2*ipb -- limb darkening q1
                                UP(0, 1, 'q2_%s' % pb)])  ##  sq2 + 2*ipb -- limb darkening q2

        # Baseline constant
        # -----------------
        self._sbl = len(self.priors)
        self.priors.extend([NP(1, 1e-4, 'bl_%i' % ilc)
                            for ilc in range(self.nlc)])  ##  sbl + ilc -- Baseline constant

        # White noise
        # -----------
        self._swn = len(self.priors)
        if noise == 'white':
            self.priors.extend([UP(1e-4, 1e-2, 'e_%i' % ilc)
                                for ilc in range(self.nlc)])  ##  sqn + ilc -- Average white noise

        self.ps = PriorSet(self.priors)
        self.set_pv_indices()

        # Limb darkening with LDTk
        # ------------------------
        if use_ldtk:
            dff = pd.read_hdf(ldf_path, 'transmission')
            self.filters = []
            for pb in self.unique_pbs:
                self.filters.append(TabulatedFilter(pb, dff.index.values, tm=dff[pb].values))
            self.sc = LDPSetCreator([4150, 100], [4.6, 0.2], [-0.14, 0.16], self.filters)
            self.lp = self.sc.create_profiles(2000)
            self.lp.set_uncertainty_multiplier(2)

    def set_pv_indices(self, sbl=None, swn=None):
        if self.constant_k:
            self.ik2 = [self._sk2]
        else:
            self.ik2 = [self._sk2 + pbid for pbid in self.gpbids]

        self.iq1 = [self._sq1 + pbid * 2 for pbid in self.gpbids]
        self.iq2 = [self._sq2 + pbid * 2 for pbid in self.gpbids]
        self.uq1 = np.unique(self.iq1)
        self.uq2 = np.unique(self.iq2)

        sbl = sbl if sbl is not None else self._sbl
        self.ibl = [sbl + ilc for ilc in range(self.nlc)]

        if self.noise == 'white':
            swn = swn if swn is not None else self._swn
            self.iwn = [swn + ilc for ilc in range(self.nlc)]
        else:
            self.iwn = []

    def setup_gp(self):
        raise NotImplementedError

    def lnprior(self, pv):
        if any(pv < self.ps.pmins) or any(pv > self.ps.pmaxs):
            return -inf
        else:
            return self.ps.c_log_prior(pv)

    def lnlikelihood(self, pv):
        raise NotImplementedError

    def lnlikelihood_wn(self, pv):
        raise NotImplementedError

    def lnlikelihood_rn(self, pv):
        raise NotImplementedError

    def lnlikelihood_ld(self, pv):
        if self.use_ldtk:
            uv = zeros([self.npb, 2])
            a, b = sqrt(pv[self.uq1]), 2. * pv[self.uq2]
            uv[:, 0] = a * b
            uv[:, 1] = a * (1. - b)
            return self.lp.lnlike_qd(uv)
        else:
            return 0.0

    def lnposterior(self, pv):
        lnp = self.lnprior(pv)
        if not isfinite(lnp):
            return lnp
        else:
            return lnp + self.lnlikelihood_ld(pv) + self.lnlikelihood(pv)


class LPF(LogPosteriorFunction):
    def __init__(self, times, fluxes, passbands, constant_k=True, noise='white', use_ldtk=False, nthreads=4, **kwargs):
        """Log-posterior function for a single dataset.
        """
        super(LPF, self).__init__(passbands, constant_k, noise, use_ldtk, **kwargs)
        self.tm = MA(interpolate=False, nthr=nthreads)
        self.nt = nthreads
        self.times = times
        self.fluxes = fluxes
        self._wrk_k = zeros(self.nlc)

        # Noise model setup
        # -----------------
        if noise == 'white':
            self.lnlikelihood = self.lnlikelihood_wn
        elif noise == 'red':
            self.setup_gp()
            self.lnlikelihood = self.lnlikelihood_rn
        else:
            raise NotImplementedError('Bad noise model')

    def apply_masks(self, masks):
        self.masks = masks
        self.mtimes = [t[~m] for t, m in zip(self.times, masks)]
        self.times = [t[m] for t, m in zip(self.times, masks)]
        self.mfluxes = [f[~m] for f, m in zip(self.fluxes, masks)]
        self.fluxes = [f[m] for f, m in zip(self.fluxes, masks)]

    def compute_baseline(self, pv):
        return pv[self.ibl]

    def compute_transit(self, pv):
        _a = as_from_rhop(pv[2], pv[1])
        _i = mt.acos(pv[3] / _a)
        self._wrk_k[:] = sqrt(pv[self.ik2])

        a, b = sqrt(pv[self.iq1]), 2. * pv[self.iq2]
        self._wrk_ld[:, 0] = a * b
        self._wrk_ld[:, 1] = a * (1. - b)

        flux_m = []
        for time, k, ldc in zip(self.times, self._wrk_k, self._wrk_ld):
            z = of.z_circular(time, pv[0], pv[1], _a, _i, self.nt)
            flux_m.append(self.tm(z, k, ldc))
        return flux_m

    def compute_lc_model(self, pv):
        return [b * m for b, m in zip(self.compute_baseline(pv), self.compute_transit(pv))]

    def lnlikelihood_wn(self, pv):
        fluxes_m = self.compute_lc_model(pv)
        return sum([ll_normal_es(fo, fm, wn) for fo, fm, wn in zip(self.fluxes, fluxes_m, pv[self.iwn])])


class CLPF(LogPosteriorFunction):
    def __init__(self, lpfs, constant_k=True, noise='white', use_ldtk=False):
        """Log-posterior function combining multiple datasets.
        """
        super(CLPF, self).__init__(concatenate([lpf.passbands for lpf in lpfs]), constant_k, noise, use_ldtk)
        self.lpfs = lpfs

        sbl = self._sbl
        swn = self._swn if self.noise == 'white' else None
        for lpf in self.lpfs:
            lpf.gpbids = pd.Categorical(lpf.passbands, categories=self.passbands.categories, ordered=True).labels
            lpf._sq1 = self._sq1
            lpf._sq2 = self._sq2
            lpf._sbl = self._sbl
            lpf._swn = self._swn
            lpf.set_pv_indices(sbl, swn)
            sbl += lpf._nbl * lpf.nlc
            if self.noise == 'white':
                swn += lpf.nlc

    def lnposterior(self, pv):
        lnp = self.lnprior(pv)
        if not isfinite(lnp):
            return lnp
        else:
            return lnp + self.lnlikelihood_ld(pv) + sum([lpf.lnlikelihood(pv) for lpf in self.lpfs])

    @property
    def times(self):
        return np.concatenate([lpf.times for lpf in self.lpfs])

    @property
    def fluxes(self):
        return np.concatenate([lpf.fluxes for lpf in self.lpfs])

    def compute_baseline(self, pv):
        return np.concatenate([lpf.compute_baseline(pv) for lpf in self.lpfs])

    def compute_transit(self, pv):
        return np.concatenate([lpf.compute_transit(pv) for lpf in self.lpfs])

    def compute_lc_model(self, pv):
        return np.concatenate([lpf.compute_lc_model(pv) for lpf in self.lpfs])
