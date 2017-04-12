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

class LPFSD(LPF):
    def __init__(self, passband, lctype='relative', use_ldtk=False, n_threads=1, night=2, pipeline='gc'):
        assert passband in ['w', 'bb', 'nb', 'K', 'Na', 'pr']
        assert lctype in ['target', 'relative']
        assert pipeline in ['hp', 'gc']
        assert night in [1,2]

        if pipeline == 'hp':
            self.df = df = pd.merge(pd.read_hdf(join(DDATA, 'aux.h5'), 'night%i'%night),
                                      pd.read_hdf(join(DRESULT, 'gtc_light_curves.h5'), 'night%i'%night),
                                      left_index=True, right_index=True)
        else:
            self.df = df = pd.read_hdf(join(DRESULT,'gtc_light_curves_gc.h5'), 'night%i'%night)

        self.night = night
        self.passband = passband        
        if passband == 'bb':
            cols = ['{:s}_{:s}'.format(lctype, pb) for pb in 'g r i z'.split()]
            self.filters = pb_filters_bb
        elif passband in ['w', 'nb']:
            cols = [c for c in self.df.columns if lctype+'_nb' in c]
            self.filters = pb_filters_nb
        elif passband == 'K':
            cols = [c for c in df.columns if lctype+'_K'  in c]
            self.filters = pb_filters_k
        elif passband == 'Na':
            cols = [c for c in df.columns if lctype+'_Na'  in c]
            self.filters = pb_filters_na

        if passband == 'w':
            pbs = ['w']
            fluxes = [df[cols].values.mean(1)]
            NN = lambda a: a / a.max()
            self.filters = [
                TabulatedFilter('white', pb_filters_bb[0].wl, NN(array([f.tm for f in pb_filters_bb]).mean(0)))]
        else:
            pbs = [c.split('_')[1] for c in cols]
            fluxes = list(df[cols].values.T)

        npb = len(pbs)
        times  = npb*[df.bjd.values-TZERO]
        fluxes = [f/nanmedian(f[:20]) for f in fluxes]
        
        ## Mask outliers
        ## -------------
        lim = 0.006
        self.mask = ones(fluxes[0].size, np.bool)
        for i in range(npb):
            f = fluxes[i]
            self.mask &= abs(f-MF(f,11)) < lim
            if lctype == 'relative' and self.night == 1:
                self.mask &= (times[0] < 855.528) | (times[0] > 855.546)

        times   = [times[i][self.mask]  for i in range(npb)]
        fluxes  = [fluxes[i][self.mask] for i in range(npb)]

        fc = pd.read_hdf(RFILE_EXT, 'vkrn_ldtk/fc')
        self.otmasks = [abs(fold(t, fc.p.mean(), fc.tc.mean(), 0.5)-0.5) < 0.0134 for t in times]
        self.rotang  = np.radians(df.rotang[self.mask].values)
        self.airmass = df.airmass[self.mask].values

        self.npt = self.airmass.size

        # Initialise the parent
        # ---------------------
        super().__init__(times, fluxes, pbs,
                         use_ldtk=False, constant_k=False, noise='red',
                         ldf_path='../data/external_lcs.h5', nthreads=n_threads)
        self.use_ldtk = use_ldtk

        self._wrk_lc = zeros([self.npb, self.fluxes[0].size])

        # Setup priors
        # ------------
        # Basic parameters
        # ----------------
        self.priors = [NP(    TC,   1e-2,   'tc'), ##  0  - Transit centre
                       NP(     P,   5e-4,    'p'), ##  1  - Period
                       UP(  3.50,   4.50,  'rho'), ##  2  - Stellar density
                       UP(  0.00,   0.99,    'b')] ##  3  - Impact parameter
        
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
        self._nbl = 1
        for ilc in range(self.nlc):
            self.priors.append(NP( 1.0, 0.001, 'bcn_%i'%ilc))  #  sbl + ilc -- Baseline constant

        # GP hyperparameters
        # ------------------
        self._sgp = len(self.priors)
        self.priors.extend([UP(-5, -1, 'log10_am_amplitude'),
                            UP(-5,  3, 'log10_am_inv_scale'),
                            UP(-5, -1, 'log10_ra_amplitude'),
                            UP(-5,  3, 'log10_ra_inv_scale')])

        self.ps = PriorSet(self.priors)
        self.set_pv_indices()

        self.wn_estimates = array([sqrt(2) * mad(diff(f)) for f in self.fluxes])

        ## Update the priors using the external data modelling
        ## ---------------------------------------------------
        fc = pd.read_hdf(RFILE_EXT, 'vkrn_ldtk/fc')
        self.priors[0] = NP(fc.tc.mean(),   fc.tc.std(),   'tc',   limsigma=15)
        self.priors[1] = NP(fc.p.mean(),    fc.p.std(),    'p',    limsigma=15)
        self.priors[2] = NP(fc.rho.mean(),  fc.rho.std(),  'rho',  limsigma=5)
        self.priors[3] = NP(fc.b.mean(),    fc.b.std(),    'b',    lims=(0,1))
        self.ps = PriorSet(self.priors)

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
        self.ibcn = [sbl + ilc for ilc in range(self.nlc)]

        if hasattr(self, '_sgp'):
            self.slgp = s_[self._sgp:self._sgp+4]


    def setup_gp(self):
        self.gp_inputs = array([self.airmass, self.rotang]).T
        self.kernel = (ConstantKernel(1e-3 ** 2, ndim=2, axes=0) * ExpSquaredKernel(.01, ndim=2, axes=0)
                     + ConstantKernel(1e-3 ** 2, ndim=2, axes=1) * ExpSquaredKernel(.01, ndim=2, axes=1))
        self.gp = GP(self.kernel)
        self.gp.compute(self.gp_inputs, yerr=5e-4)


    def map_to_gp(self, pv):
        log10_to_ln = 1./log10(e)
        gpp = zeros(4)
        gpp[0] =  2 * pv[0] * log10_to_ln
        gpp[1] =     -pv[1] * log10_to_ln
        gpp[2] =  2 * pv[2] * log10_to_ln
        gpp[3] =     -pv[3] * log10_to_ln
        return gpp


    def lnposterior(self, pv):
        return super().lnposterior(pv)

        
    def compute_transit(self, pv):
        _a  = as_from_rhop(pv[2], pv[1]) 
        _i  = mt.acos(pv[3]/_a) 
        _k = sqrt(pv[self.ik2]).mean()
        kf = pv[self.ik2]/_k**2

        a,b = sqrt(pv[self.iq1]), 2.*pv[self.iq2]
        self._wrk_ld[:,0] = a*b
        self._wrk_ld[:,1] = a*(1.-b)

        z = of.z_circular(self.times[0], pv[0], pv[1], _a, _i, self.nt) 
        f = self.tm(z, _k, self._wrk_ld)
        return (kf*(f-1.)+1.).T
        

    def compute_lc_model(self, pv, copy=False):
        bl = self.compute_baseline(pv)
        tr = self.compute_transit(pv)
        self._wrk_lc[:] = bl[:,np.newaxis]*tr
        return self._wrk_lc if not copy else self._wrk_lc.copy()


    def compute_baseline(self, pv):
        return pv[self.ibcn]


    def lnlikelihood_rn(self, pv):
        self.kernel[:] = self.map_to_gp(pv[self.slgp])
        self.gp.compute(self.gp_inputs, yerr=self.wn_estimates.mean())
        fmodel = self.compute_lc_model(pv)
        lnlike = sum([self.gp.lnlikelihood(fo - fm) for fo, fm in zip(self.fluxes, fmodel)])
        return lnlike


    def lnlikelihood_wn(self, pv):
        raise NotImplementedError


    def fit_baseline(self, pvpop):
        raise NotImplementedError


    def fit_ldc(self, pvpop, emul=1.):
        pvt = pvpop.copy()
        uv, uve = self.lp.coeffs_qd()
        us = array([normal(um, emul*ue, size=pvt.shape[0]) for um,ue in zip(uv[:,0],uve[:,0])]).T
        vs = array([normal(vm, emul*ve, size=pvt.shape[0]) for vm,ve in zip(uv[:,1],uve[:,1])]).T
        q1s, q2s = map_uv_to_qq(us, vs)
        pvt[:, self.uq1] = q1s
        pvt[:, self.uq2] = q2s
        return pvt
