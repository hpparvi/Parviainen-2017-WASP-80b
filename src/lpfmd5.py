import sys
from copy import copy
from itertools import chain
from numpy import *

from scipy.signal import medfilt as MF
from numpy.random import normal, uniform
from statsmodels.robust import mad

from .core import *
from .lpf import *
from .extcore import *

from george.kernels import CosineKernel
from george.solvers import HODLRSolver

from math import pi

class LPFMD5(LPF):
    def __init__(self, passband, lctype='target', use_ldtk=False, n_threads=1, noise='white', pipeline='hp'):
        assert passband in ['w','bb','nb','K','Na', 'pr']
        assert lctype in ['target', 'relative']
        assert noise in ['white', 'red']
        assert pipeline in ['hp','gc']

        if pipeline == 'hp':
            self.df1 = df1 = pd.merge(pd.read_hdf(join(DDATA, 'aux.h5'), 'night1'),
                                      pd.read_hdf(join(DRESULT, 'gtc_light_curves.h5'), 'night1'),
                                      left_index=True, right_index=True)
            self.df2 = df2 = pd.merge(pd.read_hdf(join(DDATA, 'aux.h5'), 'night2'),
                                      pd.read_hdf(join(DRESULT, 'gtc_light_curves.h5'), 'night2'),
                                      left_index=True, right_index=True)
        else:
            self.df1 = df1 = pd.read_hdf(join(DRESULT,'gtc_light_curves_gc.h5'), 'night1')
            self.df2 = df2 = pd.read_hdf(join(DRESULT,'gtc_light_curves_gc.h5'), 'night2')
        
        self.passband = passband        
        if passband == 'bb':
            cols = ['{:s}_{:s}'.format(lctype, pb) for pb in 'g r i z'.split()]
            self.filters = pb_filters_bb
        elif passband in ['w', 'nb']:
            cols = [c for c in self.df1.columns if lctype+'_nb' in c]
            self.filters = pb_filters_nb
        elif passband == 'K':
            cols = [c for c in df1.columns if lctype+'_K'  in c]
            self.filters = pb_filters_k
        elif passband == 'Na':
            cols = [c for c in df1.columns if lctype+'_Na'  in c]
            self.filters = pb_filters_na
        elif passband == 'pr':
            cols = [c for c in df1.columns if lctype+'_pr'  in c]
            self.filters = pb_filters_pr

        if passband == 'w':
            pbs = ['w']
            fluxes = [df1[cols].values.mean(1), df2[cols].values.mean(1)]
            NN = lambda a: a/a.max()
            self.filters = [TabulatedFilter('white', pb_filters_bb[0].wl, NN(array([f.tm for f in pb_filters_bb]).mean(0)))]
        else:
            pbs = [c.split('_')[1] for c in cols]
            fluxes = (list(df1[cols].values.T) + list(df2[cols].values.T))

        npb = len(pbs)

        times  = df1.bjd.values-TZERO, df2.bjd.values-TZERO
        fluxes = [f/nanmedian(f) for f in fluxes]

        self.otimes = times
        
        ## Mask outliers
        ## -------------
        self.masks = masks = []
        for j in range(2):
            lim = 0.006
            mask = ones(fluxes[npb*j].size, np.bool)
            for i in range(npb):
                f = fluxes[i+npb*j]
                mask &= abs(f-MF(f,11)) < lim
            if (lctype == 'relative') and (j == 0):
                mask &= (times[0] < 855.528) | (times[0] > 855.546)
            self.masks.append(mask)
            
        times   = times[0][masks[0]], times[1][masks[1]]
        fluxes  = [fluxes[i][masks[i//npb]] for i in range(2*npb)]

        fc = pd.read_hdf(RFILE_EXT, 'vkrn_ldtk/fc')
        self.otmasks = [abs(fold(t, fc.p.mean(), fc.tc.mean(), 0.5)-0.5) < 0.0134 for t in times]
        self.rotang  = np.deg2rad(df1.rotang[masks[0]].values), np.deg2rad(df2.rotang[masks[1]].values)
        self.elevat  = df1.elevat[masks[0]].values, df2.elevat[masks[1]].values
        self.airmass = df1.airmass[masks[0]].values, df2.airmass[masks[1]].values
        self.ctimes  = [t-t.mean() for t in times]
        
        self.airmass_means = [a.mean() for a in self.airmass]
        self.airmass = [a-a.mean() for a in self.airmass]

        ## Initialise the parent
        ## ---------------------
        super().__init__(times, fluxes, 2*pbs,
                                  use_ldtk=False, constant_k=False, noise=noise,
                                  ldf_path='../data/external_lcs.h5', nthreads=n_threads)
        self.use_ldtk = use_ldtk

        self.fluxes_o = copy(self.fluxes)
        self.fluxes_m = npb*[mean(self.fluxes[:npb], 0)] + npb*[mean(self.fluxes[npb:], 0)] 
        
        self._wrk_lc = (zeros([self.npb,self.fluxes[0].size]),
                        zeros([self.npb,self.fluxes[self.npb].size]))

        ## Setup priors
        ## ------------
        ## Basic parameters
        ## ----------------
        self.priors = [NP(    TC,   1e-2,   'tc'), ##  0  - Transit centre
                       NP(     P,   5e-4,    'p'), ##  1  - Period
                       UP(  3.50,   4.50,  'rho'), ##  2  - Stellar density
                       UP(  0.00,   0.99,    'b')] ##  3  - Impact parameter
        
        ## Area ratio
        ## ----------
        self._sk2 = len(self.priors)
        self.priors.extend([UP(0.165**2, 0.195**2, 'k2_%s'%pb) 
            for pb in self.unique_pbs]) ##  4  - planet-star area ratio
            
        ## Limb darkening
        ## --------------
        self._sq1 = len(self.priors)
        self._sq2 = self._sq1+1
        for ipb in range(self.npb):
            self.priors.extend([UP(0, 1, 'q1_%i'%ipb),      ##  sq1 + 2*ipb -- limb darkening q1
                                UP(0, 1, 'q2_%i'%ipb)])     ##  sq2 + 2*ipb -- limb darkening q2

        # GP hyperparameters
        # ------------------
        self.wn_estimates = array([sqrt(2)*mad(diff(f)) for f in self.fluxes])
        self._sgp = len(self.priors)
        self.priors.extend([UP(-5, -1, 'log10_am_amplitude'),
                            UP(-5,  3, 'log10_am_inv_scale'),
                            UP(-5, -1, 'log10_ra_amplitude'),
                            UP(-5,  3, 'log10_ra_inv_scale')])

        self.ps = PriorSet(self.priors)
        self.set_pv_indices()
        
        # Update the priors using the external data modelling
        # ---------------------------------------------------
        fc = pd.read_hdf(RFILE_EXT, 'vkrn_ldtk/fc')
        self.priors[0] = NP(fc.tc.mean(),   fc.tc.std(),     'tc',  limsigma=25)
        self.priors[1] = NP(fc.p.mean(),    fc.p.std(),       'p',  limsigma=25)
        self.priors[2] = NP(fc.rho.mean(),  fc.rho.std(),   'rho',  limsigma=10)
        self.priors[3] = NP(fc.b.mean(),    fc.b.std(),       'b',  lims=(0,1))
        self.ps = PriorSet(self.priors)

        self.prior_kw = NP(0.1707, 3.2e-4, 'kw', lims=(0.16,0.195))

        self.kernel = ( 1e-3**2 * ExpSquaredKernel(.01, ndim=2, axes=0)
                      + 1e-3**2 * ExpSquaredKernel(.01, ndim=2, axes=1))
        self.gps = [GP(self.kernel, solver=HODLRSolver) for i in range(2)]

        self.gp_inputs = array([self.airmass[0], self.rotang[0]]).T, array([self.airmass[1], self.rotang[1]]).T
        [gp.compute(input, yerr=5e-4) for gp,input in zip(self.gps, self.gp_inputs)]

        # Limb darkening with LDTk
        # ------------------------
        if use_ldtk:
            self.sc = LDPSetCreator([4150,100], [4.6,0.2], [-0.14,0.16], self.filters)
            self.lp = self.sc.create_profiles(2000)
            self.lp.resample_linear_z()
            #self.lp.set_uncertainty_multiplier(2)
            
            
    def set_pv_indices(self, sbl=None, swn=None):
        self.ik2 = [self._sk2+pbid for pbid in self.gpbids]
                        
        self.iq1 = [self._sq1+pbid*2 for pbid in self.gpbids]
        self.iq2 = [self._sq2+pbid*2 for pbid in self.gpbids]
        self.uq1 = np.unique(self.iq1)
        self.uq2 = np.unique(self.iq2)

        if hasattr(self, '_sgp'):
            self.slgp = s_[self._sgp:self._sgp+4]
        #self.iwn = [swn+ilc for ilc in range(self.nlc)]


    def setup_gp(self):
        raise NotImplementedError

    def compute_transit(self, pv):
        _a = as_from_rhop(pv[2], pv[1])
        _i = mt.acos(pv[3] / _a)
        _k = sqrt(pv[self.ik2]).mean()
        kf = pv[self.ik2] / _k ** 2

        a, b = sqrt(pv[self.iq1]), 2. * pv[self.iq2]
        self._wrk_ld[:, 0] = a * b
        self._wrk_ld[:, 1] = a * (1. - b)

        z1 = of.z_circular(self.times[0], pv[0], pv[1], _a, _i, self.nt)
        z2 = of.z_circular(self.times[1], pv[0], pv[1], _a, _i, self.nt)

        f1 = self.tm(z1, _k, self._wrk_ld[:self.npb])
        f2 = self.tm(z2, _k, self._wrk_ld[self.npb:])

        return (kf[self.npb:] * (f1 - 1.) + 1.).T, (kf[:self.npb] * (f2 - 1.) + 1.).T



    def compute_lc_model(self,pv):
        bl1,bl2 = self.compute_baseline(pv)
        tr1,tr2 = self.compute_transit(pv)
        self._wrk_lc[0][:] = bl1*tr1
        self._wrk_lc[1][:] = bl2*tr2
        return self._wrk_lc


    def map_to_gp(self, pv):
        log10_to_ln = 1./log10(e)
        gpp = zeros(4)
        gpp[0] =  2 * pv[0] * log10_to_ln
        gpp[1] =     -pv[1] * log10_to_ln
        gpp[2] =  2 * pv[2] * log10_to_ln
        gpp[3] =     -pv[3] * log10_to_ln
        return gpp


    def compute_baseline(self, pv):
        return 1., 1.


    def lnposterior(self, pv):
        return super().lnposterior(pv)


    def lnlikelihood_wn(self, pv):
        gpps = self.map_to_gp(pv[self.slgp])
        self.kernel[:] = gpps
        [gp.compute(inp, yerr=self.wn_estimates.mean()) for inp,gp in zip(self.gp_inputs, self.gps)]

        fmodel = self.compute_lc_model(pv)

        lnlike = 0.
        for fo, fm in zip(self.fluxes[:self.npb], fmodel[0]):
            lnlike += self.gps[0].lnlikelihood(fo-fm)
        for fo, fm in zip(self.fluxes[self.npb:], fmodel[1]):
            lnlike += self.gps[1].lnlikelihood(fo-fm)
        return lnlike

    
    def lnlikelihood_rn(self, pv):
        lnlike = 0.
        for ilc, (fo, fm) in enumerate(zip(self.fluxes, chain(*self.compute_lc_model(pv)))):
            lnlike += self.gps[ilc].gp.lnlikelihood(fo-fm)
        return lnlike

    
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
