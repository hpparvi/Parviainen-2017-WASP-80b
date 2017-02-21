import sys
from copy import copy
from itertools import chain
from numpy import *

from scipy.signal import medfilt as MF
from scipy.stats import scoreatpercentile as sap
from numpy.random import normal

from .core import *
from .lpf import *
from .extcore import *

class LPFSD(LPF):
    def __init__(self, passband, lctype='target', use_ldtk=False, n_threads=1, night=2, mask_ingress=False, noise='white', pipeline='gc'):
        assert passband in ['w', 'bb', 'nb', 'K', 'Na']
        assert lctype in ['target', 'relative']
        assert noise in ['white', 'red']
        assert pipeline in ['hp', 'gc']
        assert night in [1,2]

        self.night = night
        self.df = df = pd.merge(pd.read_hdf(join(DDATA, 'aux.h5'),'night%i'%night),
                                pd.read_hdf(join(DRESULT, 'gtc_light_curves.h5'),'night%i'%night),
                                left_index=True, right_index=True)

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
        fluxes = [f/median(f) for f in fluxes]
        
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
        self.rotang  = df.rotang[self.mask].values
        self.elevat  = df.elevat[self.mask].values
        self.airmass = df.airmass[self.mask].values
        self.ctimes  = [t-t.mean() for t in times]
        
        # Initialise the parent
        # ---------------------
        super().__init__(times, fluxes, pbs,
                                  use_ldtk=False, constant_k=False, noise='white',
                                  ldf_path='../data/external_lcs.h5', nthreads=n_threads)
        self.use_ldtk = use_ldtk

        self.fluxes_o = copy(self.fluxes)
        self.fluxes_m = npb*[mean(self.fluxes[:npb], 0)]
        #self.fluxes = [f/fm for f,fm in zip(self.fluxes, self.fluxes_m)]
        
        self._wrk_lc = zeros([self.npb,self.fluxes[0].size])

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

        # Rotator angle baseline
        # ----------------------
        self._srp = len(self.priors)
        self.priors.append(UP(-0.5 * pi, pi, 'brp'))  ##  srp -- Rotator angle phase

        ## Baseline
        ## --------
        self._sbl = len(self.priors)
        for ilc in range(self.nlc):
            self.priors.append(UP( 0.0, 2.0, 'bcn_%i'%ilc))  #  sbl + ilc -- Baseline constant
            self.priors.append(UP(-1.0, 1.0, 'btl_%i'%ilc))  #  sbl + ilc -- Linear time trend
            self.priors.append(UP(-1.0, 1.0, 'bal_%i'%ilc))  #  sbl + ilc -- Linear airmass trend
            self.priors.append(UP( 0.0, 0.5, 'bra_%i'%ilc))  #  sbl + ilc -- Rotaror angle amplitude

        ## White noise
        ## -----------
        self._swn = len(self.priors)
        self.priors.extend([UP(3e-4, 4e-3, 'e_%i'%ilc) 
                            for ilc in range(self.nlc)]) ##  sqn + ilc -- Average white noise

        self.ps = PriorSet(self.priors)
        self.set_pv_indices()
        
        ## Update the priors using the external data modelling
        ## ---------------------------------------------------
        fc = pd.read_hdf(RFILE_EXT, 'vkrn_ldtk/fc')
        self.priors[0] = NP(fc.tc.mean(),   20*fc.tc.std(),  'tc',  limsigma=15)
        self.priors[1] = NP(fc.p.mean(),    20*fc.p.std(),    'p',  limsigma=15)
        self.priors[2] = NP(fc.rho.mean(),  fc.rho.std(),   'rho',  limsigma=5)
        self.priors[3] = NP(fc.b.mean(),    fc.b.std(),       'b',  lims=(0,1))

        if self.noise == 'red':
            for ilc,i in enumerate(self.iwn):
                self.priors[i] = NP(7e-4, 2e-4, 'e_%i'%ilc, lims=(0,1))

        self.ps = PriorSet(self.priors)
        if self.noise == 'red':
            self.setup_gp()

        self.prior_kw = NP(0.1707, 3.2e-4, 'kw', lims=(0.16,0.18))
        
        ## Limb darkening with LDTk
        ## ------------------------
        if use_ldtk:
            self.filters = pb_filters_nb
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

        if hasattr(self, '_srp'):
            self.ibrp = [self._srp]

        sbl = sbl if sbl is not None else self._sbl
        self.ibcn = [sbl + 4 * ilc     for ilc in range(self.nlc)]
        self.ibtl = [sbl + 4 * ilc + 1 for ilc in range(self.nlc)]
        self.ibal = [sbl + 4 * ilc + 2 for ilc in range(self.nlc)]
        self.ibra = [sbl + 4 * ilc + 3 for ilc in range(self.nlc)]

        swn = swn if swn is not None else self._swn
        self.iwn = [swn+ilc for ilc in range(self.nlc)]


    def setup_gp(self):
        pass


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
        self._wrk_lc[:] = bl*tr
        return self._wrk_lc if not copy else self._wrk_lc.copy()


    def compute_baseline(self, pv):
        ra_term_1 = np.cos(pv[self.ibrp][:,newaxis] + self.rotang)
        ra_term_1 = pv[self.ibra][:,newaxis] * (ra_term_1 - ra_term_1.mean()) / ra_term_1.ptp()

        bl = ( pv[self.ibcn][:,newaxis]
             + pv[self.ibtl][:,newaxis] * self.ctimes[0]
             + pv[self.ibal][:,newaxis] * self.airmass
             + ra_term_1 )

        bl = pv[self.ibcn][:,newaxis] + pv[self.ibtl][:,newaxis] * self.ctimes[0] + pv[self.ibal][:,newaxis] * self.airmass
        return bl


    def lnlikelihood_wn(self, pv):
        fluxes_m = self.compute_lc_model(pv)
        return sum([ll_normal_es(fo, fm, wn) for fo,fm,wn in zip(self.fluxes, fluxes_m, pv[self.iwn])])


    def fit_baseline(self, pvpop):
        def baseline(pv):
            ra_term = np.cos(pv[4] + self.rotang)
            ra_term = pv[3] * (ra_term - ra_term.mean()) / ra_term.ptp()
            return pv[0] + pv[1] * self.ctimes[0] + pv[2] * self.airmass + ra_term

        def minfun(pv):
            if not (0. <= pv[4] <= pi) and (pv[3] > 0.):
                return inf
            return ((self.fluxes[i][m] - baseline(pv)[m])**2).sum()

        pvt = pvpop.copy()
        pvb = zeros([self.nlc, 5])
        for i in range(self.nlc):
            m = ~self.otmasks[i]
            pvb[i, :] = fmin(minfun, [1, 0, 0, 0.01, 0.1], disp=False, ftol=1e-9, xtol=1e-9)
        pvm, pvs = pvb.mean(0), pvb.std(0)

        if self.passband == 'w':
            pvs = 0.001 * np.abs(pvb.mean(0))

        pvt[:, self.ibcn] = normal(pvm[0], 0.001, size=[pvt.shape[0], self.npb])
        pvt[:, self.ibtl] = normal(pvm[1], pvs[1], size=[pvt.shape[0], self.npb])
        pvt[:, self.ibal] = normal(pvm[2], pvs[2], size=[pvt.shape[0], self.npb])
        pvt[:, self.ibra] = np.clip(normal(pvm[3], pvs[3], size=[pvt.shape[0], self.npb]), 0, 0.5)
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
