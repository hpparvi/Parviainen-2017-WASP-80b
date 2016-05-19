import sys
from copy import copy
from numpy import *

from scipy.signal import medfilt as MF
from scipy.stats import scoreatpercentile as sap
from numpy.random import normal

from core import *
from extcore import *

class LPFC(LPF):
    def __init__(self, use_ldtk=False, n_threads=4, night=2, test=False):
        self.night = night
        self.df = df = pd.merge(pd.read_hdf('../data/aux.h5','night%i'%night),
                                  pd.read_hdf('../results/gtc_light_curves.h5','night%i'%night),
                                  left_index=True, right_index=True)

        cols = [c for c in df.columns if 'relative_nb' in c]
        pbs = [c[-4:] for c in cols]
        npb = len(pbs)

        cref = [c for c in df.columns if 'comparison_nb' in c]
        ctrg = [c for c in df.columns if 'target_nb' in c]
        
        times  = npb*[df.bjd.values-TZERO]
        fref = df[cref].values.T
        ftrg = df[ctrg].values.T
        #fluxes = map(N, list(ftrg / fref.mean(0)))
        fluxes = map(lambda a: a/median(a[:25]), list(ftrg))
        
        ## Mask outliers
        ## -------------
        lim = 0.006
        self.mask = ones(fluxes[0].size, np.bool)
        for i in range(npb):
            f = fluxes[i]
            self.mask &= abs(f-MF(f,11)) < lim
            #if self.night == 1:
            #    self.mask &= (times[0] < 855.528) | (times[0] > 855.546)

        times   = [times[i][self.mask]  for i in range(npb)]
        fluxes  = [fluxes[i][self.mask] for i in range(npb)]

        fc = pd.read_hdf(RFILE_EXT, 'vkrn_ldtk/fc')
        self.otmasks = [abs(fold(t, fc.p.mean(), fc.tc.mean(), 0.5)-0.5) < 0.0134 for t in times]
        self.rotang  = df.rotang[self.mask].values
        self.elevat  = df.elevat[self.mask].values
        self.airmass = df.airmass[self.mask].values
        self.ctimes  = [t-t.mean() for t in times]
        
        ## Stuff for the rotator angle dependent baseline
        self._wrk_ra =  zeros_like(self.rotang)

        ## Stuff for the elevation dependent baseline
        ## ------------------------------------------
        self._wrk_el = zeros_like(self.elevat)
        self.celevat = self.elevat-59.25
        self.emaska  = times < t[argmax(self.elevat)]
        self.emaskb  = ~self.emaska
        
        ## Initialise the parent
        ## ---------------------
        super(LPFC,self).__init__(times, fluxes, pbs,
                                  use_ldtk=False, constant_k=False, noise='white',
                                  ldf_path='../data/external_lcs.h5', nthreads=n_threads)
        self.use_ldtk = use_ldtk

        self.fluxes_o = copy(self.fluxes)
        self.fluxes_m = npb*[mean(self.fluxes[:npb], 0)]
        self.fluxes = [f/fm for f,fm in zip(self.fluxes, self.fluxes_m)]
        
        self._wrk_lc = zeros([self.npb,self.fluxes[0].size])

        ## Setup priors
        ## ------------
        ## Basic parameters
        ## ----------------
        self.priors = [NP(    TC,   5e-3,   'tc'), ##  0  - Transit centre
                       NP(     P,   3e-4,    'p'), ##  1  - Period
                       UP(  3.50,   4.50,  'rho'), ##  2  - Stellar density
                       UP(  0.00,   0.99,    'b')] ##  3  - Impact parameter
        
        ## Area ratio
        ## ----------
        self._sk2 = len(self.priors)
        self.priors.extend([UP(0.165**2, 0.175**2, 'k2_%s'%pb) 
            for pb in self.unique_pbs]) ##  4  - planet-star area ratio
            
        ## Limb darkening
        ## --------------
        self._sq1 = len(self.priors)
        self._sq2 = self._sq1+1
        for ipb in range(self.npb):
            self.priors.extend([UP(0, 1, 'q1_%i'%ipb),      ##  sq1 + 2*ipb -- limb darkening q1
                                UP(0, 1, 'q2_%i'%ipb)])     ##  sq2 + 2*ipb -- limb darkening q2
            
        ## Baseline
        ## --------
        self._sbl = len(self.priors)
        for ilc in range(self.nlc):
            self.priors.append(UP( 0.0, 2.0, 'bcn_%i'%ilc)) ##  sbl + ilc -- Baseline constant
            self.priors.append(UP(-1.0, 1.0, 'btl_%i'%ilc)) ##  sbl + ilc -- Linear time trend
            self.priors.append(UP(-1.0, 1.0, 'bal_%i'%ilc)) ##  sbl + ilc -- Linear airmass trend

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
        self.priors[0] = NP(fc.tc.mean(),   10*fc.tc.std(),  'tc',  limsigma=5)
        self.priors[1] = NP(fc.p.mean(),    10*fc.p.std(),    'p',  limsigma=5)
        self.priors[2] = NP(fc.rho.mean(),  fc.rho.std(),   'rho',  limsigma=5)
        self.priors[3] = NP(fc.b.mean(),    fc.b.std(),       'b',  lims=(0,1))
        
        self.ps = PriorSet(self.priors)
        self.setup_gp()

        self.prior_kw = NP(0.1707, 3.2e-4, 'kw', lims=(0.16,0.18))
        
        ## Limb darkening with LDTk
        ## ------------------------
        if use_ldtk:
            self.filters = pb_filters_nb
            self.sc = LDPSetCreator([4150,100], [4.6,0.2], [-0.14,0.16], self.filters)
            self.lp = self.sc.create_profiles(2000)
            #self.lp.set_uncertainty_multiplier(2)
            
        if test:
            self.test_pv = pvt = np.load('test_pv.npz')['pv']
            pvt[self.ik2] = 0.17**2
            pvt[array(self.ik2)[[ 2, 8]]] = 0.172**2
            pvt[array(self.ik2)[[-2,-8]]] = 0.169**2
            fbl = self.compute_bl(pvt)
            ftr = self.compute_transit(pvt)

            
    def set_pv_indices(self, sbl=None, swn=None):
        self.ik2 = [self._sk2+pbid for pbid in self.gpbids]
                        
        self.iq1 = [self._sq1+pbid*2 for pbid in self.gpbids]
        self.iq2 = [self._sq2+pbid*2 for pbid in self.gpbids]
        self.uq1 = np.unique(self.iq1)
        self.uq2 = np.unique(self.iq2)

        sbl = sbl if sbl is not None else self._sbl
        self.ibcn = [sbl+3*ilc   for ilc in range(self.nlc)]
        self.ibtl = [sbl+3*ilc+1 for ilc in range(self.nlc)]
        self.ibal = [sbl+3*ilc+2 for ilc in range(self.nlc)]
        
        swn = swn if swn is not None else self._swn
        self.iwn = [swn+ilc for ilc in range(self.nlc)]


    def setup_gp(self):
        pass


    def lnposterior(self, pv):
        _k = sqrt(pv[self.ik2]).mean()
        return super(LPFC,self).lnposterior(pv) + self.prior_kw.log(_k)                

        
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
        self._wrk_lc[:] = bl*tr/tr.mean(0)
        return self._wrk_lc if not copy else self._wrk_lc.copy()


    def compute_baseline(self, pv):
        bl = pv[self.ibcn][:,newaxis] + pv[self.ibtl][:,newaxis] * self.ctimes[0] + pv[self.ibal][:,newaxis] * self.airmass
        return bl


    def lnlikelihood_wn(self, pv):
        fluxes_m = self.compute_lc_model(pv)
        return sum([ll_normal_es(fo, fm, wn) for fo,fm,wn in zip(self.fluxes, fluxes_m, pv[self.iwn])])


    def fit_baseline(self, pvpop):
        def baseline(pv, ilc):
            return pv[0] + pv[1]*self.ctimes[0] + pv[2]*self.airmass #+ pv[3]*self.airmass**2

        pvt = pvpop.copy()
        for i in range(self.nlc):
            m = ~self.otmasks[i]
            pv0 = fmin(lambda pv: ((self.fluxes[i][m]-baseline(pv,i)[m])**2).sum(), 
                       [1,0,0], disp=False, ftol=1e-9, xtol=1e-9)
            pvt[:,self.ibcn[i]] = normal(pv0[0], 0.001,            size=pvt.shape[0])
            pvt[:,self.ibtl[i]] = normal(pv0[1], 0.01*abs(pv0[1]), size=pvt.shape[0])
            pvt[:,self.ibal[i]] = normal(pv0[2], 0.01*abs(pv0[2]), size=pvt.shape[0])
        return pvt


    def fit_ldc(self, pvpop, emul=1.):
        pvt = pvpop.copy()
        uv, uve = self.lp.coeffs_qd()
        us = array([normal(um, emul*ue, size=pvt.shape[0]) for um,ue in zip(uv[:,0],uve[:,0])]).T
        vs = array([normal(vm, emul*ve, size=pvt.shape[0]) for vm,ve in zip(uv[:,1],uve[:,1])]).T
        q1s, q2s = map_uv_to_qq(us, vs)
        pvt[:, self.iq1] = q1s
        pvt[:, self.iq2] = q2s
        return pvt
