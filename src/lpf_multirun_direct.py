import sys
from copy import copy
from itertools import chain
from numpy import *

from scipy.signal import medfilt as MF
from scipy.stats import scoreatpercentile as sap
from numpy.random import normal, uniform

from core import *
from lpf import *
from extcore import *

from math import pi

class LPFC(LPF):
    def __init__(self, passband, lctype='target', use_ldtk=False, n_threads=4, mask_ingress=False, noise='white'):
        
        self.df1 = df1 = pd.merge(pd.read_hdf('../data/aux.h5','night1'),
                                  pd.read_hdf('../results/gtc_light_curves.h5','night1'),
                                  left_index=True, right_index=True)
        self.df2 = df2 = pd.merge(pd.read_hdf('../data/aux.h5','night2'),
                                  pd.read_hdf('../results/gtc_light_curves.h5','night2'),
                                  left_index=True, right_index=True)

        assert passband in ['w', 'bb','nb','K','Na']
        assert lctype in ['target', 'relative']
        assert noise in ['white', 'red']
        
        self.passband = passband        
        if passband in ['w', 'bb']:
            cols = ['{:s}_{:s}'.format(lctype, pb) for pb in 'g r i z'.split()]
            self.filters = pb_filters_bb
        elif passband == 'nb':
            cols = [c for c in self.df1.columns if lctype+'_nb' in c]
            self.filters = pb_filters_nb
        elif passband == 'K':
            cols = [c for c in df1.columns if lctype+'_K'  in c]
            self.filters = pb_filters_k
        elif passband == 'Na':
            cols = [c for c in df1.columns if lctype+'_Na'  in c]
            self.filters = pb_filters_na

        if passband == 'w':
            pbs = ['w']
            fluxes = [df1[cols].values.mean(1), df2[cols].values.mean(1)]
            NN = lambda a: a/a.max()
            self.filters = [TabulatedFilter('white', pb_filters_bb[0].wl, NN(array([f.tm for f in pb_filters_bb]).mean(0)))]
        else:
            pbs = [c.split('_')[1] for c in cols]
            fluxes = (list(df1[cols].values.T) + list(df2[cols].values.T))

        npb = len(pbs)

        times  = npb*[df1.bjd.values-TZERO]+npb*[df2.bjd.values-TZERO]
        fluxes = [f/median(f) for f in fluxes]

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
            if mask_ingress:
                mask[:180] = False
            self.masks.append(mask)
            
        times   = [times[i][masks[i//npb]]  for i in range(2*npb)]
        fluxes  = [fluxes[i][masks[i//npb]] for i in range(2*npb)]

        fc = pd.read_hdf(RFILE_EXT, 'vkrn_ldtk/fc')
        self.otmasks = [abs(fold(t, fc.p.mean(), fc.tc.mean(), 0.5)-0.5) < 0.0134 for t in times]
        self.rotang  = npb*[np.deg2rad(df1.rotang[masks[0]].values)]  + npb*[np.deg2rad(df2.rotang[masks[1]].values)]
        self.elevat  = npb*[df1.elevat[masks[0]].values]  + npb*[df2.elevat[masks[1]].values]
        self.airmass = npb*[df1.airmass[masks[0]].values] + npb*[df2.airmass[masks[1]].values]
        self.ctimes  = [t-t.mean() for t in times]
        
        ## Initialise the parent
        ## ---------------------
        super(LPFC,self).__init__(times, fluxes, 2*pbs,
                                  use_ldtk=False, constant_k=False, noise=noise,
                                  ldf_path='../data/external_lcs.h5', nthreads=n_threads)
        self.use_ldtk = use_ldtk

        self.fluxes_o = copy(self.fluxes)
        self.fluxes_m = npb*[mean(self.fluxes[:npb], 0)] + npb*[mean(self.fluxes[npb:], 0)] 
        #self.fluxes = [f/fm for f,fm in zip(self.fluxes, self.fluxes_m)]
        
        self._wrk_lc = (zeros([self.npb,self.fluxes[0].size]),
                        zeros([self.npb,self.fluxes[self.npb].size]))

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
        self.priors.extend([UP(0.165**2, 0.195**2, 'k2_%s'%pb) 
            for pb in self.unique_pbs]) ##  4  - planet-star area ratio
            
        ## Limb darkening
        ## --------------
        self._sq1 = len(self.priors)
        self._sq2 = self._sq1+1
        for ipb in range(self.npb):
            self.priors.extend([UP(0, 1, 'q1_%i'%ipb),      ##  sq1 + 2*ipb -- limb darkening q1
                                UP(0, 1, 'q2_%i'%ipb)])     ##  sq2 + 2*ipb -- limb darkening q2

        ## Rotator angle baseline
        ## ----------------------
        self._srp = len(self.priors)
        for irun in range(2):
            self.priors.append(UP( -pi,   pi, 'brp_%i'%irun)) ##  srp + irun -- Rotator angle phase
            self.priors.append(UP(0.25, 4.00, 'brf_%i'%irun)) ##  srp + irun -- Rotator angle f

        ## Baseline
        ## --------
        self._sbl = len(self.priors)
        for ilc in range(self.nlc):
            self.priors.append(UP( 0.0,  2.0, 'bcn_%i'%ilc)) ##  sbl + ilc -- Baseline constant
            self.priors.append(UP(-1.0,  1.0, 'btl_%i'%ilc)) ##  sbl + ilc -- Linear time trend
            self.priors.append(UP(-1.0,  1.0, 'bal_%i'%ilc)) ##  sbl + ilc -- Linear airmass trend
            self.priors.append(UP( 0.0, 0.50, 'bra_%i'%ilc)) ##  sbl + ilc -- Rotaror angle amplitude

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
            self.sc = LDPSetCreator([4150,100], [4.6,0.2], [-0.14,0.16], self.filters)
            self.lp = self.sc.create_profiles(2000)
            #self.lp.resample_linear_z()
            #self.lp.set_uncertainty_multiplier(2)
            
            
    def set_pv_indices(self, sbl=None, swn=None):
        self.ik2 = [self._sk2+pbid for pbid in self.gpbids]
                        
        self.iq1 = [self._sq1+pbid*2 for pbid in self.gpbids]
        self.iq2 = [self._sq2+pbid*2 for pbid in self.gpbids]
        self.uq1 = np.unique(self.iq1)
        self.uq2 = np.unique(self.iq2)

        if hasattr(self, '_srp'):
            self.ibrp = [self._srp+2*irun   for irun in range(2)]
            self.ibrf = [self._srp+2*irun+1 for irun in range(2)]
        
        sbl = sbl if sbl is not None else self._sbl
        self.ibcn = [sbl+4*ilc   for ilc in range(self.nlc)]
        self.ibtl = [sbl+4*ilc+1 for ilc in range(self.nlc)]
        self.ibal = [sbl+4*ilc+2 for ilc in range(self.nlc)]
        self.ibra = [sbl+4*ilc+3 for ilc in range(self.nlc)]

        swn = swn if swn is not None else self._swn
        self.iwn = [swn+ilc for ilc in range(self.nlc)]


    def setup_gp(self):
        if self.passband in 'bb nb Na K'.split():
            hps = pd.read_hdf(join(DRESULT,'gtc_gp_hyperparameters.h5'), self.passband)
        else:
            raise NotImplementedError()

        self.gps = []
        for ilc in range(self.nlc):
            self.gps.append(GPTime(self.times[ilc], self.fluxes[ilc]))
            if ilc < self.npb:
                self.gps[-1].compute(hps.night1.values[ilc])
            else:
                self.gps[-1].compute(hps.night2.values[ilc-self.npb])

                
    def lnposterior(self, pv):
        #_k = sqrt(pv[self.ik2]).mean()
        return super(LPFC,self).lnposterior(pv) #+ self.prior_kw.log(_k)                

        
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

        return (kf[self.npb:]*(f1-1.)+1.).T, (kf[:self.npb]*(f2-1.)+1.).T
        

    def compute_lc_model(self,pv):
        bl1,bl2 = self.compute_baseline(pv)
        tr1,tr2 = self.compute_transit(pv)
        self._wrk_lc[0][:] = bl1*tr1
        self._wrk_lc[1][:] = bl2*tr2
        return self._wrk_lc

            
    def compute_baseline(self, pv):
        ra_term_1 = np.cos(pv[self.ibrp[:1]][:,newaxis] + pv[self.ibrf[:1]][:,newaxis]*self.rotang[0])
        ra_term_2 = np.cos(pv[self.ibrp[1:]][:,newaxis] + pv[self.ibrf[1:]][:,newaxis]*self.rotang[self.npb])
        ra_term_1 = pv[self.ibra[:self.npb]][:,newaxis] * (ra_term_1 - ra_term_1.mean()) / ra_term_1.ptp()
        ra_term_2 = pv[self.ibra[self.npb:]][:,newaxis] * (ra_term_2 - ra_term_2.mean()) / ra_term_2.ptp()
        #ra_term_1 = pv[self.ibra[:self.npb]][:,newaxis] * np.cos(pv[self.ibrp[:1]][:,newaxis] + pv[self.ibrf[:1]][:,newaxis]*self.rotang[0]))

        bl1 = ( pv[self.ibcn[:self.npb]][:,newaxis]
              + pv[self.ibtl[:self.npb]][:,newaxis] * self.ctimes[0]
              + pv[self.ibal[:self.npb]][:,newaxis] * self.airmass[0]
              + ra_term_1 )
        bl2 = ( pv[self.ibcn[self.npb:]][:,newaxis]
              + pv[self.ibtl[self.npb:]][:,newaxis] * self.ctimes[self.npb]
              + pv[self.ibal[self.npb:]][:,newaxis] * self.airmass[self.npb]
              + ra_term_2 )
        return bl1, bl2


    def lnlikelihood_wn(self, pv):
        lnlike = 0.
        for iwn, fo, fm in zip(self.iwn, self.fluxes, chain(*self.compute_lc_model(pv))):
            lnlike += ll_normal_es(fo, fm, pv[iwn])
        return lnlike

    
    def lnlikelihood_rn(self, pv):
        lnlike = 0.
        for ilc, (fo, fm) in enumerate(zip(self.fluxes, chain(*self.compute_lc_model(pv)))):
            lnlike += self.gps[ilc].gp.lnlikelihood(fo-fm)
        return lnlike

    
    def fit_baseline(self, pvpop):
        def baseline(pv, ilc):
            ra_term = np.cos(pv[4]+pv[5]*self.rotang[ilc])
            ra_term = pv[3]*(ra_term-ra_term.mean()) / ra_term.ptp()
            return pv[0] + pv[1]*self.ctimes[ilc] + pv[2]*self.airmass[ilc] + ra_term

        def minfun(pv):
            if not ((-pi < pv[4] < pi) and (0.3 < pv[5] < 3.8)) and (pv[3] > 0):
                return inf
            return ((self.fluxes[i][m]-baseline(pv,i)[m])**2).sum()
         
        # pvt = pvpop.copy()
        # rp, rf = zeros(self.nlc), zeros(self.nlc)
        # for i in range(self.nlc):
        #     m = ~self.otmasks[i]
        #     pv0 = fmin(minfun, [1, 0, 0, 0.01, 0, 1], disp=False, ftol=1e-9, xtol=1e-9)
        #     pvt[:,self.ibcn[i]] = normal(pv0[0], 0.001,            size=pvt.shape[0])
        #     pvt[:,self.ibtl[i]] = normal(pv0[1], 0.01*abs(pv0[1]), size=pvt.shape[0])
        #     pvt[:,self.ibal[i]] = normal(pv0[2], 0.01*abs(pv0[2]), size=pvt.shape[0])
        #     pvt[:,self.ibra[i]] = uniform(0, 0.02, size=pvt.shape[0])
        #     #np.abs(normal(pv0[3], 0.01*abs(pv0[3]), size=pvt.shape[0]))
        #     #rp[i] = pv0[4]
        #     #ref[i] = pv0[5]
        # pvt[:, self.ibrp[0]] = normal(0, 0.1, size=pvt.shape[0])
        # pvt[:, self.ibrp[1]] = normal(0, 0.1, size=pvt.shape[0])
        # pvt[:, self.ibrf[0]] = normal(1, 0.1, size=pvt.shape[0])
        # pvt[:, self.ibrf[1]] = normal(1, 0.1, size=pvt.shape[0])
        # #pvt[:,self.ibrp[0]] = np.abs(normal(rp.mean(), rp.std(), size=pvt.shape[0]))
        # #pvt[:,self.ibrp[1]] = np.abs(normal(rp.mean(), rp.std(), size=pvt.shape[0]))
        # #pvt[:,self.ibrf[0]] = np.clip(normal(rf.mean(), rf.std(), size=pvt.shape[0]), 0.26, 3.8)
        # #pvt[:,self.ibrf[1]] = np.clip(normal(rf.mean(), rf.std(), size=pvt.shape[0]), 0.26, 3.8)
        # return pvt

        pvt = pvpop.copy()
        pvb = zeros([self.nlc, 6])
        for i in range(self.nlc):
            m = ~self.otmasks[i]
            pvb[i,:] = fmin(minfun, [1, 0, 0, 0.01, 0, 1], disp=False, ftol=1e-9, xtol=1e-9)

        s1, s2 = s_[:self.npb], s_[self.npb:]
        pvm1, pvs1 = pvb[s1,:].mean(0), pvb[s1,:].std(0)
        pvm2, pvs2 = pvb[s2,:].mean(0), pvb[s2,:].std(0)

        if self.passband == 'w':
            pvs1 = 0.001*np.abs(pvb[s1,:].mean(0))
            pvs2 = 0.001*np.abs(pvb[s2,:].mean(0))

        for s,pvm,pvs in zip([s1,s2],[pvm1,pvm2],[pvs1,pvs2]):
            pvt[:, self.ibcn[s]] = normal(pvm[0], 0.001,  size=[pvt.shape[0], self.npb])
            pvt[:, self.ibtl[s]] = normal(pvm[1], pvs[1], size=[pvt.shape[0], self.npb])
            pvt[:, self.ibal[s]] = normal(pvm[2], pvs[2], size=[pvt.shape[0], self.npb])
            pvt[:, self.ibra[s]] = np.clip(normal(pvm[3], pvs[3], size=[pvt.shape[0], self.npb]), 0, 1)
        pvt[:, self.ibrp[0]] = normal(pvm1[4], pvs1[4], size=pvt.shape[0])
        pvt[:, self.ibrp[1]] = normal(pvm2[4], pvs2[4], size=pvt.shape[0])
        pvt[:, self.ibrf[0]] = normal(pvm1[5], pvs1[5], size=pvt.shape[0])
        pvt[:, self.ibrf[1]] = normal(pvm2[5], pvs2[5], size=pvt.shape[0])
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