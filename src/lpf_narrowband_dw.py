import sys
from copy import copy
from numpy import *

from scipy.signal import medfilt as MF
from scipy.stats import scoreatpercentile as sap

from core import *
from extcore import *

class GPT(GPTime):
    @property
    def kernel(self):
        return 1e-6*ExpSquaredKernel(1)

class LPFC(LPF):
    def __init__(self, use_ldtk=False, n_threads=4):
        self.df1 = df1 = pd.merge(pd.read_hdf('../data/aux.h5','night1'),
                                  pd.read_hdf('../results/gtc_light_curves.h5','night1'),
                                  left_index=True, right_index=True)
        self.df2 = df2 = pd.merge(pd.read_hdf('../data/aux.h5','night2'),
                                  pd.read_hdf('../results/gtc_light_curves.h5','night2'),
                                  left_index=True, right_index=True)

        cols = [c for c in self.df1.columns if 'relative_nb' in c]
        pbs = [c[-4:] for c in cols]
        npb = len(pbs)

        times  = npb*[df1.bjd.values-TZERO]+npb*[df2.bjd.values-TZERO]
        fluxes = (list(df1[cols].values.T) + list(df2[cols].values.T))
        
        ## Mask outliers
        ## -------------
        self.masks = masks = []
        for j in range(2):
            lim = 0.006
            mask = ones(fluxes[npb*j].size, np.bool)
            for i in range(npb):
                f = fluxes[i+npb*j]
                mask &= abs(f-MF(f,11)) < lim
            if j==0:
                mask &= (times[0] < 855.528) | (times[0] > 855.546)
            self.masks.append(mask)

        times   = [times[i][masks[i//npb]]  for i in range(2*npb)]
        fluxes  = [fluxes[i][masks[i//npb]] for i in range(2*npb)]

        fc = pd.read_hdf(RFILE_EXT, 'vkrn_ldtk/fc')
        self.otmasks = [abs(fold(t, fc.p.mean(), fc.tc.mean(), 0.5)-0.5) < 0.0134 for t in times]
        self.rotang  = npb*[df1.rotang[masks[0]].values]  + npb*[df2.rotang[masks[1]].values]
        self.elevat  = npb*[df1.elevat[masks[0]].values]  + npb*[df2.elevat[masks[1]].values]
        self.airmass = npb*[df1.airmass[masks[0]].values] + npb*[df2.airmass[masks[1]].values]
        self.ctimes  = [t-t.mean() for t in times]
        
        ## Stuff for the rotator angle dependent baseline
        self._wrk_ra =  [zeros_like(r) for r in self.rotang]

        ## Stuff for the elevation dependent baseline
        ## ------------------------------------------
        self._wrk_el = [zeros_like(e) for e in self.elevat]
        self.celevat = [e-59.25 for e in self.elevat]
        self.emaska  = [t < t[argmax(e)] for e,t in zip(self.elevat,times)]
        self.emaskb  = [~m for m in self.emaska]
        
        ## Initialise the parent
        ## ---------------------
        super(LPFC,self).__init__(times, fluxes, 2*pbs,
                                  use_ldtk=False, constant_k=False, noise='white',
                                  ldf_path='../data/external_lcs.h5', nthreads=n_threads)

        self.fluxes_o = copy(self.fluxes)
        self.fluxes_m = npb*[mean(self.fluxes[:npb], 0)] + npb*[mean(self.fluxes[npb:], 0)] 
        self.fluxes = [f/fm for f,fm in zip(self.fluxes, self.fluxes_m)]
        
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
            self.priors.append(UP( 0.98, 1.02, 'bcn_%i'%ilc)) ##  sbl + ilc -- Baseline constant
            self.priors.append(UP(-1e-1, 1e-1, 'btl_%i'%ilc)) ##  sbl + ilc -- Linear time trend
            self.priors.append(UP(-1e-3, 1e-3, 'bel_%i'%ilc)) ##  sbl + ilc -- Linear elevation trend

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

        ## Limb darkening with LDTk
        ## ------------------------
        if use_ldtk:
            self.filters = pb_filters_nb
            self.sc = LDPSetCreator([4150,100], [4.6,0.2], [-0.14,0.16], self.filters)
            self.lp = self.sc.create_profiles(2000)
            self.lp.set_uncertainty_multiplier(2)
        
        
    def set_pv_indices(self, sbl=None, swn=None):
        self.ik2 = [self._sk2+pbid for pbid in self.gpbids]
                        
        self.iq1 = [self._sq1+pbid*2 for pbid in self.gpbids]
        self.iq2 = [self._sq2+pbid*2 for pbid in self.gpbids]
        
        sbl = sbl if sbl is not None else self._sbl
        self.ibcn = [sbl+3*ilc   for ilc in range(self.nlc)]
        self.ibtl = [sbl+3*ilc+1 for ilc in range(self.nlc)]
        self.ibel = [sbl+3*ilc+2 for ilc in range(self.nlc)]
        
        swn = swn if swn is not None else self._swn
        self.iwn = [swn+ilc for ilc in range(self.nlc)]
        

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
        self._wrk_lc[0][:] = bl1*tr1/tr1.mean(0)
        self._wrk_lc[1][:] = bl2*tr2/tr2.mean(0)
        return self._wrk_lc

        
    def bl_elevation(self, pv):
        a = 0.
        for i, (ib1,ib2) in enumerate(zip(self.ibe1,self.ibe2)):
            b1, b2 = pv[[ib1,ib2]]
            ma,mb = self.emaska[i], self.emaskb[i]
            self._wrk_el[i][ma]  = a + b1*self.celevat[i][ma]
            self._wrk_el[i][mb]  = a + b2*self.celevat[i][mb]
        return self._wrk_el


    def setup_gp(self):
        self.gps = [GPT(t,f) for t,f in zip(self.rotang, self.fluxes)]
        [gp.compute([-2.8, 0.06, -3.2]) for i,gp in enumerate(self.gps)]
        

    def compute_baseline(self, pv):
        #bl_e = self.bl_elevation(pv)
        #bl_r = self.bl_rotang(pv)
        #return [pv[icn] + be + br for icn,be,br in zip(self.ibcn,bl_e,bl_r)]
        return [pv[icn] + pv[ibe]*tc for icn,ibe,tc in zip(self.ibcn,self.ibe1,self.ctimes)]

#        return [pv[ia] + pv[ib]*t + pv[iz]*(z-1.25)+self.rafun(r, pv) for ia,ib,iz,t,r,z in 
#                zip(self.ibla,self.iblb,self.ibld,self.ctimes,self.rotang,self.airmass)]


    def compute_baseline(self, pv):
        bl1 = pv[self.ibcn[:self.npb]][:,newaxis] + pv[self.ibtl[:self.npb]][:,newaxis] * self.ctimes[0] + pv[self.ibel[:self.npb]][:,newaxis] * self.celevat[0]
        bl2 = pv[self.ibcn[self.npb:]][:,newaxis] + pv[self.ibtl[self.npb:]][:,newaxis] * self.ctimes[self.npb] + pv[self.ibel[self.npb:]][:,newaxis] * self.celevat[self.npb]
        return bl1, bl2


    def lnlikelihood_wn(self, pv):
        fluxes_m = self.compute_lc_model(pv)
        return (sum([ll_normal_es(fo, fm, wn) for fo,fm,wn in zip(self.fluxes[:self.npb], fluxes_m[0], pv[self.iwn[:self.npb]])]) +
                sum([ll_normal_es(fo, fm, wn) for fo,fm,wn in zip(self.fluxes[self.npb:], fluxes_m[1], pv[self.iwn[self.npb:]])]) )

    def lnlikelihood_rn(self, pv):
        #[gp.compute(pv[self._slgp]) for i,gp in enumerate(self.gps)]
        flux_m = self.compute_lc_model(pv)
        return sum([gp.gp.lnlikelihood(fo-fm) for gp,fo,fm in zip(self.gps,self.fluxes,flux_m)])
    