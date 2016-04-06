from copy import copy

class GPT(GPTime):
    @property
    def kernel(self):
        return 1e-6*ExpSquaredKernel(1)

class LPFC(LPF):
    def __init__(self, use_ldtk=False, constant_k=True, noise='white'):
        pbs = 'g r i z'.split()
        cols = ['relative_'+c for c in pbs]
        
        self.df1 = df1 = pd.merge(pd.read_hdf('../data/aux.h5','night1'),
                                  pd.read_hdf('../results/gtc_light_curves.h5','night1'),
                                  left_index=True, right_index=True)
        self.df2 = df2 = pd.merge(pd.read_hdf('../data/aux.h5','night2'),
                                  pd.read_hdf('../results/gtc_light_curves.h5','night2'),
                                  left_index=True, right_index=True)
        
        times  = 4*[df1.bjd.values-TZERO]+4*[df2.bjd.values-TZERO]
        fluxes = (list(df1[cols].values.T) + list(df2[cols].values.T))
        
        ## Mask outliers
        ## -------------
        self.masks = masks = []
        for j in range(2):
            lims = [0.006, 0.003, 0.005, 0.005]
            mask = ones(fluxes[4*j].size, np.bool)
            for i in range(4):
                f = fluxes[i+4*j]
                mask &= abs(f-MF(f,11)) < lims[i]
            if j==0:
                mask &= (times[0] < 855.528) | (times[0] > 855.546)
            self.masks.append(mask)

        times   = [times[i][masks[i//4]]  for i in range(8)]
        fluxes  = [fluxes[i][masks[i//4]] for i in range(8)]

        fc = pd.read_hdf(join('..',result_file), 'vkrn_ldtk/fc')
        self.otmasks = [abs(fold(t, fc.p.mean(), fc.tc.mean(), 0.5)-0.5) < 0.0134 for t in times]
        self.rotang  = 4*[df1.rotang[masks[0]].values]  + 4*[df2.rotang[masks[1]].values]
        self.elevat  = 4*[df1.elevat[masks[0]].values]  + 4*[df2.elevat[masks[1]].values]
        self.airmass = 4*[df1.airmass[masks[0]].values] + 4*[df2.airmass[masks[1]].values]
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
                                  use_ldtk=use_ldtk, constant_k=constant_k, noise=noise,
                                  ldf_path='../data/external_lcs.h5',)

        self.fluxes_o = copy(self.fluxes)
        self.fluxes_m = 4*[mean(self.fluxes[:4], 0)] + 4*[mean(self.fluxes[4:], 0)] 
        self.fluxes = [f/fm for f,fm in zip(self.fluxes, self.fluxes_m)]
        
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
        if constant_k:
            self.priors.append( UP(0.165**2, 0.175**2, 'k2'))  ##  4  - planet-star area ratio
        else:
            self.priors.extend([UP(0.165**2, 0.175**2, 'k2_%s'%pb) 
                                for pb in self.unique_pbs]) ##  4  - planet-star area ratio
            
        ## Limb darkening
        ## --------------
        self._sq1 = len(self.priors)
        self._sq2 = self._sq1+1
        for ipb in range(self.npb):
            self.priors.extend([UP(0, 1, 'q1_%i'%ipb),      ##  sq1 + 2*ipb -- limb darkening q1
                                UP(0, 1, 'q2_%i'%ipb)])     ##  sq2 + 2*ipb -- limb darkening q2
            
        ## Baseline constant
        ## -----------------
        self._sbl = len(self.priors)
        for ilc in range(self.nlc):
            self.priors.append(UP( 0.98, 1.02, 'bcn_%i'%ilc)) ##  sbl + ilc -- Baseline constant
            self.priors.append(UP(-1e-1, 1e-1, 'be1_%i'%ilc)) ##  sbl + ilc -- Linear elevation trend 1
            #self.priors.append(UP(-1e-3, 1e-3, 'be2_%i'%ilc)) ##  sbl + ilc -- Linear elevation trend 2

        ## White noise
        ## -----------
        self._swn = len(self.priors)
        if noise == 'white':
            self.priors.extend([UP(3e-4, 4e-3, 'e_%i'%ilc) 
                                for ilc in range(self.nlc)]) ##  sqn + ilc -- Average white noise
        #else:
        #    self.priors.extend([UP(-3, -2,  'log10_na'),
        #                        UP(1e-4, 1, 'its'),
        #                        UP(-3.5, -3, 'log10_wn')])

        ## Rotator angle-dependent baseline
        ## ---------------------------------
        self._srbl = len(self.priors)
        for i in range(2):
            self.priors.extend([UP( -9e-4,   9e-4, 'bra_l_%i'%i)])        ##  srbl + 0 -- Linear rotator angle trend
                                #UP(   0.0, 9.5e-3, 'bra_ea_%i'%i),       ##  srbl + 1 -- Bump amplitude
                                #UP(  -3.5,     -2, 'bra_er0_%i'%i),      ##  srbl + 2 -- Bump centre 
                                #UP(   1.3,    1.4, 'bra_epw_%i'%i),      ##  srbl + 3 -- Bump shape
                                #UP(    30,     40, 'bra_ewd_%i'%i)])     ##  srbl + 4 -- Bump width parameter
            
        self.ps = PriorSet(self.priors)
        self.set_pv_indices()
        
        ## Update the priors using the external data modelling
        ## ---------------------------------------------------
        if constant_k:
            fc = pd.read_hdf(join('..',result_file), 'ckrn_ldtk/fc')
        else:
            fc = pd.read_hdf(join('..',result_file), 'vkrn_ldtk/fc')

        self.priors[0] = NP(fc.tc.mean(),   10*fc.tc.std(),  'tc',  limsigma=5)
        self.priors[1] = NP(fc.p.mean(),    10*fc.p.std(),    'p',  limsigma=5)
        self.priors[2] = NP(fc.rho.mean(),  fc.rho.std(),   'rho',  limsigma=5)
        self.priors[3] = NP(fc.b.mean(),    fc.b.std(),       'b',  lims=(0,1))
        
        if constant_k:
            self.priors[4] = NP(fc.k2_w.mean(),   fc.k2_w.std(), 'k2_w',   limsigma=100)
        else:
            for ik2,iq1,iq2,pb in zip(self.ik2,self.iq1,self.iq2,self.unique_pbs):
                k2,q1,q2 = fc['k2_{pb:s} q1_{pb:s} q2_{pb:s}'.format(pb=pb).split()].values.T
                self.priors[ik2] = NP(k2.mean(), k2.std(), 'k2_%s'%pb,   limsigma=100)
                self.priors[iq1] = NP(q1.mean(), q1.std(), 'q1_%s'%pb,   limsigma=100)
                self.priors[iq2] = NP(q2.mean(), q2.std(), 'q2_%s'%pb,   limsigma=100)

        self.ps = PriorSet(self.priors)
        self.setup_gp()
        
        
    def set_pv_indices(self, sbl=None, swn=None):
        if self.constant_k:
            self.ik2 = [self._sk2]
        else:
            self.ik2 = [self._sk2+pbid for pbid in self.gpbids]
                        
        self.iq1 = [self._sq1+pbid*2 for pbid in self.gpbids]
        self.iq2 = [self._sq2+pbid*2 for pbid in self.gpbids]
        
        sbl = sbl if sbl is not None else self._sbl
        self.ibcn = [sbl+2*ilc   for ilc in range(self.nlc)]
        self.ibe1 = [sbl+2*ilc+1 for ilc in range(self.nlc)]
        #self.ibe2 = [sbl+3*ilc+2 for ilc in range(self.nlc)]
        
        if hasattr(self, '_srbl'):
            self.ibr1 = [self._srbl] #[self._srbl + i for i in range(5)]
            self.ibr2 = [self._srbl+1] #[self._srbl + 5 + i for i in range(5)]

        if self.noise == 'white':
            swn = swn if swn is not None else self._swn
            self.iwn = [swn+ilc for ilc in range(self.nlc)]
        #else:
        #    swn = swn if swn is not None else self._swn
        #    self.ina = [swn]
        #    self.its = [swn+1]
        #    self.iwn = [swn+2]
        #    self._slgp = s_[swn:swn+3]

        
    def compute_lc_model(self,pv):
        bl = self.compute_baseline(pv)
        tr = self.compute_transit(pv)
        trm1,trm2 = mean(tr[:4], 0), mean(tr[4:], 0)
        lc = []
        for i in range(0,4):
            lc.append(bl[i]*tr[i]/trm1)
        for i in range(4,8):
            lc.append(bl[i]*tr[i]/trm2)
        return lc

    
    def bl_rotang(self, pv):
        l,a,c,p,w = pv[self.ibr1]
        ra = self.rotang[0]
        bra1 = l*ra + a*exp(-abs(((ra-c)+30.)%60.-30.)**p/w)
        l,a,c,p,w = pv[self.ibr2]
        ra = self.rotang[4]
        bra2 = l*ra + a*exp(-abs(((ra-c)+30.)%60.-30.)**p/w)
        return 4*[bra1] + 4*[bra2]
        

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
        

    def lnlikelihood_rn(self, pv):
        #[gp.compute(pv[self._slgp]) for i,gp in enumerate(self.gps)]
        flux_m = self.compute_lc_model(pv)
        return sum([gp.gp.lnlikelihood(fo-fm) for gp,fo,fm in zip(self.gps,self.fluxes,flux_m)])
        

    def compute_baseline(self, pv):
        #bl_e = self.bl_elevation(pv)
        #bl_r = self.bl_rotang(pv)
        #return [pv[icn] + be + br for icn,be,br in zip(self.ibcn,bl_e,bl_r)]
        return [pv[icn] + pv[ibe]*tc for icn,ibe,tc in zip(self.ibcn,self.ibe1,self.ctimes)]

#        return [pv[ia] + pv[ib]*t + pv[iz]*(z-1.25)+self.rafun(r, pv) for ia,ib,iz,t,r,z in 
#                zip(self.ibla,self.iblb,self.ibld,self.ctimes,self.rotang,self.airmass)]
