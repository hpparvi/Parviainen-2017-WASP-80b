class GPT(GPTime):
    @property
    def kernel(self):
        return 1e-6*ExpSquaredKernel(1)

class LPFC(LPF):
    def __init__(self, use_ldtk=False, constant_k=True, noise='white'):
        pbs = 'g r i z'.split()
        cols = ['relative_'+c for c in pbs]
        
        self.df1 = df1 = pd.merge(pd.read_hdf('../data/aux.h5','night1'),
                                  pd.read_hdf('../results/gtc_light_curves.h5','broadband/night1'),
                                  left_index=True, right_index=True)
        self.df2 = df2 = pd.merge(pd.read_hdf('../data/aux.h5','night2'),
                                  pd.read_hdf('../results/gtc_light_curves.h5','broadband/night2'),
                                  left_index=True, right_index=True)
        
        times = [df1.bjd.values-TZERO, df2.bjd.values-TZERO]
        fluxes = df1[cols].values.mean(1), df2[cols].values.mean(1)
        
        
        ## Mask outliers
        ## -------------
        masks = []
        for i,f in enumerate(fluxes):
            mask = abs(f-MF(f,11)) < 0.003
            if i==0:
                mask &= (times[0] < 855.528) | (times[0] > 855.546)
            masks.append(mask)
        self.masks = masks
        
        times   = [t[m] for t,m in zip(times, masks)]
        fluxes  = [f[m] for f,m in zip(fluxes, masks)]
        self.rotang  = [df1.rotang[masks[0]], df2.rotang[masks[1]]]
        self.elevat  = [df1.elevat[masks[0]],df2.elevat[masks[1]]]
        self.airmass = [df1.airmass[masks[0]],df2.airmass[masks[1]]]
        self.ctimes  = [t-t.mean() for t in times]
        
        super(LPFC,self).__init__(times, fluxes, ['w','w'], 
                                  use_ldtk=False, constant_k=constant_k, noise=noise)
        
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
            self.priors.append( UP(0.16**2, 0.18**2, 'k2')) ##  4  - planet-star area ratio
        else:
            self.priors.extend([UP(0.16**2, 0.18**2, 'k2_%i'%ipb) 
                                for ipb in range(self.npb)]) ##  4  - planet-star area ratio
            
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
            self.priors.append(UP( 0.98, 1.02, 'bla_%i'%ilc)) ##  sbl + ilc -- Baseline constant
            self.priors.append(UP(-0.15, 0.15,  'blb_%i'%ilc)) ##  sbl + ilc -- Linear time trend
            self.priors.append(UP(-0.05, 0.05, 'blz_%i'%ilc)) ##  sbl + ilc -- Linear airmass trend

        ## White noise
        ## -----------
        self._swn = len(self.priors)
        if noise == 'white':
            self.priors.extend([UP(3e-4, 1e-3, 'e_%i'%ilc) 
                                for ilc in range(self.nlc)]) ##  sqn + ilc -- Average white noise
        #else:
        #    self.priors.extend([UP(-3, -2,  'log10_na'),
        #                        UP(1e-4, 1, 'its'),
        #                        UP(-3.5, -3, 'log10_wn')])

        ## Rotator angle-dependent baseline
        ## ---------------------------------
        self._srbl = len(self.priors)
        self.priors.extend([UP( -1e-3,   1e-3, 'bl_ra_l'),        ##  srbl + 0 -- Linear rotator angle trend
                            UP(6.0e-3, 8.5e-3, 'bl_ra_ea'),       ##  srbl + 1 -- Bump amplitude
                            UP(    -4,     -2, 'bl_ra_er0'),      ##  srbl + 2 -- Bump centre 
                            UP(   1.3,    1.5, 'bl_ra_epw'),      ##  srbl + 3 -- Bump shape
                            UP(    35,     55, 'bl_ra_ewd')])     ##  srbl + 4 -- Bump width parameter
            
        self.ps = PriorSet(self.priors)
        self.set_pv_indices()
        
        ## Update the priors using the external data modelling
        ## ---------------------------------------------------
        fc = pd.read_hdf(join('..',result_file), 'ckrn_ldtk/fc')
        self.priors[0] = NP(fc.tc.mean(),   10*fc.tc.std(),  'tc',  limsigma=100)
        self.priors[1] = NP(fc.p.mean(),    10*fc.p.std(),    'p',  limsigma=100)
        self.priors[2] = NP(fc.rho.mean(),  fc.rho.std(),   'rho',  limsigma=100)
        self.priors[3] = NP(fc.b.mean(),    fc.b.std(),       'b',  lims=(0,1))
        
        if constant_k:
            self.priors[4] = NP(fc.k2.mean(),   fc.k2.std(), 'k2',   limsigma=100)
        else:
            for i in self.ik2:
                self.priors[i] = NP(fc.k2.mean(), fc.k2.std(), 'k2_%i'%i,   limsigma=100)
        
        if noise == 'white':
            for i in self.iwn:
                self.priors[i] = UP(1e-4, 1e-3, 'e_%i'%i)
                    
        self.ps = PriorSet(self.priors)
        
        self.setup_gp()
        
        
        ## Limb darkening with LDTk
        ## ------------------------
        if use_ldtk:
            dff = pd.read_hdf('../data/external_lcs.h5', 'transmission')
            transmission = dff['g r i z'.split()].mean(1)
            self.filters = [TabulatedFilter('white', transmission.index.values,  tm=transmission.values)]
            self.sc = LDPSetCreator([4150,100], [4.6,0.2], [-0.14,0.16], self.filters)
            self.lp = self.sc.create_profiles(2000)
            self.lp.set_uncertainty_multiplier(2)
        
        
    def set_pv_indices(self, sbl=None, swn=None):
        if self.constant_k:
            self.ik2 = [self._sk2]
        else:
            self.ik2 = [self._sk2+pbid for pbid in self.gpbids]
                        
        self.iq1 = [self._sq1+pbid*2 for pbid in self.gpbids]
        self.iq2 = [self._sq2+pbid*2 for pbid in self.gpbids]
        
        sbl = sbl if sbl is not None else self._sbl
        self.ibla = [sbl+3*ilc   for ilc in range(self.nlc)]
        self.iblb = [sbl+3*ilc+1 for ilc in range(self.nlc)]
        self.ibld = [sbl+3*ilc+2 for ilc in range(self.nlc)]
        
        if hasattr(self, '_srbl'):
            self.iblr = [self._srbl + i for i in range(5)]
        
        if self.noise == 'white':
            swn = swn if swn is not None else self._swn
            self.iwn = [swn+ilc for ilc in range(self.nlc)]
        #else:
        #    swn = swn if swn is not None else self._swn
        #    self.ina = [swn]
        #    self.its = [swn+1]
        #    self.iwn = [swn+2]
        #    self._slgp = s_[swn:swn+3]

    def rafun(self, rotang, pv):
        l,a,c,p,w = pv[self.iblr]
        return l*rotang + a*exp(-abs(((rotang-c)+30.)%60.-30.)**p/w)
        #return l*rotang + a*exp(-abs(rotang-c)**p/w)
        
    def setup_gp(self):
        self.gps = [GPT(t,f) for t,f in zip(self.rotang, self.fluxes)]
        [gp.compute([-2.8, 0.06, -3.2]) for i,gp in enumerate(self.gps)]
        
    def lnlikelihood_rn(self, pv):
        #[gp.compute(pv[self._slgp]) for i,gp in enumerate(self.gps)]
        flux_m = self.compute_lc_model(pv)
        return sum([gp.gp.lnlikelihood(fo-fm) for gp,fo,fm in zip(self.gps,self.fluxes,flux_m)])
        
    def compute_baseline(self, pv):
        return [pv[ia] + pv[ib]*t + pv[iz]*(z-1.25)+self.rafun(r, pv) for ia,ib,iz,t,r,z in 
                zip(self.ibla,self.iblb,self.ibld,self.ctimes,self.rotang,self.airmass)]
