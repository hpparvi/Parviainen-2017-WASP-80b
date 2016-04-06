from de import DiffEvol

result_file = '../results/osiris.h5'

class Sampler(object):
    def __init__(self, run_name, lpf, npop, niter_de, niter_mc):
        self.run_name = run_name
        self.lpf = lpf
        self.niter_de = niter_de
        self.niter_mc = niter_mc
        self.de = DiffEvol(lpf, lpf.bounds, npop, maximize=True, F=0.25, C=0.1)
        self.sampler = EnsembleSampler(npop, lpf.ndim, lpf)


    def optimise(self, niter=None):
        niter = niter or self.niter_de
        try:
            for i,r in enumerate(self.de(niter)):
                if ((i+1)%20 == 0) or (i==niter-1):
                    clear_output(wait=True)
                    display(HTML('DE Iteration {:4d} max lnlike {:7.1f}'.format(i+1,-self.de.minimum_value)))
        except KeyboardInterrupt:
            pass
        finally:
            dfde = pd.DataFrame(self.de.population, columns=self.lpf.ps.names)
            dfde.to_hdf(result_file,'{:s}/de'.format(self.run_name))


    def sample(self, niter=None):
        niter = niter or self.niter_mc
        def save_chains():
            fc = self.sampler.chain[:,min(2000,self.sampler.iterations//2):self.sampler.iterations:100,:].reshape([-1,self.lpf.ndim])
            dfmc = pd.DataFrame(self.sampler.chain[:,self.sampler.iterations-1,:], columns=self.lpf.ps.names)
            dffc = pd.DataFrame(fc, columns=self.lpf.ps.names)
            dfmc.to_hdf(result_file,'{:s}/mc'.format(self.run_name))
            dffc.to_hdf(result_file,'{:s}/fc'.format(self.run_name))

        if self.sampler.chain.shape[1] == 0:
            pv0 = self.de.population.copy()
        else:
            pv0 = self.sampler.chain[:,-1,:].copy()

        try:
            for i,c in enumerate(self.sampler.sample(pv0, iterations=niter)):
                if ((i+1)%10 == 0) or (i==niter-1):
                    clear_output(wait=True)
                    display(HTML('MCMC Iteration {:d}'.format(self.sampler.iterations)))     
                if ((i+1)%200 == 0) or (i==niter-1):
                    save_chains()
        except KeyboardInterrupt:
            pass
        finally:
            save_chains()


    def plot(self, show_systematics=False):
        fc = pd.read_hdf(result_file, '{:s}/fc'.format(self.run_name))
        mp = np.median(fc, 0)
        phases = [fold(t, P, TC, 0.5) - 0.5 for t in self.lpf.times]

        if self.lpf.noise == 'red':
            fluxes_m, residuals, gpmeans = [], [], []

            for l in self.lpf.lpfs:
                fms = l.compute_lc_model(mp)
                res = [fo-fm for fo,fm in zip(l.fluxes, fms)]
                for i,gp in enumerate(l.gps):
                    gp.flux = res[i]
                    gpmeans.append(gp.predict(None))
                fluxes_m.extend(fms)
                residuals.extend(res)
        else:
            fluxes_m  = self.lpf.compute_transit(mp)
            residuals = [fo-fm for fo,fm in zip(self.lpf.fluxes, fluxes_m)]
            gpmeans   = zeros(self.lpf.nlc) 

        nfig = (4,3) if self.lpf.nlc < 28 else (7,4)

        fig,axs = pl.subplots(nfig, figsize=(14,14), sharey=True, sharex=True)
        for iax,ilc in enumerate(self.lpf.lcorder):
            a = axs.flat[iax]
            if show_systematics:
                a.plot(phases[ilc],self.lpf.fluxes[ilc],'.', alpha=0.5)
                a.plot(phases[ilc],fluxes_m[ilc]+gpmeans[ilc],'k')
            else:
                a.plot(phases[ilc],self.lpf.fluxes[ilc]-gpmeans[ilc],'.', alpha=0.5)
                a.plot(phases[ilc],fluxes_m[ilc],'k')
            a.plot(phases[ilc],self.lpf.fluxes[ilc]-fluxes_m[ilc]-gpmeans[ilc]+0.95,'.', alpha=0.5)
            a.text(0.5, 0.95, self.lpf.passbands[ilc], ha='center', va='top', size=12, transform=a.transAxes)
        pl.setp(axs, ylim=(0.94,1.01), xlim=(-0.035, 0.035))
        fig.tight_layout()
        axs.flat[-1].set_visible(False)
        return axs
