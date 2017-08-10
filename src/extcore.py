from numpy import diff, log
from scipy.optimize import minimize
from tqdm import tqdm
from george.kernels import ConstantKernel, ExpKernel, Matern32Kernel, ExpSquaredKernel

from .core import *
from .lpf import *

class Sampler(object):
    def __init__(self, result_file, run_name, lpf, npop, niter_de, niter_mc):
        self.result_file = result_file
        self.run_name = run_name
        self.lpf = lpf
        self.niter_de = niter_de
        self.niter_mc = niter_mc
        self.de = DiffEvol(lpf, lpf.bounds, npop, maximize=True, F=0.25, C=0.1)
        self.sampler = EnsembleSampler(npop, lpf.ndim, lpf)


    def optimise(self, niter=None):
        niter = niter or self.niter_de
        try:
            for r in tqdm(self.de(niter), total=niter):
                pass
        except KeyboardInterrupt:
            pass
        finally:
            dfde = pd.DataFrame(self.de.population, columns=self.lpf.ps.names)
            dfde.to_hdf(self.result_file,'{:s}/de'.format(self.run_name))


    def sample(self, niter=None):
        niter = niter or self.niter_mc
        def save_chains():
            fc = self.sampler.chain[:,min(2000,self.sampler.iterations//2):self.sampler.iterations:100,:].reshape([-1,self.lpf.ndim])
            dfmc = pd.DataFrame(self.sampler.chain[:,self.sampler.iterations-1,:], columns=self.lpf.ps.names)
            dffc = pd.DataFrame(fc, columns=self.lpf.ps.names)
            dfmc.to_hdf(self.result_file,'{:s}/mc'.format(self.run_name))
            dffc.to_hdf(self.result_file,'{:s}/fc'.format(self.run_name))

        if self.sampler.chain.shape[1] == 0:
            pv0 = self.de.population.copy()
        else:
            pv0 = self.sampler.chain[:,-1,:].copy()

        try:
            for i,c in tqdm(enumerate(self.sampler.sample(pv0, iterations=niter)), total=niter):
                if ((i+1)%200 == 0) or (i==niter-1):
                    save_chains()
        except KeyboardInterrupt:
            pass
        finally:
            save_chains()


    def plot(self, show_systematics=False):
        fc = pd.read_hdf(self.result_file, '{:s}/fc'.format(self.run_name))
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

        fig,axs = pl.subplots(*nfig, figsize=(14,14), sharey=True, sharex=True)
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


class LPFExt(LPF):
    def __init__(self, dataset, use_ldtk=False, constant_k=True, noise='white'):
        """
        Dataset should be either 'triaud2013', 'mancini2014', or 'fukui2014'.
        """
        assert dataset in 'triaud2013 mancini2014 fukui2014'.split()

        self._nbl = 1
        self.dataset = dataset
        f = pd.HDFStore(join(DDATA, 'external_lcs.h5'), 'r')
        df = pd.DataFrame([[k]+k.strip('/lc').split('/') for k in f.keys() if 'lc' in k], 
                          columns='key dataset passband name'.split())
        df = df[df.dataset == dataset]
        self.lcs = [f[n] for n in df.key]
        f.close()
        times = [s.index.values - TZERO for s in self.lcs] 
        fluxes = [s.values for s in self.lcs]   
        super().__init__(times, fluxes, df.passband, constant_k=constant_k,
                                    noise=noise, use_ldtk=use_ldtk)
        
    def setup_gp(self):
        self.hps = pd.read_hdf(self.result_file, 'gphp/{:s}'.format(self.dataset))
        self.gps = [GPTime(t,f) for t,f in zip(self.times, self.fluxes)]
        [gp.compute(pv) for gp,pv in zip(self.gps, self.hps.values[:,:-1])]
        
        
    def lnlikelihood_rn(self, pv):
        flux_m = self.compute_lc_model(pv)
        return sum([gp.gp.lnlikelihood(fo-fm) for gp,fo,fm in zip(self.gps,self.fluxes,flux_m)])


class LPFFukui2014(LPFExt):
    def __init__(self, use_ldtk=False, constant_k=True, noise='red'):
        self.dataset = 'fukui2014'
        f = pd.HDFStore(join(DDATA, 'external_lcs.h5'),'r')
        df = pd.DataFrame([[k]+k.strip('/lc').split('/') for k in f.keys() if 'lc' in k], 
                          columns='key dataset passband name'.split())
        df = df[df.dataset == self.dataset]
        data = [f[n] for n in df.key]
        f.close()
        times  = [d.index.values-TZERO for d in data]
        fluxes = [d.flux.values for d in data]
        self.airmasses = [d.airmass.values for d in data]
        self.dxs = [d.dx.values for d in data]
        self.dys = [d.dy.values for d in data]
        self.gp_inputs = [np.transpose([t,dx,dy,am]) for t,dx,dy,am in zip(times,self.dxs,self.dys,self.airmasses)]

        super().__init__(times, fluxes, df.passband, constant_k=constant_k,
                                    noise=noise, use_ldtk=use_ldtk)


    def setup_gp(self):
        self.hps = pd.read_hdf(self.result_file, 'gphp/fukui2014')
        self.gps = [GPF14(i,f) for i,f in zip(self.gp_inputs, self.fluxes)]
        [gp.compute(pv) for gp,pv in zip(self.gps, self.hps.values[:,:-1])]


class LPFTM(CLPF):
    def __init__(self, use_ldtk=False, constant_k=True, noise='white'):
        self.lpt13 = LPFExt('triaud2013',  use_ldtk=False, constant_k=constant_k, noise=noise)
        self.lpm14 = LPFExt('mancini2014', use_ldtk=False, constant_k=constant_k, noise=noise)
        super().__init__([self.lpt13, self.lpm14], use_ldtk=use_ldtk, constant_k=constant_k, noise=noise)
        self.ndim = self.ps.ndim
        self.bounds = self.ps.bounds

    def __call__(self, pv):
        return self.lnposterior(pv)



class LPFRN(CLPF):
    def __init__(self, use_ldtk=False, constant_k=True):
        self.lpt13 = LPFExt('triaud2013',  use_ldtk=False, constant_k=constant_k, noise='red')
        self.lpm14 = LPFExt('mancini2014', use_ldtk=False, constant_k=constant_k, noise='red')
        self.lpf14 = LPFFukui2014(use_ldtk=False, constant_k=constant_k)

        super().__init__([self.lpt13,self.lpm14,self.lpf14],
                                   use_ldtk=use_ldtk, constant_k=constant_k, noise='red')
        self.ndim = self.ps.ndim
        self.bounds = self.ps.bounds

    def __call__(self, pv):
        return self.lnposterior(pv)


class GPTime(object):
    def __init__(self, inputs, flux):
        self.inputs = array(inputs)
        self.flux = array(flux)
        self.wn_estimate = diff(flux).std() / sqrt(2)
        self.gp = GP(self.kernel, white_noise=log(self.wn_estimate ** 2), fit_white_noise=True)
        self.gp.compute(self.inputs)
        self._minres = None
        self.hp = self.gp.get_parameter_vector()
        self.names = 'ln_wn_var ln_output_var ln_input_scale'.split()

    def compute(self, pv=None):
        if pv is not None:
            self.gp.set_parameter_vector(pv)
        self.gp.compute(self.inputs)

    def predict(self, pv=None, flux=None):
        if pv is not None:
            self.compute(pv)
        flux = flux if flux is not None else self.flux
        return self.gp.predict(flux, self.inputs, return_cov=False)

    def lnposterior(self, pv):
        self.compute(pv)
        return self.gp.lnlikelihood(self.flux)

    def nll(self, pv):
        return self.gp.nll(pv, self.flux)

    def grad_nll(self, pv):
        return self.gp.grad_nll(pv, self.flux)

    def minfun(self, pv):
        return -self.lnposterior(pv)

    def fit(self, pv0=None, disp=False):
        self._minres = minimize(self.nll, self.gp.get_parameter_vector(), jac=self.grad_nll)
        self.hp[:] = self._minres.x.copy()
        return self.hp

    @property
    def kernel(self):
        return (ConstantKernel(log(self.flux.var()), ((-20, -2),))
                * ExpKernel(0.1, metric_bounds=((-20, 5),)))

class OldGPTime(object):
    def __init__(self, inputs, flux):
        self.inputs = inputs
        self.flux = flux.copy()
        self.gp = GP(self.kernel)
        self.pv_mapped = zeros(3)
        self.names = 'log10_amplitude inverse_time_scale log10_white_noise'.split()

        self.priors = [UP(-4.5,  -2, 'log_ta'), ##  0  - log10 time amplitude
                       UP( 5e-2, 5e5,    'its'), ##  1  - inverse time scale
                       UP(-4.0,  -2, 'log_wn')] ##  2  - log10 white noise
        self.ps = PriorSet(self.priors)


    def map(self, pv):
        self.pv_mapped[0] = (10**pv[0])**2
        self.pv_mapped[1] = 1./pv[1]
        self.pv_mapped[2] = 10**pv[2]

    def compute(self, pv):
        self.map(pv)
        self.gp.kernel[:] = np.log(self.pv_mapped[:-1])
        self.gp.compute(self.inputs, self.pv_mapped[-1])

    def predict(self, flux=None):
        flux = flux if flux is not None else self.flux
        return self.gp.predict(flux, self.inputs, return_cov=False)

    def lnposterior(self, pv):
        if any(pv < self.ps.pmins) or any(pv>self.ps.pmaxs):
            return -inf
        self.compute(pv)
        return self.ps.c_log_prior(pv) + self.gp.lnlikelihood(self.flux)

    def minfun(self, pv):
        return -self.lnposterior(pv)

    def fit(self, pv0=None, disp=False):
        pv0 = pv0 if pv0 is not None else [np.log10(self.flux.std()), 1000, np.log10(self.flux.std())]
        return fmin(self.minfun, pv0, xtol=1e-6, disp=disp)

    @property
    def kernel(self):
        return 1e-6*ExpKernel(1)


class GPF14(GPTime):
    def __init__(self, inputs, flux):
        super(GPF14, self).__init__(inputs, flux)
        self.pv_mapped = zeros(8)
        self.names = ('log10_time_amplitude inverse_time_scale log10_xy_amplitude '
                      'inverse_x_scale inverse_y_scale log10_airmass_amplitude '
                      'inverse_airmass_scale white_noise').split()

        self.priors = [UP(  -3.5,     -2,   'ta'),  ##  0  - log10 time amplitude
                       UP(     5,     20,  'its'),  ##  1  - inverse time scale
                       UP(  -3.5,     -2,   'pa'),  ##  2  - log10 xy amplitude
                       UP(     1,     50,  'ixs'),  ##  3  - inverse x scale
                       UP(     1,     50,  'iys'),  ##  4  - inverse y scale
                       UP(  -3.5,     -1,   'aa'),  ##  5  - log10 airmass amplitude
                       UP(     1,     50,  'ias'),  ##  6  - inverse airmass scale
                       UP(    -4,     -2,   'wn')]  ##  7  - log10 white noise
        self.ps = PriorSet(self.priors)


    @property
    def kernel(self):
        return (  1*ExpKernel(0.1,ndim=4,dim=0)
                 +1*ExpSquaredKernel(0.1,ndim=4,dim=1)*ExpSquaredKernel(0.1,ndim=4,dim=2)
                 +1*ExpSquaredKernel(0.1,ndim=4,dim=3))


    def map(self, pv):
        self.pv_mapped[0] = (10**pv[0])**2
        self.pv_mapped[1] = 1./pv[1]
        self.pv_mapped[2] = (10**pv[2])**2
        self.pv_mapped[3] = 1./pv[3]
        self.pv_mapped[4] = 1./pv[4]
        self.pv_mapped[5] = (10**pv[5])**2
        self.pv_mapped[6] = 1./pv[6]
        self.pv_mapped[7] = 10**pv[7]
