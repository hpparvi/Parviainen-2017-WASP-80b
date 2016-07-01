from time import time
from core import *
from w80plots import *

from matplotlib.backends.backend_pdf import PdfPages

def print_nb(str):
    clear_output(wait=True)
    display(HTML(str))

def print_tr(str):
    sys.stdout.write('\r{:s}'.format(str))
    sys.stdout.flush()

update = lambda i,interval: (i+1)%interval == 0

class Sampler(object):
    def __init__(self, result_file, run_name, lpf, npop, notebook=True, **kwargs):
        self.result_file = result_file
        self.run_name = run_name
        self.npop = npop
        self.de_path = '{:s}/de'.format(run_name)
        self.fc_path = '{:s}/fc'.format(run_name)
        self.mc_path = '{:s}/mc'.format(run_name)
        self.lpf = lpf
        self.de = DiffEvol(lpf.lnposterior, lpf.ps.bounds, npop, maximize=True, F=0.25, C=0.10)
        self.sampler = EnsembleSampler(npop, lpf.ps.ndim, lpf.lnposterior)
        self.de_iupdate = kwargs.get('de_iupdate',  10)
        self.de_isave   = kwargs.get('de_isave',   100)
        self.mc_iupdate = kwargs.get('mc_iupdate',  10)
        self.mc_isave   = kwargs.get('mc_isave',   100)
        self.mc_nruns   = kwargs.get('mc_nruns',     1) 
        self.mc_thin    = kwargs.get('mc_thin',    100)
                
        if not notebook:
            self.logger  = logging.getLogger()
            logfile = open('{:s}_{:s}.log'.format(basename(result_file), run_name.replace('/','_')), mode='w')
            fh = logging.StreamHandler(logfile)
            fh.setLevel(logging.DEBUG)
            self.logger.addHandler(fh)
            self.info = self.logger.info
        else:
            self.info = lambda str: None

        if notebook:
            self.disp = print_nb
        else:
            self.disp = self.info

        self.info('Initialised sampler with')
        self.info('  DE update interval %i', self.de_iupdate)
        self.info('  DE save interval %i', self.de_isave)
        self.info('  MCMC update interval %i', self.mc_iupdate)
        self.info('  MCMC save interval %i', self.mc_isave)

        
    def optimise(self, niter, population=None, cont=True):
        if population is not None:
            self.de._population[:] = population
        else:
            self.de._population[:] = self.lpf.fit_baseline(self.de.population)
            if self.lpf.use_ldtk:
                self.de._population[:] = self.lpf.fit_ldc(self.de.population, emul=2.)

        try:
            for i,r in enumerate(self.de(niter)):
                if update(i, self.de_iupdate):
                    self.disp('DE Iteration {:>4d} max lnlike {:8.1f}  ptp {:8.1f}'.format(i+1,-self.de.minimum_value, self.de._fitness.ptp()))
                if update(i, self.de_isave):
                    self.save_de()
        except KeyboardInterrupt:
            self.info('DE iterrupted by the user')
        finally:
            self.save_de()


    def sample(self, niter, population=None, cont=True):
        if population is not None:
            self.info('Continuing MCMC from a saved chain')
            pv0 = population
        else:
            if self.sampler.chain.shape[1] == 0 or not cont:
                self.info('Starting MCMC from the DE population')
                pv0 = self.de.population.copy()
            else:
                self.info('Continuing MCMC from the sampler')
                pv0 = self.sampler.chain[:,-1,:].copy()
            
        try:
            for j in range(self.mc_nruns):
                self.info('Starting MCMC run %i/%i', j+1, self.mc_nruns)
                tprev = time()
                for i,c in enumerate(self.sampler.sample(pv0, iterations=niter, thin=self.mc_thin)):
                    if update(i, self.mc_iupdate):
                        tcur = time()
                        titer = (tcur-tprev)/(self.mc_thin*self.mc_iupdate)
                        nlpf = (self.mc_iupdate * self.mc_thin * self.npop) / (tcur-tprev)
                        tprev = tcur
                        self.disp('MCMC Iteration {:5d}  -- {:6.2f} LPF evaluations / second, {:6.2f} seconds / MC iteration'.format(self.sampler.iterations, nlpf, titer))
                    if update(i, self.mc_isave):
                        self.save_mc()
                self.save_mc()
                if j < self.mc_nruns-1:
                    pv0 = self.sampler.chain[:,-1,:].copy()
                    self.sampler.reset()
        except KeyboardInterrupt:
            self.info('MCMC interrupted by the user')
        finally:
            self.save_mc()

            
    def save_de(self):
        self.info('Saving the DE population')
        np.savez(self.result_file+'_de.npz', population=self.de.population, columns=self.lpf.ps.names)
    
        
    def save_mc(self):
        self.info('Saving the MCMC chains with %i iterations', self.sampler.iterations)
        ns = self.sampler.iterations // self.mc_thin
        np.savez(self.result_file+'_mc.npz', chains=self.sampler.chain[:,:ns,:], columns=self.lpf.ps.names)
    

    def plot_status(self):

        def plot_blp(pars, axs):
            pal = sb.light_palette(c_ob)
            for irun in range(2):
                ps = percentile(fc[:,pars[irun]], [50,0.1,99.9,1,99,16,84], 0)
                pmin = 0.99*fc[:,pars[irun]].min()
                pmax = 1.01*fc[:,pars[irun]].max()
                for ipb,p in enumerate(ps.T):
                    dy = 1./lpf.npb
                    ymin, ymax = ipb*dy+0.1*dy, (ipb+1)*dy-0.1*dy
                    axs[irun].axvline(p[0], ymin=ymin, ymax=ymax, c='k', lw=1)
                    axs[irun].axvspan(*p[1:3], ymin=ymin, ymax=ymax, fc=pal[0], lw=0)
                    axs[irun].axvspan(*p[3:5], ymin=ymin, ymax=ymax, fc=pal[1], lw=0)
                    axs[irun].axvspan(*p[5:7], ymin=ymin, ymax=ymax, fc=pal[3], lw=0)
                    axs[irun].axvspan(*p[1:3], ymin=ymin, ymax=ymax, fill=False, ec='k', lw=1)
            setp(axs, xlim=(pmin, pmax))

        pp = PdfPages('Na_red_target_dw.pdf')

        ## Radius ratio chains
        ncols = 3
        nrows = int(ceil(lpf.npb / float(ncols)))

        fig,axs = pl.subplots(nrows, ncols, figsize=(11,11*(9./16.)), sharex=True, sharey=False)
        x = arange(chain.shape[1])
        for ipb,ax in zip(range(lpf.npb), axs.flat):
            p = percentile(chain[:,:,4+ipb], [50,16,84], 0)
            ax.fill_between(x, *p[1:3], alpha=0.1)
            ax.plot(x, p[0])
        fig.suptitle('{:s} -- Radius ratio chains'.format(lpf.passband), size=15)
        pl.setp(axs, yticks=[], xlim=x[[0,-1]])
        fig.tight_layout()
        fig.subplots_adjust(top=0.94)
        pp.savefig(fig)

        ## Baseline distributions
        fig,axs = pl.subplots(2, 4, figsize=(11,11*(9./16.)))
        plot_blp([lpf.ibcn[:lpf.npb],lpf.ibcn[lpf.npb:]], axs=axs[0,:2])
        plot_blp([lpf.ibal[:lpf.npb],lpf.ibal[lpf.npb:]], axs=axs[0,2:])
        plot_blp([lpf.ibtl[:lpf.npb],lpf.ibtl[lpf.npb:]], axs=axs[1,:2])
        plot_blp([lpf.iwn[:lpf.npb],lpf.iwn[lpf.npb:]],   axs=axs[1,2:])
        pl.setp(axs[0,:2], xlim=(0.941,1.059))
        pl.setp(axs[0,2:], xlim=(-0.05,0.05))
        pl.setp(axs[1,:2], xlim=(-0.05,0.05))
        pl.setp(axs[1,2:], xlim=( 0.0001,0.01))
        pl.setp(axs, yticks=[]);
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.01)
        sb.despine(fig, left=True)
        pp.savefig(fig)

        ## Radius ratios
        fig,ax = pl.subplots(figsize=(11,11*(9./16.)))

        if lpf.passband == 'Na':
            plot_radius_ratios(df, lpf, ax=ax, spc_scale=0.001, spc_loc=0.1718, axk=ax, 
                               xlim=(566,612), ylim=(0.167,0.175), plot_k=False)
        fig.tight_layout()
        pp.savefig(fig)

        pp.close()
