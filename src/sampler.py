from time import time
from core import *

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
        self.de = DiffEvol(lpf.lnposterior, lpf.ps.bounds, npop, maximize=True, F=0.25, C=0.1)
        self.sampler = EnsembleSampler(npop, lpf.ps.ndim, lpf.lnposterior)
        self.de_iupdate = kwargs.get('de_iupdate',  10)
        self.de_isave   = kwargs.get('de_isave',   100)
        self.mc_iupdate = kwargs.get('mc_iupdate',  10)
        self.mc_isave   = kwargs.get('mc_isave',   100)
        self.mc_nruns   = kwargs.get('mc_nruns',     1) 
        self.mc_thin    = kwargs.get('mc_thin',    100)
                
        if not notebook:
            self.logger  = logging.getLogger()
            logfile = open('{:s}_{:s}.log'.format(basename(result_file)[:-3], run_name.replace('/','_')), mode='w')
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

        
    def optimise(self, niter, cont=True):
        if cont:
            try:
                pv0 = pd.read_hdf(self.result_file, self.de_path).values
                self.info('Continuing from a previous DE result') 
            except (KeyError,IOError):
                self.info('Starting DE from scratch') 
        
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
        df = pd.DataFrame(self.de.population, columns=self.lpf.ps.names)
        df.to_hdf(self.result_file, self.de_path)        

        
    def save_mc(self):
        self.info('Saving the MCMC chains with %i iterations', self.sampler.iterations)
        mc = self.sampler.chain[:,self.sampler.iterations//self.mc_thin-2,:]
        fc = self.sampler.chain[:,:self.sampler.iterations//self.mc_thin,:].reshape([-1,self.lpf.ps.ndim])
        dfmc = pd.DataFrame(mc, columns=self.lpf.ps.names)
        dffc = pd.DataFrame(fc, columns=self.lpf.ps.names)
        dfmc.to_hdf(self.result_file, self.mc_path)
        dffc.to_hdf(self.result_file, self.fc_path)
                    
