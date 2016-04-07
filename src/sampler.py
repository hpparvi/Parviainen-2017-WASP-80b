from core import *

def print_nb(str):
    clear_output(wait=True)
    display(HTML(str))

def print_tr(str):
    sys.stdout.write('\r{:s}'.format(str))
    sys.stdout.flush()
    
class Sampler(object):
    def __init__(self, result_file, run_name, lpf, npop, notebook=True, **kwargs):
        self.result_file = result_file
        self.de_path = '{:s}/de'.format(run_name)
        self.mc_path = '{:s}/mc'.format(run_name)
        self.run_name = run_name
        self.lpf = lpf
        self.de = DiffEvol(lpf.lnposterior, lpf.ps.bounds, npop, maximize=True, F=0.25, C=0.1)
        self.sampler = EnsembleSampler(npop, lpf.ps.ndim, lpf.lnposterior)
        self.de_iupdate = kwargs.get('de_iupdate', 20)
        self.mc_iupdate = kwargs.get('mc_iupdate', 20)
        self.mc_isave = kwargs.get('mc_isave',    200)

        if notebook:
            self.disp = print_nb
        else:
            self.disp = print_tr

        if not notebook:
            self.logger  = logging.getLogger()
            logfile = open('{:s}.log'.format(run_name.replace('/','_')), mode='w')
            fh = logging.StreamHandler(logfile)
            #fh.setFormatter(logging.Formatter('%(levelname)s %(name)s: %(message)s'))
            fh.setLevel(logging.DEBUG)
            self.logger.addHandler(fh)
            np.seterrcall(lambda e,f: self.logger.info(e))
            np.seterr(invalid='ignore')
            self.info = self.logger.info
        else:
            self.info = lambda str: None

        
    def optimise(self, niter, cont=True):
        if cont:
            try:
                pv0 = pd.read_hdf(self.result_file, self.de_path).values
                self.info('Continuing from a previous DE result') 
            except (KeyError,IOError):
                self.info('Starting DE from scratch') 
        
        try:
            for i,r in enumerate(self.de(niter)):
                if ((i+1)%self.de_iupdate == 0) or (i==niter-1):
                    self.disp('DE Iteration {:>4d} max lnlike {:8.1f}  ptp {:8.1f}'.format(i+1,-self.de.minimum_value, self.de._fitness.ptp()))
        except KeyboardInterrupt:
            pass
        finally:
            dfde = pd.DataFrame(self.de.population, columns=self.lpf.ps.names)
            dfde.to_hdf(self.result_file, self.de_path)


    def sample(self, niter, thin, population=None, cont=True):
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
            for i,c in enumerate(self.sampler.sample(pv0, iterations=niter, thin=thin)):
                if ((i+1)%self.mc_iupdate == 0) or (i==niter-1):
                    self.disp('MCMC Iteration {:d}'.format(self.sampler.iterations))
                if ((i+1)%self.mc_isave == 0) or (i==niter-1):
                    self.save_chains()
        except KeyboardInterrupt:
            pass
        finally:
            self.save_chains()

    def save_chains(self): 
        fc = self.sampler.chain[:,:self.sampler.iterations,:].reshape([-1,self.lpf.ps.ndim])
        dffc = pd.DataFrame(fc, columns=self.lpf.ps.names)
        dffc.to_hdf(self.result_file, self.mc_path)
                    

