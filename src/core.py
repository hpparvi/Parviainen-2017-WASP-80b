from __future__ import division
import warnings
import os
import sys
import math as m
import logging

import pyfits as pf
import pandas as pd
import numpy as  np
import seaborn as sb
import matplotlib.pyplot as pl

from glob import glob
from os.path import join, exists, basename, abspath, relpath
from numpy.polynomial.chebyshev import Chebyshev

from matplotlib.dates import datestr2num
from matplotlib import rc
from IPython.display import clear_output, HTML, display

from astropy.time import Time
from scipy.signal import medfilt as mf
from scipy.ndimage import median_filter as MF
from scipy.ndimage import gaussian_filter1d as gf
from scipy.ndimage import binary_erosion

from numpy import pi, array, exp, abs, sum, zeros_like, arange, concatenate, argsort, s_
from scipy.constants import k, G, proton_mass
from scipy.optimize import fmin_powell, fmin

N = lambda a: a/np.median(a)

try:
    from pyfc import psf_g1d, logl_g1d
except ImportError:
    print "Couldn't import PyFC"
    
try:
    from ldtk import LDPSetCreator, BoxcarFilter, TabulatedFilter
except ImportError:
    print "Couldn't import ldTK"
    
try:
    from pyde.de import DiffEvol
except ImportError:
    print "Couldn't import PyDE"

try:
    from exotk.constants import rjup, mjup, rsun, msun
    from exotk.utils.misc import fold
except ImportError:
    print "Couldn't import ExoTK"
    print "Please install ExoTK from GitHub"
    print "    git clone git@github.com:hpparvi/PyExoTK.git"
    print "    cd PyExoTK"
    print "    python setup.py install --user "
    print ""

try:
    from emcee import EnsembleSampler
except ImportError:
    print "Couldn't import emcee, please install it..."
            
try:
    from george import GP, HODLRSolver
    from george.kernels import ExpKernel, ExpSquaredKernel
    with_george = True
except ImportError:
    print "Error, couldn't import george. Anything depending on GPs won't work"
    print "Please install george from github"
    print "    git clone https://github.com/dfm/george.git"
    print "    cd george"
    print "    python setup.py install --user "
    print ""
    with_george = False

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(name)s: %(message)s')
    
## Stellar and planetary parameters
## --------------------------------
## From http://exoplanets.org/detail/WASP-80_b
## 
## Note: the stellar density estimate is missing
MSTAR = 0.580*msun # [m]
RSTAR = 0.571*rsun # [m]
TSTAR = 4150       # [K]
TEQ   = 825        # [K]   from http://arxiv.org/pdf/1312.4982.pdf
MPLANET = 0.562*mjup
LOGGS = 4.60      
LOGGP = 3.181 

TZERO = 2456000
TC    = 125.417523 # Zero epoch
P     =   3.067861 # Orbital period

## Directories and files
## ---------------------
W80ROOT = os.path.dirname(os.path.abspath(join(__file__,'..')))
W80DATA = join(os.getenv('W80DATA'))
DATADIRS = [join(W80DATA,'OB0019'), join(W80DATA,'OB0020')]

DFLAT = [join(rawdir,'flat')   for rawdir in DATADIRS]
DBIAS = [join(rawdir,'bias')   for rawdir in DATADIRS]
DOBJ  = [join(rawdir,'object') for rawdir in DATADIRS]
DARC  = [join(rawdir,'arc')    for rawdir in DATADIRS]

l_flat = [sorted(glob(join(df,'*.fits'))) for df in DFLAT]
l_bias = [sorted(glob(join(db,'*.fits'))) for db in DBIAS]
l_arc  = [sorted(glob(join(da,'*.fits'))) for da in DARC]
l_obj  = [sorted(glob(join(do, '*.fits')))[2:] for do in DOBJ]

DRESULT = join(W80ROOT, 'results')
RFILE_EXT = abspath(join(DRESULT,'external.h5'))
RFILE_GTC = abspath(join(DRESULT,'osiris.h5'))

f_bias_dn = join(DRESULT, 'bias_denoised.npz')
f_flats   = join(DRESULT, 'masterflats.npz')

file_ids = [pd.Series([n.split('/')[-1].split('-OS')[0] for n in l_obj[i]], name='file_id') for i in [0,1]]


## CCD slices
## ----------
bias_window = np.s_[:,50:1035]
ccd1_window = np.s_[:,250:330]
ccd2_window = np.s_[:,840:890]
ccd1_w_sky  = np.s_[:,370:450]
ccd2_w_sky  = np.s_[:,770:810]

nccdkeys = 'n1ccd1 n1ccd2 n2ccd1 n2ccd2'.split()


## Matplotlib configuration
## ------------------------
AAOCW, AAPGW = 3.4645669, 7.0866142
rc('figure', figsize=(14,5))
rc(['axes', 'ytick', 'xtick'], labelsize=8)
rc('font', size=6)
rc_paper = {"lines.linewidth": 1,
            'ytick.labelsize': 6.5,
            'xtick.labelsize': 6.5,
            'axes.labelsize': 6.5,
            'figure.figsize':(AAOCW,0.65*AAOCW)}
rc_notebook = {'figure.figsize':(14,5)}
sb.set_context('notebook', rc=rc_notebook)
sb.set_style('white')
    
    
## Color definitions
## -----------------
c_ob = "#002147" # Oxford blue
c_bo = "#CC5500" # Burnt orange
cp = sb.color_palette([c_ob,c_bo], n_colors=2)+sb.color_palette('deep',n_colors=4)
sb.set_palette(cp)


## Potassium and Kalium resonance line centers [nm]
## ------------------------------------------------
wlc_k  = array([766.5,769.9])
wlc_na = array([589.0,589.6])

class GeneralGaussian(object):
    def __init__(self, name, c, s, p):
        self.name = name
        self.c = c
        self.s = s
        self.p = p
        
    def __call__(self, x):
        return np.exp(-0.5*np.abs((x-self.c)/self.s)**(2*self.p))
    
class WhiteFilter(object):
    def __init__(self, name, filters):
        self.name = name
        self.filters = filters
        
    def __call__(self, x):
        return np.sum([f(x) for f in self.filters], 0)

## Passband definitions
## --------------------
try:
    dff = pd.read_hdf('../data/external_lcs.h5', 'transmission')
    pb_filters_bb = []
    for pb in 'g r i z'.split():
        pb_filters_bb.append(TabulatedFilter(pb, dff.index.values, tm=dff[pb].values))
except IOError:
    pass
        
pb_filters_nb = [GeneralGaussian('nb{:02d}'.format(i), 530+20*i, 10, 15) for i in range(20)]
pb_filters_nb[11] = GeneralGaussian('nb11', 746.95, 7.25, 15)
pb_filters_k  = [GeneralGaussian('K{:02d}'.format(i),  768.2+4*(i-5), 2, 15) for i in range(11)]
pb_filters_k.pop(2)
pb_filters_na = [GeneralGaussian('Na{:02d}'.format(i), 589.4+4*(i-5), 2, 15) for i in range(11)]

pb_centers_bb = [np.average(f.wl, weights=f(f.wl)) for f in pb_filters_bb]
pb_centers_nb = [f.c for f in pb_filters_nb]
pb_centers_k  = [f.c for f in pb_filters_k]
pb_centers_na = [f.c for f in pb_filters_na]

fs = [f(f.wl) for f in pb_filters_bb]
fs = [np.where(f>0.01, f, np.nan) for f in fs]
tt = [500]+[pb_filters_bb[0].wl[np.nanargmin(abs(fs[i+1]-fs[i]))] for i in range(3)]+[900]
pb_bounds_bb = [(tt[i], tt[i+1]) for i in range(4)]


c_passbands = 'w g r i z J H K'.split() + ['nb%02i'%i for i in range(21)] + ['K%02i'%i for i in range(7)] +  ['Na%02i'%i for i in range(7)]

def ccd_slice(run, width=50):
    df = pd.read_hdf('data/aux.h5',('jul16' if run==0 else 'aug25'))
    centers = array(df[['center_1','center_2']].median().values)
    ymins, ymaxs = centers.round().astype(np.int)-width//2, centers.round().astype(np.int)+width//2
    return [s_[:,ymins[i]:ymaxs[i]] for i in range(2)]

class CalibrationSpectrum(object):
    def __init__(self, name, spectrum, lines, initial_guess=None):
        self.name = name
        self.spectrum = spectrum
        self.lines = lines
        self.initial_guess = array(initial_guess)
        self.nlines = len(lines)
        self._model = zeros_like(self.spectrum)
        self.pixel  = arange(spectrum.size)
        self.wl     = zeros_like(self.pixel)
        self.solution = WavelengthSolution()
        self._fit_result = None
        
        self.pixel_to_wl = self.solution.pixel_to_wl
        self.wl_to_pixel = self.solution.wl_to_pixel

        
    def model(self, pv):
        self._model.fill(0.)
        for i in xrange(self.nlines):
            self._model += psf_g1d(pv[1+i*2], pv[2+i*2], pv[0], self._model.size)
        return self._model
    
    def plot(self):
        fig,ax = pl.subplots(1,self.nlines,figsize=(13,2), sharey=True)
        cns = self.fitted_centers
        for i in range(self.nlines):
            sl = s_[cns[i]-10: cns[i]+10]
            ax.flat[i].plot(self.pixel[sl], self.spectrum[sl])
            ax.flat[i].plot(self.pixel[sl], self.model(self._fit_result)[sl])
            pl.setp(ax.flat[i], xlim=self.pixel[sl][[0,-1]])
        fig.tight_layout()
      
    
    def chi_sqr(self, pv):
        if not(5 < pv[0] < self.spectrum.size - 6):
            return np.inf
        t = np.floor((pv[0] + array([-5,5]))).astype(np.int)
        sl = s_[t[0]:t[1]]
        model = psf_g1d(pv[0]-sl.start, pv[1], fwhm=pv[2], npx=10)
        return ((self.spectrum[sl]-model)**2).sum()

            
    def fit(self, offset=0, minargs={}, disp_min=False):
        self.initial_guess[1::2] += offset
        pvs = [fmin_powell(self.chi_sqr, self.initial_guess[[1+2*i,2+2*i,0]], disp=disp_min) for i in range(self.nlines)]
        #pvs = [fmin(self.chi_sqr, self.initial_guess[[1+2*i,2+2*i,0]], disp=disp_min) for i in range(self.nlines)]
        pva = array(pvs)
        self._fit_result = concatenate([[np.median(pva[:,-1])],pva[:,:2].ravel()])
        self.solution.fit(self.fitted_centers, self.lines) 

            
    @property
    def fitted_centers(self):
        return self._fit_result[1::2]
    
    
class WLFitter(object):
    def __init__(self, spectra):
        self.spectra = spectra
        self.lines = concatenate([s.lines for s in self.spectra])
        self.solution = WavelengthSolution()
        
        self.pixel_to_wl = self.solution.pixel_to_wl
        self.wl_to_pixel = self.solution.wl_to_pixel
    
    def fit(self, offset=0):
        [s.fit(offset=offset, disp_min=False) for s in self.spectra]
        lines_fit = self.fitted_centers
        lines_ref = self.lines
        sids = argsort(lines_ref)
        self.solution.fit(lines_fit[sids], lines_ref[sids])
        
    @property
    def fitted_centers(self):
        return concatenate([s.fitted_centers for s in self.spectra])


class WavelengthSolution(object):
    def __init__(self):
        self.fitted_lines = None
        self.reference_lines = None
        self._cp2w = None
        self._cw2p = None

    def fit(self, fitted_lines, reference_lines):
        self.fitted_lines = fitted_lines
        self.reference_lines = reference_lines
        self._cp2w = Chebyshev.fit(self.fitted_lines, self.reference_lines, 5, domain=[0,2051])
        self._cw2p = Chebyshev.fit(self.reference_lines, self.fitted_lines, 5, domain=[500,950])
        
    def pixel_to_wl(self, pixels):
        return self._cp2w(pixels)
    
    def wl_to_pixel(self, wavelengths):
        return self._cw2p(wavelengths)


def H(T,g,mu=None):
    """Atmospheric scale height [m]"""
    mu = mu or 2.3*proton_mass
    return T*k/(g*mu)



