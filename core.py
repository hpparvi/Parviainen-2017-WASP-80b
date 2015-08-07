import pyfits as pf
import pandas as pd
import numpy as  np

from glob import glob
from os.path import join, exists
from numpy.polynomial.chebyshev import Chebyshev

<<<<<<< HEAD
rawdir = '/home/mert3269/soft/Parviainen-WASP-80b-Osiris/data/'
reddir = '/home/mert3269/soft/Parviainen-WASP-80b-Osiris/data/'
=======
rawdir = '/home/bootha/Project/Parviainen-WASP-80b-Osiris/data/gct_transmission_spectra/OB0020'
reddir  = 'data'
>>>>>>> 883a8263199adf8014049c5a0bbd8e932cc58b0d

d_flat = join(rawdir,'flat')
d_bias = join(rawdir,'bias')
d_obj  = join(rawdir,'object')

l_flat = sorted(glob(join(d_flat,'*.fits')))
l_bias = sorted(glob(join(d_bias,'*.fits')))
l_obj  = sorted(glob(join(d_obj, '*.fits')))[2:]

bias_window = np.s_[:,50:1035]

ccd1_window = np.s_[:,250:330]
ccd2_window = np.s_[:,840:890]

period = pper = 3.0678504

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
        self._cw2p = Chebyshev.fit(self.reference_lines, self.fitted_lines, 5, domain=[400,1000])
        
    def pixel_to_wl(self, pixels):
        return self._cp2w(pixels)
    
    def wl_to_pixel(self, wavelengths):
        return self._cw2p(wavelengths)

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
