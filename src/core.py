import pyfits as pf
import pandas as pd
import numpy as  np

from glob import glob
<<<<<<< HEAD:src/core.py
from os.path import join, exists, basename
from numpy.polynomial.chebyshev import Chebyshev

import sys
import math as m
import seaborn as sb
import pyfits as pf

from matplotlib.dates import datestr2num
from matplotlib import rc
from numpy import pi, array, exp, abs, sum
from scipy.constants import k, G, proton_mass

from exotk.constants import rjup, mjup, rsun, msun


rawdir = '/home/mert3269/soft/Parviainen-WASP-80b-Osiris/data/'
reddir = '/home/mert3269/soft/Parviainen-WASP-80b-Osiris/data/'
dir_results = '/home/mert3269/soft/Parviainen-WASP-80b-Osiris/results/'


d_flat = join(rawdir,'flat')
d_bias = join(rawdir,'bias')
d_obj  = join(rawdir,'object')
=======
from os import getenv
from os.path import join, exists
from numpy.polynomial.chebyshev import Chebyshev

d_raw = getenv('W80_DATA_PATH')
d_res = 'results'

d_flat = join(d_raw,'flat')
d_bias = join(d_raw,'bias')
d_obj  = join(d_raw,'object')
>>>>>>> dc167223fa662ac813f53c97b6bb69808e4604c7:core.py

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
    
## Matplotlib configuration
AAOCW, AAPGW = 3.4645669, 7.0866142
rc('figure', figsize=(13,5))
rc(['axes', 'ytick', 'xtick'], labelsize=8)
rc('font', size=6)

rc_paper = {"lines.linewidth": 1,
            'ytick.labelsize': 6.5,
            'xtick.labelsize': 6.5,
            'axes.labelsize': 6.5,
            'figure.figsize':(AAOCW,0.65*AAOCW)}

rc_notebook = {'figure.figsize':(13,5)}

sb.set_context('notebook', rc=rc_notebook)
sb.set_style('white')
    
    
## Color definitions
##
c_ob = "#002147" # Oxford blue
c_bo = "#CC5500" # Burnt orange
cp = sb.color_palette([c_ob,c_bo], n_colors=2)+sb.color_palette(name='deep',n_colors=4)
sb.set_palette(cp)

## Potassium and Kalium resonance line centers [nm]
##
wlc_k  = array([766.5,769.9])
wlc_na = array([589.4])

## Narrow-band filters
##
pb_centers    = 540. + np.arange(16)*25
pb_filters_nb = [GeneralGaussian('', c, 12, 20) for c in pb_centers]
pb_filter_bb  = WhiteFilter('white', pb_filters_nb)


