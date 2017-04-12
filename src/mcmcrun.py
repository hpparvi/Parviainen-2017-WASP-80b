import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

from astropy.table import Table
from numpy import ones, sqrt, percentile, median

class MCMCRun(object):
    def __init__(self, filename, burn=0, clean=False):
        self.filename = filename
        self.burn = burn
        self.clean = clean
        self.tb = Table.read(filename)
        self.npop = self.tb.meta['NPOP']
        self.ndim = self.tb.meta['NDIM']
        self.__create_arrays(burn, clean)


    def __create_arrays(self, burn=None, clean=None):
        self.chain = self.tb.to_pandas().values.reshape([self.npop, -1, self.ndim])
        mask = ((self.chain[:, :, 0].std(1) > 1e-8)
                if (clean or self.clean) else ones(self.npop, np.bool))
        self.fc = self.chain[mask, (burn or self.burn):, :].reshape([-1, self.ndim])
        self.df = pd.DataFrame(self.fc, columns=self.tb.colnames)

        k2cols = [c for c in self.df.columns if 'k2' in c]
        for k2col in k2cols:
            self.df[k2col.replace('k2', 'k')] = sqrt(self.df[k2col])

        self.kcols = [c for c in self.df.columns if 'k_' in c]
        self.gpcols = [c for c in self.df.columns if 'log10' in c]

        self.pes = percentile(self.df[self.kcols], [50, 0.5, 99.5, 16, 84], 0)
        self.pv = median(self.fc, 0)


    def plot_chain(self, pid=0, ax=None):
        if ax is None:
            fig, ax = pl.subplots(1, 1)
        else:
            fig = None
        ax.plot((self.chain[:, :, pid]).T, 'k', alpha=0.1);
        ax.plot(median((self.chain[:, :, pid]), 0), 'w', lw=4)
        ax.plot(median((self.chain[:, :, pid]), 0), 'k', lw=2)
        ax.axhline(median((self.chain[:, :, pid])), c='w')
        pl.setp(ax, xlim=(0, self.chain.shape[1]))
        if fig:
            fig.tight_layout()
        return ax


    def plot_dist(self, pid=0, ax=None, hargs={}):
        if ax is None:
            fig, ax = pl.subplots(1, 1)
        else:
            fig = None
        ax.hist((self.fc[:, pid]), alpha=0.1, **hargs);
        if fig:
            fig.tight_layout()
        return ax
