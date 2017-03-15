import numpy as np
import matplotlib.pyplot as pl
import seaborn as sb
from seaborn.palettes import get_color_cycle
from numpy import asarray, zeros
from .core import *

def wavelength_to_rgb(wavelength, gamma=0.8):

    '''This converts a given wavelength of light to an 
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    '''

    wavelength = asarray(wavelength)
    rgb = zeros((wavelength.size, 3))
    
    
    mask = (wavelength >= 380) & (wavelength <= 440)
    attenuation = 0.3 + 0.7 * (wavelength[mask] - 380.) / (440. - 380.)
    rgb[mask, 0] = ((-(wavelength[mask] - 440) / (440 - 380)) * attenuation) ** gamma
    rgb[mask, 1] = 0.0
    rgb[mask, 2] = (1.0 * attenuation) ** gamma
    
    mask = (wavelength >= 440) & (wavelength <= 490)
    rgb[mask, 0] = 0.0
    rgb[mask, 1] = ((wavelength[mask] - 440.) / (490. - 440.)) ** gamma
    rgb[mask, 2] = 1.0
    
    mask = (wavelength >= 490) & (wavelength <= 510)
    rgb[mask, 0] = 0.0
    rgb[mask, 1] = 1.0
    rgb[mask, 2] = (-(wavelength[mask] - 510) / (510. - 490.)) ** gamma
    
    mask = (wavelength >= 510) & (wavelength <= 580)
    rgb[mask, 0] = ((wavelength[mask] - 510) / (580 - 510.)) ** gamma
    rgb[mask, 1] = 1.0
    rgb[mask, 2] = 0.0
 
    mask = (wavelength >= 580) & (wavelength <= 645)
    rgb[mask, 0]= 1.0
    rgb[mask, 1]= (-(wavelength[mask] - 645) / (645. - 580.)) ** gamma
    rgb[mask, 2]= 0.0
    
    mask = (wavelength >= 645) & (wavelength <= 750)
    attenuation = 0.3 + 0.7 * (750 - wavelength[mask]) / (750 - 645)
    rgb[mask, 0] = (1.0 * attenuation) ** gamma
    rgb[mask, 1] = 0.0
    rgb[mask, 2] = 0.0

    return rgb

def plot_blp(lpf, fc, pars, axs=None):
    if axs is None:
        fig, axs = subplots(1, 2, figsize=(14,4), sharex=True, sharey=True)
    pal = sb.light_palette(c_ob)
    pmin, pmax = np.inf, -np.inf
    for irun in range(2):
        pmin = min(pmin, 0.99*fc[:,pars[irun]].min())
        pmax = max(pmax, 1.01*fc[:,pars[irun]].max())
        ps = np.percentile(fc[:,pars[irun]], [50,0.1,99.9,1,99,16,84], 0)
        for ipb,p in enumerate(ps.T):
            dy = 1./lpf.npb
            ymin, ymax = ipb*dy+0.1*dy, (ipb+1)*dy-0.1*dy
            axs[irun].axvline(p[0], ymin=ymin, ymax=ymax, c='k', lw=1)
            axs[irun].axvspan(*p[1:3], ymin=ymin, ymax=ymax, fc=pal[0], lw=0)
            axs[irun].axvspan(*p[3:5], ymin=ymin, ymax=ymax, fc=pal[1], lw=0)
            axs[irun].axvspan(*p[5:7], ymin=ymin, ymax=ymax, fc=pal[3], lw=0)
            axs[irun].axvspan(*p[1:3], ymin=ymin, ymax=ymax, fill=False, ec='k', lw=1)
    pl.setp(axs, xlim=(pmin, pmax))
    try:
        fig.tight_layout()
    except:
        pass
    return axs


def plot_light_curves(lpf, pv, pbc, yoffset=0.019, **kwargs):
    figsize = kwargs.get('figsize', (14,8))
    xlim    = kwargs.get('xlim', (0.473,0.533))
    ylim    = kwargs.get('ylim', (0.955,1.175))
    title   = kwargs.get('title', '')

    cc = get_color_cycle()
    fms = lpf.compute_lc_model(pv)
    fms[0][:,141] = np.nan
    fos = lpf.fluxes
    fig,axs = pl.subplots(1, 2, figsize=figsize, sharey=True, sharex=True)
    for i in range(2):
        for j in range(lpf.npb):
            ilc = lpf.npb*i+j
            phase = fold(lpf.times[ilc], pv[1], pv[0], 0.5)
            res = fos[ilc]-fms[i][j]
            axs[i].plot(phase, fos[ilc]+j*yoffset, 'o', c='k', ms=5)
            axs[i].plot(phase, fos[ilc]+j*yoffset, 'o', c=cc[j], ms=4)
            axs[i].plot(phase, fms[i][j]+j*yoffset, c='w', lw=3)
            axs[i].plot(phase, fms[i][j]+j*yoffset, c='k', lw=2)
            if i==1:
                axs[1].text(0.473, fms[i][j][0]+j*yoffset, int(round(pbc[j])), va='center')
            else:
                axs[0].text(0.532, fms[i][j][-1]+j*yoffset, int(round(pbc[j])), va='center')

    fig.suptitle(title, size=16)
    pl.setp(axs, ylim=ylim, yticks=[], xlim=xlim, xlabel='Phase [d]')
    pl.setp(axs[0], ylabel='Normalised flux')
    sb.despine(ax=axs[0], top=True)
    sb.despine(ax=axs[1], left=True, right=False, top=True)
    fig.tight_layout()
    return fig


def plot_radius_ratios(df, lpf, df2=None, axk=None, plot_spectrum=True, plot_kprior=True, **kwargs):
    if axk is None:
        fig, axk = pl.subplots(1,1)
    axh = axk.twinx()

    xlims = dict(bb = (450,950), nb = (515,925), Na = (560,620), K = (735,797))
    pbcs  = dict(bb = pb_centers_bb, nb = pb_centers_nb, K = pb_centers_k, Na = pb_centers_na)
    spss  = dict(bb = 0.0015, nb=0.0015, Na=0.001, K=0.001)
    spls  = dict(bb = 0.1, nb=0.1, Na=0.25, K=0.25)

    k2cols = [c for c in df.columns if 'k2' in c]
    pes_k = np.asarray(np.percentile(np.sqrt(df[k2cols]), [50, 0.5, 99.5, 16, 84], 0))
    yc = pes_k[0].mean()

    yspan = kwargs.get('yspan', 0.008)
    ylim = np.asarray(kwargs.get('ylim', np.array([yc-0.5*yspan, yc+0.5*yspan])))
    xlim = np.asarray(kwargs.get('xlim', xlims[lpf.passband]))

    spc_scale = kwargs.get('spc_scale', spss[lpf.passband])
    spc_loc = kwargs.get('spc_loc', yc+spls[lpf.passband]*yspan)

    fs = kwargs.get('fs', 14)
    lyoffset = kwargs.get('lyoffset', 2e-4)
    lxoffset = kwargs.get('lxoffset', 10)
    plot_h = kwargs.get('plot_h', False)

    plot_k  = kwargs.get('plot_k',  lpf.passband in ['K','bb','nb'])
    plot_na = kwargs.get('plot_na', lpf.passband in ['Na','bb','nb'])

    scaleheight = H(TEQ)
    pbc = pbcs[lpf.passband]

    if df2 is not None:
        pes_k2 = np.asarray(np.percentile(np.sqrt(df2[k2cols]), [50, 0.5, 99.5, 16, 84], 0))
    
    if plot_spectrum:
        d  = np.load(join(DRESULT, 'gtc_spectra_n2.npz'))
        sp = d['ccd2'].mean(0)
        wl = d['wl2']

        spn = spc_scale*N(sp)
        spf = [flt(wl)*spn for flt in lpf.filters]
        spf = [np.where(f>1e-4, f+spc_loc, f) for f in spf]
    
        axk.plot(wl, spc_loc+spn, alpha=0.2, c=c_bo)
        [axk.fill(wl, f, alpha=1, lw=1, fc='w', ec=c_ob) for f in spf]
                  
    if plot_kprior:
        kmean, kstd = lpf.prior_kw.mean, lpf.prior_kw.std
        axk.axhline(kmean, c=c_ob)
        [axk.axhspan(kmean-i*kstd, kmean+i*kstd, alpha=0.15-0.02*i, fc=c_bo) for i in range(1,4)]
          
    if plot_na:
        [axk.axvline(l, c='k', lw=3, ls='-', ymax=0.1, zorder=10) for l in wlc_na]
        axk.text(wlc_na.mean()+lxoffset, ylim[0]+lyoffset, 'Na I', size=fs, weight='bold', ha='center', 
                bbox=dict(facecolor='w', edgecolor='w'), zorder=9)

    if plot_k:
        [axk.axvline(l, c='k', lw=3, ls='-', ymax=0.1, zorder=10) for l in wlc_k]
        axk.text(wlc_k.mean()+lxoffset, ylim[0]+lyoffset, 'K I', size=fs, weight='bold', ha='center',
               bbox=dict(facecolor='w', edgecolor='w'), zorder=9)

    if df2 is None:
        axk.errorbar(pbc, pes_k[0], yerr=abs(pes_k[1:3]-pes_k[0]).mean(0), fmt=',', capsize=0, capthick=0, c=c_ob, lw=4, alpha=0.1)
        axk.errorbar(pbc, pes_k[0], yerr=abs(pes_k[3:5]-pes_k[0]).mean(0), fmt='o', capsize=2, capthick=1, c=c_ob)
    else:
        pbc = np.asarray(pbc)
        axk.errorbar(pbc-2.5, pes_k[0],  yerr=abs(pes_k[1:3]-pes_k[0]).mean(0), fmt=',', capsize=0, capthick=0, c=c_ob, lw=4, alpha=0.1)
        axk.errorbar(pbc-2.5, pes_k[0],  yerr=abs(pes_k[3:5]-pes_k[0]).mean(0), fmt='o', capsize=2, capthick=1, c=c_ob)
        axk.plot(pbc-2.5, pes_k[0], 'D', ms=2, mfc='r', mec=c_ob, mew=0.5)
        axk.errorbar(pbc+2.5, pes_k2[0], yerr=abs(pes_k2[1:3]-pes_k2[0]).mean(0), fmt=',', capsize=0, capthick=0, c=c_ob, lw=4, alpha=0.1)
        axk.errorbar(pbc+2.5, pes_k2[0], yerr=abs(pes_k2[3:5]-pes_k2[0]).mean(0), fmt='o', capsize=2, capthick=1, c=c_ob)
        axk.plot(pbc+2.5, pes_k2[0], 'o', ms=2, mfc=c_bo, mec=c_ob, mew=0.5)
  
    #[axh.axhline(v, alpha=0.1, zorder=-100) for v in arange(-6,7,4)]

    pl.setp(axk, xlim=xlim, ylim=ylim, ylabel='Radius ratio', xlabel='Wavelength [nm]')
    pl.setp(axh, xlim=xlim, ylim=(ylim-yc)*RSTAR/scaleheight, ylabel='$\Delta$ Scale height', yticks=arange(-6,7,2))
    return axk, axh

pal_na = sb.color_palette(wavelength_to_rgb(pb_centers_na))
pal_k  = sb.color_palette(wavelength_to_rgb(np.array(pb_centers_k)-150))
pal_nb = sb.color_palette(wavelength_to_rgb(np.array(pb_centers_nb)*0.82-30))
pal_bb = sb.color_palette(wavelength_to_rgb(pb_centers_bb))
