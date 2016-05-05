from core import *

from numpy import load, median, std, ones, ones_like, average, linspace, argmax, inf, poly1d, polyfit, zeros, nan, full_like, log10, arange

def create_dc(night, ccd, sl, istart=0, iend=1000):
    dc = []
    k = 'n%iccd%i'%(night,ccd)
    for fo in l_obj[night-1][istart:iend]:
        t = pf.getdata(fo, ext=ccd).astype(float64)[:,sl]-bias[k][:,np.newaxis]
        dc.append(t)
    return array(dc)

def create_lc(night, ccd, sl, maxpts=50):
    sp1, sp2 = [],[]
    k = 'n%iccd%i'%(night,ccd)
    for fo in l_obj[night-1][:maxpts]:
        t = pf.getdata(fo, ext=ccd).astype(float64)[:,sl]-bias[k][:,np.newaxis]
        sp1.append(t.mean(1))
        t /= flats[k][:,sl]
        sp2.append(t.mean(1))
    return array(sp1), array(sp2)

class DataCube(object):
    def __init__(self, night, ccd, filters, width=60, istart=0, iend=1000, force=False):
        self._cname = 'dc_n%iccd%i.npy'%(night,ccd)
        
        gain = pf.getval(l_obj[0][0], 'gain')
        bias  = {k:v for k,v in load(f_bias_dn).items()}
        flats = {k:v-bias[k][:,np.newaxis] for k,v in load(f_flats).items()}
        flats = {k:v/median(v) for k,v in flats.items()}

        self._sl  = ccd_slice(night-1, width)[ccd-1][1]
        self._wls = load(join('results','wl_calibration.pkl'))['n%iccd%i'%(night,ccd)]
        
        self.filters = filters
        self.pixels = arange(2051)
        self.wl = self._wls.pixel_to_wl(self.pixels)
            
        if not exists(join('results',self._cname)) or force:
            self._data  = create_dc(night, ccd, self._sl, istart, iend)
            save(join('results',self._cname), self._data)
        else:
            self._data = load(join('results',self._cname))
            
        self.width  = width
        self.hwidth = width//2
        self._med   = median(self._data, 0)
        self._std   = std(self._data, 0)
        self._mask  = abs(self._data - self._med) < 5*self._std
        self._spectrum_mask = ones(self._data.shape[1])
        
        self._flat  = flats['n%iccd%i'%(night,ccd)][:,self._sl]
        
        if ccd == 1:
            self._sky = concatenate([self._data[:,:,:5],self._data[:,:,-10:]], 2)
        if ccd == 2:
            self._sky = concatenate([self._data[:,:,:10],self._data[:,:,-5:]], 2)
            
        msky = median(self._sky,0)
        vsky = 1.482*median(abs(self._sky-msky),0)
        m = self._sky-msky > 5*vsky
        self._sky[m] = MF(self._sky, 4)[m]
        self._sky = self._sky.mean(2)
        
        self.aperture = ones(self._data.shape[1:])
        self.apt_mf =  1. 
        self.apt_p  =  1. 

        self.apply_flat = True
        self.apply_mask = True
        self.apply_aperture = True
        self.apply_sky = True
        self.apply_spectrum_mask = True
        self.calculate_cache()
                
        ## Outlier masking
        ## ---------------
        f   = self.spectra.mean(1)
        fm  = MF(f, 15)
        mad = median(abs(f-fm))
        self.mask = abs(f-fm) < 15*mad

        
    def set_flags(self, afl=None, ams=None, aap=None, ask=None, asm=None):
        vset = lambda v,a: v if v is not None else a
        self.apply_flat          = vset(afl, self.apply_flat)
        self.apply_mask          = vset(ams, self.apply_mask)
        self.apply_aperture      = vset(aap, self.apply_aperture)
        self.apply_sky           = vset(ask, self.apply_sky)
        self.apply_spectrum_mask = vset(asm, self.apply_spectrum_mask)
        self.calculate_cache()
        

    def calculate_cache(self):
        if self.apply_flat:
            self._cdata = (self._data-self.sky[:,:,np.newaxis]) / self._flat
        else:
            self._cdata = self._data-self.sky[:,:,np.newaxis]
            
        pixel_weights = ones_like(self._data)
        if self.apply_mask:
            pixel_weights *= self._mask
        if self.apply_aperture:
            pixel_weights *= self.aperture
            
        self._cspectra = average(self._cdata,2,weights=pixel_weights)
            
        
    @property
    def data(self):
        return self._cdata
        
    @property
    def spectra(self):
        return self._cspectra
        
    @property
    def filtered_spectra(self):
        spectra = self.spectra
        smask = self._spectrum_mask if self.apply_spectrum_mask else 1.
        return [spectra*smask*f(self.wl)[np.newaxis,:] for f in self.filters]
        
    @property
    def light_curves(self):
        return [s.mean(1) for s in self.filtered_spectra]
    
    @property
    def sky(self):
        return self._sky if self.apply_sky else 0.

    
    def trace_spectrum(self):
        pts = linspace(50,1950,10).astype(int)
        img = self.data.mean(0)
        profiles = img[pts,:]
        centers = argmax(img[pts], 1)
        amplitudes = img[pts].max(1)
        npts = img.shape[1]
        
        def minfun(pv,i):
            if any(pv < 0) or (pv[0]>npts):
                return inf
            return -logl_g1d(*pv, fobs=profiles[i])

        pvs = array([fmin(minfun, [centers[i], amplitudes[i], 8, 200, 0], args=(i,), maxfun=3000, disp=False) 
                     for i in range(10)])

        self.apt_x = pts
        self.apt_y = pvs[:,0]
        self.apt_fwhm = pvs[:,2].mean()
        self.apt_poly = poly1d(polyfit(self.apt_x, self.apt_y, 2))
        
        
    def create_aperture(self, mf=None, p=None):
        self.apt_mf = mf if mf is not None else self.apt_mf
        self.apt_p  = p if p is not None else self.apt_p

        mf,p = self.apt_mf, self.apt_p
        y = arange(self.width)
        self.aperture[:] =  array([exp(-0.5*((y-self.apt_poly(i))/(mf*self.apt_fwhm/2.355))**(2*p)) 
                      for i in xrange(self._data.shape[1])])
        self.calculate_cache()
        return self.aperture

    def plot_spectrum_trace(self, ax):
        ax.imshow(log10(self.data.mean(0)).T, aspect='auto', vmin=1)
        x = arange(self._data.shape[1])
        ax.plot(x, self.apt_poly(x), 'w', lw=5)
        ax.plot(x, self.apt_poly(x), 'k', lw=3)
        ax.plot(self.apt_x, self.apt_y, 'wo', markeredgecolor='k', markeredgewidth=2)
        ax.contour(self.create_aperture().T, levels=[0.0, 0.25, 0.5, 0.75, 0.95])
        
        
    def create_spectrum_mask(self, lims=None, sl=None, elim=None, erosion_iterations=2, plot=True, figsize=(13,8), fs=12):
        s = self.spectra
        x = arange(s.shape[0])

        if lims:
            tmask = (x < lims[0]) | (x > lims[1])
        else:
            tmask = ones(x.size, np.bool)

        bmask = self.mask    
        means = s.mean(0)
        lc_std = zeros(s.shape[1])
        sl = sl if sl is not None else s_[:]

        for i, lc in enumerate(s.T):
            lc = N(lc[bmask])
            lc_std[i] = (lc/gf(lc,35)).std()

        lc_mean = N(s.mean(1))
        lc_mean[~bmask] = nan
        lc_gf = full_like(lc_mean, nan)
        lc_gf[bmask] = gf(lc_mean[bmask], 25, mode='nearest')

        self._spectrum_mask = smask  = binary_erosion(lc_std < elim, iterations=erosion_iterations, border_value=1)

        if plot:
            fig = pl.figure(figsize=figsize)
            gs = pl.GridSpec(2,2, width_ratios=(0.85,0.15), height_ratios=(0.6,0.4))
            axs = ae, ai, at = pl.subplot(gs[0,0]), pl.subplot(gs[1,0]), pl.subplot(gs[1,1])
            ai.imshow(s[bmask,:], aspect='auto', interpolation='nearest', origin='lower')
            fs = 16

            ae.plot(lc_std, c=cp[0], lw=2, alpha=0.24)
            ae.plot(np.where(smask, lc_std, nan), lw=2)
            ae.plot(np.where(smask, nan, -0.00), lw=3, c='k')   

            spec  = s[tmask&bmask,:].mean(0)
            spec -= spec.min()
            spec /= spec.max()
            ae.plot(0.03+0.075*spec, alpha=0.5, c=cp[2], lw=2)

            at.plot(lc_mean, x, '.', alpha=1, ms=10)
            at.plot(lc_gf,   x, c=cp[0], lw=4)
            at.plot(lc_gf,   x, 'w', lw=2, alpha=1)

            if lims is not None:
                ai.axhspan(*lims, ec='k', lw=2, fill=False, alpha=0.5)
                ai.axhspan(*lims, fc='w', ec='w', lw=1, fill=True, alpha=0.25)
                at.axhspan(*lims, ec='k', lw=2, fill=False, alpha=0.5)
                at.axhspan(*lims, fc='w', fill=True, alpha=0.25)

            if elim:
                ae.axhline(elim, ls='-', c=c_bo, alpha=0.25, zorder=-100, lw=2)

            ae.text(0.01,0.94,'a', transform=ae.transAxes, size=fs)
            ai.text(0.97,0.88,'b', transform=ai.transAxes, size=fs)
            at.text(0.85,0.88,'c', transform=at.transAxes, size=fs)

            pl.setp(ai, xlim=[0,s.shape[1]], ylim=x[[0,-1]], ylabel='Exposure', xlabel='Pixel')
            pl.setp(ae, xlim=[0,s.shape[1]], xticks=[], ylim=(-0.0025,0.11), ylabel='Flux')
            pl.setp(at, ylim=x[[0,-1]], yticks=[], xticks=[], xlabel='Flux')

            fig.tight_layout()
            sb.despine(ax=ae)

        if plot:
            return fig, axs
