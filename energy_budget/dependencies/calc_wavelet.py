import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.dates as mdates
import datetime as dt
import scipy.signal as signal
import scipy.interpolate as intrp
import pycwt as wavelet

import sys
sys.path.insert(0, "/home/bieito/Documents/SCIENCE/EPFL/python_functions/")
from get_colormap import get_colormap
CMCV = get_colormap('/home/bieito/Documents/SCIENCE/EPFL/python_functions/WhiteBlueGreenYellowRed.ncmap') 

def calc_crosswavelet(date, x0, y0, steps = 32, PLOT = False, figname = "cross_wavelet.png", nquiverx = 120, nquiverz = 30):
    deltaT = (date[1] - date[0]).total_seconds()/60./60/24
    N = x0.size
    mother = wavelet.Morlet(6)
    dj = 1 / float(steps)        # sub-octaves per octaves
    s0 = 2 * deltaT       # Starting scale, here 2 months
    
    W12, coi, freq, signif = wavelet.xwt(x0, y0, deltaT, dj=dj, s0=s0, J=-1,
                                           significance_level=0.95,
                                               wavelet='morlet', normalize=True)

    cross_power = np.abs(W12)**2
    cross_sig = np.ones([1, N]) * signif[:, None]
    cross_sig = cross_power / cross_sig  # Power is significant where ratio > 1
    period = 1/freq

    glbl_cpower = cross_power.mean(axis=1)

    angle = np.angle(W12)
    u, v = np.cos(angle), np.sin(angle)
    u[cross_sig<1] = 0
    v[cross_sig<1] = 0

    OUT = dict()
    OUT["date"] = np.copy(date)
    OUT["PW"] = np.copy(cross_power)
    OUT["period"] = np.copy(period)
    OUT["steps"] = np.copy(steps)
    OUT["glPW"] = np.copy(glbl_cpower)
    OUT["sig"] = np.copy(cross_sig)
    OUT["fr"] = np.copy(freq)
    OUT["coi"] = np.copy(coi)
    OUT["angle"] = np.copy(angle)
    OUT["u"] = np.copy(u)
    OUT["v"] = np.copy(v)
    
    if PLOT:

        PW=np.log10(cross_power)
        #LP2=np.clip(LP2,0,np.max(LP2))
        PW=np.clip(PW,np.min(np.log10(2*glbl_cpower)),np.log10(np.max(2*glbl_cpower)))
        #LP2=np.clip(LP2,-5,5)
                
        
        iR = np.arange(0,N,N//nquiverx)
        dateR = [date[i] for i in iR]
        
        Nper = period.size
        Nper = Nper//nquiverz
        
        fig, ax = plt.subplots(figsize = (8,4))
        cc=ax.contourf(date, period,PW, steps,cmap = cm.jet)
        plt.colorbar(cc)
        coicoi=np.clip(coi,0,coi.max())
        ax.contour(date, period,cross_sig, np.arange(0,2,1), linewidths = 0.5, colors = "k")
        ax.quiver(dateR, period[::Nper], u[::Nper, iR], v[::Nper, iR], units='height',
                  angles='uv', pivot='mid', linewidth=0.5, edgecolor='k', color = "k",
                  headwidth=5, headlength=5, headaxislength=5, minshaft=2,
                  minlength=0, scale =30)
        ax.fill_between(date,coicoi,period.max(),alpha=0.8, color='w', hatch='x')
        yt = 2**np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
        #ax.set_xlim(xl)
        ax.set_yscale('log')
        ax.set_yticks(yt)
        ax.set_ylim((np.min(period), np.max(period)))
        ax.tick_params(axis='y', which='minor', left=False)
        ax.set_yticklabels(yt)
        ax.set_ylabel("time [days]")
        ax.grid(b=None, which='major', axis='y', alpha=0.2, antialiased=True, c='k', linestyle='-.')
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
        fig.savefig(figname, dpi = 300)
        plt.close(fig)

    return OUT
    
    
    

def calc_wavelet(date, x00, steps = 32, ylab = "", annotation = "", figtitle = "",units = "m s$^{-1}$",figname = "wavelet.png",**kargs):
    deltaT = (date[1] - date[0]).total_seconds()/60./60/24
    mean = np.mean(x00)
    var = np.var(x00)
    x0 = (x00 - mean)/var**0.5
    N = x0.size
    
    mother = wavelet.Morlet(6)
    dj = 1 / float(steps)        # sub-octaves per octaves
    s0 = 2 * deltaT       # Starting scale, here 2 months
    wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(x0, deltaT, dj, s0, -1, mother)
    # Normalized wavelet power spectra
    power = (np.abs(wave)) ** 2
    period = 1/freqs

    #significance
    alpha, _,_ = wavelet.ar1(x00)
    signif, fft_theor = wavelet.significance(1.0, deltaT, scales, 0, alpha,
                                         significance_level=0.95,
                                         wavelet=mother)
    sig95 = np.ones([1, N]) * signif[:, None]
    sig95 = power/sig95


    #mean spectrum
    glbl_power = power.mean(axis=1)
    dof = N - scales  # Correction for padding at edges
    glbl_signif, tmp = wavelet.significance(var, deltaT, scales, 1, alpha,
                                        significance_level=0.95, dof=dof,
                                        wavelet=mother)
    
    LP2=np.log10(var*power)
    #LP2=np.clip(LP2,0,np.max(LP2))
    LP2=np.clip(LP2,np.min(np.log10(2*var*glbl_power)),np.log10(2*np.max(var*glbl_power)))
    #LP2=np.clip(LP2,np.min(np.log10(2*var*glbl_signif)),np.log10(2*np.max(var*signif_power)))
    #LP2=np.clip(LP2,-5,5)
    
    fig = plt.figure(figsize = (8,4))
    ax1 = fig.add_axes([0.10,0.72,0.50,0.23])
    ax1.plot(date, x00, color = "k", lw = 1)
    ax1.set_xticklabels([])
    ax1.grid(True)
    xl = ax1.get_xlim()
    ax1.set_ylabel(ylab)
    
    ax2 = fig.add_axes([0.10,0.08,0.62,0.60])
    cc=ax2.contourf(date, period,LP2, steps,cmap = cm.viridis)
    cb=plt.colorbar(cc)
    cb.set_label("Log10 Wavelet Power [(%s)$^2]$"%(units))
    ax2.contour(date, period,sig95, np.arange(0,2,1), linewidths = 0.5, colors = "k")
    coicoi=np.clip(coi,0,coi.max())
    ax2.fill_between(date,coicoi,period.max(),alpha=0.8, color='w', hatch='x')
    period=1/freqs
    yt = 2**np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
    ax2.set_xlim(xl)
    ax2.set_yscale('log')
    ax2.set_yticks(yt)
    ax2.set_ylim((np.min(period), np.max(period)))
    ax2.tick_params(axis='y', which='minor', left=False)
    ax2.set_yticklabels(yt)
    ax2.set_ylabel("time [days]")
    ax2.grid(b=None, which='major', axis='y', alpha=0.2, antialiased=True, c='k', linestyle='-.')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
    ax2.annotate(annotation, xy = (0.01,0.02), xycoords = "axes fraction", fontweight = "bold", color = "k")

    """
    ax3 = fig.add_axes([0.82,0.08,0.17,0.63])
    ax3.plot(glbl_signif, period, 'k--')
    #ax3.plot(var * fft_theor, period, '--', color='#cccccc')
    #ax3.plot(var * fft, 1./fftfreqs, '-', color='#cccccc',
    #        linewidth=1.)
    ax3.plot(var * glbl_power, period, 'k-', linewidth=1.5)
    #ax3.set_xlabel(r'Power [({})^2]'.format(units))
    #ax3.set_xlim([0, glbl_power.max() + var])
    #ax3.set_ylim(np.log2([period.min(), period.max()]))
    #ax3.set_yticks(np.log2(Yticks))
    ax3.set_yticklabels([])
    ax3.set_yscale("log")
    ax3.set_xscale("log")
    ax3.set_yticks(yt)
    ax3.set_ylim((np.min(period), np.max(period)))
    ax3.tick_params(axis='y', which='minor', left=False)
    ax3.set_yticklabels(yt)
    ax3.xaxis.tick_top()
    ax3.set_xlabel('Mean Power [(%s)$^2$]'%(units))    
    ax3.xaxis.set_label_position('top') 
    #plt.setp(cx.get_yticklabels(), visible=False)
    """
    #fig.suptitle(figtitle)
    if "xy" in kargs:
        xy = kargs["xy"]
        ax2.plot(xy[0],xy[1], color = cm.tab10(1))
    
    fig.savefig(figname, dpi = 300, bbox_inches = "tight")
    plt.close(fig)
