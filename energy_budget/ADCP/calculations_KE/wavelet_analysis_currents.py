import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import scipy.io as io
import scipy.signal as signal
import copy
sys.path.insert(0, "/home/bieito/Documents/SCIENCE/EPFL/python_functions/")
from time_conversion import *
import vertical_modes as VM
import scipy.interpolate as intrp
import datetime as DT
from calc_wavelet import *
import seawater as sw
from useful_functions import moving_average
ADCP = pickle.load( open("ADCP_Geneva_all_2019_2020_smooth_2h_4m_NM.pi") )
ST = pickle.load(open("smooth_temperature_evolution.pi"))
MET = pickle.load(open("meteo/Meteo_Leman_all_MO.pi"))
WW = MET["WS_ms_S_WVT"][::6]

D_range = ST["depth"][-1] - ST["depth"][0]
ADCP["NM"]["Au"] /= np.sqrt(D_range)#normalices to m/s
ADCP["NM"]["Av"] /= np.sqrt(D_range)


#Kelvin wave

bath = io.loadmat("./bath/BathGeneva.mat")
ycent = np.full( bath["LongitudeBath"].size, np.nan )
#for i in range(bath["LongitudeBath"].size):
#    ycent[i] = np.nansum( bath["LatitudeBath"] )
maxDepth = np.nanmax(-bath["Bathymetry"], axis = 0)
Lz = np.full( ST["depth"].size, np.nan )
for i in range(ST["depth"].size):
    ij = np.where((maxDepth>ST["depth"][i]))[0]
    Lz[i] = bath["LongitudeBath"][0,ij[-1]] - bath["LongitudeBath"][0,ij[0]]


#L20 = 62*1000. #basin length

dz = ST["depth"][1]-ST["depth"][0]
z = np.tile(ST["depth"], (len(ST["dateGRs"]),1)).T
zth = np.nansum( ST["N2GRs"]*dz*z, axis = 0 )/np.nansum(dz*ST["N2GRs"], axis = 0)
L = np.full(zth.size, np.nan)
for j in range(zth.size):
    ij = np.nanargmin( np.abs(z[:,j]-zth[j] ))
    L[j] = Lz[ij]

TKB = 2*L/ST["bcMOD"]["c"][:,0]/60./60./24
Ro = ST["bcMOD"]["c"][:,0]/sw.f(46)/1000.


#wavelet BC mode
Au = np.copy(ADCP["NM"]["Au"][0,:])
date= np.copy(ADCP["date"])

"""
calc_wavelet(date,Au, steps = 64,xy = (ST["dateGRs"],TKB), figtile = "Barotropic mode",figname = "wavelet_bt.png", ylab = "$A_u^{\\mathrm{bt}}$ [m$^2$ s$^{-1}$]")

#wavelet first mode
Au = ADCP["NM"]["Au"][1,:]
calc_wavelet(date,Au, steps = 64,xy = (ST["dateGRs"],TKB), figtile = "First baroclinic mode",figname = "wavelet_bc1.png", ylab = "$A_u^{\\mathrm{bc1}}$ [m$^2$ s$^{-1}$]")

#wavelet second mode
Au = ADCP["NM"]["Au"][2,:]
#calc_wavelet(date,Au, steps = 64,xy = (ST["dateGRs"],TKB), figtile = "Second baroclinic mode",figname = "wavelet_bc2.png", ylab = "$A_u^{\\mathrm{bc1}}$ [m$^2$ s$^{-1}$]")
"""

#wavelet for fixed depth
#wavelet for fixed depth
z0 = 4.
iz = np.where(ADCP["z"] == z0)[0][0]
#u0 = ADCP["u"][iz,:]
u0 = np.nansum(ADCP["NM"]["u"][:,iz,:], axis = 0)
u0[np.isnan(u0)] = 0.
calc_wavelet(date,u0, steps = 64,xy = (ST["dateGRs"],TKB), figtile = "$u = \mathrm{%d m}$"%(z0), annotation = "a.",figname = "wavelet_u_%dm.png"%(z0), ylab = "$u_{\\mathrm{%d m}}}$ [m s$^{-1}$]"%(z0))

#v0 = ADCP["NM"]["v"][iz,:]
v0 = np.nansum(ADCP["NM"]["v"][:,iz,:], axis = 0)
v0[np.isnan(u0)] = 0.
calc_wavelet(date,u0, steps = 64,xy = (ST["dateGRs"],TKB), annotation = "a.",figtile = "$v = \mathrm{%d m}$"%(z0),figname = "wavelet_v_%dm.png"%(z0), ylab = "$v_{\\mathrm{%d m}}}$ [m s$^{-1}$]"%(z0))


z0 = 100.
iz = np.where(ADCP["z"] == z0)[0][0]
#u0 = ADCP["u"][iz,:]
u0 = np.nansum(ADCP["NM"]["u"][:,iz,:], axis = 0)
u0[np.isnan(u0)] = 0.
calc_wavelet(date,u0, steps = 64,xy = (ST["dateGRs"],TKB), figtile = "$u = \mathrm{%d m}$"%(z0), annotation = "b.",figname = "wavelet_u_%dm.png"%(z0), ylab = "$u_{\\mathrm{%d m}}}$ [m s$^{-1}$]"%(z0))

#v0 = ADCP["NM"]["v"][iz,:]
v0 = np.nansum(ADCP["NM"]["v"][:,iz,:], axis = 0)
v0[np.isnan(u0)] = 0.
calc_wavelet(date,u0, steps = 64,xy = (ST["dateGRs"],TKB), annotation = "b.",figtile = "$v = \mathrm{%d m}$"%(z0),figname = "wavelet_v_%dm.png"%(z0), ylab = "$v_{\\mathrm{%s m}}}$ [m s$^{-1}$]"%(z0))



fw,Sw = signal.welch(signal.detrend(WW), fs = 24., nperseg = 1024)
f = []
S = []
for (Au,Av) in zip(ADCP["NM"]["Au"],ADCP["NM"]["Av"]):
    fu,Su = signal.welch( signal.detrend(Au), fs = 24., nperseg = 1024)
    fv,Sv = signal.welch(signal.detrend( Av), fs = 24., nperseg = 1024)
    f.append(fu)
    S.append(0.5*(Su+Sv))
    

fig, ax = plt.subplots()

l = []
for (f0,S0) in zip(f,S):
    l0,=ax.loglog(f0,S0, lw = 1)
    l.append(l0)
ax.axvline(np.max(1/TKB), color = "k", lw = 0.5)
ax.annotate("$KW$", xy = (np.max(1/TKB),2e-8), ha = "left")
ax.axvline(sw.f(46)*60*60*24/2/np.pi, color = "k",lw = 0.5)
ax.annotate("$f$", xy = (sw.f(46)*60*60*24/2/np.pi,2e-8), ha = "left")
ii = np.where( (f0>3) & (f0<10))[0]

ax.loglog(f0[ii],1e-3*f0[ii]**-2, color = "k")
xx = np.array([f0[ii][0],f0[ii][-1]])
yy = 1e-3*np.array([xx[0]]*2)**-2
ax.loglog(xx,yy, color = "k" )
xx1 = np.array([f0[ii][-1],f0[ii][-1]])
yy1 = 1e-3*np.array(xx)**-2
ax.loglog(xx1,yy1, color = "k"  )
ax.annotate("1", xy = (10**np.nanmean(np.log10(xx)), yy[0]), va = "bottom" )
ax.annotate("-2", xy = (xx1[0],10**np.nanmean(np.log10(yy1))), ha = "left" )

l2,=ax.loglog(fw,Sw*1e-6, lw = 1, color = "k")
l.append(l2)
ax.legend(l,["M0","M1","M2","M3","wind $[\\times 10^{-6}]$"], loc = "lower left")

xtk = np.array([0.125,0.25,0.5,1.,2.,4.,8.,16.,32.])
xtkl = []
for x in xtk:
    if x<1:
        xtkl.append("%d h"%(x*24))
    else:
        xtkl.append("%d d"%(x))
ax.set_xticks(1/xtk)
ax.set_xticks([],minor = True)
ax.set_xticklabels(xtkl)

ax.set_ylabel("Spectral density [m$^2$ s$^{-2}$ day]")
fig.savefig("Spectral_density_modes.pdf", dpi = 300, bbox_inches = "tight")
plt.show()


###
##DIEL SIGNAL ANALYSIS
##
z0 = 4.
iz = np.where(ADCP["z"] == z0)[0][0]
u00 = ADCP["u"][iz,:]
v00 = ADCP["v"][iz,:]

iigood = np.isfinite(u0+v0)
intu = intrp.interp1d( ADCP["time"][iigood], u00[iigood], bounds_error = False )
intv = intrp.interp1d( ADCP["time"][iigood], v00[iigood], bounds_error = False )

u0 = intu(ADCP["time"])
v0 = intv(ADCP["time"])
u0[np.isnan(u0)] = 0.
v0[np.isnan(v0)] = 0.

fmax = 24./20.
fmin = 24./28.
fN = 24*0.5
b,a = signal.butter(4,[fmin/fN, fmax/fN], btype = "band" )
u024h = signal.filtfilt(b,a,u0)
v024h = signal.filtfilt(b,a,v0)


fNW = 24*6*0.5
Wu = np.copy(MET["u"])
Wu[np.isnan(Wu)] = 0.
Wv = np.copy(MET["v"])
Wv[np.isnan(Wv)] = 0.
b,a = signal.butter(4,[fmin/fNW, fmax/fNW], btype = "band" )
Wu24h = signal.filtfilt(b,a,Wu)
Wv24h = signal.filtfilt(b,a,Wv)

Au24h = moving_average(u024h**2,24)
Av24h = moving_average(v024h**2,24)
AWu24h = moving_average(Wu24h**2,6*24)
AWv24h = moving_average(Wv24h**2,6*24)


fig, ax = plt.subplots(2,1, sharex = True, figsize = (12,5))
ax[0].plot(ADCP["date"], 0.5*Au24h*1000, label = "Eastward")
ax[0].plot(ADCP["date"], 0.5*Av24h*1000, label = "Northward")
ax[0].legend()
ax[0].set_ylabel("Surface water KE diel [J m$^{-3}$]")
ax[1].plot(MET["date_dt"], 0.5*AWu24h*1.2)
ax[1].plot(MET["date_dt"], 0.5*AWv24h*1.2)
ax[1].set_ylabel("Wind KE diel [J m$^{-3}$]")
fig.savefig("Diel_currents_and_winds.png", dpi = 300)
plt.show()
