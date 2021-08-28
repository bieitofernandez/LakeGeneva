import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import scipy.io as io
import copy
sys.path.insert(0, "/home/bieito/Documents/SCIENCE/EPFL/python_functions/")
from time_conversion import *
from useful_functions import *
import vertical_modes as VM
import scipy.interpolate as intrp
from scipy import optimize
import glob
import seawater as sw
import datetime as DT

ST = pickle.load(open("smooth_temperature_evolution.pi"))
ADCP = pickle.load( open("ADCP_Geneva_all_2019_2020_smooth_2h_4m.pi") )
#ADCP = pickle.load( open("ADCP_Geneva_all_2019_2020.pi") )
Nm = ST["bcMOD"]["c"].shape[1]
nt = len(ADCP["date"])
z = np.copy(ADCP["z"])

sh2 = np.full(ADCP["u"].shape, np.nan)

Au = np.full( (Nm+1,nt), np.nan )
Av = np.full( (Nm+1,nt), np.nan )
um = np.full( (Nm+1,ADCP["z"].size, nt), np.nan )
vm = np.full( (Nm+1,ADCP["z"].size, nt), np.nan )
sh2m = np.full( (ADCP["z"].size, nt), np.nan )
N2 = np.full( (ADCP["z"].size, nt), np.nan ) 
Ri = np.full( (ADCP["z"].size, nt), np.nan )
sh2m_norm = np.full( (Nm+1, nt), 0.)

KEu = np.full(nt, np.nan)
KEv = np.full(nt, np.nan)
us = np.full(ADCP["u"].shape, np.nan)
vs = np.full(ADCP["v"].shape, np.nan)
shxs = np.full(ADCP["u"].shape, np.nan)
shys = np.full(ADCP["v"].shape, np.nan)

PLOT = False
indices = range(0,nt)
#indices = range(10000,20000)
D_range = ST["depth"][-1] - ST["depth"][0]
KEfactor = 3.
#ss = 0.001
ss = 0.001
for i in indices:
    print i
    print ADCP["date"][i]
    iT = np.argmin( np.abs(ADCP["time"][i]-ST["timeGRs"]) )
    u = ADCP["u"][:,i]
    v = ADCP["v"][:,i]
    shx = first_centered_differences( -z,u )
    shy = first_centered_differences( -z,v )
    sh2[:,i] = shx**2 + shy**2
    
    iif = np.where(np.isfinite(u))[0]
    if len(iif)>10:
        intU = intrp.UnivariateSpline(z[iif], u[iif], s= ss)
        intDU = intU.derivative()
        us[iif[0]:iif[-1]+1,i] = intU(z[iif[0]:iif[-1]+1])
        shxs[iif[0]:iif[-1]+1,i] = intDU(z[iif[0]:iif[-1]+1])
        #us[:,i] = intU(z)
        #shxs[:,i] = intDU(z)

    iif = np.where(np.isfinite(v))[0]
    if len(iif)>10:
        intV = intrp.UnivariateSpline(z[iif], v[iif], s= ss)
        intDV = intV.derivative()
        vs[iif[0]:iif[-1]+1,i] = intV(z[iif[0]:iif[-1]+1])
        shys[iif[0]:iif[-1]+1,i] = intDV(z[iif[0]:iif[-1]+1])
        #vs[:,i] = intV(z)
        #shys[:,i] = intDV(z)


    
    Au[:,i], um[:,:,i],KEu[i] = VM.__fits_modes_progressive__(z,u, ST["depth"], ST["bcMOD"]["v"][:,iT,:], min_dat = 0.25, KEfactor = KEfactor)
    Av[:,i], vm[:,:,i],KEv[i] = VM.__fits_modes_progressive__(z,v, ST["depth"], ST["bcMOD"]["v"][:,iT,:], min_dat = 0.25, KEfactor = KEfactor)

    #Au[:,i], um[:,:,i],KEu[i] = VM.__fits_modes_progressive_fitBT__(z,u, ST["depth"], ST["bcMOD"]["v"][:,iT,:], min_dat = 0.25, KEfactor = KEfactor)
    #Av[:,i], vm[:,:,i],KEv[i] = VM.__fits_modes_progressive_fitBT__(z,v, ST["depth"], ST["bcMOD"]["v"][:,iT,:], min_dat = 0.25, KEfactor = KEfactor)

    umt = np.sum(um[:,:,i], axis = 0)
    vmt = np.sum(vm[:,:,i], axis = 0)
    shxm = first_centered_differences(-z,umt)
    shym = first_centered_differences(-z,vmt)
    sh2m[:,i] = shxm**2 + shym**2

    iif = np.isfinite(ST["N2GRs"][:,iT])
    intN2 = intrp.interp1d( ST["depth"][iif],ST["N2GRs"][iif,iT], bounds_error = False)
    N2[:,i] = intN2(z)
    Ri[:,i] = N2[:,i]/sh2m[:,i]
    
    for j in range(Nm):
        sh0 = first_centered_differences(-ST["depth"],ST["bcMOD"]["v"][:,iT,j])
        #fig, ax = plt.subplots(1,2, sharey = True)
        #ax[0].plot(ST["bcMOD"]["v"][:,i,j], -ST["depth"])
        #ax[1].plot(sh0, -ST["depth"])
        #plt.show()
        sh2m_norm[j+1,i] = np.nanmean(sh0**2)
    

    if PLOT:
        fig, ax = plt.subplots(1,2, sharey = True)
        ax[0].plot(u,-z,'.')
        ax[0].plot(us[:,i],-z,'k-')
        ax[0].plot(np.cumsum(um[:,:,i], axis = 1),-ST["depth"],"-") #+Au[0,i]
        ax[1].plot(v,-z,'.')
        ax[1].plot(vs[:,i],-z,'k-')
        ax[1].plot(np.cumsum(vm[:,:,i], axis = 1),-ST["depth"],"-") #+Av[0,i]
        plt.show()

KE = 1000*(KEu+KEv)
KEs = 1000*0.5*np.nanmean(us**2 + vs**2, axis = 0) * D_range
mKE = 1000*0.5*( Au**2 + Av**2)
intmKE = np.cumsum(mKE, axis = 0)


SH2s = np.nanmean(shxs**2 + shys**2, axis = 0)
mSH2 = ( Au**2 + Av**2)*sh2m_norm
intmSH2 = np.cumsum(mSH2, axis = 0)

#Kelvin wave
L = 62*1000. #basin length
TKB = 2*L/ST["bcMOD"]["c"][:,0]/60./60./24
Ro = ST["bcMOD"]["c"][:,0]/sw.f(46)/1000.


#saves output
ADCP["KE"] = KE
ADCP["sh2"] = sh2
ADCP["NM"] = dict()
ADCP["NM"]["u"] = np.copy(um)
ADCP["NM"]["v"] = np.copy(vm) 
ADCP["NM"]["Au"] = np.copy(Au)
ADCP["NM"]["Av"] = np.copy(Av)
ADCP["NM"]["KE"] = np.copy(mKE)
ADCP["NM"]["int_KE"] = np.copy(intmKE)
ADCP["NM"]["msh2_norm"] = np.copy(sh2m_norm)
ADCP["NM"]["msh2"] = np.copy(mSH2)
ADCP["NM"]["int_sh2"] = np.copy(intmSH2)
ADCP["NM"]["Ri"] = np.copy(Ri)
ADCP["NM"]["N2"] = np.copy(N2)
ADCP["NM"]["sh2"] = np.copy(sh2m)



ADCP["smooth"] = dict()
ADCP["smooth"]["u"] = np.copy(us)
ADCP["smooth"]["v"] = np.copy(vs)
ADCP["smooth"]["KE"] = np.copy(KEs)
ADCP["smooth"]["sh2"] = shxs**2 + shys**2

pickle.dump(ADCP,open("ADCP_Geneva_all_2019_2020_smooth_2h_4m_NM.pi","w+"))
#pickle.dump(ADCP,open("ADCP_Geneva_all_2019_2020_NM.pi","w+"))



"""
fig, ax = plt.subplots()
ax.plot(ADCP["time"][2175:2920],intmKE[:,2175:2920].T)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
plt.show()
"""





fig = plt.figure(figsize = (10,8))
ax0 = fig.add_axes([0.08,0.78,0.77,0.20])
ax1 = fig.add_axes([0.08,0.56,0.77,0.20]) 
ax2 = fig.add_axes([0.08,0.30,0.96,0.24])
ax3 = fig.add_axes([0.08,0.04,0.96,0.24])
l= []
for i in range(Nm+1):
    ii = Nm-i
    l0 = ax0.fill_between(ADCP["date"], intmKE[ii,:], color = cm.tab10(ii))
    l.append(l0)
    print ii
#ax0.plot(ADCP["date"], KEs,"k", lw = 0.5)
ax0.set_xticklabels([])
xl = ax0.get_xlim()
ax0.set_ylabel("KE [J m$^{-2}$]")
ax0.set_ylim((0,1000))
ax0.legend(l[::-1],["Bt","bc1","bc2","bc3"], ncol = 4)

l= []
for i in range(Nm+1):
    ii = Nm-i
    l0 = ax1.fill_between(ADCP["date"], 1e4*intmSH2[ii,:], color = cm.tab10(ii))
    l.append(l0)
    print ii
#ax1.plot(ADCP["date"], 1e4*SH2s,"k", lw = 0.5)
ax1.set_xticklabels([])
ax1.set_ylabel("$sh^2$ [$10^{-4}$ s$^{-2}$]")
ax1.set_ylim((0.,1))
ax1.set_xlim(xl)
ax1b = ax1.twinx()
ax1b.plot(ST["dateGRs"], TKB,"k")
ax1b.set_xticklabels([])
ax1b.set_ylabel("K-W Period [days]")

ccu1 = ax2.contourf(ADCP["date"],ADCP["z"], ADCP["u"]*100, np.arange(-25,26,1), cmap = cm.RdBu_r)
#ccu1 = ax2.contourf(ADCP["date"],ADCP["z"], us*100, np.arange(-25,26,1), cmap = cm.RdBu_r)
ax2.contour(ST["dateGRs"], ST["depth"], ST["TGRs"], np.arange(8,30,2), colors = "gray",linewidths = 0.5)
ax2.set_ylim((110,0))
ax2.set_xticklabels([])
ax2.set_xlim(ax0.set_xlim())
ax2.set_ylabel("Depth [m]")
cb=plt.colorbar(ccu1, ax = ax2)
cb.set_label("$u_{meas}$ [cm s$^{-1}$]")
ax2.set_xlim(xl)

ccu2 = ax3.contourf(ADCP["date"],ADCP["z"], np.sum(um,axis = 0)*100, np.arange(-25,26,1), cmap = cm.RdBu_r)
cb=plt.colorbar(ccu2, ax = ax3)
ax3.contour(ST["dateGRs"], ST["depth"], ST["TGRs"], np.arange(8,30,2), colors = "gray",linewidths = 0.5)
ax3.set_ylim((110,0))
ax3.set_xlim(ax0.set_xlim())
ax3.set_ylabel("Depth [m]")
cb.set_label("$u_{mod}$ [cm s$^{-1}$]")
ax3.set_xlim(xl)
fig.savefig("LeXPLORE_currents.png", dpi = 300)
plt.show()
plt.close("all")

fig, ax = plt.subplots(figsize = (6,4))
cc = ax.contourf(ADCP["date"],-z, np.log10(Ri), np.arange(-1,2.1,0.1), extend = "both", cmap = cm.hot)
ax.contour(ADCP["date"],-z, Ri, [1], colors = "green", linewidths = 0.5)
cb = plt.colorbar(cc, ax = ax)
cb.set_label("$\log_{10} Ri$")
plt.show()

j = 4234

print ADCP["date"][j]
fig = plt.figure()
plt.plot(ADCP["u"][:,j],-ADCP["z"],'.-')
plt.plot(us[:,j],-ADCP["z"],'--')
plt.plot(np.cumsum(um[:,:,j], axis = 0).T,-ADCP["z"],'-')
plt.plot(np.sum(um[:,:,j], axis = 0),-ADCP["z"],'k-')
plt.show()

