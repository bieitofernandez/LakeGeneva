import numpy as np
import datetime 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import matplotlib as mpl
import scipy.io as io
import datetime as dt
from scipy import signal
import scipy.interpolate as intrp
import pickle
import seawater as sw
import os
import sys
import glob
sys.path.insert(0, "/home/bieito/Documents/SCIENCE/EPFL/python_functions/")
from extract_Tmoor import *
from time_conversion import *
from useful_functions import first_centered_differences
import vertical_modes as VM
#from vertical_modes import *

def __fits_modes__(  Y, X):
    #X = np.c_[np.ones((Y.size,1)),X]
    #print X.shape
    jj = np.isfinite(Y)
    Nm = X.shape[1]
    if np.sum(jj)<0.75*jj.size:
        return np.full( Nm, 0.)
    Y0 = np.copy(Y)
    Y = Y[jj]
    X0 = np.copy(X)
    X = X[jj,:]
    
    invXTX = np.linalg.inv(np.dot(X.T,X))
    A = np.dot(np.dot(invXTX, X.T),Y)
    
    #mod = sm.OLS(Y,X).fit()
    #A = mod.params
    return A


LON = 6.657764
LAT = 46.50094
ELEV = 372

accT = 0.01
### PLOTS temperature time series
ST = pickle.load(open("smooth_temperature_evolution.pi"))
#TM = extracts_all_Tmoor("Tmoor")#pickle.load(open("Tmoor/temperature_mooring_Leman_all.pi"))


#########
## Reads and gridds the mooring
####

TM = extracts_all_Tmoor("Tmoor")#pickle.load(open("Tmoor/temperature_mooring_Leman_daily.pi"))
start_date = max(ST["timeGRs"][0], TM["time"][0])
end_date = np.floor(min(ST["timeGRs"][-1], TM["time"][-1])+1)#, RHONE["date"][-1])

dt = 60.
dt /=(60.*24.)
dts =30. #days
time = np.arange(start_date, end_date+dt,dt)
date =  convert_time_to_date(time)
nt = time.size
nz = TM["T"].shape[0]

z = np.copy(TM["depth"])
T = np.full((nz,nt), np.nan)
sT = np.full((nz,nt), np.nan)
dTdz = np.full((nz,nt), np.nan)

mindTdz = first_centered_differences( z,accT*np.arange(z.size))
mindTdz = np.tile(mindTdz, (nt,1)).T
for i in range(nt):
    ii = np.where( (TM["time"]>= time[i]-dt) & (TM["time"]<= time[i]+dt))[0]
    iis = np.where( (TM["time"]>= time[i]-dts) & (TM["time"]<= time[i]+dts))[0]
    for j in range(nz):
        T[j,i] = np.nanmean(TM["T"][j,ii])
        sT[j,i] = np.nanmean(TM["T"][j,iis])
    dTdz[:,i] = first_centered_differences(z, np.sort(sT[:,i])[::-1])

dTdzF = np.copy(dTdz)
dTdzF[np.abs(dTdz)<5*mindTdz] = np.nan
disp = - (T-sT)/dTdzF

OUT = dict()
OUT["time"] = np.copy(time)
OUT["date"] = np.copy(date)
OUT["z"] = np.copy(z)
OUT["T"] = np.copy(T)
OUT["sT"] = np.copy(sT)
OUT["dTdz"] = np.copy(dTdz)
OUT["dTdzF"] = np.copy(dTdzF)
OUT["mindTdz"] = np.copy(mindTdz)
OUT["disp"] = np.copy(disp)

pickle.dump(OUT,open("thermistor_mooring_gridded_1hour.pi","w+"))



IN = pickle.load(open("thermistor_mooring_gridded_1hour.pi"))
for var in IN.keys():
    exec("%s = np.copy(IN[\"%s\"])"%(var,var))

#dTdzF[np.abs(dTdz)<10*mindTdz] = np.nan
#disp = - (T-sT)/dTdzF

nt = time.size
nz = z.size
#modal decomposition
Nm = 2#ST["bcMOD"]["c"].shape[1]
Ad = np.full( (Nm,nt), 0. )
mD = np.full( nt, 0. )
dispm = np.full( (Nm,nz,nt), 0. )

time_steps = np.where( np.nanmean(np.abs(dTdz), axis = 0)>0.156 )[0]

for i in time_steps:
    iT = np.argmin( np.abs(time[i]-ST["timeGRs"]) )
    mD[i] = np.nanmean(disp[:,i])
    w = np.full( (nz,Nm), np.nan )
    for j in range(Nm):
        iif = np.where(np.isfinite(ST["bcMOD"]["w"][:,iT,j]))[0]
        intw = intrp.interp1d( ST["depth"], ST["bcMOD"]["w"][:,iT,j])
        w[:,j] = intw(z)
    Ad[:,i]= __fits_modes__(disp[:,i],w[:,:])
    #dispm[:,:,i] = np.c_[np.ones((w.shape[0],1))*Ad[0,i], (Ad[1:,i].T*w)].T
    dispm[:,:,i] = (Ad[:,i].T*w).T
    """
    fig, ax = plt.subplots()
    ax.plot( disp[:,i], -z,'.')
    ax.plot( mD[i]+np.cumsum(dispm[:,:,i], axis = 0).T,-z)
    plt.show()
    """

mPE = 0.5*1000*Ad**2
intmPE = np.cumsum(mPE, axis = 0)

IN["Ad"] = np.copy(Ad)
IN["mPE"] = np.copy(mPE)
IN["intmPE"] = np.copy(intmPE)
IN["dispm"] = np.copy(dispm)

pickle.dump(IN,open("thermistor_mooring_gridded_1hour_NM.pi","w+"))

MLDs = mixed_layer_depth(z, -sT, Dd = 0.5)

fig = plt.figure(figsize = (17,8))
ax0 = fig.add_axes([0.05,0.69, 0.50,0.28])
ax1 = fig.add_axes([0.05,0.37, 0.50,0.28])
ax2 = fig.add_axes([0.05,0.05, 0.50,0.28])
ax3 = fig.add_axes([0.53,0.69, 0.50,0.28])
ax4 = fig.add_axes([0.53,0.37, 0.50,0.28])
ax5 = fig.add_axes([0.53,0.05, 0.43,0.28])

ccT=ax0.contourf(date, z, sT, np.arange(5,28,1), cmap = cm.inferno)
plt.colorbar(ccT, ax = ax0)
ax0.plot(date, MLDs)
xl = ax0.get_xlim()
ax0.set_ylim((30,0))
ax0.set_xticklabels([])
ax0.set_ylabel("Depth [m]")
ax0.annotate("$T_s$ [$^{\circ}$C]", xy = (0.02,0.07), xycoords = "axes fraction", bbox = dict(facecolor = "w"))

ccTs=ax1.contourf(date, z, T, np.arange(5,28,1), cmap = cm.inferno)
plt.colorbar(ccTs, ax = ax1)
plt.plot(TM["date"], TM["MLDdif"],"w")
ax1.set_xlim(xl)
ax1.set_ylim((30,0))
ax1.set_xticklabels([])
ax1.set_ylabel("Depth [m]")
ax1.annotate("$T$ [$^{\circ}$C]", xy = (0.02,0.07), xycoords = "axes fraction", bbox = dict(facecolor = "w"))

ccdT=ax2.contourf(date, z,dTdz, np.arange(-1,0.05,0.05), cmap = cm.Blues)
plt.colorbar(ccdT, ax = ax2)
ax2.set_xlim(xl)
ax2.set_ylim((30,0))
ax2.set_ylabel("Depth [m]")
ax2.annotate("$dT/dz$ [$^{\circ}$C/m]", xy = (0.02,0.07), xycoords = "axes fraction", bbox = dict(facecolor = "w"))
#ax2.set_xticklabels([])

ccd=ax3.contourf(date, z, disp, np.arange(-30,32,2), cmap = cm.RdBu_r)
plt.colorbar(ccd, ax = ax3)
ax3.set_xlim(xl)
ax3.set_xticklabels([])
ax3.set_yticklabels([])
ax3.set_ylim((30,0))
ax3.annotate("$\\delta$ [m]", xy = (0.02,0.07), xycoords = "axes fraction", bbox = dict(facecolor = "w"))

ccd=ax4.contourf(date, z, np.sum(dispm, axis = 0), np.arange(-30,32,2), cmap = cm.RdBu_r)
plt.colorbar(ccd, ax = ax4)
ax4.set_xlim(xl)
ax4.set_xticklabels([])
ax4.set_yticklabels([])
ax4.set_ylim((30,0))
ax4.annotate("$\\delta_{NM}$ [m]", xy = (0.02,0.07), xycoords = "axes fraction", bbox = dict(facecolor = "w"))

l= []
for i in range(Nm):
    ii = Nm-1-i
    l0 = ax5.fill_between(date, intmPE[ii,:], color = cm.tab10(ii))
    l.append(l0)
    print ii
ax5.set_ylabel("PE [J m$^{-2}$]")
ax5.set_ylim((0,1000))
ax5.legend(l[::-1],["bc1","bc2","bc3"], ncol = 4)
ax5.yaxis.set_label_position("right")
ax5.yaxis.tick_right()
ax5.set_xlim(xl)

"""
ccd=ax[4].contourf(date, -z, np.sum(dispm, axis = 0), np.arange(-20,21,1), cmap = cm.RdBu_r)
plt.colorbar(ccd, ax = ax[4])
"""
fig.savefig("PE_energy.png", dpi = 300)
plt.close(fig)

    
#ADCP = pickle.load( open("ADCP_Geneva_all_2019_2020_smooth_2h_4m_NM.pi") )
ADCP = pickle.load( open("ADCP_Geneva_all_2019_2020_NM.pi") )

Au = np.full( (4,time.size), np.nan )
Av = np.full( (4,time.size), np.nan )
for i in range(4):
    intAu = intrp.interp1d( ADCP["time_r"], ADCP["NM"]["Au"][i,:], bounds_error = False )
    Au[i,:] = intAu(time)
    Au[i,~(intmPE[0,:]>0)] = 0.

    intAv = intrp.interp1d( ADCP["time_r"], ADCP["NM"]["Av"][i,:], bounds_error = False )
    Av[i,:] = intAv(time)
    Av[i,~(intmPE[0,:]>0)] = 0.

    
D_range = ST["depth"][-1] - ST["depth"][0]

fd = []
Sd = []
for Ad0 in Ad:
    f0,S0 = signal.welch( Ad0[time_steps]/np.sqrt(D_range), fs = 24., nperseg = 1024)
    Sd.append(0.5*S0)
    fd.append(f0)
Sd = np.array(Sd)
Sdt = np.nansum(Sd,axis = 0)

fu = []
Su = []
for Au0, Av0 in zip(Au,Av):
    f0,Su0 = signal.welch( Au0[time_steps]/np.sqrt(D_range), fs = 24., nperseg = 1024)
    Sv0 = signal.welch( Av0[time_steps]/np.sqrt(D_range), fs = 24., nperseg = 1024)[1]
    fu.append(f0)
    Su.append(0.5*Su0+0.5*Sv0)
fu = f0
Su = np.array(Su)
Sut = np.nansum(Su,axis = 0)

print np.trapz(Su, x=fu, axis = 1)*D_range*1000
print np.trapz(Sd, x=fd, axis = 1)*D_range*1000

#Kelvin wave
L = 62*1000. #basin length
TKB = 2*L/ST["bcMOD"]["c"][:,0]/60./60./24
Ro = ST["bcMOD"]["c"][:,0]/sw.f(46)/1000.

fig, ax = plt.subplots()
l = []
i=1
for (f0,S0) in zip(fd,Sd):
    l0,=ax.loglog(f0,S0, lw = 1, color = cm.tab10(i))
    i+=1
    l.append(l0)
l0,=ax.loglog(f0,Sdt, lw = 2, color = cm.tab10(0))
l.append(l0)
l0,=ax.loglog(f0,Sut, lw = 2, color = "k", ls = "-")
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

#l2,=ax.loglog(fw,Sw*1e-6, lw = 1, color = "k")
#l.append(l2)
ax.legend(l,["bc1","bc2","PE","KE"], loc = "lower left")

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
fig.savefig("Spectral_density_modes_PE.pdf", dpi = 300, bbox_inches = "tight")
plt.show()





KE0 = np.copy(ADCP["NM"]["KE"][0:,:])
timeAD = np.copy(ADCP["time_r"])

iif = np.isfinite(np.sum(KE0, axis = 0))

x = timeAD[iif]
y = KE0[:,iif]

bcKE = np.full( mPE.shape[1], np.nan )
iifT = mPE[0,:]>0
intbcKE = intrp.interp1d( timeAD[iif],np.nansum(KE0[1:,iif], axis = 0), bounds_error = False )
bcKE[iifT] = intbcKE(IN["time"][iifT]) 

KE = np.full( mPE.shape[1], np.nan )
intKE = intrp.interp1d( timeAD[iif],np.nansum(KE0[:,iif], axis = 0), bounds_error = False )
KE[iifT] = intKE(IN["time"][iifT])

tmPE = np.sum(mPE, axis = 0)
iif = (tmPE>0) & (KE>0)

x = bcKE[iif]
y = tmPE[iif]

pp1 = np.polyfit( np.log10(x), np.log10(y),1 )
pp2 = np.polyfit( np.log10(y), np.log10(x),1 )

pp2b = np.array([1/pp2[0], -pp2[1]/pp2[0]])
ppP1 = np.poly1d(pp1)
ppP2 = np.poly1d(pp2b)

fig = plt.figure()
plt.loglog( bcKE,tmPE,'.' )
plt.plot( plt.xlim(),  plt.ylim(),"k")
#plt.loglog( plt.xlim(), 10**ppP1(np.log10(plt.xlim()))  )
#plt.loglog( plt.xlim(), 10**ppP2(np.log10(plt.xlim()))  )

plt.show()

