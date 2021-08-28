import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cmocean
import matplotlib.colors as colors
import matplotlib.dates as mdates
import scipy.io as io
import copy
sys.path.insert(0, "/home/bieito/Documents/SCIENCE/EPFL/python_functions/")
import salinity_density as SD
from time_conversion import *
import scipy.interpolate as intrp
from scipy import optimize
from useful_functions import *
import seawater as sw
from vertical_modes import *

def generates_random_matrix(x,sx):
    rx = np.full(x.shape, np.nan)
    d1,d2 = x.shape
    for i in range(d1):
        for j in range(d2):
            if np.isnan(x[i,j]) or np.isnan(sx[i,j]):
                continue
            rx[i,j] = x[i,j] + sx[i,j]*np.random.randn()
    return rx

def harmonic_func(t,*a):
    if len(a)%2 == 0:
        F = a[0]+a[1]*t #linear term
        ind = 2
    else:
        F = a[0]
        ind = 1
    i = 1.
    while ind+1<len(a):
        F += a[ind]*np.cos(2*np.pi*(t/(365./i)+a[ind+1]))
        ind+=2
        i+=1.
    return F


def smooth_gradient(X,Y,N):
    nf, nc = Y.shape
    dYdX = np.copy(Y)
    for i in range(nc):
        x = Y[:,i]
        for j in range(N//2,nf-N//2):
            ii = np.arange(j - N//2, j+ N//2+1)
            y = Y[ii,i]
            x = X[ii,i]
            pp = np.polyfit(x,y,1)
            dYdX[j,i] = pp[0]
    dYdX[0:N//2,:] = dYdX[N//2,:]
    dYdX[(nf-N//2):,:] = dYdX[nf-N//2,:]
    return dYdX

def deep_T(X,x0,x1,y0,y1 = 999.9677, shape = "inverse"):
    if shape == "inverse":
        b = (y0-y1)*( x0*x1 )/(x1-x0)
        a = y1 - b/x1
        yM = a+b/X
    elif shape == "linear":
        yM = y0 + (y1-y0)/(x1-x0)*(X-x0)
    return yM


micro = pickle.load( open("CTD_micro/microstructure_down_Lac_Leman.pi"))
CTD = pickle.load( open("CTD_micro/gridded_CTD_Leman_profiles.pi") )
TM = pickle.load(open("Tmoor/temperature_mooring_Leman_all_daily.pi"))
#DOM = pickle.load(open("DO/DO_mooring_Leman_all_daily.pi"))

nT= CTD["time"].size + micro["date"].size + TM["time"].size

flgCTD = np.ones(CTD["time"].size)
flgmicro = 2*np.ones(micro["date"].size)
flgmoor = 3*np.ones(TM["time"].size)
flg = np.r_[flgCTD,flgmicro, flgmoor]

indexCTD = np.arange( CTD["time"].size )
indexmicro = np.arange( micro["date"].size)
indexmoor = np.arange( TM["time"].size)
index = np.r_[indexCTD,indexmicro, indexmoor]

time = np.r_[CTD["time"]-366, micro["date"][0,:]-366, TM["time"]]
iisort = np.argsort(time)
flg = flg[iisort]
time = time[iisort]
index = index[iisort]
date = convert_time_to_date(time)

dd = 0.5
depth = np.arange(0.5,110,0.5)
Nd = depth.size

#ipsometric function
BAT = io.loadmat("bath/BathGeneva.mat")
bat = -BAT["Bathymetry"]
dx = 25.#*1000#grid is 25 m
Area = np.full(depth.size,np.nan)
for (i,z) in enumerate(depth):
    Area[i] = np.sum(bat>=z-dd)*dx**2
zCM = np.sum( depth*Area )/np.sum(Area)#center of mass

#including all depths
depth_deep = np.arange(depth[0],310,dd)
Area_deep = np.full(depth_deep.size,np.nan)
for (i,z) in enumerate(depth_deep):
    Area_deep[i] = np.sum(bat>=z-dd)*dx**2
zCM_deep = np.sum( depth_deep*Area_deep )/np.sum(Area_deep)#center of mass

#merges data together
z = np.tile(depth,(nT,1)).T
T = np.full( (Nd,nT), np.nan )
S = np.full( (Nd,nT), np.nan )
DO = np.full( (Nd,nT), np.nan )
calcS = False
calcDO = False
calcT = False
for i in np.arange(nT):
    #deteles some profiles that produce spikes shit
    #if int(time[i]) == dt.datetime(2019,7,9,0,0).toordinal() or int(time[i]) == dt.datetime(2019,7,17,0,0).toordinal() or int(time[i]) == dt.datetime(2019,11,5,0,0).toordinal():
    #    continue
    if flg[i] == 1:
        z0 = np.copy(CTD["depth"])
        T0 = CTD["Temp"][:, index[i]]
        S0 = CTD["S"][:, index[i]]
        DO0 = CTD["DO_mg"][:, index[i]]
        calcT = True
        calcS = True
        calcDO = True
    elif flg[i]==2:
        z0 = np.copy(micro["depth"][:,index[i]])
        T0 = micro["T"][:, index[i]]
        S0 = micro["S"][:, index[i]]
        calcT = True
        calcS = True
        calcDO = False
    elif flg[i]==3:
        z0 = np.copy(TM["depth"])
        T0 = TM["T"][:,index[i]]
        calcT = True
        calcS = False
        calcDO = False

    if calcT :
        iif = np.isfinite(T0)
        if np.sum(iif) == 0:
            continue
        z0 = z0[iif]
        T0 = T0[iif]
        
        intT = intrp.interp1d(z0,T0, bounds_error = False)
        T[:,i] = intT(depth)

    if calcS:
        iif = np.isfinite(S0)
        if np.sum(iif) == 0:
            continue
        S0 = S0[iif]
        intS = intrp.interp1d(z0,S0, bounds_error = False)
        S[:,i] = intS(depth)

    if calcDO:
        iif = np.isfinite(DO0)
        if np.sum(iif) == 0:
            continue
        DO0 = DO0[iif]
        intDO = intrp.interp1d(z0,DO0, bounds_error = False)
        DO[:,i] = intDO(depth)

#density
den = SD.density(T, S, lake = "Leman")
ddendz = FCD_2d(- z, np.sort(den, axis = 0) )
N2 = - sw.g(46)/den*ddendz
#stability
Sc = np.full( nT, np.nan )
for i in range(nT):
    ngood = np.sum(np.isfinite(den[:,i]))
    if ngood < 30./dd:
        continue
    #Sc[i] = np.nansum( sw.g(46.)*( den[:,i] - np.nanmean(den[:,i]) )*(depth-zCM)*dd*Area  )/Area[0]
    #ii = np.where( np.isfinite(den[:,i]) )[0]
    #mden = np.nansum(den[ii,i]*Area[ii])/np.nansum(Area[ii])
    #Sc[i] = np.nansum( sw.g(46.)*( den[:,i] - mden )*(depth)*dd*Area  )/Area[0]

    den0 = den[:,i]
    z0 = depth
    jj = np.where(np.isfinite(den0))[0]
    den0 = den0[jj]
    z0 = z0[jj]
    den1 = np.full(depth_deep.size, np.nan)
    den1[jj] = den0
    j0 = jj[-1]
    den1[j0+1:] = deep_T( depth_deep[j0+1:], z0[-1], depth_deep[-1], den0[-1],1000.1537)
    Ar = np.copy(Area_deep)
    Ar[np.isnan(den1)] = np.nan

    Sc[i] = np.nansum( 9.81*( den1 - np.nansum(den1*Ar)/np.nansum(Ar) )*(depth_deep-zCM_deep)*dd*Area_deep)/Area_deep[0]
    
    #Sc[i] = np.nansum( sw.g(46.)*( den[:,i] )*(depth-zCM)*dd*Area  )/Area[0]


timeR = convert_date_to_time(date)
timeGR = np.arange( int(np.min(timeR)), int(np.max(timeR))+1 )
#start_date = dt.datetime(2019,05,1,0,0).toordinal()
#timeGR = np.arange( start_date, int(np.max(timeR))+1 )
#timeGR = np.arange( int(np.min(micro["date"]-366))-1, int(np.max(timeR))+1 )
dateGR = convert_time_to_date(timeGR)

nTGR = timeGR.size
zGR = np.tile(depth,(timeGR.size,1)).T
AreaGR = np.tile(Area,(timeGR.size,1)).T
TGR = np.full( (Nd,timeGR.size), np.nan )
SGR = np.full( (Nd,timeGR.size), np.nan )
DOGR = np.full( (Nd,timeGR.size), np.nan )

for i in range(timeGR.size):
    ii = np.where( np.floor(timeR) == timeGR[i] )[0]
    TGR[:,i] = np.nanmean(T[:,ii], axis = 1)
    SGR[:,i] = np.nanmean(S[:,ii], axis = 1)
    DOGR[:,i] = np.nanmean(DO[:,ii], axis = 1)
#density
denGR = SD.density(TGR, SGR, lake = "Leman")

ddenGRdz = FCD_2d(- zGR, np.sort(denGR, axis = 0) )
N2GR = - sw.g(46)/denGR*ddenGRdz

#################
#smoothed fields
#################
A00 = [10.,10]
PLOT = False

time00 = dt.datetime(2019,4,15,0,0,0).toordinal()
time01 = dt.datetime(2020,4,15,0,0,0).toordinal()
timeGRs = np.arange( time00, time01 )
#timeGRs = np.arange( int(np.min(micro["date"]-366))-1, int(np.max(timeR))+1 )
#timeGRs = np.arange( int(np.min(micro["date"]-366))-1, int(np.max(micro["date"]-366))+1 )
nTGRs = timeGRs.size
#timeGRs = np.arange(start_date, int(np.max(timeR))+1 )
dateGRs = convert_time_to_date(timeGRs)
zGRs = np.tile(depth,(timeGRs.size,1)).T
AreaGRs = np.tile(Area,(timeGRs.size,1)).T

TGRs = np.full( (Nd,timeGRs.size), np.nan )
#eTGRs = np.full( (2,Nd,timeGRs.size), np.nan )
eTGRs = np.full( (Nd,timeGRs.size), np.nan )
SGRs = np.full( (Nd,timeGRs.size), np.nan )
eSGRs = np.full( (Nd,timeGRs.size), np.nan )
denGRs = np.full( (Nd,timeGRs.size), np.nan )
edenGRs = np.full( (Nd,timeGRs.size), np.nan )
DOGRs = np.full( (Nd,timeGRs.size), np.nan )
eDOGRs = np.full( (Nd,timeGRs.size), np.nan )

nmtc = 200

for iz in range(1,Nd):
    print iz
    y1 = TGR[iz,:]
    y2 = SGR[iz,:]
    y3 = DOGR[iz,:]
    x = np.copy(timeGR)
    
    if depth[iz]>=30:
        iifT = (np.isfinite(y1)) & (timeGR>=np.min( micro["date"]-366)-1)
        iifS = (np.isfinite(y1)) & ( np.isfinite(y2))
        iifDO = (np.isfinite(y1)) & ( np.isfinite(y3))
        NharmsT = 0
        NharmsS = 0
        NharmsDO = 0
        DT_smooth = 2.*30
        if depth[iz]>50:
            NharmsT = 0
            NharmsS = 0
            NharmsDO = 0
            DT_smooth = 3.*30
        if depth[iz]>90:
            NharmsT = 0
            NharmsS = 0
            NharmsDO = 0
            DT_smooth = 5.*30

    else:
        iifT = (np.isfinite(y1)) #& (timeGR>=np.min( micro["date"]-366)-1) 
        iifS = (np.isfinite(y1)) &  (np.isfinite(y2))
        iifDO = (np.isfinite(y1)) &  (np.isfinite(y3)) 
        NharmsT = 0
        NharmsS = 0
        NharmsDO = 0
        DT_smooth = 1.5*30
        
    if np.sum(iifT)<0.05*iifT.size:
        continue
    
    A0T = A00 + [1.,90]*NharmsT
    A0T = tuple(A0T)
    A0S = A00 + [1.,90]*NharmsS
    A0S = tuple(A0S)
    A0DO = A00 + [1.,90]*NharmsDO
    A0DO = tuple(A0DO)
    
    y1 = y1[iifT]
    x1 = x[iifT]
    iis = np.argsort(x1)
    x1 = x1[iis]
    y1 = y1[iis]

    y2 = y2[iifS]
    x2 = x[iifS]
    iis = np.argsort(x2)
    x2 = x2[iis]
    y2 = y2[iis]

    y3 = y3[iifDO]
    x3 = x[iifDO]
    iido = np.argsort(x3)
    x3 = x3[iido]
    y3 = y3[iido]
    #temperature
    if NharmsT == 0:
        XT = np.full( (nmtc,timeGRs.size), np.nan  )
        for it in range(timeGRs.size):
            jj = np.arange(x1.size)#np.where( np.abs(timeGRs[it]-x1)<2*DT_smooth)[0]
            w = np.exp(-np.abs(timeGRs[it]-x1[jj])**2/(0.5*DT_smooth)**2)
            w /= float(np.nansum(w))
            TGRs[iz,it] = np.nansum(w*y1[jj])#np.nanmean(y1[jj])
            nn = float(jj.size)
            eTGRs[iz,it] = np.nanstd(y1[jj])/nn**0.5

    else:
        params, params_cov = optimize.curve_fit(harmonic_func, x1, y1, p0=A0T)
        XT = np.full((nmtc,timeGRs.size), np.nan)
        for i in range(nmtc):
            PP = np.random.multivariate_normal(params,params_cov)
            XT[i,:] = harmonic_func(timeGRs,*PP)
        TGRs[iz,:] =  np.nanmedian(XT, axis = 0)#harmonic_func(timeGR, *params)
        eTGRs[iz,:] =  np.nanstd(XT, axis = 0)
            
    #salinity
    if np.sum(iifS)<0.05*iifS.size:
        continue
    if NharmsS == 0:
        for it in range(timeGRs.size):
            jj = np.arange(x2.size)#np.where( np.abs(timeGRs[it]-x1)<2*DT_smooth)[0]
            w = np.exp(-np.abs(timeGRs[it]-x2[jj])**2/(0.5*DT_smooth)**2)
            w /= float(np.nansum(w))
            SGRs[iz,it] = np.nansum(w*y2[jj])#np.nanmean(y1[jj])
            nn = float(jj.size)
            eSGRs[iz,it] = np.nanstd(y2[jj])/nn**0.5
            
    else:
        params, params_cov = optimize.curve_fit(harmonic_func, x2, y2, p0=A0S)
        XS = np.full((nmtc,timeGRs.size), np.nan)
        for i in range(nmtc):
            PP = np.random.multivariate_normal(params,params_cov)
            XS[i,:] = harmonic_func(timeGRs,*PP)
        SGRs[iz,:] =  np.nanmedian(XS, axis = 0)#harmonic_func(timeGR, *params)
        eSGRs[iz,:] =  np.nanstd(XS, axis = 0)
    #XD =  SD.density(XT, XS, lake = "Leman")
    denGRs[iz,:] = SD.density(TGRs[iz,:], SGRs[iz,:], lake = "Leman")#np.nanmean(XD, axis = 0)

    #dissolved oxygen
    if NharmsDO == 0:
        XDO = np.full( (nmtc,timeGRs.size), np.nan  )
        for it in range(timeGRs.size):
            jj = np.arange(x3.size)#np.where( np.abs(timeGRs[it]-x3)<2*DT_smooth)[0]
            w = np.exp(-np.abs(timeGRs[it]-x3[jj])**2/(0.5*DT_smooth)**2)
            w /= float(np.nansum(w))
            DOGRs[iz,it] = np.nansum(w*y3[jj])#np.nanmean(y1[jj])
            nn = float(jj.size)
            eDOGRs[iz,it] = np.nanstd(y3[jj])/nn**0.5

    else:
        params, params_cov = optimize.curve_fit(harmonic_func, x3, y3, p0=A0DO)
        XDO = np.full((nmtc,timeGRs.size), np.nan)
        for i in range(nmtc):
            PP = np.random.multivariate_normal(params,params_cov)
            XDO[i,:] = harmonic_func(timeGRs,*PP)
        DOGRs[iz,:] =  np.nanmedian(XDO, axis = 0)#harmonic_func(timeGR, *params)
        eDOGRs[iz,:] =  np.nanstd(XDO, axis = 0)
        
    
    
    

    if PLOT and iz%10 == 0:
        fig = plt.figure()
        plt.plot(x1, y1,'.', color = "gray")
        plt.plot(dateGRs, TGRs[iz,:],'k-')
        #plt.plot(dateGRs, TGRs[iz,:]-2*eTGRs[iz,:],'--', color = "gray")
        #plt.plot(dateGRs, TGRs[iz,:]+2*eTGRs[iz,:],'--', color = "gray")
        plt.fill_between(dateGRs, TGRs[iz,:]-2*eTGRs[iz,:], TGRs[iz,:]+2*eTGRs[iz,:], color = "gray", alpha = 0.5)
        plt.title(depth[iz])
        plt.ylabel("T")
        #plt.show()

        fig = plt.figure()
        plt.plot(x3, y3,'.', color = "gray")
        plt.plot(dateGRs, DOGRs[iz,:],'k-')
        #plt.plot(dateGRs, TGRs[iz,:]-2*eTGRs[iz,:],'--', color = "gray")
        #plt.plot(dateGRs, TGRs[iz,:]+2*eTGRs[iz,:],'--', color = "gray")
        plt.fill_between(dateGRs, DOGRs[iz,:]-2*eDOGRs[iz,:], DOGRs[iz,:]+2*eDOGRs[iz,:], color = "gray", alpha = 0.5)
        plt.title(depth[iz])
        plt.ylabel("DO")
        plt.show()
    

#adiabatical sorting
for i in range(nTGRs):
    ii = np.where(np.isfinite(denGRs[:,i]))[0]
    jj = np.argsort(denGRs[ii,i])
    denGRs[ii,i] = denGRs[ii[jj],i]
    edenGRs[ii,i] = edenGRs[ii[jj],i]
    jj = np.argsort(TGRs[ii,i])[::-1]
    TGRs[ii,i] = TGRs[ii[jj],i]
    eTGRs[ii,i] = eTGRs[ii[jj],i]


fig, ax = plt.subplots(2,2, figsize = (22,8), sharex = True, sharey = True)
cc = ax[0,0].pcolor(dateGR,depth,TGR, vmin = 6, vmax = 25, cmap = cm.plasma)
plt.colorbar(cc, ax = ax[0,0])
ax[0,0].set_ylim((100,0))
ax[0,0].set_ylabel("Depth [m]")
cc=ax[0,1].contourf(dateGRs, depth, TGRs, np.arange(5,26,1),  vmin = 6, vmax = 25, cmap = cm.plasma)
cb=plt.colorbar(cc, ax = ax[0,1])
cb.set_label("Temperature [$^{\circ}$]")
ccD=ax[0,1].contour(dateGRs, depth, denGRs-1000, np.arange(-2.25,0.25,0.25), colors = "white", linestyles = "solid", linewidths = 1)
plt.clabel(ccD, fmt = "%1.1f")
cc = ax[1,0].pcolor(dateGR,depth,SGR, vmin = 0.2, vmax = 0.26,  cmap = cm.viridis)
plt.colorbar(cc, ax = ax[1,0])
ax[1,0].set_ylim((100,0))
ax[1,0].set_ylabel("Depth [m]")
cc=ax[1,1].contourf(dateGRs, depth, SGRs, np.arange(0.21,0.257,0.002),cmap = cm.viridis)
cb=plt.colorbar(cc, ax = ax[1,1])
cb.set_label("Salinity")
ax[1,0].xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
plt.savefig("Temperature_reconstruction.png", dpi = 300, bbox_inches = "tight")
plt.show()
plt.close(fig)



#density and N2
minT = np.nanmin(TGRs)
#dTGRsdz_0 = FCD_2d(-zGRs, TGRs )
dTGRsdz = smooth_gradient(-zGRs, TGRs,10 )
ddenGRsdz = smooth_gradient(-zGRs, denGRs,10)
N2GRs = - sw.g(46)/denGRs*ddenGRsdz

HC = np.nancumsum( denGRs[::-1,:]*sw.cp(SGRs[::-1,:],TGRs[::-1,:],zGRs[::-1,:])*(TGRs[::-1,:]-minT)*dd*AreaGRs[::-1,:], axis = 0)/Area[0]
HC = HC[::-1,:]

dHdt = FCD_2d( np.tile(timeGRs*60*60*24,(depth.size,1)),HC, axis = 1)
Kz = dHdt/dTGRsdz/denGRs/sw.cp(SGRs,TGRs,zGRs)


ScGRs = np.full( timeGRs.size, np.nan )
for i in range(ScGRs.size):
    den0 = denGRs[:,i]
    z0 = zGRs[:,i]
    jj = np.where(np.isfinite(den0))[0]
    den0 = den0[jj]
    z0 = z0[jj]
    den1 = np.full(depth_deep.size, np.nan)
    den1[jj] = den0
    j0 = jj[-1]
    den1[j0+1:] = deep_T( depth_deep[j0+1:], z0[-1], depth_deep[-1], den0[-1],1000.1537)
    Ar = np.copy(Area_deep)
    Ar[np.isnan(den1)] = np.nan

    ScGRs[i] = np.nansum( 9.81*( den1 - np.nansum(den1*Ar)/np.nansum(Ar) )*(depth_deep-zCM_deep)*dd*Area_deep)/Area_deep[0]
    
    
    
HCinv = np.tile(HC[0,:],(depth.size,1)) - HC
    
fig, ax = plt.subplots(2,1,figsize = (8,8), sharex = True)
ax[0].plot(date, Sc,'o', color = "gray", ms = 3, mfc = "none")
ax[0].plot(dateGRs, ScGRs,'k-',lw = 3)
#ax[0].fill_between(dateGRs,ScGRs-2*eScGRs,ScGRs+2*eScGRs, color = "gray", alpha = 0.5)
ax[0].set_ylabel("$Sc$ [J m$^{-2}$]")

normalize = colors.Normalize(vmin=depth.min(), vmax=depth.max())
#ax[1].fill_between(dateGR, (HC[0,:])*1e-9, color = cmocean.cm.deep(normalize(depth[0])))
for i in np.arange(200,-20,-20):
    print i
    ax[1].fill_between(dateGRs, HCinv[i,:]*1e-9, color = cmocean.cm.deep(normalize(depth[i])))
ax[1].plot(dateGRs, HCinv[::20,:].T*1e-9, color = "k", lw = 0.5)
ax[1].set_ylabel("$H$ [$10^9$ J m$^{-2}$]")
#plt.colorbar(clrs)
ax[1].set_xlim((dt.datetime(2019,2,1,0,0,0),dt.datetime(2020,4,1,0,0,0)))
for i in [20,40,60,80,200]:
    print i
    ax[1].annotate("%d m"%(depth[i]), xy = (dateGRs[160], HCinv[i,160]*1e-9), horizontalalignment = "center", verticalalignment = "middle", fontsize = 7)
plt.savefig("lake_stability_Leman.png", dpi = 300,bbox_inches = "tight")
plt.close(fig)

plt.pcolor(dateGRs,-depth, np.log10(Kz), vmin = -7, vmax = -4)
plt.show(block = False)

#vertical modes
nNM = 3
NMw = np.full((Nd, nTGRs,nNM), np.nan)
NMv = np.full((Nd, nTGRs,nNM), np.nan)
NMlb = np.full((nTGRs,nNM), np.nan) 
for i in range(timeGRs.size):
    iif = np.where(np.isfinite(N2GRs[:,i]))[0]
    if iif.size == 0:
        continue
    iif2 = np.r_[0,iif]
    NMlb[i,:],NMv[iif2,i,:], NMw[iif2,i,:] = vertical_modes(depth[iif], N2GRs[iif,i], nNM)
    print "vertical modes  %s, c = %1.3f m/s"%(dateGRs[i].strftime("%Y %b %d"), np.sqrt(1/NMlb[i,0]))

L = 62*1000. #basin length
NMc = NMlb**-0.5
TKB = 2*L/NMc[:,0]/60./60./24
Rr = NMc[:,0]/sw.f(46)/1000.



OUT = dict()
OUT["date"] = date
OUT["time"] = time
OUT["depth"] = depth
OUT["T"] = T
OUT["S"] = S
OUT["den"] = den
OUT["N2"] = N2
OUT["Area"] = Area

OUT["dateGR"] = dateGR
OUT["timeGR"] = timeGR
OUT["TGR"] = TGR
OUT["SGR"] = SGR
OUT["denGR"] = denGR
OUT["N2GR"] = N2GR
OUT["DOGR"] = DOGR
OUT["AreaGR"] = AreaGR

OUT["dateGRs"] = dateGRs
OUT["timeGRs"] = timeGRs
OUT["TGRs"] = TGRs
OUT["SGRs"] = SGRs
OUT["denGRs"] = denGRs
OUT["dTdz"] = dTGRsdz
OUT["N2GRs"] = N2GRs
OUT["DOGRs"] = DOGRs
OUT["eTGRs"] = eTGRs
OUT["eSGRs"] = eSGRs
OUT["edenGRs"] = edenGRs
OUT["eDOGRs"] = eTGRs

OUT["AreaGRs"] = AreaGRs

OUT["HCGR"] = HC
OUT["Sc"] = Sc
OUT["ScGR"] = ScGRs

OUT["bcMOD"] = dict()
OUT["bcMOD"]["lb"]= NMlb
OUT["bcMOD"]["c"]= NMc
OUT["bcMOD"]["v"]= NMv
OUT["bcMOD"]["w"]= NMw


pickle.dump(OUT, open("smooth_temperature_evolution.pi","w+"))





