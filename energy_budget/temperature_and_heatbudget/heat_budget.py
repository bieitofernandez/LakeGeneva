import numpy as np
import datetime 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib as mpl
import scipy.io as io
import datetime as dt
from scipy import signal
from scipy.stats import norm
import scipy.interpolate as intrp
import pickle
import seawater as sw
import os
import sys
import glob
sys.path.insert(0, "/home/bieito/Documents/SCIENCE/EPFL/python_functions/")
from scipy import optimize
from time_conversion import *
from useful_functions import *
#from vertical_modes import *
import xlrd
from get_colormap import get_colormap
CMCV = get_colormap('/home/bieito/Documents/SCIENCE/EPFL/python_functions/WhiteBlueGreenYellowRed.ncmap')
import cmocean


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


"""
##secchi depth
wb = xlrd.open_workbook("secchi/transparency_data_2000-2016-original.xlsx")
print wb.sheet_names()
sh = wb.sheet_by_index(0)
sh.nrows
sd_datexls0 = [sh.cell_value(i,3) for i in range(1,sh.nrows)] 
sd_date = [dt.datetime(*xlrd.xldate_as_tuple(date0, wb.datemode)) for date0 in sd_datexls0]
sd_time = np.array(convert_date_to_time(sd_date))
sd_sd = np.array([])
for i in range(1, sh.nrows):
    x = sh.cell_value(i,6)
    if type(x) == float:
        sd_sd = np.r_[sd_sd, np.float(x)]
    else:
        sd_sd = np.r_[sd_sd, np.nan]

ii = np.isfinite(sd_sd)
sd_dateF = [sd_date[i] for i in np.where(ii)[0]]
sd_sd = sd_sd[ii]
sd_time = sd_time[ii]

params, params_cov = optimize.curve_fit(harmonic_func, sd_time, sd_sd, p0=[5,1,5,180,5])
sd_harm = harmonic_func(sd_time, *params)
secchi = harmonic_func(ST["timeGRs"], *params)

"""
#kd from mooring
kd_table = open("secchi/Kd_2018-10-12_to_2020-02-12.csv")
kd0_time = []
kd0 = []
i = 0
for line in kd_table.readlines():
    i+=1
    if i<2:
        continue
    lineS = line.split(";")
    kd0_time.append(datetime.datetime.strptime(lineS[0],'%Y-%m-%d').toordinal())
    kd0.append(float(lineS[1]))
kd0_time = np.array(kd0_time)
kd0 = np.array(kd0)
params, params_cov = optimize.curve_fit(harmonic_func, kd0_time, kd0, p0=[5,1,5,180,5])
kd_harm = harmonic_func(kd0_time, *params)

LON = 6.657764
LAT = 46.50094
ELEV = 372

### Loads temperature
ST = pickle.load(open("smooth_temperature_evolution.pi"))

fig, ax = plt.subplots(2,1, figsize = (8,8), sharex = True, sharey = True)
cc = ax[0].pcolor(ST["dateGR"],ST["depth"],ST["TGR"], vmin = 6, vmax = 25, cmap = cm.inferno)
ax[0].set_ylim((100,0))
ax[0].set_ylabel("Depth [m]")
cb=plt.colorbar(cc, ax = ax[0])
cb.set_label("Temperature [$^{\circ}$C]")
cc=ax[1].contourf(ST["dateGRs"], ST["depth"], ST["TGRs"], np.arange(5,26,1),  vmin = 6, vmax = 25, cmap = cm.inferno)

ccD=ax[1].contour(ST["dateGRs"], ST["depth"], ST["denGRs"]-1000, np.arange(-2.25,0.25,0.25), colors = "white", linestyles = "solid", linewidths = 1)
plt.clabel(ccD, fmt = "%1.1f")
ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
ax[1].set_xlim((ST["dateGRs"][0], ST["dateGRs"][-1]))
ax[1].set_ylabel("Depth [m]")
ax[1].tick_params(axis='x', rotation=45)
cb=plt.colorbar(cc, ax = ax[1])
cb.set_label("Temperature [$^{\circ}$C]")
plt.savefig("Temperature_cycle.pdf", dpi = 300, bbox_inches = "tight")
plt.show()
plt.close(fig)


HCinv = np.tile(ST["HCGR"][0,:],(ST["depth"].size,1)) - ST["HCGR"]
fig, ax = plt.subplots(2,1,figsize = (7,7), sharex = True)
ax[1].plot(ST["date"], ST["Sc"],'o', color = "gray", ms = 3, mfc = "none")
ax[1].plot(ST["dateGRs"], ST["ScGR"],'k-',lw = 3)
ax[1].set_ylabel("$Sc$ [J m$^{-2}$]")

normalize = colors.Normalize(vmin=ST["depth"].min(), vmax=ST["depth"].max())
#ax[1].fill_between(dateGR, (HC[0,:])*1e-9, color = cmocean.cm.deep(normalize(depth[0])))
for i in np.arange(200,-20,-20):
    print i
    ax[0].fill_between(ST["dateGRs"], HCinv[i,:]*1e-9, color = cmocean.cm.deep(normalize(ST["depth"][i])))
ax[0].plot(ST["dateGRs"], HCinv[::20,:].T*1e-9, color = "k", lw = 0.5)
ax[0].set_ylabel("$H$ [$10^9$ J m$^{-2}$]")
#plt.colorbar(clrs)
ax[0].set_xlim((dt.datetime(2019,2,1,0,0,0),dt.datetime(2020,4,1,0,0,0)))
for i in [20,40,60,80,200]:
    print i
    ax[0].annotate("%d m"%(ST["depth"][i]), xy = (ST["dateGRs"][160], HCinv[i,160]*1e-9), horizontalalignment = "center", verticalalignment = "middle", fontsize = 10)
plt.savefig("lake_stability_Leman.pdf", dpi = 300,bbox_inches = "tight")
plt.close(fig)

        

time = ST["timeGRs"].astype(float)
date = ST["dateGRs"]
z = np.copy(ST["depth"])
d_z = z[1]-z[0]
ST["MLD"] = mixed_layer_depth(ST["depth"], -ST["TGRs"], Dd = 0.5)
zM = np.tile(z,(time.size,1)).T
cp = sw.cp(ST["SGRs"],ST["TGRs"],zM)
alpha = sw.alpha(ST["SGRs"],ST["TGRs"],zM)

MMLD = np.tile( ST["MLD"], (z.size,1) )
TML = np.copy(ST["TGRs"])
#TMLD[ zM> np.maximum(10.,MMLD)] = np.nan
TML[ zM> MMLD] = np.nan
mTML = np.nanmean(TML, axis = 0)
denML = np.copy(ST["denGRs"])
denML[ zM> MMLD] = np.nan
mdenML = np.nanmean(denML, axis = 0)

nt = ST["timeGRs"].size
ST["THD"] = np.full(nt, np.nan)
for i in range(nt):
    N2 = np.copy(ST["N2GRs"][:,i])
    if np.isnan(N2).all():
        continue
    ST["THD"][i] = ST["depth"][np.nanargmax(N2)]

#radiative flux
kd = -harmonic_func(ST["timeGRs"], *params)

"""
COSMO = pickle.load(open("meteo/COSMO_Leman_all_MO.pi"))
nt = time.size
I0 = np.full(nt, np.nan)
sigma = 15.
for i in range(nt):
    jj = np.where(np.abs(COSMO["datenum"]-time[i])<2*sigma)
    x = COSMO["sHFluxes"]["Qsw"][jj]
    dd = np.abs(COSMO["datenum"][jj]-time[i])
    ww = np.exp(- dd**2/sigma**2)
    ii = np.where(np.isfinite(x))
    I0[i] = np.nansum(ww[ii]*x[ii])/np.nansum(ww[ii])
"""


MET = pickle.load(open("meteo/Meteo_Leman_all_MO.pi"))
nt = time.size
I0 = np.full(nt, np.nan)
sigma = 15
for i in range(nt):
    jj = np.where(np.abs(MET["datenum"]-time[i])<2*sigma)
    x = MET["sHFluxes"]["Qsw"][jj]
    dd = np.abs(MET["datenum"][jj]-time[i])
    ww = np.exp(- dd**2/sigma**2)
    ii = np.where(np.isfinite(x))
    I0[i] = np.nansum(ww[ii]*x[ii])/np.nansum(ww[ii])

iif = np.isfinite(I0)
params, params_cov = optimize.curve_fit(harmonic_func, time[iif], I0[iif], p0=[5,180]*4)
I0s = harmonic_func(time,*params)

dHdt = FCD_2d(np.tile(time*60*60*24,(z.size,1)),ST["HCGR"], axis = 1)
#kd = np.log(0.1)/secchi
#kd = -1.44/secchi #Schwefel et al. (2016)
Iz = np.tile(I0s,(z.size,1))*np.exp( np.tile(kd,(z.size,1))* zM)

#rhone heat flux
RH = pickle.load(open("meteo/rhone_heat_budget.pi"))
ii = np.where(RH["time"]>dt.datetime(2018,9,1,0,0,0).toordinal())[0]
paramsQ, params_covQ = optimize.curve_fit(harmonic_func, RH["time"][ii], RH["Q_in"][ii], p0=[1]+[5,180]*4)
RH_Q_in = harmonic_func(time,*paramsQ)
paramsT, params_covT = optimize.curve_fit(harmonic_func, RH["time"][ii], RH["T_in"][ii], p0=[1]+[5,180]*4)
RH_T_in = harmonic_func(time,*paramsT)

#calculates the dispersion in temperature
RH_T_in0 = harmonic_func(RH["time"],*paramsT)
dT2 = ( RH["T_in"] - RH_T_in0)**2
paramssT, params_covsT = optimize.curve_fit(harmonic_func, RH["time"][ii], dT2[ii], p0=[5,180]*4)
RH_sT_in = harmonic_func(time,*paramssT)**0.5

#uses the turbidity peak to calculate the dilution
micro = pickle.load( open("CTD_micro/microstructure_down_Lac_Leman.pi"))
micro["time"] = micro["date"][0,:]-366
micro["date_dt"] = convert_time_to_date(micro["time"])


T_intr = np.full( micro["time"].size, np.nan )
d_intr = np.full( micro["time"].size, np.nan )
for i in range(micro["time"].size):
    if (np.nanmax( micro["Turb"][:,i])<4.) or (np.nanmin( micro["Turb"][:,i])>3.):
        continue
    imax = np.nanargmax( micro["Turb"][:,i] )
    if micro["depth"][imax,i]<3:
        continue
    T_intr[i] = micro["T"][imax,i]
    d_intr[i] = micro["depth"][imax,i]

Trhone = harmonic_func(micro["time"],*paramsT)
intTML = intrp.interp1d(ST["timeGRs"], mTML, bounds_error = False)
TML0 = intTML(micro["time"].astype(int))
#TML0 = np.nanmax(micro["T"], axis = 0)

fig, ax = plt.subplots(2,1, sharex = True)
cc=ax[0].pcolor(micro["date_dt"], micro["pres"][:,0],micro["Turb"], vmin = 0, vmax = 4)
ccT = ax[0].contour(micro["date_dt"], micro["pres"][:,0],micro["T"], np.arange(10,18,2), colors = "w", linewidths = 0.5)
plt.clabel(ccT,fmt = "%d")
ax[0].plot(micro["date_dt"], d_intr,marker = ".", color = "r")
ax[0].set_ylim((100,0))
ax[0].annotate("Turbidity", xy = (0.01,0.03), xycoords = "axes fraction", color = "k")
ax[0].set_ylabel("Depth [m]")
ax[1].plot(micro["date_dt"], T_intr,".")
ax[1].set_ylabel("T at intrusion")
fig.savefig("Rhone_intrusion.png")
plt.show()

x = TML0 - T_intr
y = TML0 - Trhone

Gamma0 = y/x
ii = np.isfinite(Gamma0)
paramsG, params_covG = optimize.curve_fit(harmonic_func, micro["time"][ii], Gamma0[ii], p0=[1,1,180])
Gamma = harmonic_func(time,*paramsG)


"""
fig, ax = plt.subplots()
ax.plot(micro["time"], Gamma0,".")
ax.plot(time,Gamma)
plt.show()
"""

"""
#Gamma1 = 1/np.nanmedian(x/y)
#Gamma2 = np.nanmedian(y/x)
Gamma = np.nanmedian(y/x)
#Gamma = 0.5*Gamma1+ 0.5*Gamma2
print Gamma
fig, ax = plt.subplots()
ax.scatter( x, y,10, micro["time"] )
ax.plot(np.sort(x), Gamma*np.sort(x),color = "gray", lw = 2)
ax.plot(np.sort(x), np.sort(x),color = "gray", lw = 1)
ax.set_xlabel("$T_s - T_i$ [$^{\\circ}$C]")
ax.set_ylabel("$T_s - T_r$ [$^{\\circ}$C]")
plt.show()
"""


fig, ax = plt.subplots(2,1, sharex = True, figsize = (10,7))
ax[0].plot(RH["date"], RH["T_in"],"k", alpha = 0.5)
ax[0].plot(date, RH_T_in,"k")
ax[0].plot(date, RH_T_in-RH_sT_in,"k--", lw = 0.5)
ax[0].plot(date, RH_T_in+RH_sT_in,"k--", lw = 0.5)
ax[0].set_ylabel("Tin [degC]")
#ax[0].plot(RH["date"], RH["T_out"], "b", alpha = 0.5)
ax[1].plot(RH["date"], RH["Q_in"],"k", alpha = 0.5)
ax[1].plot(date, RH_Q_in,"k")
ax[1].set_ylabel("Qin [m3/s]")
#ax[2].plot(RH["date"],dT2, "k", alpha = 0.5)
#ax[2].plot(date,RH_sT_in**2,"k", alpha = 0.5)
#ax[2].set_ylabel("varTin [degC2]")
#ax[1].plot(RH["date"], RH["Q_out"], "b")
fig.savefig("rhoneTandQ.png", bbox_inches = "tight")
plt.close(fig)

QHrhone0 = np.nanmean(ST["denGRs"])*np.nanmean(cp)*(ST["TGRs"][1,:]-RH_T_in)*RH_Q_in/ST["AreaGRs"][0,0]

#Ignores dilution
Gamma[:] = np.nanmedian(y/x)#1.57
#Gamma[:] = 1.
#Gamma[:] = 1.57

T_intrusion = (RH_T_in - (1-Gamma)*mTML)/Gamma

#vertical heat flux
QRH_depth = np.full( ST["TGRs"].shape, 0. )
hadvHF = np.full( (ST["TGRs"].shape[0]-1,ST["TGRs"].shape[1]), 0. )
inthadvHF = np.full( ST["TGRs"].shape, 0. )
for i in range(nt):
    #integrated mass convergence
    xx = (ST["TGRs"][:,i]-T_intrusion[i])/RH_sT_in[i]
    #xx = (ST["TGRs"][:,i]-T_intrusion[i])/T_intrusion[i]
    #jj = np.where( ST["TGRs"][:,i]>RH_T_in[i] )[0]
    QRH_depth[:,i] = Gamma[i]*RH_Q_in[i]*norm.cdf(xx)
    ilast = np.where(np.isfinite(QRH_depth[:,i]))[0][-1]
    Qdeep = QRH_depth[ilast,i]

    #integrated heat convergence
    t_c = 0.5*(ST["TGRs"][1:,i]+ST["TGRs"][:-1,i])
    t_c[ilast] = t_c[ilast-1]
    cp_c = 0.5*(cp[1:,i]+cp[:-1,i])
    cp_c[ilast] = cp_c[ilast-1]
    den_c = 0.5*(ST["denGRs"][1:,i]+ST["denGRs"][:-1,i])
    den_c[ilast] = den_c[ilast-1]
    #xxc = (t_c-T_intrusion[i])/RH_sT_in[i]
    Q_conv = -np.diff(QRH_depth[:,i])#Gamma[i]*RH_Q_in[i]*norm.pdf(xxc)
    Q_conv[ilast] = np.copy(Qdeep) 

    hadvHF[:,i] = t_c*cp_c*den_c*Q_conv/ST["AreaGRs"][0,0]/d_z ############################ d_z added 17 Nov, IS IT CORRECT????
    inthadvHF[-2::-1,i] = np.nancumsum( hadvHF[::-1,i]*d_z )

#deletes river
#hadvHF[:] = 0.
#inthadvHF[:] = 0.
#QRH_depth[:] = 0.

w = QRH_depth/ST["AreaGRs"][0,0]#I am not sure about the 0, but I think it is needed to be coherent with the rest
vadvHF = w*ST["TGRs"]*cp*ST["denGRs"]
  
qq = FCD_2d(-zM,QRH_depth)*d_z
#qq /=RH_Q_in

fig = plt.figure(figsize = (12,4))
cc=plt.contourf(date,z,qq,20, cmap = CMCV)
cb=plt.colorbar(cc)
ccT = plt.contour(date,z,ST["TGRs"], np.arange(7,29,2),colors = "gray", linewidths = 0.5)
plt.clabel(ccT, fmt = "%d")
cb.set_label("Q [m$^{3}$ s$^{-1}$]")
#cb.set_label("%%")
plt.ylim((110,0))
plt.ylabel("Depth [m]")
fig.savefig("Rhone inflow.png", bbox_inches = "tight")
plt.close(fig)




#calculation of turbulent flux and Kz
totHflux = - np.copy(dHdt)
radHF = Iz*ST["AreaGRs"]/ST["Area"][0]
turbHflux = totHflux-radHF -vadvHF +inthadvHF
Kz = -turbHflux/ST["dTdz"]/ST["denGRs"]/cp

MLD = np.tile(ST["MLD"],(z.size,1))
neg_flux_depth = np.full(ST["TGRs"].shape, np.nan)
for i in range(nt):
    ii = np.where(turbHflux[:,i]<0)[0]
    if len(ii) == 0:
        neg_flux_depth[:,i] = MLD[:,i]
    else:
        neg_flux_depth[:,i] = z[ ii ][0]

ULD = np.maximum(MLD,neg_flux_depth)

Kz[Kz<0] = np.nan
#Kz[(zM<ULD)] = np.nan
#Kz[((np.isnan(Kz)) | (Kz<1.4e-7)) & (zM>ULD) & (zM<105)] = 1.4e-7

#Turbulent buoyancy flux

turbBF = turbHflux*sw.g(LAT)*alpha/cp # W/m2
turbBF_th = np.copy(turbBF)
turbBF_th[turbBF_th>0] = np.nan
#turbBF0[(zM<MLD)] = np.nan
#intturbBF_th = np.nansum(turbBF_th*d_z*ST["AreaGRs"]/ST["Area"][0], axis = 0) #W/m2 #I think this was wrong
intturbBF_th = -np.nansum(turbBF_th*d_z, axis = 0) #W/m2
print np.nanmean(intturbBF_th)*1000.
#print np.nanmean(intturbBF_th0)*1000.

turbBF0nr = (turbHflux +vadvHF-inthadvHF) *sw.g(LAT)*alpha/cp # W/m2
turbBF0nr[turbBF0nr>0] = np.nan
#turbBF0nr[(zM<MLD)] = np.nan
#intturbBF_thnr = np.nansum(turbBF0nr*d_z*ST["AreaGRs"]/ST["Area"][0], axis = 0) #W/m2
intturbBF_thnr = -np.nansum(turbBF0nr*d_z, axis = 0) #W/m2
print np.nanmean(intturbBF_thnr)*1000.

turbBF_K = np.nansum( Kz*ST["N2GRs"]*d_z*ST["AreaGRs"]/ST["Area"][0], axis = 0)

print np.nanmean(turbBF_K)*1e6

##divergences, contribution to buoyancy 
cp_c = 0.5*cp[1:,:]+0.5*cp[:-1,:]
alpha_c = 0.5*alpha[1:,:]+0.5*alpha[:-1,:]

radBF = radHF*sw.g(LAT)*alpha/cp #radiative
vadvBF = vadvHF*sw.g(LAT)*alpha/cp #vertical advection
totBF = -dHdt*sw.g(LAT)*alpha/cp

hadvBJ = hadvHF*alpha_c/cp_c*sw.g(LAT) #convergence as a source
turbBJ = np.diff(turbHflux, axis = 0)*alpha_c/cp_c*sw.g(LAT)/d_z
vadvBJ = np.diff(vadvHF, axis = 0)*alpha_c/cp_c*sw.g(LAT)/d_z
radBJ = np.diff(radHF, axis = 0)*alpha_c/cp_c*sw.g(LAT)/d_z
totBJ = -np.diff(dHdt, axis = 0)*alpha_c/cp_c*sw.g(LAT)/d_z


fig, ax = plt.subplots(3,2, sharex = True, sharey = True, figsize = (12,7))
cc=ax[0,0].contourf(date,z,totHflux, np.arange(-100,110,10), cmap = cm.RdBu_r, extend = "both")
cb = plt.colorbar(cc, ax= ax[0,0])
ax[0,0].contour(date,z,totHflux, [0], colors = "gray", linewidths = 0.5)
ax[0,0].set_title("Total heat flux [W m$^{-2}$]")
ax[0,0].set_ylabel("Depth [m]")
ax[0,0].plot(date,ST["MLD"],'-', color = "k")
ax[0,0].plot(date,ST["MLD"],'--', color = "green")

cc = ax[0,1].contourf(date,z,Iz, np.arange(-100,110,10), cmap = cm.RdBu_r, extend = "both")
cb = plt.colorbar(cc, ax = ax[0,1])
ax[0,1].set_title("Radiative flux [W m$^{-2}$]")
ax[0,1].contour(date,z,Iz, [0], colors = "gray", linewidths = 0.5)
#ax[0,1].set_ylabel("Depth [m]")
ax[0,1].plot(date,ST["MLD"],'-', color = "k")
ax[0,1].plot(date,ST["MLD"],'--', color = "green")

cc = ax[1,0].contourf(date,z,vadvHF, np.arange(-100,110,10), cmap = cm.RdBu_r, extend = "both")
cb=plt.colorbar(cc, ax = ax[1,0])
ax[1,0].set_title("River uplift [W m$^{-2}$]")
ax[1,0].contour(date,z,vadvHF, [0], colors = "gray", linewidths = 0.5)
ax[1,0].plot(date,ST["MLD"],'-', color = "k")
ax[1,0].plot(date,ST["MLD"],'--', color = "green")
ax[1,0].set_ylabel("Depth [m]")


cc = ax[1,1].contourf(date,z,inthadvHF, np.arange(-100,110,10), cmap = cm.RdBu_r, extend = "both")
cb=plt.colorbar(cc, ax = ax[1,1])
ax[1,1].set_title("Depth integrated convergence (river) [W m$^{-2}$]")
ax[1,0].contour(date,z,inthadvHF, [0], colors = "gray", linewidths = 0.5)
ax[1,1].plot(date,ST["MLD"],'-', color = "green")
ax[1,1].plot(date,ST["MLD"],'-', color = "k")
ax[1,1].set_ylim((105,0))

cc = ax[2,0].contourf(date,z,turbHflux, np.arange(-100,110,10), cmap = cm.RdBu_r, extend = "both")
cb=plt.colorbar(cc, ax = ax[2,0])
ax[2,0].set_title("Turbulent flux [W m$^{-2}$]")
ax[2,0].contour(date,z,turbHflux, [0], colors = "gray", linewidths = 0.5)
ax[2,0].plot(date,ST["MLD"],'-', color = "k")
ax[2,0].plot(date,ST["MLD"],'--', color = "green")
ax[2,0].set_ylabel("Depth [m]")

cc = ax[2,1].contourf(date,z,np.log10(Kz), np.arange(-7,-2.5,0.5), vmin = -7, vmax = -3, cmap = cm.viridis)
cc2 = ax[2,1].contour(date,z,np.log10(Kz), np.arange(-7,-2.5,0.5), colors = "k", linewidths = 0.5, linestyles = "-")
cb = plt.colorbar(cc, ax = ax[2,1])
plt.clabel(cc2, fmt = "%1.1f", color = "k")
ax[2,1].set_title("$\log_{10} K_T$ [m$^{2}$ s$^{-1}$]")
ax[2,1].plot(date,ST["MLD"],'-', color = "k",lw = 2)
ax[2,1].plot(date,ST["MLD"],'--', color = "darkred",lw = 2)

ax[2,0].xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
ax[2,1].xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
ax[2,0].tick_params(axis='x', rotation=45)
ax[2,1].tick_params(axis='x', rotation=45)
fig.tight_layout()

let = ["a.","b.","c.","d.","e.","f."]
for ax0, le in zip(ax.ravel(),let):
    if le != "f.":
        ax0.annotate(le, xy = (0.02,0.02), xycoords = "axes fraction", fontweight = "bold")
    else:
        ax0.annotate(le, xy = (0.02,0.02), xycoords = "axes fraction", fontweight = "bold", color = "w")
fig.savefig("Heat_budget.png", dpi = 300)
fig.savefig("Heat_budget.pdf", dpi = 300)
plt.close(fig)


fig, ax = plt.subplots(1, figsize = (7,3))
cc = ax.contourf(date,z,np.log10(Kz), np.arange(-7,-2.5,0.5), vmin = -7, vmax = -3, cmap = cm.viridis)
cb = plt.colorbar(cc, ax = ax)
#plt.clabel(cc, fmt = "%1.1f", color = "w")
cb.set_label("$\log_{10} K_T$ [m$^{2}$ s$^{-1}$]")
ax.set_ylabel("Depth [m]")
ax.set_ylim((105,0))
ax.plot(date,ST["MLD"],'-', color = "k",lw = 2)
ax.plot(date,ST["MLD"],'--', color = "darkred",lw = 2)
ax.tick_params(axis='x', rotation=45)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
fig.savefig("vertical_diffusivity_HB.pdf", dpi = 300, bbox_inches = "tight")


##CONTRIBUTION OF THE FLUXES TO PE CHANGE
Z = np.nanmax(ST["depth"])-ST["depth"]
dens = np.copy(ST["denGRs"])
T = np.copy(ST["TGRs"])
ZM = np.tile(Z,(time.size,1)).T
ZC = 0.5*(Z[1:]+Z[:-1])
ZCM = np.tile(ZC,(time.size,1)).T
Area = np.copy(ST["AreaGRs"])
dZ = np.abs(Z[1]-Z[0])
Zv = np.nansum( Z*Area[:,0]*dZ )/np.nansum( Area[:,0]*dZ ) 


 
ii = np.where( np.isfinite(np.nanmean(turbBF, axis = 1)))[0]
isurf = ii[0]
ibot = ii[-1]
A0 = Area[isurf,0]
AB = Area[ibot,0]
Zsurf = Z[isurf]
Zbot = Z[ibot]

mdens = np.nanmean(np.nanmax(ST["denGR"], axis = 0))
APE = sw.g(LAT)*np.nansum( (dens-mdens)*(ZM-Zv)*dZ, axis = 0)
APE0 =sw.g(LAT)*np.nansum( (ST["denGR"]-mdens)*(np.tile(Z,(ST["timeGR"].size,1)).T-Zv)*dZ, axis = 0)
#APE = sw.g(LAT)*np.nansum( sw.dens0(0.2,T)*(ZM-Zv)*dZ, axis = 0)
APE0[APE0 == 0] = np.nan


surf_turbBF = turbBF[isurf,:]*(Zsurf-Zv)
surf_radBF = radBF[isurf,:]*(Zsurf-Zv)
surf_vadvBF = vadvBF[isurf,:]*(Zsurf-Zv)
surf_totBF = totBF[isurf,:]*(Zsurf-Zv)

bot_turbBF = -turbBF[ibot,:]*(Zbot-Zv)*A0/AB
bot_radBF = -radBF[ibot,:]*(Zbot-Zv)*A0/AB
bot_vadvBF = -vadvBF[ibot,:]*(Zbot-Zv)*A0/AB
bot_totBF = -totBF[ibot,:]*(Zbot-Zv)*A0/AB

int_turbBF = -np.nansum(turbBF*dZ, axis = 0)
int_radBF = -np.nansum(radBF*dZ, axis = 0)
int_vadvBF = -np.nansum(vadvBF*dZ, axis = 0)
int_hadvBJ = -np.nansum(hadvBJ*dZ*(ZCM-Zv), axis = 0)
int_totBF = -np.nansum(totBF*dZ, axis = 0)

int_radBJ = -np.nansum(radBJ*dZ*(ZCM-Zv), axis = 0)
int_turbBJ = -np.nansum(turbBJ*dZ*(ZCM-Zv), axis = 0)
int_vadvBJ = -np.nansum(vadvBJ*dZ*(ZCM-Zv), axis = 0)
int_totBJ = -np.nansum(totBJ*dZ*(ZCM-Zv), axis = 0)

change_PE = surf_turbBF + surf_radBF + surf_vadvBF + \
            bot_turbBF + bot_radBF + bot_vadvBF + \
            int_turbBF + int_radBF+int_vadvBF+int_hadvBJ
change_PE2 = int_radBJ + int_turbBJ + int_vadvBJ + int_hadvBJ
#change_PE = surf_turbBF + surf_radBF+  \
#            bot_turbBF + bot_radBF + \
#            int_turbBF + int_radBF

fig, ax = plt.subplots(2,1)
ax[0].plot(date, surf_radBF+bot_radBF, color = cm.tab10(0), ls = "--")
ax[0].plot(date, int_radBF, color = cm.tab10(0), ls = "-")

ax[0].plot(date, surf_turbBF+bot_turbBF, color = cm.tab10(1), ls = "--")
ax[0].plot(date, int_turbBF, color = cm.tab10(1), ls = "-")

ax[0].plot(date, surf_vadvBF+bot_vadvBF, color = cm.tab10(2), ls = "--")
ax[0].plot(date, int_vadvBF, color = cm.tab10(2), ls = "--")
ax[0].plot(date, int_hadvBJ, color = cm.tab10(2), ls = "-")
ax[0].plot(date, change_PE, color = "k")
ax[0].plot(date, change_PE2, color = "red")
ax[0].axhline(0, color = "k", lw = 1)

ax[1].plot(date, APE)
ax[1].plot(date, APE[0]+np.cumsum(change_PE*60*60*24), color = "k")
ax[1].plot(date, APE[0]+np.cumsum(change_PE2*60*60*24), color = "r")
plt.show()


####POTENTIAL ENERGY
#mean depth of intrussion
intr_depth0 = np.full( ST["TGRs"].shape[1], 0. )
intr_depth = np.full( ST["TGRs"].shape[1], 0. )
mdensdiff = np.full( ST["TGRs"].shape[1], 0. )
mdensdiff_ent = np.full( ST["TGRs"].shape[1], 0. )
QPE =  np.full( ST["TGRs"].shape[1], 0. )
QPE_ent =  np.full( ST["TGRs"].shape[1], 0. )
for i in range(nt):
    iint0 = np.nanargmin(np.abs(RH_T_in[i]-ST["TGRs"][:,i]))
    intr_depth0[i] = ST["depth"][iint0]
    mdensdiff[i] = np.nanmean( (ST["denGRs"][0:iint0,i] - ST["denGRs"][iint0,i]))

    iint = np.nanargmin(np.abs(T_intrusion[i]-ST["TGRs"][:,i]))
    intr_depth[i] = ST["depth"][iint]
    dens_ML = sw.dens0(0.2, mTML[i] )
    mdensdiff_ent[i] = np.nanmean( ST["denGRs"][0:iint,i] - dens_ML)
    
QPE =- RH_Q_in*sw.g(LAT)*mdensdiff*intr_depth0*1000/ST["Area"][0] #mW m-2

QPE2 = -sw.alpha(0.2,mTML,0)*1000*sw.g(LAT)*RH_Q_in*RH_T_in*(Z[0]-Zv)/ST["Area"][0]*1000
QPE_ent = (1-Gamma)*RH_Q_in*sw.g(LAT)*mdensdiff_ent*(ST["MLD"]/2-intr_depth)*1000/ST["Area"][0] #mW m-2

ST["HB"] = dict()
ST["HB"]["kd"] = kd
ST["HB"]["Iz"] = Iz
ST["HB"]["I0"] = I0s
ST["HB"]["w"] = w
ST["HB"]["Q"] = QRH_depth
ST["HB"]["vadvHF"] = vadvHF
ST["HB"]["hadvHF"] = hadvHF
ST["HB"]["int_hadv"] = inthadvHF
ST["HB"]["totHF"] = totHflux
ST["HB"]["turbHF"] = turbHflux
ST["HB"]["Kz"] = Kz
ST["HB"]["int_turbBF"] = intturbBF_th
ST["HB"]["turbBF"] = turbBF
ST["HB"]["turbBF_th"] = turbBF_th
ST["HB"]["ULD"] = ULD[0,:]
ST["HB"]["QPE"] = QPE
ST["HB"]["QPE_ent"] = QPE_ent

ST["HB"]["APE"] = dict()
ST["HB"]["APE"]["APE"] = APE
ST["HB"]["APE"]["APE0"] = APE0
ST["HB"]["APE"]["surf_turbBF"] = surf_turbBF
ST["HB"]["APE"]["surf_radBF"] = surf_radBF
ST["HB"]["APE"]["surf_vadvBF"] = surf_vadvBF
ST["HB"]["APE"]["surf_totBF"] =surf_totBF

ST["HB"]["APE"]["bot_turbBF"] = bot_turbBF
ST["HB"]["APE"]["bot_radBF"] = bot_radBF
ST["HB"]["APE"]["bot_vadvBF"] = bot_vadvBF
ST["HB"]["APE"]["bot_totBF"] = bot_totBF

ST["HB"]["APE"]["int_turbBF"] = int_turbBF
ST["HB"]["APE"]["int_radBF"] = int_radBF
ST["HB"]["APE"]["int_vadvBF"] = int_vadvBF
ST["HB"]["APE"]["int_hadvBJ"] = int_hadvBJ
ST["HB"]["APE"]["int_totBF"] = int_totBF


#pickle.dump(ST, open("./DO/smooth_temperature_evolution_heat_budget_DO.pi","w+"))
pickle.dump(ST, open("smooth_temperature_evolution_heat_budget.pi","w+"))

OUTk = dict()
OUTk["time"] = ST["timeGRs"]+366
OUTk["depth"] = ST["depth"]
OUTk["Kz"] = Kz
io.savemat("Leman_diffusivity_heat_budget.mat",OUTk)
 
fig,ax = plt.subplots(figsize = (6,4))
l1,=ax.plot(date,intturbBF_th*1000)
l2,=ax.plot(date,intturbBF_thnr*1000)
l4,=ax.plot(date,intturbBF_thnr*1000+QPE_ent)
l3,=ax.plot(date, QPE,"k")
#ax.plot(date,(intBF-intBFnr)*1000)
ax.legend(["Rhone (mean %1.2f mW m$^{-2}$)"%(np.nanmean(intturbBF_th)*1000.),"No Rhone (mean %1.2f mW m$^{-2}$)"%(np.nanmean(intturbBF_thnr)*1000.), "$Q_{PE}$"])
ax.set_ylabel("B and $Q_{PE}$ [mW m$^{-2}$]")

fig.savefig("Buoyancy_flux.png",dpi = 300)
plt.show()



#Extremelly sensitive to alpha choice
#Potential energy balance
SC = np.copy(ST["ScGR"])
zCM = np.nansum(ST["Area"]*ST["depth"])/np.nansum(ST["Area"])
HF0 = dHdt[0,:] + QHrhone0
JB0 = HF0*alpha[1,:]/cp[1,:]*sw.g(LAT)
#JB0 = HF0*np.nanmean(alpha)/np.nanmean(cp)*sw.g(LAT)

"""
JB0 = np.full(nt, np.nan)
for i in range(nt):
    jj = np.where(np.abs(COSMO["datenum"]-time[i])<2*sigma)
    x = COSMO["JB"][jj]
    dd = np.abs(COSMO["datenum"][jj]-time[i])
    ww = np.exp(- dd**2/sigma**2)
    ii = np.where(np.isfinite(x))
    JB0[i] = np.nansum(ww[ii]*x[ii])/np.nansum(ww[ii])

HF00 = np.full(nt, np.nan)
for i in range(nt):
    jj = np.where(np.abs(COSMO["datenum"]-time[i])<2*sigma)
    x = -COSMO["sHFnet"][jj]
    dd = np.abs(COSMO["datenum"][jj]-time[i])
    ww = np.exp(- dd**2/sigma**2)
    ii = np.where(np.isfinite(x))
    HF00[i] = np.nansum(ww[ii]*x[ii])/np.nansum(ww[ii])
"""

#JB0 *= ST["denGRs"][1,:]
dt = 60*60*24.
fig = plt.figure()
plt.plot(date,SC)
plt.plot(date, np.cumsum(JB0*zCM*dt))
plt.show()

