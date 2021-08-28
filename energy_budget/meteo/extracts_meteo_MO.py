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
from scipy import optimize
import pickle
import seawater as sw
import os
import sys
import glob
sys.path.insert(0, "/home/bieito/Documents/SCIENCE/EPFL/python_functions/")
import HB_functions as hbf
import HB_functions_Geneva as hbfG
sys.path.insert(0, "../")
from extract_Tmoor import *
import seawater as sw
#from vertical_modes import *



#defines an harmonic function to smooth the river variables
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


LON = 6.657764
LAT = 46.50094
ELEV = 372
  

##reads file with mean smoothed temperature profiles and heat content
STdata = pickle.load(open("../smooth_temperature_evolution.pi"))

##extracts temperature data from mooring
TM = extracts_all_Tmoor("../Tmoor")


##reads the meteo station file
MET = hbf.import_weather_station("Lexplore_Weather.toa5")

##calculates cloud cover fraction and clear skyradiation
MET["clf"],MET["Ics"] = hbf.cloud_cover(MET["date_dt"],LON,LAT,ELEV,MET["SlrW_Avg"],MET["AirTC_Avg"],MET["RH"],MET["BP_mbar_Avg"])

##selects the time range where temperature measurements are available
start_date = max(STdata["dateGRs"][0], MET["date_dt"][0])
end_date = min(STdata["dateGRs"][-1], MET["date_dt"][-1])#, RHONE["date"][-1])
MET = hbfG.extract_period_from_dict(MET, start_date, end_date)

##interpolates surface temperature (two options: smoothed temperature profiles, or mooring)
#(1)smoothed profiles
#intTs = intrp.interp1d(STdata["timeGRs"], np.array(STdata["TGRs"][1,:]), bounds_error = False)
#MET["swT"] = intTs(MET["datenum"])

#(2)mooring data
tmoor = TM["time"]
swT = TM["T"][0,:]
iif = np.where(np.isfinite(swT))[0]
intswT = intrp.interp1d(tmoor[iif], swT[iif], bounds_error = False )
MET["swT"] = intswT(MET["datenum"])

#Interpolates mixed layer depth from mooring (not required for heat fluxes)
intML = intrp.interp1d(tmoor[iif], TM["MLDdif"][iif], bounds_error = False )
MET["MLDdif"] = intML(MET["datenum"])
intML = intrp.interp1d(tmoor[iif], TM["MLDgr"][iif], bounds_error = False )
MET["MLDgr"] = intML(MET["datenum"])

##cloud cover cosmo
#intCC = intrp.interp1d(COSMO["datenum"], COSMO["CLCT"]/100.)
#MET["clf_COSMO"] = intCC(MET["datenum"])

##saves and loads the data (usefull to avoid repeating this first part of the code)
pickle.dump(MET, open("Meteo_raw.pi","w+"))


MET = pickle.load(open("Meteo_raw.pi"))

##some time operations
MET["datenum"] = np.array(MET["datenum"])
time_step = (MET["datenum"][1]-MET["datenum"][0])*24*60*60

##Interpolates the contribution of rhone heat flux
RH = pickle.load(open("rhone_heat_budget.pi"))
ii = np.where(RH["time"]>dt.datetime(2018,9,1,0,0,0).toordinal())[0]
paramsQ, params_covQ = optimize.curve_fit(harmonic_func, RH["time"][ii], RH["Q_in"][ii], p0=[1]+[5,180]*4)
RH_Q_in = harmonic_func(MET["datenum"],*paramsQ)
paramsT, params_covT = optimize.curve_fit(harmonic_func, RH["time"][ii], RH["T_in"][ii], p0=[1]+[5,180]*4)
RH_T_in = harmonic_func(MET["datenum"],*paramsT)

zc = np.nansum(STdata["Area"]*STdata["depth"])/np.nansum(STdata["Area"])
cp = sw.cp( np.nanmean(STdata["SGRs"]),np.nanmean( STdata["TGRs"]), zc)
rho = np.nanmean(STdata["denGRs"])

intsT = intrp.interp1d(STdata["timeGRs"], STdata["TGRs"][1,:])
sT = intsT(MET["datenum"])

Rhone_HF = rho*cp*( (RH_T_in - sT )*RH_Q_in)/STdata["Area"][0]
int_Rhone_HF = np.cumsum(Rhone_HF*time_step)

##Interpolates the total heat content from temperature profiles
intHC = intrp.interp1d(STdata["timeGRs"], STdata["HCGR"][0,:]) 
HC = intHC(MET["datenum"])
HC -= HC[0]

"""
#loads optimized
#optP = pickle.load( open("optimized_parameters_meteo_MO.pi"))
optP = dict()
optP["Adiff"] = 0.10443109399822727
optP["Cc"] = 0.39197190718339225
optP["a"] = 0.9125406479313491
optP["alpha2"] = 0.010578198797004012
optP["b1"] = 0.6972379351173725
"""
##Runs with optimized parameters
HF = hbf.HeatFluxCalculator( zu = 5., zt = 5., zq = 5., Tw = MET["swT"], Ta = MET["AirTC_Avg"], Wsp = MET["WS_ms_S_WVT"],RH = MET["RH"], P = MET["BP_mbar_Avg"], C = MET["clf"], SRad = MET["SlrW_Avg"], date = MET["date_dt"], a = 0.99)#, a = 0.97,#a = optP["a"] , Cc = optP["Cc"], Adiff = optP["Adiff"], alpha2 = optP["alpha2"], b1 =optP["b1"])

"""
##Runs with Abolfalz optimized parameters
HF = hbf.HeatFluxCalculator( zu = 5., zt = 5., zq = 5., Tw = MET["swT"], Ta = MET["AirTC_Avg"], Wsp = MET["WS_ms_S_WVT"],RH = MET["RH"], P = MET["BP_mbar_Avg"], C = MET["clf"], SRad = MET["SlrW_Avg"], date = MET["date_dt"],a = 0.98, Cc = 0.11, alpha2 = 0.01, b1 = 1.52)
"""
HF.shortwave_Fink()
HF.longwave_Fink()
HF.latent_sensible_windstress_MO()

HF.Qnet = HF.Qsen + HF.Qlat + HF.Qlw_in + HF.Qlw_out + HF.Qsw        
intHF = np.nancumsum(-time_step*HF.Qnet)

#Plots the result
fig, ax = plt.subplots(figsize = (12,4))
ax.plot(MET["date_dt"],HC, color = "k", label = "Integrated heat content")
ax.plot(MET["date_dt"],intHF, label = "Cumulative atm. heat flux")
ax.plot(MET["date_dt"],intHF+int_Rhone_HF, label = "Cumulative atm. heat flux + Rhone")
ax.plot(MET["date_dt"],int_Rhone_HF, label = "Rhone heat flux")
ax.axhline(0, color = "gray")
ax.set_ylabel("Heat [J m$^{-2}$]")
plt.legend()
fig.savefig("Heat_fluxes_meteo_MO.png", dpi = 300)
plt.show()

#Calculates derived variables and stores

#wind stress
MET["rho_air"] = np.copy(HF.rhoa)
MET["Cd"] = np.copy(HF.Cd10)
MET["WS_10m"] = np.copy(HF.u10)
MET["tau"] = np.copy(HF.tau)
MET["u"], MET["v"] = hbf.wdir_to_uv(MET["WS_10m"],MET["WindDir_D1_WVT"])
#MET["tau"] = MET["rho_air"]*MET["Cd"]*MET["WS_10m"]**2
MET["P10"] = MET["tau"]*MET["WS_10m"]
MET["tau_x"] = MET["rho_air"]*MET["Cd"]*MET["WS_10m"]*MET["u"]
MET["tau_y"] = MET["rho_air"]*MET["Cd"]*MET["WS_10m"]*MET["v"]
MET["rho0"] = sw.dens0(0.2,MET["swT"])
MET["ustar"] = np.sqrt(MET["tau"]/MET["rho0"])


MET["sHFnet"] = np.copy(HF.Qnet)
MET["sHFturb0"] = np.copy(HF.Qnet-HF.Qsw)
MET["sHFluxes"] = dict()
MET["sHFluxes"]["Qsw"] = np.copy(HF.Qsw)
MET["sHFluxes"]["Qlw_in"] = np.copy(HF.Qlw_in)
MET["sHFluxes"]["Qlw_out"] = np.copy(HF.Qlw_out)
MET["sHFluxes"]["Qlat"] = np.copy(HF.Qlat)
MET["sHFluxes"]["Qsen"] = np.copy(HF.Qsen)

MET["JB"] = sw.alpha(0.2,MET["swT"],0)*9.81/sw.cp(0.2,MET["swT"],0)*MET["sHFnet"]/MET["rho0"]
MET["LMO"] = MET["ustar"]**3/0.4/MET["JB"]

#pickle.dump(MET, open("Meteo_Leman_all_MO.pi","w+"))

MET.pop("date_dt")
MET["datenum"]+=366
io.savemat("Meteo_Leman_all_MO.mat", MET)
