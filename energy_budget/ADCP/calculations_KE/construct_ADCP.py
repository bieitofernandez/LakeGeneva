import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.io as io
import copy
sys.path.insert(0, "/home/bieito/Documents/SCIENCE/EPFL/python_functions/")
from time_conversion import *
import scipy.interpolate as intrp
from scipy import optimize
import glob
import datetime 

def mean_filt(x, factor = 2.):
    if np.isnan(x).all():
        return np.nan
    m = np.nanmean(x)
    s = np.nanstd(x)
    #if np.abs(s)>2*np.abs(m):
    #    return np.nan
    igood = (x<= (m + factor*s)) & (x>= (m - factor*s))
    m = np.nanmean(x[igood])
    return m

files = sorted(glob.glob("ADCP/*.pi"))

AD0 = pickle.load(open(files[0]))
timeSTART = float(datetime.datetime(2019,4,15,0,0,0).toordinal())#np.floor(AD0["time"][0])

"""
plt.pcolor(AD0["u"].mask)
plt.show()
AD0["u"].mask = (AD0["u"].mask) | (np.abs(AD0["eu"])>0.05)
AD0["v"].mask =  (AD0["v"].mask) | (np.abs(AD0["eu"])>0.05)
plt.pcolor(AD0["u"].mask)        
plt.show()
"""
del AD0
AD0 = pickle.load(open(files[-1]))
timeEND = float(datetime.datetime(2020,4,16,0,0,0).toordinal())#np.floor(AD0["time"][-1])+1
del AD0

dz0 = 2.
OUT = dict()
OUT["z"] = np.arange(0,110,dz0)
#dt0 = 15.
dt0 = 60.
dt0 = dt0/60./24.
OUT["time"] = np.arange(timeSTART, timeEND+dt0,dt0)

OUT["date"] = convert_time_to_date(OUT["time"])
Nz = OUT["z"].size
Nt = OUT["time"].size
OUT["time_r"] = np.full(Nt, np.nan)
OUT["u"] = np.full((Nz,Nt), np.nan)
OUT["v"] = np.full((Nz,Nt), np.nan)
for (ifil,fil) in enumerate(files):
    print fil
    AD0 = pickle.load(open(fil))
    if ifil<2:
        #lots of spikes in the first deployment
        #AD0["u"].mask = (AD0["u"].mask) | (np.abs(AD0["eu"])>0.004)
        #AD0["v"].mask =  (AD0["v"].mask) | (np.abs(AD0["eu"])>0.004)
        AD0["u"].mask = (AD0["u"].mask) | (np.abs(AD0["eu"])>0.01)
        AD0["v"].mask =  (AD0["v"].mask) | (np.abs(AD0["eu"])>0.01)
        dt = dt0*2
        dz = dz0*2
    else:
        dt = np.copy(dt0)
        dz = np.copy(dz0)
    tI = AD0["time"][0]
    tE = AD0["time"][-1]
    ind_t = np.where( (OUT["time"]>= tI) & (OUT["time"]<= tE) )[0]
    for it in ind_t:
        iit = np.where( (AD0["time"]>= OUT["time"][it]-dt) & (AD0["time"]<= OUT["time"][it]+dt))[0]
        if "z" in AD0:
            z = np.nanmean(AD0["z"][:,iit], axis = 1)
        else:
            z = np.copy( AD0["z0"])
        OUT["time_r"][it] = np.nanmean(AD0["time"][iit])
        for iz in range(Nz):
            iiz = np.where( (z>=OUT["z"][iz]-dz)& (z<=OUT["z"][iz]+dz) )[0]
            if iit.size == 0 or iiz.size == 0:
                continue
            u00 = AD0["u"][iiz][:,iit].compressed()
            v00 = AD0["v"][iiz][:,iit].compressed()

            OUT["u"][iz,it] = mean_filt(u00)
            OUT["v"][iz,it] = mean_filt(v00)

        #delete values at the bottom for the first deployment
        if ifil<2:
            ilastgood = np.where(np.isfinite(OUT["u"][:,it]))[0]
            if ilastgood.size>5:
                ilastgood = ilastgood[-1]
                OUT["u"][ilastgood-4:ilastgood+1,it] = np.nan
            ilastgood = np.where(np.isfinite(OUT["v"][:,it]))[0]
            if ilastgood.size>5:
                ilastgood = ilastgood[-1]
                OUT["v"][ilastgood-4:ilastgood+1,it] = np.nan
            
    del AD0
#OUT["date_t"] = convert_time_to_date(OUT["time_r"])
#
pickle.dump(OUT, open("ADCP_Geneva_all_2019_2020_smooth_2h_4m.pi","w+") )
#pickle.dump(OUT, open("ADCP_Geneva_all_2019_2020.pi","w+") )


OUT = pickle.load( open("ADCP_Geneva_all_2019_2020_smooth_2h_4m.pi") )
KE = 0.5*np.nanmean( OUT["u"]**2 + OUT["v"]**2, axis = 0)
iif = np.where(np.isfinite(KE))[0]
dateIF = [OUT["date"][i] for i in iif]


fig , ax= plt.subplots(2,1)

ccu = ax[0].contourf(OUT["date"], OUT["z"], OUT["u"]*100, np.arange(-30,31,1), cmap = cm.RdBu_r)
ax[0].set_ylabel("Depth [m]")
ax[0].set_ylim((110,0))
plt.colorbar(ccu, ax = ax[0])

ccu = ax[1].contourf(OUT["date"], OUT["z"], OUT["v"]*100, np.arange(-30,31,1), cmap = cm.RdBu_r)
ax[1].set_ylabel("Depth [m]")
ax[1].set_ylim((110,0))
plt.colorbar(ccu, ax = ax[1])

plt.show()

