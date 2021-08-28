import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
import seawater as sw
import scipy.stats as stats
import scipy.interpolate as intrp
import copy
from scipy import io
import datetime as DT



ADfiles = sorted( glob.glob("../*DEP*.pi"))

ADf = []
AD_time_start = np.array([])
AD_time_end = np.array([])
nf = len(ADfiles)
for fil in ADfiles:
    ADf.append( pickle.load(open(fil)) )
    AD_time_start = np.r_[ AD_time_start,  ADf[-1]["mDATA"]["time"][0]]
    AD_time_end = np.r_[ AD_time_end,  ADf[-1]["mDATA"]["time"][-1]]
    

dz = 0.25
TH = pickle.load(open("merged_thermistors_AD.pi"))
TH["pres_2mab"] = np.full(TH["time"].size, np.nan)
TH["heading"] = np.full(TH["time"].size, np.nan)
TH["pres"] = np.full((TH["mab"].size, TH["time"].size), np.nan)
TH["u"] = np.full((TH["mab"].size, TH["time"].size), np.nan)
TH["v"] = np.full((TH["mab"].size, TH["time"].size), np.nan)
TH["v_pc1"] = np.full((TH["mab"].size, TH["time"].size), np.nan)
TH["v_pc2"] = np.full((TH["mab"].size, TH["time"].size), np.nan)
TH["U"] = np.full((TH["mab"].size, TH["time"].size), np.nan)
TH["q"] = np.full((TH["mab"].size, TH["time"].size), np.nan)
TH["epsilon"] = np.full((TH["mab"].size, TH["time"].size), np.nan)
TH["time_AD"] = np.full(TH["time"].size, np.nan)
print "START"
for i,tt in enumerate(TH["time"]):
    print TH["date"][i]
    for jf in range(nf):
        if (tt>=AD_time_start[jf]) and (tt<=AD_time_end[jf]):
            break
    it = np.where( np.abs(ADf[jf]["mDATA"]["time"]-tt)<=20./60/24 )[0]
    if it.size == 0:
        continue
    it = it[0]
    TH["time_AD"][i] = ADf[jf]["mDATA"]["time"][it]
    TH["pres_2mab"][i] = ADf[jf]["mDATA"]["pres"][it]
    TH["pres"][:,i] = TH["pres_2mab"][i]+9.81*(2-TH["mab"])/10.
    TH["heading"][i] = ADf[jf]["mDATA"]["head"][it]

    for k in range(TH["mab"].size):
        #print np.abs( ADf[jf]["mDATA"]["mab"][it,:] - TH["mab"][k])
        iimab = np.where( np.abs( ADf[jf]["mDATA"]["mab"][it,:] - TH["mab"][k])<=dz )[0]
        #print iimab
        if iimab.size>=3:
            TH["u"][k,i] = np.nanmean(ADf[jf]["mDATA"]["u"][it,iimab])
            TH["v"][k,i] = np.nanmean(ADf[jf]["mDATA"]["v"][it,iimab])
            TH["v_pc1"][k,i] = np.nanmean(ADf[jf]["mDATA"]["v_pc1"][it,iimab])
            TH["v_pc2"][k,i] = np.nanmean(ADf[jf]["mDATA"]["v_pc2"][it,iimab])
            TH["U"][k,i] = (np.nanmean(ADf[jf]["mDATA"]["v_pc1"][it,iimab])**2+np.nanmean(ADf[jf]["mDATA"]["v_pc2"][it,iimab])**2)**0.5
            TH["epsilon"][k,i] = np.nanmean(ADf[jf]["mDATA"]["SF"]["epsilonMP"][iimab,it])
        
        iimab1 = np.where( np.abs( ADf[jf]["mDATA"]["mab1"][it,:] - TH["mab"][k])<=dz )[0]
        if iimab1.size>=0:
            q1 = np.nanmean(ADf[jf]["mDATA"]["sv1"][it,iimab1]**2)
        iimab2 = np.where( np.abs( ADf[jf]["mDATA"]["mab2"][it,:] - TH["mab"][k])<=dz )[0]
        if iimab1.size>=0:
            q2 = np.nanmean(ADf[jf]["mDATA"]["sv2"][it,iimab2]**2)
        iimab3 = np.where( np.abs( ADf[jf]["mDATA"]["mab3"][it,:] - TH["mab"][k])<=dz )[0]
        if iimab1.size>=0:
            q3 = np.nanmean(ADf[jf]["mDATA"]["sv3"][it,iimab3]**2)
        q = np.array([q1,q2,q3])
        if np.sum(np.isfinite(q)) >= 2:
            TH["q"][k,i] = np.nansum(q)**0.5

TH["PT"] = sw.ptmp(0.2,TH["mTemp"],TH["pres"],np.nanmin(TH["pres"]))
TH["pdens"] = sw.pden(0.2,TH["mTemp"],TH["pres"],np.nanmin(TH["pres"])) 
pickle.dump(TH,open("merged_thermistors_AD_with_Aquadopp.pi","w+"))
