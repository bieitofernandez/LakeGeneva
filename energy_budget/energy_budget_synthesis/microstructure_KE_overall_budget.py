import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
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
import cmocean
import pybootstrap as bs
import scipy.stats as stats
import seaborn as sns #colors
import scipy
colorsCB = sns.color_palette("colorblind")
nu_air = 1.45e-5 #kinematic viscosity of air approx

def viscosity(T):
    #seawater viscosity
    v=1.792747-0.052126103*T+0.0005918645*T**2;
    v=v*10**-6;
    return v

def merge_sensors(sn1,sn2,fl1=0,fl2=0, factor = 2.7):
    if type(fl1) == int:
        fl1 = np.full(sn1.shape, True)
    if type(fl2) == int:
        fl2 = np.full(sn2.shape, True)
    SN = np.full(sn1.shape, np.nan)
    d1,d2 = sn1.shape
    for i in range(d1):
        for j in range(d2):
            #print i
            #print j
            #print fl1[i,j]
            #print sn1[i,j]
            if (fl1[i,j] == 0 or np.isnan(sn1[i,j])) and (fl2[i,j] == 0 or np.isnan(sn2[i,j])):
                #print "case 1"
                SN[i,j] == np.nan
            elif fl1[i,j] == 0 or np.isnan(sn1[i,j]) or sn1[i,j]>factor*sn2[i,j]:
                #print "case 2"
                SN[i,j] = sn2[i,j]
            elif fl2[i,j] == 0 or np.isnan(sn2[i,j]) or sn2[i,j]>factor*sn1[i,j]:
                #print "case 3"
                SN[i,j] = sn1[i,j]
            else:
                #print "case 4"
                SN[i,j] = 0.5*(sn1[i,j]+sn2[i,j])
    return SN


def logmean(x):
    x = x[np.isfinite(x)]
    x = np.log(x)
    mx = np.mean(x)
    sx = np.std(x)
    M = np.exp( mx + sx**2/2.  )
    return M


def mixing_layer_thorpe(z,LT, zs = 0., ov_coef = 10.):
    iif = np.isfinite(LT+z)
    z = z[iif]
    LT = LT[iif]
    dz = z[1]-z[0]
    ii = np.argmin(np.abs(z-zs))
    if np.abs(z[ii]-zs)>10:
        return np.nan
    #print z
    mLT = LT[ii]#np.nanmean(LT[z<=zs])
    #mLT = LT[z==zs]

    while True:
        ii +=1
        if ii>=np.size(z):
            zs = np.nanmax(z)
            break
        zs = z[ii]
        mLT = LT[ii]#np.nanmean(LT[z<=zs])
        #fig, ax = plt.subplots()
        #ax.semilogx(LT,-z)
        #ax.axhline(-zs, color = "r")
        #ax.axhline( -mLT*3, color = "green")
        #plt.show()
        
        if zs == ov_coef*mLT:
            break
        if zs>ov_coef*mLT:
            zs -= 0.5*dz
            break

    print zs
    return zs
        
               
    

z_max = 109 #maximum depth

ST = pickle.load(open("smooth_temperature_evolution_heat_budget.pi"))
ADCP = pickle.load( open("ADCP_Geneva_all_2019_2020_smooth_2h_4m_NM_WW.pi") )
AD = pickle.load( open("Aquadopp/merged_aquadop.pi") )
MET= pickle.load(open("meteo/Meteo_Leman_all_MO.pi"))
#ADCP = pickle.load( open("ADCP_Geneva_all_2019_2020_NM.pi") )
 
int_KE_ADCP = np.sum(ADCP["NM"]["KE"], axis = 0)
msh2ADCP = np.nansum(ADCP["NM"]["msh2"], axis = 0) 

#int_KE_ADCP = ADCP["smooth"]["KE"]
sh2ADCP = ADCP["smooth"]["sh2"]
#msh2ADCP = np.nanmean(sh2ADCP, axis = 0)

#int_KE_ADCP = ADCP["KE"]
#sh2ADCP = ADCP["sh2"]
#msh2ADCP = np.nanmean(sh2ADCP, axis = 0)

#sh2ADCP = np.copy(ADCP["sh2"])
#sh2ADCP = np.copy(ADCP["smooth"]["sh2"])
#sh2ADCP = np.copy(ADCP["NM"]["sh2"])
iish = np.isfinite(sh2ADCP)

#Velocities as per modal decomposition
U_ADCP = np.nansum(ADCP["NM"]["u"], axis = 0)
V_ADCP = np.nansum(ADCP["NM"]["v"], axis = 0)

U_angle = np.full(U_ADCP.shape[0], np.nan)
U_var = np.full(U_ADCP.shape[0], np.nan)
for i in range(U_ADCP.shape[0]):
    xu = U_ADCP[i,:]
    xv = V_ADCP[i,:]
    if (xu==0).all():
        continue
    cov = np.cov(xu,xv)
    eig_val, eig_vec = np.linalg.eig(cov)
    U_var[i] = eig_val[0]/np.sum(eig_val) 
    U_angle[i] = np.arccos(eig_vec[0,0])*180/np.pi

###
##Bottom dissipation
####
#current from ADCP
u_bot = np.nansum(ADCP["NM"]["u"][:,-5,:], axis = 0)
v_bot = np.nansum(ADCP["NM"]["v"][:,-5,:], axis = 0)
V_bot = (u_bot**2 + v_bot**2)**0.5
z_bot = ADCP["z"][-5]
h_bot = z_max - z_bot


V_bot_AD = np.nansum(AD["vpc_1m"]**2, axis = 0)**0.5
Cd = np.copy(AD["Cd_SF"])
z0 = np.copy(AD["mz0_SF"])
#z0 = np.exp( np.log(1) - 0.41*(Cd)**-0.5 )
us_AD = (Cd*V_bot_AD**2)**0.5

#factor to extrapolate
ifrq = sw.f(46.)
zll_AD = 0.41*us_AD/ifrq #ekman layer height?
#zll_AD = 9.
deltaB = 11/us_AD*viscosity(7.)
#factor_AD = np.log( zll_AD/z0 )
factor_AD = np.log( zll_AD/deltaB ) #OLLO January 2021
epsilon_AD = np.copy(AD["epsilon_SF_1m"][0,:])
us_AD_meas = (0.41*1*epsilon_AD)**(1./3.)
diss_AD_bot = factor_AD*epsilon_AD*1e6

#with dragg coeficient
epsilon_ADCd = us_AD**3/0.41
diss_ADCd_bot = factor_AD*epsilon_ADCd*1e6

#ADCP
#[1] optimize the Cd coefficient
def cost_cd(Cd, V_bot, ifrq, z0, ADCP_time, AD_time, AD_diss):
    us0 = (Cd*V_bot**2)**0.5
    zll_ADCP0 = 0.41*us0/ifrq
    #zll_ADCP0 = 9.
    deltaB_ADCP0 = 11/us0*viscosity(7.)
    factor_ADCP0 = np.log( zll_ADCP0/deltaB_ADCP0 ) # JAN 2021
    #factor_ADCP0 = np.log( zll_ADCP0/z0 )
    us0[us0==0] = np.nan
    epsilon_ADCP_bot0 = us0**3/0.41
    diss_ADCP_bot0 = epsilon_ADCP_bot0*factor_ADCP0*1e6
    iif = np.isfinite(diss_ADCP_bot0)
    intBD = intrp.interp1d( ADCP_time[iif], diss_ADCP_bot0[iif], bounds_error = False )
    diss_ADCP_bot0_AD = intBD(AD_time)
    #cost[i] = np.nansum( (np.log10(diss_ADCP_bot0_AD) - np.log10(diss_AD_bot))**2 )
    cost = np.nansum( (diss_ADCP_bot0_AD - AD_diss)**2 )
    return cost

op = optimize.minimize( cost_cd, Cd, args = (V_bot, ifrq, z0, ADCP["time"], AD["time"], diss_AD_bot), bounds = [(1e-4,1e-2)] )
Cd_ADCP = op.x[0]  

"""
#some monte carlo
MC = 1000
CdMC = np.full(MC, np.nan)
for i in range(MC):
    rr = 0
    print i
    while rr<0.25:
        rr = np.random.random()
    jj_ADCP = np.array([])
    for j in range(ADCP["time"].size):
        if np.random.random()<rr:
            jj_ADCP = np.r_[jj_ADCP,j]
    jj_AD = np.array([])
    for j in range(AD["time"].size):
        if np.random.random()<rr:
            jj_AD = np.r_[jj_AD,j]
    jj_ADCP = jj_ADCP.astype(int)
    jj_AD = jj_AD.astype(int)
    op = optimize.minimize( cost_cd, Cd, args = (V_bot[jj_ADCP], ifrq, z0, ADCP["time"][jj_ADCP], AD["time"][jj_AD], diss_AD_bot[jj_AD]), bounds = [(1e-4,1e-2)] )
    CdMC[i] = op.x[0]

eCd_ADCP = np.nanstd(CdMC)    
"""

eCd_ADCP = 0.0013
print Cd_ADCP
us = (Cd_ADCP*V_bot**2)**0.5
zll_ADCP = 0.41*us/ifrq
#zll_ADCP = 9.
#factor_ADCP = np.log( zll_ADCP/z0 )
deltaB_ADCP = 11/us*viscosity(7.)
deltaB_ADCP[~np.isfinite(deltaB_ADCP)] = np.nan
factor_ADCP = np.log( zll_ADCP/deltaB_ADCP ) # JAN 2021
us[us==0] = np.nan
epsilon_ADCP_bot = us**3/0.41
diss_ADCP_bot = epsilon_ADCP_bot*factor_ADCP*1e6

print np.nanmean(diss_ADCP_bot)
print np.nanmean(diss_AD_bot)
print np.nanmean(epsilon_ADCd*factor_AD*1e6)

##
## Microstructure
##

micro = pickle.load( open("CTD_micro/microstructure_down_Lac_Leman.pi"))
micro["time"] = micro["date"][0,:]-366
micro["date_dt"] = convert_time_to_date(micro["time"])

micro["MLDgr"] = mixed_layer_depth( micro["depth"], micro["sigmat"], 0.05, crit = "grad" )
micro["MLD"] = mixed_layer_depth( micro["depth"], np.sort(micro["sigmat"], axis = 0), 0.02)

M_MLD = np.tile(micro["MLD"],(micro["pres"][:,0].size,1))

#iINT = micro["depth"]>np.maximum( np.ones(M_MLD.shape)*10, M_MLD )
iINT = micro["depth"]>M_MLD
Nnm = ADCP["NM"]["Au"].shape[0]
micro["N2s"] = np.full(micro["N2"].shape,np.nan)
micro["sh2"] = np.full(micro["N2"].shape,np.nan)
micro["Ri"] = np.full(micro["N2"].shape,np.nan)
micro["intKE"] = np.full(micro["N2"].shape[1],np.nan)
micro["intsh2"] = np.full(micro["N2"].shape[1],np.nan)
ns = 5
vconv = np.full(ns,1.)
vconv /= np.sum(vconv)
for i in range(len(micro["time"])):
    jj = np.argmin( np.abs(micro["time"][i]-ADCP["time"]) )

    micro["intKE"][i] = int_KE_ADCP[jj]
    micro["intsh2"][i] = msh2ADCP[jj]
    sh20 = sh2ADCP[:,jj]
    iif = np.isfinite(sh20)
    if np.sum(iif) == 0 or np.abs(micro["time"][i]-ADCP["time"][jj])>3./24.:
        continue
    intsh2 = intrp.interp1d( ADCP["z"][iif], sh20[iif], bounds_error = False )
    micro["sh2"][:,i] = intsh2(micro["depth"][:,i])

    den = np.copy(micro["sigmat"][:,i])
    sden = np.full(den.size, np.nan)
    sden[int(ns//2):-int(ns//2)] = np.convolve(np.sort(den), vconv, mode = "valid")
    micro["N2s"][:,i] = 9.81/sden*first_centered_differences(micro["depth"][:,i], sden)

    
micro["Ri"] = micro["N2"]/micro["sh2"]

micro["epsilon"] = merge_sensors( micro["epsT1"], micro["epsT2"], micro["fit_flag_T1"] , micro["fit_flag_T2"])
micro["epsilonSH"] = merge_sensors( micro["epsSH1"], micro["epsSH2"], micro["fit_flag_sh1"] , micro["fit_flag_sh2"])
micro["chi"] = merge_sensors( micro["Xif1"], micro["Xif2"], micro["fit_flag_T1"] , micro["fit_flag_T2"])
#micro["LT"] = merge_sensors( micro["LTuT1"], micro["LTuT2"])
micro["KT"] = 0.5*micro["chi"]/micro["grT"]**2
micro["Krho"] = 0.2*micro["epsilon"]/micro["N2"]
micro["BF"] = 0.5*micro["chi"]/micro["grT"]**2*micro["N2"]


zmixLT = np.full(micro["time"].size,np.nan)
for ilt in range(zmixLT.size):
    zmixLT[ilt] = mixing_layer_thorpe( micro["depth"][:,ilt], micro["LT"][:,ilt], ov_coef = 10.)
    """
    fig,ax = plt.subplots()
    ax.semilogx(micro["LT"][:,ilt],-micro["depth"][:,ilt])
    ax.axhline(-zmixLT[ilt], color = "red")
    ax1 = ax.twiny()
    ax1.plot(micro["T"][:,ilt],-micro["depth"][:,ilt], color = "green")
    plt.show()
    """

"""
plt.pcolor(np.log10(micro["LT"]))
plt.plot(2*zmixLT,"r")
plt.show()
fig, ax = plt.subplots()
plt.loglog(micro["epsilonSH"].ravel(), micro["epsilon"].ravel(),'.', ms = 0.5)
plt.xlim((1e-12, 1e-6))
plt.ylim((1e-12, 1e-6))
plt.loglog(plt.xlim(),plt.xlim(),"k")
plt.show()
"""
##integrated water column dissipation
BF_clean = np.copy(micro["BF"])
#BF_clean[ micro["BF"]>5*mix_eff*micro["epsilon"] ] = np.nan
BF_clean[ (micro["N2"]<5e-6) | (9.81*sw.alpha(0.2,10.,0)*micro["grT"]<5e-6) ] = np.nan

H = z_max - np.nanmean(zll_ADCP)
mdepth = np.nanmean(micro["depth"], axis = 1)
#easy
#ii = np.where( (mdepth>5.) & (mdepth<H)  )[0]
#int_diss = H*np.nanmean(micro["epsilon"][ii,:], axis = 0)*1000
#int_BF = H*np.nanmean(BF_clean[ii,:], axis = 0)*1000

#ipsometric function
BAT = io.loadmat("bath/BathGeneva.mat")
bat = -BAT["Bathymetry"]
dx = 25.#*1000#grid is 25 m

Area = np.full(mdepth.size,np.nan)
for (i,z) in enumerate(mdepth):
    if np.isnan(z):
        continue
    Area[i] = np.sum(bat>=z)*dx**2


#volume weighted
int_diss = np.full(micro["time"].size, np.nan)
int_BF = np.full(micro["time"].size, np.nan)
hz = np.nanmean(np.diff(mdepth))
Vol = np.nansum(Area*hz)
for i in range(micro["time"].size):
    jj = np.where( (micro["depth"][:,i]>max([5.,zmixLT[i]])) & (micro["depth"][:,i]<H) & (np.isfinite(micro["epsilon"][:,i])))[0]
    hz = np.nanmean(np.diff(micro["depth"][:,i]))
    int_diss[i] = np.nansum( micro["epsilon"][jj,i]*hz*Area[jj]  )/np.nansum(hz*Area[jj])*Vol/Area[0]*1000
    #int_diss[i] = np.nansum( micro["epsilon"][jj,i]*hz  )/np.nansum(np.ones(jj.size)*hz)*H*1000

    jj = np.where( (micro["depth"][:,i]>max([5.,zmixLT[i]])) & (micro["depth"][:,i]<H) & (np.isfinite(micro["BF"][:,i])))[0]
    int_BF[i] = np.nansum( BF_clean[jj,i]*hz*Area[jj]  )/np.nansum(hz*Area[jj])*Vol/Area[0]*1000
    #int_BF[i] = np.nansum( BF_clean[jj,i]*hz  )/np.nansum(np.ones(jj.size)*hz)*H*1000
    

micro["Area"] = Area
micro["int_diss"] = int_diss
pickle.dump(micro, open("CTD_micro/microstructure_down_Lac_Leman_merged.pi","w+"))

intKE = np.copy(micro["intKE"])
intKE[intKE == 0.] = np.nan
intSH2 = np.copy(micro["intsh2"])

dz = 2.
z = np.arange(dz*2,102,dz)
mepsilon_prof = np.full( z.size, np.nan)
cimepsilon_prof = np.full( (2,z.size), np.nan)
MLEepsilon_prof = np.full( z.size, np.nan)

for i in range(z.size):
    jj =  (micro["depth"]>=z[i]-dz/2.) & (micro["depth"]<=z[i]+dz/2.) & (np.isfinite(micro["epsilon"])) 
    xx = micro["epsilon"][jj]
    mx = np.mean(xx)
    sx = np.std(xx)
    
    print xx.shape
    mepsilon_prof[i] = np.nanmean(xx)
    MLEepsilon_prof[i] = logmean(xx)
    #print xx
    cimepsilon_prof[:,i] = bs.bootstrap(xx, confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.mean)


#######################
#fits epsilon vs TKE #
#######################
x = np.log10(intKE)
y = np.log10(int_diss)
iif = (intKE>0) & (np.isfinite(y)) #(x>0) #& (x<3.3) & (intSH2>1e-6)

r = np.corrcoef(x[iif],y[iif])
pp0 = np.polyfit( x[iif], y[iif], 1 )
#ppP = np.poly1d(pp)
ppi = np.polyfit( y[iif], x[iif], 1 )
pp1 = np.array([1/ppi[0], -ppi[1]/ppi[0]  ])
ppP0 = np.poly1d(pp0)
ppP1 = np.poly1d(pp1)

#pp = 0.5*np.array( [pp0[0]+pp1[0], pp0[1]+pp1[1]])

pp, spp, r = linfit_modelII(x,y)
r2,p2 = stats.pearsonr(x[iif],y[iif])
#pp = np.array([1/pp[0], -pp[1]/pp[0]  ])
ppP = np.poly1d(pp)

ypred = ppP(x)
dissKEmod = 10**ypred
rmse = np.nanmean( (10**y-10**ypred)**2)**0.5

xKE = np.linspace(-1,4,20)
xKEc =  0.5*(xKE[1:]+xKE[0:-1])
ydiss = ppP(xKE)
ydiss0 = ppP0(xKE)
ydiss1 = ppP1(xKE)

xKE = 10**xKE
xKEc = 10**xKEc

a = 10**pp[1]
b = pp[0]

A1 = np.nanmean(int_diss/intKE)
A2 = np.nanmean(intKE/int_diss)**-1

A = (A1+A2)/2.
tau = 1/A/60/60/24
tau1 = 1/A1/60/60/24
tau2 = 1/A2/60/60/24
print tau

A3 = np.nanmean(int_diss)/np.nanmean(intKE)

dissKEiso = A*intKE

#######################
#fits epsilon vs sh2 #
#######################
x = np.log10(intSH2)
y = np.log10(int_diss)
iif = (intSH2>0) & (np.isfinite(y))

rsh = np.corrcoef(x[iif],y[iif])
ppsh0 = np.polyfit( x[iif], y[iif], 1 )
#ppP = np.polyd(pp)
ppshi = np.polyfit( y[iif], x[iif], 1 )
ppsh1 = np.array([1/ppshi[0], -ppshi[1]/ppshi[0]  ])
ppshP0 = np.poly1d(ppsh0)
ppshP1 = np.poly1d(ppsh1)


ppsh, sppsh, rsh = linfit_modelII(x,y)
ppshP = np.poly1d(ppsh)

ypred_sh = ppshP(x)
dissshmod = 10**ypred_sh
rmse_sh = np.nanmean( (10**y-10**ypred_sh)**2)**0.5

xSH2 = np.linspace(-8,-3.5,20)
xSH2c =  0.5*(xSH2[1:]+xSH2[0:-1])
ydiss_sh = ppshP(xSH2)
ydiss0_sh = ppshP0(xSH2)
ydiss1_sh = ppshP1(xSH2)

ash = 10**ppsh[1]
bsh = ppsh[0]

xSH2 = 10**xSH2
xSH2c = 10**xSH2c

fig = plt.figure(figsize = (9,5))
ax0 = fig.add_axes([0.08,0.12,0.42,0.72])
cc=ax0.scatter(intKE, int_diss,20, np.log10(intSH2), cmap = cmocean.cm.thermal)
ax0.plot(xKE, 10**ydiss,"k-")
ax0.fill_between(xKE, y1 = 10**ydiss0, y2= 10**ydiss1, color = "gray", alpha = 0.3, zorder = -1 )
ax0.plot(xKE, A*(xKE),"r")
cbaxes = fig.add_axes([0.12, 0.85, 0.34, 0.03]) 
cb = plt.colorbar(cc, ax = ax0, orientation = "horizontal",cax = cbaxes)
cb.set_ticks(np.arange(-7,-3))
cb.set_ticklabels(["10$^{%d}$"%(i) for i in range(-7,-3)])
cbaxes.xaxis.set_ticks_position("top")
cbaxes.xaxis.set_label_position("top")
cb.set_label("$\\langle sh^2 \\rangle$ [s$^{-2}$]")
ax0.set_xscale("log")
ax0.set_yscale("log")
ax0.set_xlim((5e-1,3e3))
ax0.set_ylim((1e-5,2e-3))
ax0.set_xlabel("KE [J m$^{-2}$]")
ax0.set_ylabel("$\\varepsilon_{\\mu}$ [W m$^{-2}$]")
ax0.annotate("$\\varepsilon = %1.2f \\times 10^{-5} \\times \\mathrm{KE}^{%1.2f \\pm %1.2f}$"%(a*1e5,b,spp[0]), xy = (0.05,0.93), xycoords = "axes fraction")
#ax0.annotate("$\\varepsilon = %1.2e \\times \\mathrm{KE}$\n$\\tau = %1.1f\\,[%1.1f-%1.1f]$ days"%(A, tau, tau1, tau2), xy = (0.05,0.83), xycoords = "axes fraction", color = "r")
ax0.annotate("$\\varepsilon = %1.2f \\times 10^{-5} \\times \\mathrm{KE}$\n"%(A*1e5), xy = (0.05,0.83), xycoords = "axes fraction", color = "r")
ax0.annotate("$r = %1.2f$, $p<0.01$\n$\\mathrm{rmse} = %1.2f$ mW m$^{-2}$"%(r,rmse*1000), xy = (0.95,0.05), xycoords = "axes fraction", ha = "right")

ax1 = fig.add_axes([0.54,0.12,0.42,0.72])
cc=ax1.scatter(intSH2, int_diss,20, np.log10(intKE), cmap = cmocean.cm.thermal)
ax1.plot(xSH2, 10**ydiss_sh,"k-")
ax1.fill_between(xSH2, y1 = 10**ydiss0_sh, y2= 10**ydiss1_sh, color = "gray", alpha = 0.3, zorder = -1 )

cbaxes = fig.add_axes([0.58, 0.85, 0.34, 0.03]) 
cb = plt.colorbar(cc, ax = ax1, orientation = "horizontal",cax = cbaxes)
cb.set_ticks(np.arange(1,4))
cb.set_ticklabels(["10$^{%d}$"%(i) for i in range(1,4)])
cbaxes.xaxis.set_ticks_position("top")
cbaxes.xaxis.set_label_position("top")
cb.set_label("KE [J m$^{-2}$]")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlim((2e-8,1e-3))
ax1.set_ylim((2e-5,2e-3))
ax1.set_yticklabels([])
ax1.set_xlabel("$\\langle sh^2 \\rangle$ [s$^{-2}$]")
ax1.annotate("$\\mathcal{E} = %1.2e \\times \\langle sh^2 \\rangle^{%1.2f}$"%(ash,bsh), xy = (0.05,0.92), xycoords = "axes fraction")
ax1.annotate("$r = %1.2f$, $\\mathrm{rmse} = %1.2f$ mW m$^{-2}$"%(rsh,rmse_sh*1000), xy = (0.95,0.02), xycoords = "axes fraction", ha = "right")
fig.savefig("dissipation_vs_KE_and_sh2.pdf",dpi = 300)
plt.close(fig)


##
# Mixing efficiency
##

LO = (micro["epsilon"]/micro["N2"]**(3./2.))**0.5
LK = (viscosity(micro["T"])**3/ micro["epsilon"])**0.25
FrT = (LO/micro["LT"])**(2./3.)
ReT = ( micro["LT"]/LK )**(4./3.)

Rf = micro["BF"]/(micro["BF"]+micro["epsilon"])
ii =  (micro["N2"]>5e-6) & (9.81*sw.alpha(0.2,10.,0)*micro["grT"]>5e-6) & (Rf>0)
mRf = np.nanmean(Rf[ii])
med_Rf = np.nanmedian(Rf[ii])
mix_eff = mRf/(1-mRf)

ciRf = bs.bootstrap(Rf[ii], confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean)
ci_mix_eff = np.array(ciRf)/(1-np.array(ciRf))


p_Rf,xb_Rf = np.histogram( np.log10( Rf[ii])   , 20)
x_Rf = 10**(0.5*(xb_Rf[1:]+xb_Rf[:-1]))
intp = intrp.interp1d( x_Rf, p_Rf,"nearest" )

eps_x = micro["epsilon"][ii]
BF_y = micro["BF"][ii]

#mean interior
mean_int_epsilon = np.nanmean(eps_x)
ci_int_epsilon = bs.bootstrap(eps_x, confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean)

x_bin = np.arange(-12,-5.5,0.5)
y_bin = np.arange(-13,-6.5,0.5)
hist2d = np.full( (x_bin.size, y_bin.size), np.nan )
for i, x0 in enumerate(x_bin):
    for j, y0 in enumerate(y_bin):
        ii = (np.abs(np.log10(eps_x)-x0)<0.5) & (np.abs(np.log10(BF_y)-y0)<0.5)
        hist2d[i,j] = np.nansum(ii)

#mix_eff = 0.25   

fig = plt.figure(figsize = (9,4))
ax0 = fig.add_axes([0.08,0.14,0.41,0.70])
cc=ax0.scatter(intKE, int_diss,20, np.log10(intSH2), cmap = cmocean.cm.thermal)
ax0.plot(xKE, 10**ydiss,"k-")
ax0.fill_between(xKE, y1 = 10**ydiss0, y2= 10**ydiss1, color = "gray", alpha = 0.3, zorder = -1 )
ax0.plot(xKE, A*(xKE),"r")
cbaxes = fig.add_axes([0.12, 0.87, 0.34, 0.03]) 
cb = plt.colorbar(cc, ax = ax0, orientation = "horizontal",cax = cbaxes)
cb.set_ticks(np.arange(-7,-3))
cb.set_ticklabels(["10$^{%d}$"%(i) for i in range(-7,-3)])
cbaxes.xaxis.set_ticks_position("top")
cbaxes.xaxis.set_label_position("top")
cb.set_label("$\\langle sh^2 \\rangle$ [s$^{-2}$]")
ax0.set_xscale("log")
ax0.set_yscale("log")
ax0.set_xlim((5e-1,3e3))
ax0.set_ylim((1e-5,2e-3))
ax0.set_xlabel("KE [J m$^{-2}$]")
ax0.set_ylabel("$\\mathcal{E}_{\\mathrm{int}}$ [W m$^{-2}$]")
ax0.annotate("$\\mathcal{E}_{\\mathrm{int}} = %1.2f \\times 10^{-5} \\times \\mathrm{KE}^{%1.2f \\pm %1.2f}$"%(a*1e5,b,spp[0]), xy = (0.05,0.93), xycoords = "axes fraction")
#ax0.annotate("$\\varepsilon = %1.2e \\times \\mathrm{KE}$\n$\\tau = %1.1f\\,[%1.1f-%1.1f]$ days"%(A, tau, tau1, tau2), xy = (0.05,0.83), xycoords = "axes fraction", color = "r")
ax0.annotate("$\\mathcal{E}_{\\mathrm{int}} = %1.2f \\times 10^{-5}  \\times \\mathrm{KE}$\n"%(A*1e5), xy = (0.05,0.83), xycoords = "axes fraction", color = "r")
ax0.annotate("$r = %1.2f$, $p<0.01$\n$\\mathrm{rmse} = %1.2f$ mW m$^{-2}$"%(r,rmse*1000), xy = (0.57,0.05), xycoords = "axes fraction", ha = "left")
ax0.annotate("a.", xy = (0.02,0.02), xycoords = "axes fraction", fontweight = "bold")

ax1 = fig.add_axes([0.58,0.14,0.41,0.70])
#ax1.plot(x_Rf, p_Rf, "k-",drawstyle='steps-mid')
ax1.bar(x_Rf, p_Rf, color = "slateblue", ec = "k", width = np.diff(10**xb_Rf))
#ax1.fill_between(x_Rf, p_Rf,color = cm.tab10(0), step = "mid")
ax1.plot(mRf, 180,"kv", mfc = "k", mec = "k", ms = 10)
ax1.plot(med_Rf, 180,"kv", mfc = "w", mec = "k", ms = 10)
ax1.set_xlabel("$R_f = \\dfrac{K_T N^2}{\\varepsilon  + K_T N^2}$")
ax1.set_ylabel("Counts")
ax1.set_xscale("log")
ax1.set_ylim((0,8000))
ax1.set_xlim((1e-4,2))
ax1.annotate( "$\\langle R_f \\rangle = %1.3f$\n$[%1.3f-%1.3f]$\n\n$\\langle \\Gamma \\rangle = %1.3f$\n$[%1.3f-%1.3f]$"%(mRf,ciRf[0], ciRf[1],mix_eff, ci_mix_eff[0], ci_mix_eff[1]), xy = (0.24,0.12), xycoords = "axes fraction", ha = "center" )
ax1.annotate("b.", xy = (0.02,0.02), xycoords = "axes fraction", fontweight = "bold")

ax2 = fig.add_axes([0.67,0.56,0.14,0.24])
ax2.contourf( 10**x_bin, 10**y_bin , hist2d.T, cmap = cm.hot_r)
ax2.plot(np.array([1e-11, 1e-7]), mix_eff*np.array([1e-11, 1e-7]) )
ax2.annotate("$\\Gamma = %1.2f$"%(mix_eff), xy = (0.05,0.8), xycoords = "axes fraction", color = cm.tab10(0))
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_ylabel("$b = K_T N^2$")
ax2.set_xlabel("$ \\varepsilon$")

fig.savefig("KE_epsilon_Gamma.pdf", dpi = 300, bbox_inches = "tight")
plt.close(fig)


####
"""
jj =  (micro["N2"]>5e-6) & (9.81*sw.alpha(0.2,10.,0)*micro["grT"]>5e-6) & (Rf>0) & (np.isfinite(FrT))

x_FrT = FrT[jj]
y_Rf = Rf[jj]


mFrT = np.nanmean(FrT[jj])
ciFrT = bs.bootstrap(FrT[jj], confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean)
med_FrT = np.nanmedian(FrT[jj])


mRf = np.nanmean(Rf[jj])
ciRf = bs.bootstrap(Rf[jj], confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean)
med_Rf = np.nanmedian(Rf[jj])
mGamma = mRf/(1-mRf)


xxb = np.logspace(-1,1,21)
xx = 0.5*xxb[1:]+0.5*xxb[:-1]
yy = np.full(xx.size, np.nan)
nn = np.full(xx.size, np.nan)
for i in range(xx.size):
    ii = (x_FrT>=xxb[i-1]) & (x_FrT<=xxb[i])
    yy[i] = np.nanmedian(y_Rf[ii])
    nn[i] = np.nansum(np.isfinite(y_Rf[ii]))
    
fig = plt.figure()
ax = fig.add_axes([0.12,0.12,0.6,0.6])

ax.loglog(x_FrT, y_Rf,".", ms =1, color = "gray")
l1,=ax.loglog(xx,yy,"s", mec = "k", mfc = "w", mew = 1.5)
l2,=ax.loglog(mFrT, mRf, "*", mec = "k", mfc = "orange", ms = 15)
l,=ax.loglog(np.arange(1,8,1),0.17*np.arange(1,8,1.)**-2, color = "darkred", lw = 2 )
#ax.legend((l1,l2,l),["median","mean","$R_f = 0.17\\, Fr_T^{-2}$"], loc = "upper right")
ax.legend((l,),["$R_f = 0.17\\, Fr_T^{-2}$"], loc = "upper right")
#ax.annotate("$R_f = 0.17\\, Fr_T^{-2}$", xy = (5,0.25))
ax.set_yscale("log")
ax.set_xlim((1e-1,2.5e1))
ax.set_xlabel("$Fr_T = (L_O/L_E)^{2/3}$")
ax.set_ylabel("$R_f$")#
yl = ax.get_ylim()

ax1 = fig.add_axes([0.74,0.12,0.2,0.6])
ax1.hist( np.log10(Rf[Rf>0]), 20,ec = "k" , color = "slateblue",orientation = "horizontal")
ax1.plot(90, np.log10(mRf), "<", mfc = "k", mec = "k", ms = 8)
ax1.plot(90, np.log10(med_Rf), "<", mfc = "w", mec = "k", ms = 8)
ax1.set_ylim(( np.log10(yl[0]),np.log10(yl[1]) ))
ax1.set_yticklabels([])
ax1.annotate("$R_f$ mean:\n%1.3f\n[%1.3f-%1.3f]\nmed:\n$%1.3f$"%(mRf,ciRf[0],ciRf[1],med_Rf), xy = (0.5,0.2), xycoords = "axes fraction", ha = "center", va = "center", fontsize = 9)

ax2 = fig.add_axes([0.12,0.74,0.6,0.2])
ax2.bar( np.log10(xx),nn, width = 0.08, color = "slateblue", ec = "k" )
ax2.plot( np.log10(mFrT), 120,"v", mfc = "k", mec = "k", ms = 8)
ax2.plot( np.log10(med_FrT),120, "v", mfc = "w", mec = "k", ms = 8)
#ax2.set_ylim((0,4000))
ax2.set_xlim((-1, np.log10(25.)))
ax2.yaxis.tick_right()
ax2.set_ylabel("Counts")
ax2.yaxis.set_label_position("right")
ax2.set_xticks(np.log10([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,20]))
ax2.set_xticklabels([])
ax2.annotate("$Fr_T$ mean:\n$%1.3f\\,[%1.3f-%1.3f]$\nmed:\n$%1.3f$"%(mFrT,ciFrT[0],ciFrT[1],med_FrT), xy = (0.2,0.60), xycoords = "axes fraction", ha = "center", va = "center", fontsize = 9)

ax3 = fig.add_axes([0.24,0.22,0.18,0.18])
ax3.contourf( 10**x_bin, 10**y_bin , hist2d.T, cmap = cm.hot_r)
ax3.plot(np.array([1e-10, 3e-7]), mGamma*np.array([1e-10, 3e-7]) )
ax3.annotate("$\\Gamma = %1.2f$"%(mGamma), xy = (0.05,0.8), xycoords = "axes fraction", color = cm.tab10(0))
ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_ylabel("$\\mathcal{B} = K_T N^2$")
ax3.set_xlabel("$ \\varepsilon$")

fig.savefig("Rf_FrT.pdf", dpi = 300)
fig.savefig("Rf_FrT.png", dpi = 300)
plt.show()
"""


#residence time
ytau = 1/a*xKE**(1-b)


#Dissipation from ADCP
epsilon_ADCP_KE_fit = a* int_KE_ADCP**b
epsilon_ADCP_KE_iso = A* int_KE_ADCP
epsilon_ADCP_SH2_fit = ash*msh2ADCP**bsh


print np.nanmean(int_diss)*1000
print np.nanmean(epsilon_ADCP_KE_fit)*1000
print np.nanmean(epsilon_ADCP_KE_iso)*1000
print np.nanmean(epsilon_ADCP_SH2_fit)*1000


"""
#Plots mean epsilon profile
fig, ax = plt.subplots(figsize = (4,6))
ax.plot(mepsilon_prof*1e9, z,'ks-', mfc = "w")
#ax.plot(np.nanmean(us**3)/0.41/(z_max-z[30:])*1e9, z[30:])
ax.fill_betweenx(z, cimepsilon_prof[0,:]*1e9, cimepsilon_prof[1,:]*1e9, color = "gray", alpha = 0.2)
ax.set_ylim((110,0))
ax.set_ylabel("Depth [m]")
ax.grid(True)
#ax.set_xscale("log")
ax.set_xlabel("$\\varepsilon$ [$10^{-9}$ W kg$^{-1}$]")
fig.savefig("mean_epsilon_profile.png", dpi = 300, bbox_inches = "tight")
plt.show()
"""

intQPE = intrp.interp1d(ST["timeGRs"],ST["HB"]["QPE"], bounds_error = False)
QPE = intQPE(ADCP["time"])


intBF = intrp.interp1d(ST["timeGRs"],ST["HB"]["int_turbBF"], bounds_error = False)
BF = intBF(ADCP["time"])

#total_Prod = ADCP["wind_work"]["gamma_fit"]*ADCP["wind_work"]["smooth_P10"]*1e3 + QPE
#total_diss = BF*1e3 + diss_bottom_smooth*1000 + diss_KEiso_smooth*1000 


######
##### METEO VARIABLES
#####
nnmet = 7*24*6
WS = moving_average( MET["WS_10m"], nnmet, window = "gauss" )
P10 = moving_average( MET["P10"], nnmet, window = "gauss" )
Qnet = moving_average( MET["sHFnet"], nnmet, window = "gauss" )

WS_micro = np.full(micro["time"].size, np.nan)
P10_micro = np.full(micro["time"].size, np.nan)
WS_micro_week = np.full(micro["time"].size, np.nan)
P10_micro_week = np.full(micro["time"].size, np.nan)
for i in range(micro["time"].size):
    ii = np.where( np.abs( micro["time"][i]-MET["datenum"] )<=1 )[0]
    WS_micro[i] = np.nanmean( MET["WS_10m"][ii] )
    P10_micro[i] = np.nanmean( MET["P10"][ii] )

    WS_micro_week[i] = np.nanmean( WS[ii] )
    P10_micro_week[i] = np.nanmean( P10[ii] )


#Atmospheric - Surface dissipation
usa = MET["ustar"]*(MET["rho0"]/MET["rho_air"])**0.5
delta_at = 11*nu_air/usa
epsilon_atm_1m = usa**3/0.41
diss_atm = epsilon_atm_1m*np.log(10./delta_at)*MET["rho_air"]*1000 #mW/m2
dissmol_atm = 11*MET["rho_air"]*usa**3*1000

delta_surf = 11*viscosity(MET["swT"])/MET["ustar"]
ek_surf = 0.41*MET["ustar"]/ifrq
diss_surf = MET["ustar"]**3*np.log(ek_surf/delta_surf)/0.41*MET["rho0"]*1000 #mW/m2
dissmol_surf = 11*MET["rho0"]*MET["ustar"]**3*1000

print "Mean turbulent dissipation ABL %1.3f mW m-2"%( np.nanmean( diss_atm ) )
print "Mean laminar dissipation ABL %1.3f mW m-2"%( np.nanmean( dissmol_atm ) )

print "Mean turbulent dissipation SBL %1.3f mW m-2"%( np.nanmean( diss_surf ) )
print "Mean laminar dissipation SBL %1.3f mW m-2"%( np.nanmean( dissmol_surf ) )

#################################
###############################
##Averages every 3 months
##############################
##############################

#Potential energy
TM = pickle.load(open("thermistor_mooring_gridded_1hour_NM.pi"))
TC = pickle.load(open("thermistor_chain/Thermistor_chain_NM.pi"))

dScdt = first_centered_differences( ST["timeGRs"]*60*60*24, ST["ScGR"] )
Sc_cor = ST["ScGR"] + np.cumsum( ST["HB"]["int_turbBF"]*60*60*24 )


#time_avg = np.arange( ADCP["time"][0],ADCP["time"][-1],62)
ndays = ADCP["time"][-1] - ADCP["time"][0]
nn = 4
dint = ndays//nn
time_avg = np.arange( ADCP["time"][0],ADCP["time"][-1],dint)
date_avg = convert_time_to_date(time_avg)

time_avgC = 0.5*(time_avg[1:]+time_avg[:-1])
date_avgC = convert_time_to_date(time_avgC)
Navg = time_avgC.size



###
#mean profiles per season
zKE_ADCP = 0.5*(U_ADCP)**2+0.5*(V_ADCP)**2
zKE_ADCP[zKE_ADCP == 0] = np.nan

dz = 2.
dzadcp = 2.
z = np.arange(dz*2,102,dz)
sea_m_KE_prof = np.full( (z.size,nn), np.nan)
sea_ci_KE_prof = np.full( (2,z.size,nn), np.nan)

sea_m_us_prof = np.full( (z.size,nn), np.nan)
sea_ci_us_prof = np.full( (2,z.size,nn), np.nan)

sea_m_epsilon_prof = np.full( (z.size,nn), np.nan)
sea_ci_epsilon_prof = np.full( (2,z.size,nn), np.nan)

sea_m_chi_prof = np.full( (z.size,nn), np.nan)
sea_ci_chi_prof = np.full( (2,z.size,nn), np.nan)

sea_m_LT_prof = np.full( (z.size,nn), np.nan)
sea_ci_LT_prof = np.full( (2,z.size,nn), np.nan)

sea_m_grT_prof = np.full( (z.size,nn), np.nan)
sea_ci_grT_prof = np.full( (2,z.size,nn), np.nan)

sea_m_N2_prof = np.full( (z.size,nn), np.nan)
sea_ci_N2_prof = np.full( (2,z.size,nn), np.nan)

sea_m_T_prof = np.full( (z.size,nn), np.nan)
sea_ci_T_prof = np.full( (2,z.size,nn), np.nan)

sea_m_S_prof = np.full( (z.size,nn), np.nan)
sea_ci_S_prof = np.full( (2,z.size,nn), np.nan)

sea_m_BFhb_prof = np.full( (z.size,nn), np.nan)
sea_ci_BFhb_prof = np.full( (2,z.size,nn), np.nan)

sea_m_KThb_prof = np.full( (z.size,nn), np.nan)
sea_ci_KThb_prof = np.full( (2,z.size,nn), np.nan)

sea_m_hmix = np.full( (nn), np.nan)
sea_ci_hmix = np.full( (2,nn), np.nan)
for i in range(Navg):
    imicro = (micro["time"]>=time_avg[i]) & (micro["time"]<=time_avg[i+1])
    iHB = (ST["timeGRs"]>=time_avg[i]) & (ST["timeGRs"]<=time_avg[i+1])
    iadcp = (ADCP["time"]>=time_avg[i]) & (ADCP["time"]<=time_avg[i+1])
    sea_m_hmix[i] = np.nanmean(zmixLT[imicro])
    sea_ci_hmix[:,i] = bs.bootstrap(zmixLT[imicro], confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean)

    for j in range(z.size):
        #depths index adcp
        jjadcp =  (ADCP["z"]>=z[j]-dzadcp/2.) & (ADCP["z"][:]<=z[j]+dzadcp/2.)
        #epsilon
        xx = zKE_ADCP[:, iadcp]
        xx = xx[jjadcp]
        xx = xx[np.isfinite(xx)]
        if xx.size >= 10:
            sea_m_KE_prof[j,i] = np.nanmean(xx)
            sea_ci_KE_prof[:,j,i] = bs.bootstrap(xx, confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.mean)

        xx = (Cd_ADCP*2*zKE_ADCP[:, iadcp])**(3/2.)
        xx = xx[jjadcp]
        xx = xx[np.isfinite(xx)]
        if xx.size >= 10:
            sea_m_us_prof[j,i] = np.nanmean(xx)
            sea_ci_us_prof[:,j,i] = bs.bootstrap(xx, confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.mean)

            
        ##depth index microprofiler
        jj =  (micro["depth"][:, imicro]>=z[j]-dz/2.) & (micro["depth"][:, imicro]<=z[j]+dz/2.)

        #epsilon
        xx = micro["epsilon"][:, imicro]
        xx = xx[jj]
        xx = xx[np.isfinite(xx)]
        mx = np.mean(xx)
        sx = np.std(xx)
        #xx = xx[xx<=mx+10*sx]
        #mx = np.mean(xx)
        #sx = np.std(xx)
        if xx.size >= 10:
            sea_m_epsilon_prof[j,i] = np.nanmean(xx)
            sea_ci_epsilon_prof[:,j,i] = bs.bootstrap(xx, confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.mean)

        #chi
        xx = micro["chi"][:, imicro]
        xx = xx[jj]
        xx = xx[np.isfinite(xx)]
        mx = np.mean(xx)
        sx = np.std(xx)
        #xx = xx[xx<=mx+5*sx]
        #mx = np.mean(xx)
        #sx = np.std(xx)
        if xx.size >= 10:
            sea_m_chi_prof[j,i] = np.nanmean(xx)
            sea_ci_chi_prof[:,j,i] = bs.bootstrap(xx, confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.mean)

        #LT
        xx = micro["LT"][:, imicro]
        xx = xx[jj]
        xx = xx[np.isfinite(xx)]
        mx = np.mean(xx)
        sx = np.std(xx)
        #xx = xx[xx<=mx+5*sx]
        #mx = np.mean(xx)
        #sx = np.std(xx)
        if xx.size >= 10:
            sea_m_LT_prof[j,i] = np.nanmean(xx)
            sea_ci_LT_prof[:,j,i] = bs.bootstrap(xx, confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.mean)

        #grT
        xx = micro["grT"][:, imicro]
        xx = xx[jj]
        xx = xx[np.isfinite(xx)]
        mx = np.mean(xx)
        sx = np.std(xx)
        if xx.size >= 10:
            sea_m_grT_prof[j,i] = np.nanmean(xx)
            sea_ci_grT_prof[:,j,i] = bs.bootstrap(xx, confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.mean)

        #grT
        xx = micro["N2"][:, imicro]
        xx = xx[jj]
        xx = xx[np.isfinite(xx)]
        mx = np.mean(xx)
        sx = np.std(xx)
        if xx.size >= 10:
            sea_m_N2_prof[j,i] = np.nanmean(xx)
            sea_ci_N2_prof[:,j,i] = bs.bootstrap(xx, confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.mean)

        #T
        xx = micro["T"][:, imicro]
        xx = xx[jj]
        xx = xx[np.isfinite(xx)]
        mx = np.mean(xx)
        sx = np.std(xx)
        if xx.size >= 10:
            sea_m_T_prof[j,i] = np.nanmean(xx)
            sea_ci_T_prof[:,j,i] = bs.bootstrap(xx, confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.mean)

        #S
        xx = micro["S"][:, imicro]
        xx = xx[jj]
        xx = xx[np.isfinite(xx)]
        mx = np.mean(xx)
        sx = np.std(xx)
        if xx.size >= 10:
            sea_m_S_prof[j,i] = np.nanmean(xx)
            sea_ci_S_prof[:,j,i] = bs.bootstrap(xx, confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.mean)


        ##depth heat budget
        jj =  (ST["depth"]>=z[j]-dz/2.) & (ST["depth"]<=z[j]+dz/2.)

        #buoyancy flux heat budget
        xx = -ST["HB"]["turbBF_th"][:, iHB]
        xx = xx[jj]
        xx[np.isnan(xx)] = 0.
        
        mx = np.mean(xx)
        sx = np.std(xx)
        if xx.size > 10:
            sea_m_BFhb_prof[j,i] = np.nanmean(xx)
            sea_ci_BFhb_prof[:,j,i] = bs.bootstrap(xx, confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.mean)

        #buoyancy flux heat budget
        xx = ST["HB"]["Kz"][:, iHB]
        xx = xx[jj]
        xx = xx[np.isfinite(xx)]
        mx = np.mean(xx)
        sx = np.std(xx)
        if xx.size >= 10:
            sea_m_KThb_prof[j,i] = np.nanmean(xx)
            sea_ci_KThb_prof[:,j,i] = bs.bootstrap(xx, confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.mean)

sea_alpha = sw.alpha(sea_m_S_prof, sea_m_T_prof,0)
sea_m_KT_prof = 0.5*sea_m_chi_prof/sea_m_grT_prof**2
sea_m_BF_prof = 0.5*sea_m_chi_prof/sea_m_grT_prof*sea_alpha*sw.g(46.)
sea_m_Reb_prof = sea_m_epsilon_prof/viscosity(sea_m_T_prof)/sea_m_N2_prof
sea_m_LO_prof = sea_m_epsilon_prof**0.5/sea_m_N2_prof**(3./4.)

sea_ci_KT_prof = 0.5*sea_ci_chi_prof/sea_m_grT_prof**2
sea_ci_BF_prof = 0.5*sea_ci_chi_prof/sea_m_grT_prof*sea_alpha*sw.g(46.)
sea_ci_Reb_prof = sea_m_epsilon_prof/viscosity(sea_m_T_prof)/sea_m_N2_prof

sea_m_us_prof = sea_m_us_prof**(1/3.)
sea_ci_us_prof = sea_ci_us_prof**(1/3.)


Area_ = np.full(z.size,np.nan)
for (i,zz) in enumerate(z):
    if np.isnan(zz):
        continue
    Area_[i] = np.sum(bat>=zz)*dx**2

dArea_ = Area_[0:-1]-Area_[1:]
Vol_ = dz*(Area_[0:-1]+Area_[1:])/2.

intDISS_meas_avg_0 = np.full((3,Navg), np.nan)
intBF_meas_avg_0 = np.full((3,Navg), np.nan)
intBF_hb_avg_0 = np.full((3,Navg), np.nan)
for i in range(Navg):
    jj = np.where( (z>max([5.,sea_ci_hmix[1,i]])) & (z<H) & (np.isfinite(sea_m_epsilon_prof[:,i])))[0]
    intDISS_meas_avg_0[0,i] = np.nansum( sea_m_epsilon_prof[jj,i]*dz*Area_[jj]  )/np.nansum(dz*Area_[jj])*Vol/Area[0]*1e6
    intDISS_meas_avg_0[1,i] = np.nansum( sea_ci_epsilon_prof[0,jj,i]*dz*Area_[jj]  )/np.nansum(dz*Area_[jj])*Vol/Area[0]*1e6
    intDISS_meas_avg_0[2,i] = np.nansum( sea_ci_epsilon_prof[1,jj,i]*dz*Area_[jj]  )/np.nansum(dz*Area_[jj])*Vol/Area[0]*1e6

    intBF_meas_avg_0[0,i] = np.nansum( sea_m_BF_prof[jj,i]*dz*Area_[jj]  )/np.nansum(dz*Area_[jj])*Vol/Area[0]*1e6
    intBF_meas_avg_0[1,i] = np.nansum( sea_ci_BF_prof[0,jj,i]*dz*Area_[jj]  )/np.nansum(dz*Area_[jj])*Vol/Area[0]*1e6
    intBF_meas_avg_0[2,i] = np.nansum( sea_ci_BF_prof[1,jj,i]*dz*Area_[jj]  )/np.nansum(dz*Area_[jj])*Vol/Area[0]*1e6

    intBF_hb_avg_0[0,i] = np.nansum( sea_m_BFhb_prof[jj,i]*dz*Area_[jj]/Area[0]  )*1e3 #/np.nansum(dz*Area_[jj])*Vol/Area[0]*1e3
    intBF_hb_avg_0[1,i] = np.nansum( sea_ci_BFhb_prof[0,jj,i]*dz*Area_[jj]/Area[0]  )*1e3#/np.nansum(dz*Area_[jj])*Vol/Area[0]*1e3
    intBF_hb_avg_0[2,i] = np.nansum( sea_ci_BFhb_prof[1,jj,i]*dz*Area_[jj]/Area[0]  )*1e3#/np.nansum(dz*Area_[jj])*Vol/Area[0]*1e3



#gamma = ADCP["wind_work"]["gamma_fit"]
#gamma = ADCP["wind_work"]["gamma"]
#gamma= 0.5*(ADCP["wind_work"]["gamma_fit"] + ADCP["wind_work"]["gamma"])
#gamma = 0.4/100
KE_avg = np.full((4,Navg), np.nan)
btKE_avg = np.full((4,Navg), np.nan)
bcKE_avg = np.full((4,Navg), np.nan)

PE_TM_avg = np.full((4,Navg), np.nan)
PE_TC_avg = np.full((4,Navg), np.nan)

U10_avg = np.full((4,Navg), np.nan)

Vbot_avg = np.full((4,Navg), np.nan)
Vbot_AD_avg = np.full((4,Navg), np.nan)


RW_avg = np.full((4,Navg), np.nan)
P10_avg = np.full((4,Navg), np.nan)
P10micro_avg = np.full((4,Navg), np.nan)
QPE_avg = np.full((4,Navg), np.nan)

surfDISS_avg = np.full((4,Navg), np.nan)
atmDISS_avg = np.full((4,Navg), np.nan)

intDISS_meas_avg = np.full((4,Navg), np.nan)
intDISS_KE_avg = np.full((4,Navg), np.nan)
intDISS_KEiso_avg = np.full((4,Navg), np.nan)

intBF_meas_avg = np.full((4,Navg), np.nan)
intBF_KE_avg = np.full((4,Navg), np.nan)
intBF_KEiso_avg = np.full((4,Navg), np.nan)

botDISS_meas_avg = np.full((4,Navg), np.nan)
botDISS_ADCP_avg = np.full((4,Navg), np.nan)
botDISS_ADCd_avg = np.full((4,Navg), np.nan)

botmolDISS_meas_avg = np.full((4,Navg), np.nan)
botmolDISS_ADCP_avg = np.full((4,Navg), np.nan)
botmolDISS_ADCd_avg = np.full((4,Navg), np.nan)


mix_eff_bot = 0.13
BF_bb = np.full((4,Navg), np.nan)#BF_avg - intBF_KE_avg
BF_tot = np.full((4,Navg), np.nan)#BF_avg - intBF_KE_avg

BF_avg = np.full((4,Navg), np.nan)
APEchange_avg = np.full((4,Navg), np.nan)
Sc_avg = np.full((4,Navg), np.nan)

bottom_area = Area[0] - Area[np.nanargmin(np.abs(mdepth-z_max))]
#bot_scale = bottom_area/Area[0]
bot_scale = 1
for i in range(Navg):
    imet = (MET["datenum"]>=time_avg[i]) & (MET["datenum"]<=time_avg[i+1])
    xx = MET["WS_10m"]
    U10_avg[0,i] = np.nanmean(xx[imet])
    U10_avg[1:3,i] = bs.bootstrap(xx[imet], confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean)
    U10_avg[3,i] = np.nanstd(xx[imet])
    
    xx = diss_atm
    atmDISS_avg[0,i] = np.nanmean(xx[imet])
    atmDISS_avg[1:3,i] = bs.bootstrap(xx[imet], confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean)
    atmDISS_avg[3,i] = np.nanstd(xx[imet])
    
    xx = diss_surf
    surfDISS_avg[0,i] = np.nanmean(xx[imet])
    surfDISS_avg[1:3,i] = bs.bootstrap(xx[imet], confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean)
    surfDISS_avg[3,i] = np.nanstd(xx[imet])
    
    iadcp = (ADCP["time"]>=time_avg[i]) & (ADCP["time"]<=time_avg[i+1])
    #Vbot
    Vbot_avg[0,i] = np.nanmean(V_bot[iadcp]**2)**0.5*100
    Vbot_avg[1:3,i] = bs.bootstrap(V_bot[iadcp]**2, confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean)
    Vbot_avg[1:3,i] = Vbot_avg[1:3,i]**0.5*100
    Vbot_avg[3,i] = np.nanstd(V_bot[iadcp]**2)**0.5*100
    
    #KE
    xx = np.sum(ADCP["NM"]["KE"], axis = 0)
    xx[xx==0] = np.nan
    KE_avg[0,i] = np.nanmean(xx[iadcp])
    KE_avg[1:3,i] = bs.bootstrap(xx[iadcp], confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean)
    KE_avg[3,i] = np.nanstd(xx[iadcp])
    
    #barotropic
    xx = np.copy(ADCP["NM"]["KE"][0,:])
    xx[xx==0] = np.nan
    btKE_avg[0,i] = np.nanmean(xx[iadcp])
    btKE_avg[1:3,i] = bs.bootstrap(xx[iadcp], confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean)
    btKE_avg[3,i] = np.nanstd(xx[iadcp])
    
    #baroclinic
    xx = np.sum(ADCP["NM"]["KE"][1:,:], axis = 0)
    xx[xx==0] = np.nan
    bcKE_avg[0,i] = np.nanmean(xx[iadcp])
    bcKE_avg[1:3,i] = bs.bootstrap(xx[iadcp], confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean)
    bcKE_avg[3,i] = np.nanstd(xx[iadcp])
    
    #PE
    iTM = (TM["time"]>=time_avg[i]) & (TM["time"]<=time_avg[i+1])                
    xx = np.sum(TM["mPE"], axis = 0)
    xx[xx==0] = np.nan
    PE_TM_avg[0,i] = np.nanmean(xx[iTM])
    PE_TM_avg[1:3,i] = bs.bootstrap(xx[iTM], confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean)
    PE_TM_avg[3,i] = np.nanstd(xx[iTM])
    
    #PETC
    iTC = (TC["time"]>=time_avg[i]) & (TC["time"]<=time_avg[i+1])                
    xx = np.sum(TC["mPE"], axis = 0)
    xx[xx==0] = np.nan
    xx=xx[iTC]
    xx = xx[np.isfinite(xx)]
    if xx.size>0:
        PE_TC_avg[0,i] = np.nanmean(xx)
        PE_TC_avg[1:3,i] = bs.bootstrap(xx, confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean)
        PE_TC_avg[3,i] = np.nanstd(xx)
        
    #production
    RW_avg[0,i] = np.nanmean( ADCP["wind_work"]["RW"][iadcp]*1000 )
    RW_avg[1:3,i] = bs.bootstrap(ADCP["wind_work"]["RW"][iadcp]*1000, confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean) 
    RW_avg[3,i] = np.nanstd( ADCP["wind_work"]["RW"][iadcp]*1000 )
    
    P10_avg[0,i] = np.nanmean( ADCP["wind_work"]["P10"][iadcp]*1000 )
    P10_avg[1:3,i] = bs.bootstrap(ADCP["wind_work"]["P10"][iadcp]*1000, confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean) 
    P10_avg[3,i] = np.nanstd( ADCP["wind_work"]["P10"][iadcp]*1000 )
    
    QPE_avg[0,i] = np.nanmean( QPE[iadcp] )
    QPE_avg[1:3,i] = bs.bootstrap(QPE[iadcp], confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean) 
    QPE_avg[3,i] = np.nanstd( QPE[iadcp] )
    
    #water column dissipation
    imicro = (micro["time"]>=time_avg[i]) & (micro["time"]<=time_avg[i+1])

    P10micro_avg[0,i] = np.nanmean( P10_micro[imicro]*1000 )
    P10micro_avg[1:3,i] = bs.bootstrap(P10_micro[imicro]*1000, confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean)
    P10micro_avg[3,i] = np.nanstd( P10_micro[imicro]*1000 )
    
    intDISS_meas_avg[0,i] = np.nanmean(int_diss[imicro]*1000)
    intDISS_meas_avg[1:3,i] = bs.bootstrap(int_diss[imicro]*1000, confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean) 
    intDISS_meas_avg[3,i] = np.nanstd(int_diss[imicro]*1000)
    
    x = epsilon_ADCP_KE_fit[iadcp]*1000
    iif = np.isfinite(x)
    intDISS_KE_avg[0,i] = np.nanmean(x[iif])
    intDISS_KE_avg[1:3,i] = bs.bootstrap(x[iif], confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean) 
    intDISS_KE_avg[3,i] = np.nanstd(x[iif])
    
    x = epsilon_ADCP_KE_iso[iadcp]*1000
    iif = np.isfinite(x)
    intDISS_KEiso_avg[0,i] = np.nanmean(x[iif])
    intDISS_KEiso_avg[1:3,i] = bs.bootstrap(x[iif], confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean)
    intDISS_KEiso_avg[3,i] = np.nanstd(x[iif])
    #water buoyancy flux


    intBF_meas_avg[0,i] = np.nanmean(int_BF[imicro]*1000)
    intBF_meas_avg[1:3,i] = bs.bootstrap(int_BF[imicro]*1000, confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean) 
    intBF_meas_avg[3,i] = np.nanstd(int_BF[imicro]*1000)
    
    x = mix_eff*epsilon_ADCP_KE_fit[iadcp]*1000
    iif = np.isfinite(x)
    intBF_KE_avg[0,i] = np.nanmean(x[iif])
    intBF_KE_avg[1:3,i] = bs.bootstrap(x[iif], confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean) 
    intBF_KE_avg[3,i] = np.nanstd(x[iif])
    
    x = mix_eff*epsilon_ADCP_KE_iso[iadcp]*1000
    iif = np.isfinite(x)
    intBF_KEiso_avg[0,i] = np.nanmean(x[iif])
    intBF_KEiso_avg[1:3,i] = bs.bootstrap(x[iif], confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean)
    intBF_KEiso_avg[3,i] = np.nanstd(x[iif])
    

    #bottom dissipation
    iAD = (AD["time"]>=time_avg[i]) & (AD["time"]<=time_avg[i+1])
    if np.sum(iAD) > 0:
        #Vbot
        Vbot_AD_avg[0,i] = np.nanmean(V_bot_AD[iAD]**2)**0.5*100
        Vbot_AD_avg[1:3,i] = bs.bootstrap( np.abs(V_bot_AD[iAD]), confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean)
        Vbot_AD_avg[1:3,i] = Vbot_AD_avg[1:3,i]**0.5*100 #cm/s
        Vbot_AD_avg[3,i] = np.nanstd(V_bot_AD[iAD]**2)**0.5*100

        xx = diss_AD_bot[iAD]*bot_scale
        botDISS_meas_avg[0,i] = np.nanmean(xx )
        botDISS_meas_avg[1:3,i] =  bs.bootstrap(xx , confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean)
        botDISS_meas_avg[3,i] = np.nanstd(xx)
        
        xx = diss_ADCd_bot[iAD]*bot_scale
        botDISS_ADCd_avg[0,i] = np.nanmean(xx)
        botDISS_ADCd_avg[1:3,i] =  bs.bootstrap(xx, confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean)
        botDISS_ADCd_avg[3,i] = np.nanstd(xx)
        
        xx = us_AD_meas[iAD]**3*11*1e6*bot_scale
        botmolDISS_meas_avg[0,i] = np.nanmean(xx)
        botmolDISS_meas_avg[1:3,i] =  bs.bootstrap(xx, confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean)
        botmolDISS_meas_avg[3,i] = np.nanstd(xx)
        
        xx = us_AD[iAD]**3*11*1e6*bot_scale
        botmolDISS_ADCd_avg[0,i] = np.nanmean( xx)
        botmolDISS_ADCd_avg[1:3,i] =  bs.bootstrap( xx, confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean)
        botmolDISS_ADCd_avg[3,i] = np.nanstd( xx)
        
    botDISS_ADCP_avg[0,i] = np.nanmean(diss_ADCP_bot[iadcp]*bot_scale )
    botDISS_ADCP_avg[1:3,i] =  bs.bootstrap(diss_ADCP_bot[iadcp]*bot_scale, confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean)
    botDISS_ADCP_avg[3,i] = np.nanstd(diss_ADCP_bot[iadcp]*bot_scale )
    
    botmolDISS_ADCP_avg[0,i] = np.nanmean(us[iadcp]**3*11*1e6*bot_scale)
    botmolDISS_ADCP_avg[1:3,i] =  bs.bootstrap(us[iadcp]**3*11*1e6*bot_scale, confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean)
    botmolDISS_ADCP_avg[3,i] = np.nanstd(us[iadcp]**3*11*1e6*bot_scale)
    
    BF_bb[0,i] = np.nanmean(mix_eff_bot*diss_ADCP_bot[iadcp]*bot_scale )
    BF_bb[1:3,i] = bs.bootstrap(mix_eff_bot*diss_ADCP_bot[iadcp]*bot_scale, confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean)
    BF_bb[3,i] = np.nanstd(mix_eff_bot*diss_ADCP_bot[iadcp]*bot_scale )
    
    BF_tot[0,i]  = np.nanmean(mix_eff_bot*diss_ADCP_bot[iadcp]*bot_scale + mix_eff*epsilon_ADCP_KE_fit[iadcp]*1000 )
    BF_tot[1:3,i] = bs.bootstrap(mix_eff_bot*diss_ADCP_bot[iadcp]*bot_scale +  mix_eff*epsilon_ADCP_KE_fit[iadcp]*1000, confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean)
    BF_tot[3,i]  = np.nanstd(mix_eff_bot*diss_ADCP_bot[iadcp]*bot_scale + mix_eff*epsilon_ADCP_KE_fit[iadcp]*1000 )
    
    #Buoyancy flux
    iHB = (ST["timeGRs"]>=time_avg[i]) & (ST["timeGRs"]<=time_avg[i+1])
    BF_avg[0,i] = np.nanmean(ST["HB"]["int_turbBF"][iHB]*1e3)
    BF_avg[1:3,i] =  bs.bootstrap(ST["HB"]["int_turbBF"][iHB]*1e3, confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean)
    BF_avg[3,i] = np.nanstd(ST["HB"]["int_turbBF"][iHB]*1e3)
    
    #Potential energy change
    APEchange_avg[0,i] = np.nanmean( dScdt[iHB]*1e3)
    APEchange_avg[1:3,i] =  bs.bootstrap(dScdt[iHB]*1e3, confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean)
    APEchange_avg[3,i] = np.nanstd( dScdt[iHB]*1e3)
    
    #Potential energy
    Sc_avg[0,i] = np.nanmean( ST["ScGR"][iHB]*1e-3) #kJ/m-2
    Sc_avg[1:3,i] =  bs.bootstrap(ST["ScGR"][iHB]*1e-3, confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean)
    Sc_avg[3,i] = np.nanstd( ST["ScGR"][iHB]*1e-3) #kJ/m-2
    
    gamma_avg = RW_avg[0,:]/P10_avg[0,:]*100


tot_diss = np.nanmean(epsilon_ADCP_KE_fit*1000)+np.nanmean(diss_ADCP_bot*bot_scale) + np.nanmean(us**3*11*1e6*bot_scale)+np.nanmean(mix_eff*epsilon_ADCP_KE_fit*1000)+ np.nanmean(mix_eff_bot*diss_ADCP_bot*bot_scale)

##SUMMARY PLOT
#BF_avg = np.copy(intBF_hb_avg_0)
BF_int = intBF_KE_avg
#BF_bb = mix_eff_bot*botDISS_ADCP_avg#BF_avg - intBF_KE_avg
#BF_bb[0,BF_bb[0,:]<0] = 0.

#BF_loc =intDISS_KE_avg[0,:]*mRf/(1-mRf)
#BF_tot[BF_tot<BF_loc] = BF_loc[BF_tot<BF_loc]
Dissipation = np.c_[intDISS_KE_avg[0,:], botDISS_ADCP_avg[0,:],botmolDISS_ADCP_avg[0,:],  BF_int[0,:],BF_bb[0,:]].T
mDiss = np.nanmean(Dissipation, axis =1)

gamma = np.nansum(np.nanmean(tot_diss))/np.nanmean(P10*1000)

Production = np.c_[gamma*P10_avg[0,:],QPE_avg[0,:]].T
mProd = np.nanmean(Production, axis =1)



eProduction = np.c_[ 0.5*(gamma*P10_avg[2,:]-gamma*P10_avg[1,:]),0.5*(QPE_avg[2,:]-QPE_avg[1,:])].T
eDissipation = np.c_[0.5*(intDISS_KE_avg[2,:]- intDISS_KE_avg[1,:]), 0.5*(botDISS_ADCP_avg[2,:]-botDISS_ADCP_avg[1,:]),0.5*(botmolDISS_ADCP_avg[2,:]-botmolDISS_ADCP_avg[1,:]), 0.5*(BF_int[2,:]-BF_int[1,:]),0.5*(BF_bb[2,:]-BF_bb[1,:])].T



PE_to_KE = 0.5*(np.nanmean(PE_TM_avg[0,:])+ np.nanmean(PE_TC_avg[0,:]))/np.nanmean(KE_avg[0,:])
PKE = KE_avg[0,:]*(1+PE_to_KE)
#decay time
bot_diss_scaled = (botDISS_ADCP_avg[0,:]+botmolDISS_ADCP_avg[0,:]+ BF_bb[0,:])#*bottom_area/Area[0]
tot_diss_scaled = intDISS_KE_avg[0,:] + BF_int[0,:]  + bot_diss_scaled
res_time = PKE/tot_diss_scaled/60/60/24*1000 #enery residence time




anm_P10 = np.nanmean(P10)*1000
ans_P10 = np.nanstd(P10)*1000
anm_RW = np.nanmean( ADCP["wind_work"]["RW"])*1000
ans_RW = np.nanstd( ADCP["wind_work"]["RW"])*1000
anm_epsint = np.nanmean(epsilon_ADCP_KE_fit*1000)
ans_epsint = np.nanstd(epsilon_ADCP_KE_fit*1000)
anm_epsbbl = np.nanmean(diss_ADCP_bot*bot_scale)
ans_epsbbl = np.nanstd(diss_ADCP_bot*bot_scale)
anm_epsbbllam = np.nanmean(us**3*11*1e6*bot_scale)
ans_epsbbllam = np.nanstd(us**3*11*1e6*bot_scale)
anm_mixint = np.nanmean(mix_eff*epsilon_ADCP_KE_fit*1000)
ans_mixint = np.nanstd(mix_eff*epsilon_ADCP_KE_fit*1000)
anm_mixbbl = np.nanmean(mix_eff_bot*diss_ADCP_bot*bot_scale)
ans_mixbbl = np.nanstd(mix_eff_bot*diss_ADCP_bot*bot_scale)
anm_mixtot = np.nanmean(mix_eff*epsilon_ADCP_KE_fit*1000 + mix_eff_bot*diss_ADCP_bot*bot_scale)
ans_mixtot = np.nanmean(mix_eff*epsilon_ADCP_KE_fit*1000 + mix_eff_bot*diss_ADCP_bot*bot_scale)

                        
anm_epsintmeas = np.nanmean(int_diss*1000)
ans_epsintmeas = np.nanstd(int_diss*1000)
anm_epsbblmeas = np.nanmean(diss_AD_bot*bot_scale)
ans_epsbblmeas = np.nanmean(diss_AD_bot*bot_scale)



xx = np.sum(ADCP["NM"]["KE"], axis = 0)

anm_KE = np.nanmean( xx  )
ans_KE = np.nanstd( xx )
xx = ADCP["NM"]["KE"][0,:]
xx[xx == 0] = np.nan
anm_btKE = np.nanmean( xx )
ans_btKE = np.nanstd( xx )

xx = np.sum(TM["mPE"], axis = 0)
xx[xx== 0] = np.nan
anm_PE_TM = np.nanmean(xx )
ans_PE_TM = np.nanstd( np.sum(TM["mPE"], axis = 0))
xx = np.sum(TC["mPE"], axis = 0)
xx[xx== 0] = np.nan
anm_PE_TC = np.nanmean(xx)
ans_PE_TC = np.nanstd( xx)
anm_PKE = (1+PE_to_KE)*anm_KE
ans_PKE = (1+PE_to_KE)*ans_KE

anm_Vbot_AD = np.nanmean(V_bot_AD**2)**0.5*100
ans_Vbot_AD = np.nanstd(V_bot_AD**2)**0.5*100

anm_Vbot = np.nanmean(V_bot**2)**0.5*100
ans_Vbot = np.nanstd(V_bot**2)**0.5*100


mres_time = anm_PKE/(anm_epsint+anm_epsbbl+anm_epsbbllam+anm_mixint+anm_mixbbl)*1000/60/60/24.


print "#################################################"
print "########## MEAN VALUES #########################"
print "#################################################"
print "Wind Work P10 %1.1f +- %1.1f"%(anm_P10, ans_P10)
print "Wind Work RW %1.3f +- %1.3f"%(anm_RW, ans_RW)
print "Water column dissipation %1.2f +- %1.2f mW m-2"%(anm_epsint,ans_epsint)
print "BBL turbulent dissipation %1.2f +- %1.2f mW m-2"%(anm_epsbbl, ans_epsbbl)
print "BBL laminar dissipation %1.2f +- %1.2f mW m-2"%(anm_epsbbllam ,ans_epsbbllam )
print "Water column mixing %1.3f +- %1.3f mW m-2"%( anm_mixint,ans_mixint)
print "Water column mixing %1.3f +- %1.3f mW m-2"%(anm_mixbbl , ans_mixbbl)
print "#################################################"
print "Total turbulent dissipation %1.2f"%( anm_epsint+anm_epsbbl )
print "Measured bottom dissipation %1.2f +- %1.2f"%( np.nanmean(diss_AD_bot*bot_scale), np.nanstd(diss_AD_bot*bot_scale) )






colors_st = ["seagreen", "gold", "saddlebrown", "royalblue" ]
 
fig,ax = plt.subplots(figsize = (7,3))
l1=ax.bar(time_avgC-19, Production[0,:], yerr = eProduction[0,:], width = 34, fc = cm.Accent(0), ec = "k", ecolor = "gray")
#l2=ax.bar(time_avgC-19, Production[1,:], yerr = eProduction[1,:],  bottom = Production[0,:], width = 34, fc = cm.Accent(1), ec = "k")
l3=ax.bar(time_avgC+19, Dissipation[0,:], yerr = eDissipation[0,:], width = 34, fc= cm.Accent(6), ec = "k", ecolor = "gray")
l4=ax.bar(time_avgC+19, Dissipation[1,:], yerr = eDissipation[1,:], bottom = Dissipation[0,:], width = 34, fc= cm.Accent(2), ec = "k", ecolor = "gray")
l5=ax.bar(time_avgC+19, Dissipation[2,:], yerr = eDissipation[2,:], bottom = Dissipation[1,:]+Dissipation[0,:], width = 34, fc= cm.Accent(3), ec = "k", ecolor = "gray")
l6=ax.bar(time_avgC+19, Dissipation[3,:], yerr = eDissipation[3,:], bottom = Dissipation[2,:]+Dissipation[1,:]+Dissipation[0,:], width = 34, fc= "royalblue", ec = "k", ecolor = "gray") #cm.Paired(0),
l7=ax.bar(time_avgC+19, Dissipation[4,:], yerr = eDissipation[4,:], bottom =  Dissipation[3,:]+Dissipation[2,:]+Dissipation[1,:]+Dissipation[0,:], width = 34, fc= "lightskyblue", ec = "k", ecolor = "gray")
ax.bar(time_avgC[-1]+88, mProd[0], width = 34, fc = cm.Accent(0), ec = "k")

ax.bar(time_avgC[-1]+126, mDiss[0], width = 34, fc = cm.Accent(6), ec = "k")
ax.bar(time_avgC[-1]+126, mDiss[1],   bottom = mDiss[0], width = 34, fc = cm.Accent(2), ec = "k")
ax.bar(time_avgC[-1]+126, mDiss[2],   bottom = mDiss[0]+mDiss[1], width = 34, fc = cm.Accent(3), ec = "k")
ax.bar(time_avgC[-1]+126, mDiss[3],   bottom = mDiss[0]+mDiss[1]+mDiss[2], width = 34, fc = "royalblue",  ec = "k")
ax.bar(time_avgC[-1]+126, mDiss[4],   bottom = mDiss[0]+mDiss[1]+mDiss[2]+mDiss[3], width = 34, fc = "lightskyblue", ec = "k")

for i in range(time_avgC.size):
    ax.annotate("%1.2f"%(Production[0,i]), xy = (time_avgC[i]-19,Production[0,i]), ha = "center", va = "bottom", fontweight = "bold"  )
    ax.annotate("%1.2f"%( Dissipation[0,i]), xy = (time_avgC[i]+19,Dissipation[0,i]/2 ), ha = "center", va = "center", fontsize = 9  )
    ax.annotate("%1.2f"%( np.nansum(Dissipation[:,i], axis = 0)), xy = (time_avgC[i]+19,np.nansum(Dissipation[:,i], axis = 0)), ha = "center", va = "bottom", fontweight = "bold"  )
    if Dissipation[1,i]>0.1:
        ax.annotate("%1.2f"%( Dissipation[1,i]), xy = (time_avgC[i]+19,Dissipation[0,i]+Dissipation[1,i]/2 ), ha = "center", va = "center", fontsize = 9  )
    if Dissipation[2,i]>0.05:
        ax.annotate("%1.2f"%( Dissipation[2,i]), xy = (time_avgC[i]+19,Dissipation[0,i]+Dissipation[1,i]+Dissipation[2,i]/2 ), ha = "center", va = "center", fontsize = 9  )
            
ax.annotate("%1.2f"%(anm_epsint), xy = (time_avgC[-1]+126, mDiss[0]/2.), ha = "center", va = "center", fontsize = 9, color = "k")
ax.annotate("%1.2f"%(anm_epsbbl), xy = (time_avgC[-1]+126, mDiss[0]+mDiss[1]/2.), ha = "center", va = "center", fontsize = 9, color = "k")
ax.annotate("%1.2f"%(anm_epsbbllam), xy = (time_avgC[-1]+126, mDiss[0]+mDiss[1]+mDiss[2]/2.), ha = "center", va = "center", fontsize = 9, color = "k")
ax.annotate("%1.2f"%(anm_epsint+anm_epsbbl+anm_epsbbllam+anm_mixint+anm_mixbbl), xy = (time_avgC[-1]+126, np.sum(mDiss)), ha = "center", va = "bottom", fontsize = 9, color = "k", fontweight = "bold")
ax.annotate("%1.2f"%(mProd[0]), xy = (time_avgC[-1]+88, mProd[0]), ha = "center", va = "bottom", fontsize = 9, color = "k", fontweight = "bold")
ax.annotate("%3.0f J m$^{-2}$\n$\\tau  = %1.1f$ d"%(anm_PKE,mres_time), xy = ( time_avgC[-1]+110, 1.29 ), ha = "center", va = "top")

ax.annotate("%1.0f%%"%(mDiss[0]/np.nansum(mDiss)*100), xy = (time_avgC[-1]+161, mDiss[0]/2.), ha = "center", va = "center", fontsize = 9, color = "darkred")
ax.annotate("%1.0f%%"%(mDiss[1]/np.nansum(mDiss)*100), xy = (time_avgC[-1]+161, mDiss[0]+mDiss[1]/2.), ha = "center", va = "center", fontsize = 9, color = "darkred")
ax.annotate("%1.0f%%"%(mDiss[2]/np.nansum(mDiss)*100), xy = (time_avgC[-1]+161, mDiss[0]+mDiss[1]+mDiss[2]/2.), ha = "center", va = "center", fontsize = 9, color = "darkred")
ax.annotate("%1.0f%%"%(np.nansum(mDiss[3:])/np.nansum(mDiss)*100), xy = (time_avgC[-1]+161, mDiss[0]+mDiss[1]+mDiss[2]), ha = "center", va = "bottom", fontsize = 9, color = "darkred")

ax.legend((l1,l3,l4,l5,l6,l7),["%1.2f%% $P_{10}$"%(gamma*100),"$\\mathcal{E}_{\\mathrm{int}}$","$\\mathcal{E}_{\\mathrm{BBL}}$","$\\mathcal{E}_{\\mathrm{BBL}}^{\mathrm{lam}}$","$\\mathcal{B}_{int}$","$\\mathcal{B}_{BBL}$"], ncol = 7, loc = "upper center", bbox_to_anchor = (0.5,1.18))
ax.set_ylim((0,1.3))
for i in range(Navg):
    ax.fill_between([time_avg[i], time_avg[i+1]], y1 = 1.2,  y2 = 1.3, color = colors_st[i], alpha = 0.3)
    ax.annotate("%3.0f J m$^{-2}$\n$\\tau  \\approx %1.1f$ d"%(PKE[i],res_time[i]), xy = ( time_avgC[i], 1.29 ), ha = "center", va = "top")
    ax.axvline(time_avg[i], lw = 1, color = "gray", ls = "dotted" )
ax.set_ylabel("Energy sources/sinks [mW m$^{-2}$]")
ax.axvline(time_avg[i+1], lw = 1, color = "gray", ls = "dotted"  )
#ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
ax.set_xticks(np.r_[time_avg,time_avg[-1]+62])
xtkb = [date0.strftime("%d-%b") for date0 in date_avg]
xtkb.append("Annual\nmean")
ax.set_xticklabels(xtkb)
xl = ax.get_xlim()
ax.set_xlim((xl[0],xl[1]+20))
ax.annotate("d.", xy = (0.01,0.02), xycoords = "axes fraction", fontweight = "bold")

#ax.xaxis.set_minor_locator(mdates.MonthLocator())

fig.savefig("energy_budget_bar_plot.pdf", dpi = 300, bbox_inches = "tight")
fig.savefig("energy_budget_bar_plot.png", dpi = 300, bbox_inches = "tight")
plt.show()




fig, ax = plt.subplots( 2,nn, figsize = (12,8), sharey = True )
ax[0,0].set_ylim((105,0))
for i in range(nn):
    l1,=ax[0,i].plot( sea_m_epsilon_prof[:,i], z, "-", color = cm.Accent(6), mfc = "w" )
    ax[0,i].fill_betweenx(z, sea_ci_epsilon_prof[0,:,i], sea_ci_epsilon_prof[1,:,i],  alpha = 0.4, color = cm.Accent(6))
    #ax[0,i].plot( sea_m_epsilon_ADCP_prof[:,i], z, "--", color = cm.Accent(6), mfc = "w" )
    l2,=ax[0,i].plot( sea_m_chi_prof[:,i], z, "-", color = cm.Dark2(0), mfc = "w" )
    ax[0,i].fill_betweenx(z, sea_ci_chi_prof[0,:,i], sea_ci_chi_prof[1,:,i],  alpha = 0.4, color = cm.Accent(0))
    #ax[0,i].set_xlim((1e-10,2e-8))
    #ax0.set_xlim((5e-10,3e-6))
    
    l3,=ax[0,i].semilogx( sea_m_BF_prof[:,i], z, "-", color ="deepskyblue", mfc = "w" )
    ax[0,i].fill_betweenx(z, sea_ci_BF_prof[0,:,i], sea_ci_BF_prof[1,:,i],  alpha = 0.4, color ="deepskyblue")
    #l4,=ax[0,i].semilogx( sea_m_BFhb_prof[:,i]/1000., z, "-", color = cm.Accent(4), mfc = "w" )
    #ax[0,i].fill_betweenx(z, sea_ci_BFhb_prof[0,:,i]/1000., sea_ci_BFhb_prof[1,:,i]/1000.,  alpha = 0.4, color = cm.Accent(4))
    ax[0,i].set_xlabel("$\\varepsilon$, $b$ [W kg$^{-1}$], $\\chi$ [K$^2$ s$^{-1}$]")

    ax[0,i].set_xlim((1e-11,5e-6))
    ax[0,i].set_xticks(10**np.arange(-11.,-5.))
    ax[0,i].set_xticklabels(["","$10^{-10}$","","$10^{-8}$","","$10^{-6}$"])
    locmin = mpl.ticker.LogLocator(base=10.0,subs=np.arange(0,1.,0.1),numticks=12)
    ax[0,i].xaxis.set_minor_locator(locmin)
    ax[0,i].xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax[0,i].grid(True)

    ax[0,i].set_title("%s $-$ %s"%(date_avg[i].strftime("%d %b"),date_avg[i+1].strftime("%d %b")))
    #ax[0,i].set_xlim((1e-11,3e-8))

    l5,=ax[1,i].semilogx( sea_m_KT_prof[:,i], z, "-", color =cm.Accent(4), mfc = "w" )
    ax[1,i].fill_betweenx(z, sea_ci_KT_prof[0,:,i], sea_ci_KT_prof[1,:,i],  alpha = 0.4, color =cm.Accent(4))
    l6,=ax[1,i].semilogx( sea_m_N2_prof[:,i], z, "-", color = "darkorange", mfc = "w" )
    ax[1,i].fill_betweenx(z, sea_ci_N2_prof[0,:,i], sea_ci_N2_prof[1,:,i],  alpha = 0.4, color = "darkorange")
    ax[1,i].set_xlim((5e-7,2e-3))
    ax[1,i].grid(True)
    ax[1,i].set_xlabel("$K_T$ [m$^2$ s$^{-1}$], $N^2$ [s$^{-2}$]")
ax[0,0].set_ylabel("Depth [m]")
ax[1,0].set_ylabel("Depth [m]")
ax[0,-1].legend(["$\\varepsilon$", "$\\chi$","$b$"] )
ax[1,0].legend(["$K_T$", "$N^2$"] )

#ax[0,-1].legend(["$\\varepsilon$", "$\\chi$","$\\mathcal{B}^{\\mathrm{\\mu}}$", "$\\mathcal{B}^{hb}$"] )
#ax[1,-1].legend(["$K_T^{\\mathrm{\\mu}}$", "$K_T^{hb}$"] )
fig.savefig("mean_profiles_season.pdf", dpi = 300, bbox_inches = "tight")
plt.close(fig) 






#### THE FIGURE
CTD = pickle.load( open("CTD_micro/gridded_CTD_Leman_profiles.pi") )

time_u = micro["time"].astype(int)
time_u_d, iun = np.unique(time_u, return_index = True)
WS_micro_d = WS_micro_week[iun]
date_u_d = convert_time_to_date(time_u_d)

time_CTD = CTD["time"].astype(int)-366
time_CTD = np.unique(time_CTD)
date_CTD = convert_time_to_date(time_CTD)

jj = np.isfinite( ST["ScGR"] )
intSc = intrp.interp1d(ST["timeGRs"][jj], -ST["ScGR"][jj], bounds_error = False)
Sc00 = intSc(ST["timeGR"])
jj = np.isfinite( Sc00+ST["HB"]["APE"]["APE0"])
pp = np.polyfit(ST["HB"]["APE"]["APE0"][jj],Sc00[jj],2 )
ppp=np.poly1d(pp)
ScGR = ppp( ST["HB"]["APE"]["APE0"] )

fig, ax = plt.subplots()
ax.plot(ST["HB"]["APE"]["APE0"]/1000., Sc00/1000.,".")
ax.plot( np.sort( ST["HB"]["APE"]["APE0"]/1000. ), ppp(np.sort(ST["HB"]["APE"]["APE0"]))/1000. )
plt.show()

Nm = 2
fig = plt.figure(figsize = (7,8))
ax0 = fig.add_axes([0.1, 0.88, 0.80,0.12])
ax1 = fig.add_axes([0.1, 0.64, 0.80,0.22])
ax2 = fig.add_axes([0.1, 0.46, 0.80,0.16])
ax3 = fig.add_axes([0.1, 0.28, 0.80,0.16])
ax4 = fig.add_axes([0.1, 0.04, 0.97,0.22])

#ax0.plot(MET["date_dt"], MET["sHFnet"], color = "gray", lw = 0.5)
#ax0.plot(MET["date_dt"], Qnet, color = "k", lw = 1)
ax0.fill_between(MET["date_dt"], Qnet, where = Qnet<0, color = "darkred", alpha = 0.4)
ax0.fill_between(MET["date_dt"], Qnet, where = Qnet>0, color = "darkblue", alpha = 0.4)
ax0.axhline(0, color = "gray", lw= 1)
ax0.set_xticklabels([])
ax0.set_ylabel("$Q_{\\mathrm{net}}$ [W m$^{-2}$]")
ax0.annotate("a.", xy = (0.01,0.87), xycoords = "axes fraction", color = "k", fontweight = "bold")
sea_names = ["Spring","Summer","Fall","Winter"];
for i in range(Navg):
    #if i%2 == 0:
    ax0.axvline(time_avg[i], color = "k", lw = 0.5, ls = "dotted")
    ax0.fill_between([time_avg[i], time_avg[i+1]], y1 = 240,  y2 =300, color = colors_st[i], alpha = 0.5, zorder = -1)
    ax0.annotate( sea_names[i], xy = ( 0.5*(time_avg[i]+ time_avg[i+1]),290 ), color = "k", ha = "center", fontweight = "bold"  )
ax0.set_ylim((300,-300))
ax0.axvline(time_avg[-1], color = "k", lw = 0.5, ls = "dotted")
ax0b = ax0.twinx()
#l0,=ax0b.plot(ST["dateGR"], -ST["HB"]["APE"]["APE0"]/1000,'o', color = "k", ms = 3, mfc = "none")
#l1, = ax0b.plot(ST["dateGRs"], -ST["HB"]["APE"]["APE"]/1000,'k-',lw = 2)
l0,=ax0b.plot(ST["dateGR"], -ScGR/1000.,'o', color = "k", ms = 3, mfc = "none")
l1, = ax0b.plot(ST["dateGRs"], ST["ScGR"]/1000,'k-',lw = 2)

ax0b.set_xlim((time_avg[0]-10,time_avg[-1]+10))
ax0b.set_ylabel("Sc [kJ m$^{-2}$]")
#ax0b.set_ylim((-5,15))
ax0b.set_ylim((-5,30))
ax0b.set_xticklabels([])

ax1.contourf( ST["dateGRs"], ST["depth"], ST["TGRs"], np.arange(5,27,1),cmap = cmocean.cm.thermal )
cc = ax1.contour( ST["dateGRs"], ST["depth"], ST["TGRs"], np.arange(5,27,2),colors = "w", linewidths = 0.5)
ax1.plot(ST["dateGRs"], mixed_layer_depth(ST["depth"], -ST["TGRs"], Dd = 0.5), color = "w")
ax1.plot(ST["dateGRs"], mixed_layer_depth(ST["depth"], -ST["TGRs"], Dd = 0.5), "--",color = "g")
plt.clabel(cc, fmt = "%d", fontsize = 6)
ax1.set_xlim(xl)
ax1.set_ylabel("Depth [m]")
ax1.set_xticklabels([])
ax1.annotate("b.", xy = (0.01,0.92), xycoords = "axes fraction", color = "k", fontweight = "bold")
ax1.annotate("Temperature [$^{\\circ}$C]", xy = (0.05,0.08), xycoords = "axes fraction", color = "w")
for i in range(Navg):
    #if i%2 == 0:
    ax1.fill_between([time_avg[i], time_avg[i+1]], y1 = 106.,  y2 =110, color = colors_st[i], alpha = 0.5)
ax1.set_ylim((110,0))
ax1.plot( date_CTD, 107.5*np.ones(time_CTD.size), "|", color = "w")
ax1.plot( date_u_d, 107.5*np.ones(iun.size), "|", color = "k")


ax2.plot(MET["date_dt"], MET["WS_10m"], color = "gray", lw = 0.5)
ax2.plot(MET["date_dt"], WS, color = "k", lw = 1)
ax2.plot(date_u_d, WS_micro_d, "ok", mfc = "w", ms = 4)
ax2.set_xticklabels([])
ax2.set_ylabel("$W_{\\mathrm{10}}$ [m s$^{-1}$]")
xl = ax2.get_xlim()
ax2.set_ylim((-1,18))
ax2.annotate("c.", xy = (0.01,0.87), xycoords = "axes fraction", color = "k", fontweight = "bold")
for i in range(Navg):
    #if i%2 == 0:
    ax2.axvline(time_avg[i], color = "k", lw = 0.5, ls = "dotted")
    ax2.fill_between([time_avg[i], time_avg[i+1]], y1 = -1,  y2 = -0.1, color = colors_st[i], alpha = 0.5, zorder = -1)
    ax2.annotate("$%1.1f$ m s$^{-1}$\n$P_{10} = %1.0f$ mW m$^{-2}$"%(U10_avg[0,i],P10_avg[0,i]), xy = ( time_avgC[i],15), va = "center", ha = "center", color = "k", bbox = dict( boxstyle = 'round,pad=0.2', fc = "w", ec = "k", alpha = 1.), fontsize = 9)
ax2.axvline(time_avg[-1], color = "k", lw = 0.5, ls = "dotted")
l= []
#colors_bars = [cm.Accent(4), cm.Set1(0),cm.Dark2(5), cm.Dark2(0) ]
for i in range(Nm+2):
    ii = Nm+1-i
    l0 = ax3.fill_between(ADCP["date"], ADCP["NM"]["int_KE"][ii,:], color = colorsCB[ii])
    l.append(l0)
    print ii
#ax3.plot(ADCP["date"], KEs,"k", lw = 0.5)
ax3.set_xticklabels([])
ax3.set_xlim(xl)
ax3.set_ylabel("KE [J m$^{-2}$]")
ax3.set_ylim((0,1000))
ax3.legend(l[::-1],["M$_0$","M$_1$","M$_2$","M$_3$"], ncol = 1, bbox_to_anchor=(0.97, 0.90))
for i in range(Navg):
    l0,=ax3.plot([time_avg[i], time_avg[i+1]], [KE_avg[0,i],KE_avg[0,i]], lw = 1, zorder = 10, color = "k")
    ax3.annotate("%1.0f (M$_0: %1.0f \\%%)$"%(KE_avg[0,i],btKE_avg[0,i]/KE_avg[0,i]*100), xy = ( time_avgC[i],900), va = "center", ha = "center", color = "k", bbox = dict( boxstyle = 'round,pad=0.2', fc = "w", ec = "k", alpha = 1.))
    #print i%2
    ax3.fill_between([time_avg[i], time_avg[i+1]], y1 = 950,  y2 =1000, color = colors_st[i], alpha = 0.5)
    #if i%2 == 0:
    #    ax3.fill_between([time_avg[i], time_avg[i+1]], y1 = 0,  y2 = 1000, color = "gold", alpha = 0.2, zorder = -1)
for i in range(Navg+1):
    ax3.axvline(time_avg[i], color = "k", lw = 0.5, ls = "dotted")
ax3.annotate("d.", xy = (0.01,0.87), xycoords = "axes fraction", color = "k", fontweight = "bold")
ax3.axvline(time_avg[-1], color = "k", lw = 0.5, ls = "dotted")

ccu2 = ax4.contourf(ADCP["date"],ADCP["z"], np.sum(ADCP["NM"]["u"],axis = 0)*100, np.arange(-15,16,1), extend = "both",cmap = cm.RdBu_r)
cb=plt.colorbar(ccu2, ax = ax4,pad = 0.02)
cb.set_label("$u_{NM}$ [cm s$^{-1}$]")
#ax4.contour(TM["date"], TM["z"], TM["T"], np.arange(8,30,2), colors = "gray",linewidths = 0.5)
#ax4.contour(TC["date"], TC["z"], TC["T"], np.arange(8,30,2), colors = "gray",linewidths = 0.5)
ax4.contour( ST["dateGRs"], ST["depth"], ST["TGRs"], np.arange(5,27,2),colors = "gray", linewidths = 0.5)

ax4.set_xlim(xl)
ax4.set_ylabel("Depth [m]")
#ax4.annotate("$u_{NM}$ [cm s$^{-1}$]", xy = (0.04,0.05), xycoords = "axes fraction", color = "k")
ax4.annotate("e.", xy = (0.01,0.92), xycoords = "axes fraction", color = "k", fontweight = "bold")
ax4.annotate("$u_{NM}$ [cm s$^{-1}$]", xy = (0.05,0.08), xycoords = "axes fraction", color = "k")
ax4.set_xlim(xl)
for i in range(Navg):
    #if i%2 == 0:
    ax4.fill_between([time_avg[i], time_avg[i+1]], y1 = 106.,  y2 =110, color = colors_st[i], alpha = 0.5)
ax4.set_ylim((110,0))
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
ax0.set_xlim(xl)

ax0.xaxis.set_minor_locator(mdates.MonthLocator())
ax1.xaxis.set_minor_locator(mdates.MonthLocator())
ax2.xaxis.set_minor_locator(mdates.MonthLocator())
ax3.xaxis.set_minor_locator(mdates.MonthLocator())
ax4.xaxis.set_minor_locator(mdates.MonthLocator())

fig.savefig("Figure_T_ADCP.png", dpi = 300, bbox_inches = "tight")


#

#Potential energy mooring
Nm = 2
fig = plt.figure(figsize = (13,4.5))
ax0 = fig.add_axes([0.06,0.60,0.41,0.33])
ax1 = fig.add_axes([0.06,0.08,0.41,0.47])
ax2 = fig.add_axes([0.50,0.60,0.41,0.33])
ax3 = fig.add_axes([0.50,0.08,0.52,0.47])

l= []
for i in range(Nm):
    ii = Nm-1-i
    l0 = ax0.fill_between(TM["date"], TM["intmPE"][ii,:], color = colorsCB[ii])
    l.append(l0)
    print ii
ax0.set_ylabel("PE [J m$^{-2}$]")
ax0.set_ylim((0,1000))
#ax0.legend(l[::-1],["M$_1$","M$_2$"], ncol = 1, bbox_to_anchor=(0.80, 0.75))
ax0.set_xticklabels([])
for i in range(Navg):
    l0,=ax0.plot([time_avg[i], time_avg[i+1]], [PE_TM_avg[0,i],PE_TM_avg[0,i]], lw = 3, zorder = 10, color = "k")
    ax0.annotate("%1.0f J m$^{-2}$"%(PE_TM_avg[0,i]), xy = ( time_avgC[i],900), va = "center", ha = "center", color = "k",  bbox = dict( boxstyle = 'round,pad=0.2', fc = "w", ec = "k", alpha = 1))
    ax0.fill_between([time_avg[i], time_avg[i+1]], y1 = 920,  y2 = 1000, color = colors_st[i],  zorder = -1, alpha = 0.5)
    ax0.axvline(time_avg[i], lw = 1, color = "gray", ls = "dotted" )
#xl = ax0.get_xlim()]
ax0.set_xlim()
ax0.axvline(time_avg[i+1], lw = 1, color = "gray", ls = "dotted" )
ccT=ax1.contourf(TM["date"], TM["z"], TM["T"], np.arange(5,28,2), cmap = cmocean.cm.thermal)
ax1.plot(TM["date"], mixed_layer_depth(TM["z"], -TM["sT"], Dd = 0.5),color = "w", lw = 2)
ax1.plot(TM["date"], mixed_layer_depth(TM["z"], -TM["sT"], Dd = 0.5),color = "g", ls = "--", lw = 2)
#ax1.plot(np.ones(TM["z"].size)*TM["time"][0]-15,TM["z"], ">", mec ="k", mfc = "red" )
#xl = ax1.get_xlim()
ax1.set_ylim((100,0))
ax1.set_xticklabels([])
ax1.set_ylabel("Depth [m]")
ax1.set_xlim(xl)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
ax0.annotate("a.", xy = (0.01,0.02), xycoords = "axes fraction", fontweight = "bold")
ax1.annotate("b.", xy = (0.01,0.02), xycoords = "axes fraction", fontweight = "bold")
ax0.set_title("Thermistor mooring (0-30 m)")
#Thermistor chain

l= []
for i in range(Nm):
    ii = Nm-1-i
    l0 = ax2.fill_between(TC["date"], TC["intPEm"][ii,:], color = colorsCB[ii])
    l.append(l0)
    
    print ii
ax2.set_ylim((0,1000))
ax2.legend(l[::-1],["M$_1$","M$_2$"], ncol = 1, bbox_to_anchor=(1.14, 0.75))
ax2.set_xlim(xl)
ax2.set_yticklabels([])

for i in range(Navg):
    l0,=ax2.plot([time_avg[i], time_avg[i+1]], [PE_TC_avg[0,i],PE_TC_avg[0,i]], lw = 3, zorder = 10, color = "k")
    ax2.annotate("%1.0f J m$^{-2}$"%(PE_TC_avg[0,i]), xy = ( time_avgC[i],900), va = "center", ha = "center", color = "k",  bbox = dict( boxstyle = 'round,pad=0.2', fc = "w", ec = "k", alpha = 1))
    #print i%2
    #if i%2 == 0:
    ax2.axvline(time_avg[i], lw = 1, color = "gray", ls = "dotted" )
    ax2.fill_between([time_avg[i], time_avg[i+1]], y1 = 920,  y2 = 1000, color = colors_st[i], zorder = -1,  alpha = 0.5)
ax2.set_xticklabels([])
ax2.axvline(time_avg[i+1], lw = 1, color = "gray", ls = "dotted" )
ccT=ax3.contourf(TC["date"], TC["z"], TC["T"], np.arange(5,28,2), cmap = cmocean.cm.thermal)
cb = plt.colorbar(ccT, ax = ax3)
cb.set_label("Temperature [$^{\circ}$C]")
ax3.plot(TC["date"], mixed_layer_depth(TC["z"], -TC["sT"], Dd = 0.5),color = "w", lw = 2)
ax3.plot(TC["date"], mixed_layer_depth(TC["z"], -TC["sT"], Dd = 0.5),color = "g", ls = "--", lw = 2)
ax3.set_xlim(xl)
ax3.set_ylim((100,0))
ax3.set_xticklabels([])
ax3.set_yticklabels([])
#ax3.set_ylabel("Depth [m]")
#ax3.annotate("Temperature [$^{\circ}$C]", xy = (0.02,0.07), xycoords = "axes fraction", bbox = dict(facecolor = "w"))
#ax3.set_xticks(time_avg)
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
ax2.annotate("c.", xy = (0.01,0.02), xycoords = "axes fraction", fontweight = "bold")
ax3.annotate("d.", xy = (0.01,0.02), xycoords = "axes fraction", fontweight = "bold")
ax2.set_title("Thermistor chain (0-100 m)")

ax0.xaxis.set_minor_locator(mdates.MonthLocator())
ax1.xaxis.set_minor_locator(mdates.MonthLocator())
ax2.xaxis.set_minor_locator(mdates.MonthLocator())
ax3.xaxis.set_minor_locator(mdates.MonthLocator())
#ax4.xaxis.set_minor_locator(mdates.MonthLocator())

fig.savefig("PE_timeseries.png", dpi = 300)
#plt.show()
plt.close("all")


###

fig = plt.figure(figsize = (7,6))
ax6 = fig.add_axes([0.12, 0.69, 0.85,0.30])
ax7 = fig.add_axes([0.12, 0.37, 0.85,0.30])
ax8 = fig.add_axes([0.12, 0.05, 0.85,0.30])
ax6.plot(ADCP["date"], ADCP["wind_work"]["P10"]*1000,'-', color = "gray", lw = 1,alpha = 0.8)
ax6.plot(ADCP["date"], ADCP["wind_work"]["smooth_P10"]*1000,'-', color = "k", lw = 1)
ax6.set_xlim((time_avg[0]-10,time_avg[-1]+10))
ax6.set_ylabel("$P_{10}$ [mW m$^{-2}$]")
#ax6.set_yscale("log")
for i in range(Navg):
    l0,=ax6.plot([time_avg[i], time_avg[i+1]], [P10_avg[0,i],P10_avg[0,i]], lw = 2, zorder = 10, color = colors_st[i])
    #ax6.annotate("%1.1f (%1.1f)"%(P10_avg[0,i],P10micro_avg[0,i]), xy = ( time_avgC[i],1750), va = "center", ha = "center", color = "k", bbox = dict( boxstyle = 'round,pad=0.3', fc = "w", ec = "k", alpha = 1.)) #
    ax6.annotate("%1.0f"%(P10_avg[0,i]), xy = ( time_avgC[i],1800), va = "center", ha = "center", color = "k", bbox = dict( boxstyle = 'round,pad=0.3', fc = "w", ec = "k", alpha = 1.), fontsize = 9) 
    ax6.fill_between([time_avg[i], time_avg[i+1]], y1 = 1900,  y2 =2000, color = colors_st[i], alpha = 0.5)
#for i in range(Navg+1):
#    ax6.axvline(time_avg[i], color = "gray", lw = 0.5)
ax6.set_ylim((-100,2000))
ax6.set_xticklabels([])
ax6.annotate("a.", xy = (0.005,0.90), xycoords = "axes fraction", color = "k", fontweight = "bold")

l1,=ax7.plot(ADCP["date"],epsilon_ADCP_KE_fit*1000, color = "gray", lw = 0.5)
#ax7.plot(ADCP["date"],epsilon_ADCP_KE_iso*1000, color = "red", lw = 0.5)
l2,=ax7.plot(micro["date_dt"], int_diss*1000,"k.", ms = 3)
ax7.set_xlim((time_avg[0]-10,time_avg[-1]+10))
for i in range(Navg):
    ax7.plot([time_avg[i], time_avg[i+1]], [intDISS_KE_avg[0,i],intDISS_KE_avg[0,i]], lw = 2, zorder = 10, color = colors_st[i])
    ax7.plot([time_avg[i], time_avg[i+1]], [intDISS_meas_avg[0,i],intDISS_meas_avg[0,i]], lw = 2, ls ="--", zorder = 10, color = colors_st[i])
    ax7.annotate("%1.2f\n($\mu$CTD: %1.2f)"%(intDISS_KE_avg[0,i],intDISS_meas_avg[0,i]), xy = ( time_avgC[i],1.80), va = "center", ha = "center", color = "k",  bbox = dict( boxstyle = 'round,pad=0.3', fc = "w", alpha = 1), fontsize = 9) #
    ax7.fill_between([time_avg[i], time_avg[i+1]], y1 = 1.9,  y2 =2., color = colors_st[i], alpha = 0.5)
    #ax7.fill_between([time_avg[i], time_avg[i+1]], y1 = ax7.get_ylim()[0],  y2 = ax7.get_ylim()[1], color = cm.tab10(i), alpha = 0.1)
lg=ax7.legend((l2,l1), ["$\\mu$CTD","ADCP"], loc = "center right", fontsize = 9, markerscale = 2)
for line in lg.get_lines():
    line.set_linewidth(3.0)
ax7.set_ylim((-0.05,2.))
ax7.set_ylabel("$\\mathcal{E}_{\\mathrm{int}}$ [mW m$^{-2}$]")
ax7.set_xticklabels([])
ax7.annotate("b.", xy = (0.005,0.90), xycoords = "axes fraction", color = "k", fontweight = "bold")

l1,=ax8.plot(ADCP["date"], epsilon_ADCP_bot*factor_ADCP*1e6,color = "gray", lw = 0.5)
l2,=ax8.plot(AD["date"],epsilon_AD*factor_AD*1e6, color = "k", lw = 0.5)
#ax8.plot(ADCP["date"],epsilon_ADCP_KE_iso*1000, color = "red", lw = 0.5)
ax8.set_xlim((time_avg[0]-10,time_avg[-1]+10))
for i in range(Navg):
    ax8.plot([time_avg[i], time_avg[i+1]], [botDISS_ADCP_avg[0,i],botDISS_ADCP_avg[0,i]], lw = 2, zorder = 10, color = colors_st[i])
    ax8.plot([time_avg[i], time_avg[i+1]], [botDISS_meas_avg[0,i],botDISS_meas_avg[0,i]], lw = 2, ls ="--", zorder = 10,  color = colors_st[i])
    #l1,=ax8.plot([time_avg[i], time_avg[i+1]], [botDISS_ADCd_avg[0,i],botDISS_ADCd_avg[0,i]], lw = 3, ls ="-.", zorder = 10, color = l0.get_color())
    if np.isfinite(botDISS_meas_avg[0,i]):
        lbl = "%1.2f (AD:%1.2f)"%(botDISS_ADCP_avg[0,i],botDISS_meas_avg[0,i])
    else:
        lbl = "%1.2f (AD: -)"%(botDISS_ADCP_avg[0,i])
    ax8.annotate(lbl, xy = ( time_avgC[i],2.7), va = "center", ha = "center", color = "k",  bbox = dict( boxstyle = 'round,pad=0.3', fc = "w", ec = "k", alpha = 1), fontsize = 9) #
    ax8.fill_between([time_avg[i], time_avg[i+1]], y1 = 2.85,  y2 =3., color = colors_st[i], alpha = 0.5)
ax8.set_ylabel("$\\mathcal{E}_{\\mathrm{bbl}}$ [mW m$^{-2}$]")
lg =ax8.legend((l2,l1), ["Aquadopp","ADCP"], loc = "center right", fontsize = 9, markerscale = 2)
for line in lg.get_lines():
    line.set_linewidth(3.0)
ax8.set_ylim((-0.05,3))
ax8.set_xticklabels([])
ax8.annotate("c.", xy = (0.005,0.90), xycoords = "axes fraction", color = "k", fontweight = "bold")
ax8.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
ax6.set_xlim(xl)
ax7.set_xlim(xl)
ax8.set_xlim(xl)

ax6.xaxis.set_minor_locator(mdates.MonthLocator())
ax7.xaxis.set_minor_locator(mdates.MonthLocator())
ax8.xaxis.set_minor_locator(mdates.MonthLocator())
fig.savefig("Figure_input_output.pdf", dpi = 300)
plt.close(fig)




##############
#Makes a table
#############
RWgamma_avg = gamma*P10_avg

variables = ["P10_avg","RW_avg","RWgamma_avg",\
             #"atmDISS_avg","surfDISS_avg",\
             "KE_avg","btKE_avg","PE_TM_avg","PE_TC_avg",\
             "PKE", "Vbot_AD_avg","Vbot_avg",\
             "intDISS_meas_avg","intDISS_KE_avg",\
             "botDISS_meas_avg", "botDISS_ADCP_avg","botmolDISS_ADCP_avg",\
             "BF_int","BF_bb","BF_tot"]

anm_RWgamma = anm_P10*gamma
ans_RWgamma = ans_P10*gamma
anvar = ["P10","RW","RWgamma","KE","btKE","PE_TM","PE_TC","PKE","Vbot_AD","Vbot","epsintmeas","epsint","epsbblmeas","epsbbl","epsbbllam","mixint","mixbbl","mixtot"]
          #"$\\mathcal{B}_{\\mathrm{int+bbl}}^{\mathrm{HB}}$",\
            #"BF_avg",\   
             
var_names = ["$P_{10}$ ","RW"," %1.2f\\%% $P_{10}$"%(gamma*100),\
             #"$\\mathcal{E}^{\mathrm{abl}}_{\\mathrm{LOW}}$","$\\mathcal{E}^{\mathrm{sbl}}_{\\mathrm{LOW}}$",\
             "KE","KE$_{\mathrm{M0}}$ ","PE$_{\\mathrm{mooring}}$","PE$_{\\mathrm{T-chain}}$","ME $\\approx$ %1.2f KE"%(1+PE_to_KE),\
             "$q_{\\mathrm{BBL}}$","$q_{\\mathrm{BBL}}^{\mathrm{ADCP}}$",\
             "$\\mathcal{E}_{\mathrm{int}}$", "$\\mathcal{E}_{\mathrm{int}}^{\mathrm{ADCP}}$",\
             "$\\mathcal{E}_{\mathrm{BBL}}$", "$\\mathcal{E}_{\mathrm{BBL}}^{\mathrm{ADCP}}$","$\\mathcal{E}_{\mathrm{BBL, lam}}^{\mathrm{ADCP}}$",\
             "$\\mathcal{B}_{\\mathrm{int}}$","$\\mathcal{B}_{\\mathrm{BBL}}$","$\\mathcal{B}_{\\mathrm{int+BBL}}$"]
ffactor = 1.
ftab = open("table_mean_values.txt","w+")
ftab.write("\\toprule\n")
for i in range(Navg):
    ftab.write("&%s -- %s"%(date_avg[i].strftime("%d %b"),date_avg[i+1].strftime("%d %b")))
ftab.write("&Annual")
ftab.write("\\\\\n\\midrule\n")
for i in range(len(variables)):
    if  variables[i] == "P10_avg":
        ftab.write("&\\multicolumn{5}{c}{{\\it Energy input} [mW~m$^{-2}$]} \\\ \n \\midrule\n")
    #if  variables[i] == "atmDISS_avg":
    #    ftab.write("\\midrule \n &\\multicolumn{4}{c}{ {\\it Atmospheric and surface boundary layer dissipation} [mW~m$^{-2}$]} \\\ \n \\midrule\n")
    if  variables[i] == "KE_avg":
        ftab.write("\\midrule \n &\\multicolumn{5}{c}{ {\\it Energy stock} [J~m$^{-2}$]} \\\ \n \\midrule\n")
    if  variables[i] == "Vbot_AD_avg":
        ftab.write("\\midrule \n &\\multicolumn{5}{c}{ {\\it Near-bed root-mean-square velocities}  [cm~s$^{-1}$]} \\\ \n \\midrule\n")
    if  variables[i] == "intDISS_meas_avg":
        ftab.write("\\midrule \n &\\multicolumn{5}{c}{ {\\it Energy dissipation}  [mW~m$^{-2}$]} \\\ \n \\midrule\n")
    if  variables[i] == "BF_int":
        ftab.write("\\midrule \n &\\multicolumn{5}{c}{ {\\it Buoyancy flux}  [mW~m$^{-2}$]} \\\ \n \\midrule\n")
    # variables[i] == "intDISS_meas_avg" or variables[i] == "BF_int":         
    SD = True
    if variables[i] == "PKE":
        SD = False
    
    ftab.write("%s"%(var_names[i]))
    exec("xx = %s"%(variables[i]))
    exec("mxx = anm_%s"%(anvar[i]))
    

    if SD:
        for j in range(Navg):
            if xx[0,j]>10:
                ftab.write("&%1.0f\\,$\\pm$%1.0f"%(xx[0,j],xx[3,j]))
            elif xx[0,j]>1:
                ftab.write("&%1.1f\\,$\\pm$%1.1f"%(xx[0,j],xx[3,j]))
            elif xx[0,j]>0.1:
                ftab.write("&%1.2f\\,$\\pm$%1.2f"%(xx[0,j],xx[3,j]))
            elif np.isfinite(xx[0,j]):
                ftab.write("&%1.3f\\,$\\pm$%1.3f"%(xx[0,j],xx[3,j]))
            else:
                ftab.write("&--")
        exec("sxx = ans_%s"%(anvar[i]))
        if mxx>10:
            ftab.write("&%1.0f\\,$\\pm$%1.0f"%(mxx,sxx))
        elif mxx>1:
            ftab.write("&%1.1f\\,$\\pm$%1.1f"%(mxx,sxx))
        elif mxx>0.1:
            ftab.write("&%1.2f\\,$\\pm$%1.2f"%(mxx,sxx))
        elif np.isfinite(mxx):
            ftab.write("&%1.3f\\,$\\pm$%1.3f"%(mxx,sxx))
        else:
            ftab.write("&--")
    else:
        for j in range(Navg):
            if xx[j]>10:
                ftab.write("&%1.0f"%(xx[j]*ffactor))
            elif xx[j]>1:
                ftab.write("&%1.1f"%(xx[j]*ffactor))
            elif np.isfinite(xx[j]):
                ftab.write("&%1.2f"%(xx[j]*ffactor))
            else:
                ftab.write("&--")
        if mxx>10:
            ftab.write("&%1.0f"%(mxx))
        elif mxx>1:
            ftab.write("&%1.1f"%(mxx))
        elif mxx>0.1:
            ftab.write("&%1.2f"%(mxx))
        elif np.isfinite(mxx):
            ftab.write("&%1.3f"%(mxx))
        else:
            ftab.write("&--")
            
    ftab.write("\\\\\n")
ftab.write("\\bottomrule")

ftab.close()



##
# Sc plot
intSc = intrp.interp1d( ST["timeGRs"], ST["ScGR"]/1000, bounds_error = False ) #extrapolated with depth
#intSc = intrp.interp1d( ST["timeGRs"], -ST["HB"]["APE"]["APE"]/1000, bounds_error = False )
ScADCP = intSc(ADCP["time"])

diss_ADCP_int = epsilon_ADCP_KE_fit*1000
#diss_ADCP_bot
diss_ADCP_botlam = us**3*11*1e6*bot_scale
bot_ratio = (diss_ADCP_botlam+diss_ADCP_bot+diss_ADCP_bot*mix_eff_bot)/(diss_ADCP_botlam+diss_ADCP_bot+diss_ADCP_bot*mix_eff_bot+diss_ADCP_int+diss_ADCP_int*mix_eff)


diss_ADCP_int_week = moving_average( diss_ADCP_int, 7*24 )
diss_ADCP_bbl_week = moving_average( diss_ADCP_bot, 7*24 )
diss_ADCP_bbllam_week = moving_average( diss_ADCP_botlam, 7*24 )
P10_week = moving_average(ADCP["wind_work"]["P10"], 7*24)
bot_ratio_week = (diss_ADCP_bbllam_week+diss_ADCP_bbl_week+diss_ADCP_bbl_week*mix_eff_bot)/(diss_ADCP_bbllam_week+diss_ADCP_bbl_week+diss_ADCP_bbl_week*mix_eff_bot+diss_ADCP_int_week+diss_ADCP_int_week*mix_eff)

xx = ScADCP
yy = bot_ratio_week
zz = P10_week#ADCP["wind_work"]["P10"]
pp = np.polyfit(xx[np.isfinite(xx+yy)],yy[np.isfinite(xx+yy)],1)
print pp
ppp = np.poly1d(pp)
X = np.linspace(2,25,20)
#X = np.linspace(0,10,20)
Y = ppp(X)

r = scipy.stats.pearsonr(xx[np.isfinite(xx+yy)],yy[np.isfinite(xx+yy)])

rP10 =scipy.stats.pearsonr(zz[np.isfinite(zz+yy)],yy[np.isfinite(zz+yy)])

fig, ax = plt.subplots(figsize = (4,3))
cc = ax.scatter( ScADCP[::24], bot_ratio_week[::24], 40,P10_week[::24]*1000, vmax = 700, cmap =cm.inferno, edgecolor = "k", linewidth = 0.5)
cb = plt.colorbar(cc)
ax.plot(X,Y, color = "gray", ls = "-", lw = 3)
cb.set_label("$P_{10}$ [mW m$^{-2}$]")
#ax.set_xscale("log")
ax.set_xlabel("Sc [kJ m$^{-2}$]")
ax.set_ylabel("$\\dfrac{(\\mathcal{E}+\\mathcal{B})_{\\mathrm{BBL}}} {(\\mathcal{E}+\\mathcal{B})_{\\mathrm{BBL+int}}}$ ", rotation = 90)
lab = "Sc: $r = %1.2f$ ($p<0.01$)\n$P_{10}$: $r = %1.2f$ ($p<0.01$)"%(r[0],rP10[0])
ax.annotate(lab, xy = (0.98,0.98), xycoords = "axes fraction", va = "top", ha = "right")
ax.annotate("e.", xy = (0.02,0.02), xycoords = "axes fraction", va = "bottom", ha = "left", fontweight = "bold")
ax.set_ylim((-0.05,1.05))
fig.savefig("Bottom_ratio.pdf", dpi = 300, bbox_inches = "tight")
plt.show()

mix_int = diss_ADCP_int*mix_eff
mix_bbl = diss_ADCP_bot*mix_eff
dissmix_int = diss_ADCP_int + diss_ADCP_int*mix_eff
dissmix_bbl = diss_ADCP_botlam + diss_ADCP_bot +diss_ADCP_bot*mix_eff

iiSc = np.argsort( ScADCP )
sSc = ScADCP[iiSc]

cum_diss_int = np.nancumsum( diss_ADCP_int[iiSc]*60*60/1000. )
cum_diss_bbl = np.nancumsum( (diss_ADCP_bot[iiSc]+diss_ADCP_botlam[iiSc])*60*60/1000. )
