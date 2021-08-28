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
from scipy import signal
from scipy.special import erf
import matplotlib.cm as cm
import pybootstrap as bs
import scipy.optimize as opt
from scipy.stats import chi2
import sys
sys.path.insert(0, "/home/bieito/Documents/SCIENCE/EPFL/python_functions/")
from time_conversion import *

def viscosity(T):
    #seawater viscosity
    v=1.792747-0.052126103*T+0.0005918645*T**2;
    v=v*10**-6;
    return v

def Tspec(Xi, KB, K, spec = "K"):
     D = 1.44e-7
     if spec != "B" and spec != "K":
          spec = "B"
     
     if spec == "K":
          q = 5.26
          phi = KB/np.sqrt(2*q)
          y = K/phi
          f = y*np.exp( - np.sqrt(3)*y )
     else:
          q = 3.9
          phi = KB/np.sqrt(2*q)
          y = K/phi
          f = y*(np.exp(-y**2/2.)*np.sqrt(np.pi/2)*(1-erf( y/np.sqrt(2.))) )
     SK = Xi/(2*D*KB)*np.sqrt(2*q)*f
     return SK

def gets_spectrum(T,fs = 2., nps = 256, nov = 128, PLOT = False):
     f = np.full( nov+1, np.nan )
     S = np.full( nov+1, np.nan )
     if np.sum(np.isnan(T))>0:
          return S, f
     T = signal.detrend(T)
     f,S = signal.welch(T, fs=fs,  nperseg= nps, noverlap=nov)
     if PLOT:
          Steo = f**(-5/3)
          Steo = Steo*np.nanmean(S[1:])/np.nanmean(Steo[1:])
          fig, ax = plt.subplots()
          ax.loglog(f,S)
          ax.loglog(f,Steo)
          plt.show()
     return S, f

#inertial
def inertial_T_spectrum(chi, epsilon, K):
     C = 0.4
     S = C *chi * epsilon**(-1/3.) * K**(-5/3.)
     #S = C * chi * epsilon**(-1/3.) * K**(-5/3.)
     return S


def chi_inertial_calculation(K, S, epsilon):
     
     chiOP = np.nanmean( S / (0.4*epsilon**(-1./3.)*(K)**(-5./3.)) )
     Steo = inertial_T_spectrum(chiOP, epsilon, K)
     R2 = 1 - np.nanvar( (np.log10(S) - np.log10(Steo)))/ np.var( np.log10(S) )
     MAD = np.mean(np.abs(S/Steo - np.mean(S/Steo)))
     return chiOP, R2, MAD


def calculates_Xi(fr, S, Sn, U, epsilon, LO = np.nan,T = 8., fmax0 = 0.1, varcut = 0.75,varcut_in = 1.0, theos = "K", PLOT = False, figname = ""):
     D = 1.44e-7
     if np.sum(np.isnan(S))>0 or np.isnan(U) or np.isnan(epsilon):
          return np.nan, np.nan, np.nan
     visco = viscosity(T)

     KB = (epsilon/visco/D**2)**(0.25)/2/np.pi
     KLO = 1/LO/2/np.pi

     #tau = 1.
     #H = 1/(1 + (fr*2*np.pi*tau)**2)
     #S /= H
     #Sn /=H
     #if fmax0 == 0.:
     #    ifmax = np.where( H>=0.5 )[0][-1]
     #    fmax0 = fr[ifmax]
     #    print "Fmax"
     #    print fmax0
     
     k = fr/U
     Sk = S*U
     Snk = Sn*U
     Skgr = Sk*(2*np.pi*k)**2
     Snkgr = Snk*(2*np.pi*k)**2

     #Kraichnan/Batchelor
     hnoise = np.where( Skgr<2*Snkgr )[0]
     if hnoise.size == 0:
          hnoise = k.size-1
     else:
          hnoise = hnoise[0]
     fnoise = fr[hnoise]
     cumS = np.nancumsum(Skgr)
     i50 = np.where( cumS>=varcut*cumS[hnoise] )[0][0]
     f50 = fr[i50]
     k50 = f50/U
  
     fmax = min([fmax0, fnoise, f50])
     kmax = fmax/U
     i_int = np.where( (fr>fr[0]) & (fr<fmax)  )
     
     Xi0 = 6*D*np.trapz( Skgr[i_int], x = k[i_int] )

     Steo0 = Tspec(Xi0, KB, k, spec = theos)
     XiT = 6*D*np.trapz( Steo0[i_int], x = k[i_int] )
     Xi = Xi0**2/XiT

     Steo = Tspec(Xi, KB, k, spec = theos)

     #Inertial
     cumS_in = np.nancumsum(Sk)
     i50_in = np.where( cumS_in>=varcut_in*cumS_in[hnoise] )[0][0]
     f50_in = fr[i50_in]
     k50_in = f50_in/U
     fmax_in = min([fmax0, fnoise, f50_in])
     kmax_in = fmax_in/U
     i_int_in = np.where( (fr>fr[0]) & (fr<fmax_in) )
     Xi_in, R2_in, MAD_in = chi_inertial_calculation(k[i_int_in]*2*np.pi, Sk[i_int_in]/2/np.pi, epsilon = epsilon)
     #Xi_in = np.nanmean( Sk[i_int_in]/2/np.pi / (0.4*epsilon**(-1./3.)*(2*np.pi*k[i_int_in])**(-5./3.)) )
     Steo_in = inertial_T_spectrum(Xi_in, epsilon, k)
     
     if PLOT:
          print "Chi before correction %1.2e K2/s"%(Xi0)
          print "Chi after correction %1.2e K2/s"%(Xi)
          print "Chi from inertial %1.2e K2/s"%(Xi_in)
          print "epsilon %1.2e W/kg"%(epsilon)
          label = "$\\chi_0  = %1.2e$ K$^{2}$ s$^{-1}$\n$\\chi  = %1.2e$ K$^{2}$ s$^{-1}$\n$\\varepsilon  = %1.2e$ W kg$^{-1}$"%(Xi0, Xi, epsilon)
          label_in = "$\\chi = %1.2e$ K$^{2}$ s$^{-1}$\n$R^2 = %1.3f$\n$MAD = %1.2f$"%(Xi_in,R2_in,MAD_in)
          
          fig, ax = plt.subplots(1,2, figsize = (12,4))
          ax[0].loglog( k, Skgr, "k")
          ax[0].loglog( k, Snkgr, color = "gray", lw = 2, ls = "--")
          ax[0].loglog(k, Steo, lw = 2, color = "gray")
          ax[0].axvline(kmax, color = "k", lw = 1)
          ax[0].axvline(k50,color = "gray", lw = 1)
          if np.isfinite(KLO):
              ax[0].axvline(KLO, color = "k", ls = "--")
              ax[0].annotate( "$L_O$", xy = (KLO*1.05,np.nanmean(Skgr)), ha = "left" )
          ax[0].set_ylabel( "$S_{grT}$ [K$^2$ m$^{-2}$ cpm$^{-1}$]" )
          ax[0].set_xlabel( "$k$ [cpm]" )
          #ax[0].set_ylim([1e-9,1e-1])
          ax[0].annotate(label, xy = (0.95,0.05), xycoords = "axes fraction", va = "bottom", ha = "right", bbox = dict(color = "w", ec = "k"))
          if theos == "K":
              ax[0].set_title("Kraichnan")
          elif theos == "B":
              ax[0].set_title("Batchelor")

          ax[1].loglog(k, Sk,"k")
          ax[1].loglog(k[i_int_in], Steo_in[i_int_in]/2/np.pi, lw = 2, color = "gray")
          ax[1].axvline(kmax_in, color = "k", lw = 1)
          ax[1].axvline(k50_in,color = "gray", lw = 1)
          if np.isfinite(KLO):
              ax[1].axvline(KLO, color = "k", ls = "--")
              ax[1].annotate( "$L_O$", xy = (KLO*1.05,np.nanmean(Sk)), ha = "left" )
          ax[1].annotate(label_in, xy = (0.05,0.05), xycoords = "axes fraction", va = "bottom", ha = "left", bbox = dict(color = "w", ec = "k"))
          ax[1].set_ylabel( "$S_{T}$ [K$^2$ cpm$^{-1}$]" )
          ax[1].set_xlabel( "$k$ [cpm]" )
          ax[1].set_title("Inertial")
          if len(figname)>0:
               fig.savefig(figname, dpi = 300, bbox_inches = "tight")
          plt.close()

     return Xi_in, Xi, Xi0
#######################################


TH = pickle.load(open("merged_thermistors_AD_with_Aquadopp.pi"))


#################################
### Gets the temperature gradient
#################################
grT = np.full(TH["PT"].shape,np.nan)
N2 = np.full(TH["PT"].shape,np.nan)
for i in range(TH["time"].size):
     x = np.copy(TH["mab"])
     y = TH["PT"][:,i]
     #y = TH["pdens"][:,i]
     if np.sum(np.isfinite(y))<3:
          continue
     ii = np.isfinite(y)
     x = x[ii]
     y = y[ii]
     pp = np.polyfit(x,y,1)
     grT[:,i] = pp[0]
     N2[:,i] = sw.g(46.)*sw.alpha(0.2, np.nanmean(TH["PT"][:,i]),TH["pres_2mab"][i])*grT[:,i]

"""
grD = np.full(TH["pdens"].shape,np.nan)
for i in range(TH["time"].size):
     x = np.copy(TH["mab"])
     y = TH["pdens"][:,i]
     if np.sum(np.isfinite(y))<3:
          continue
     ii = np.isfinite(y)
     x = x[ii]
     y = y[ii]
     pp = np.polyfit(x,y,1)
     grD[:,i] = pp[0]

N2 = -sw.g(46.)*grD/np.nanmean(TH["pdens"])
"""
####################
#gets the spectra
###################
SL = 256
spec = np.full( (TH["mab"].size, TH["time"].size, SL+1), np.nan )
freq = np.full(SL+1, np.nan)
l = 0
for i in range(TH["time"].size):
     print "Calculates spectrum %s"%(TH["date"][i])
     for j in range(TH["mab"].size):
          while np.nansum(np.isnan(freq))>0:
               spec[j,i,:], freq = gets_spectrum(TH["Temp"][j,i,:], fs = 2.,nov = SL, nps = SL*2)
          else:
               spec[j,i,:] = gets_spectrum(TH["Temp"][j,i,:], fs = 2.,nov = SL, nps = SL*2)[0]

##########################
##Gets the noise spectra
#########################
spec_min = np.nanpercentile(spec,1,axis = 1)
spec_noise = np.full(spec_min.shape, np.nan)
min_freq = 0.05
PLOT = False
for j in range(TH["mab"].size):
     x = np.copy(freq)
     y = spec_min[j,:]
     #min_freq = freq[np.nanargmin(y*x**2)]
     x = x[freq>min_freq]
     y = y[freq>min_freq]
     x =  np.log10(x)
     y = np.log10(y)
     pol = np.polyfit(x,y,2)
     ppol = np.poly1d(pol)
     spec_noise[j,:] = 10**ppol(np.log10(freq))

     if PLOT:
          fig, ax = plt.subplots()
          ax.loglog(freq, spec_min[j,:]*freq**2)
          ax.loglog(freq[freq>min_freq], spec_noise[j, freq>min_freq]*freq[freq>min_freq]**2)
          plt.show()
spec_noise[np.isnan(spec_noise)] = 0.


##############
#Calculates Xi
##############

epsilon = np.copy(TH["epsilon"])
LO = (epsilon/N2**(3./2.))**0.5

FRMAX = 0.1 ##paper
#FRMAX = 0.05
THEOS = "K"

i = int(np.random.random()*TH["time"].size)
#i = 1375
#i = 2800 #joli
#i = 4300
#i = 3

#i = 3694
j = 1
print TH["U"][j,i]
print TH["q"][j,i]
calculates_Xi(freq, spec[j,i,:], spec_noise[j,:], TH["U"][j,i], TH["epsilon"][j,i], LO = LO[j,i],T = TH["mTemp"][j,i], fmax0 = 0., theos = THEOS, figname = "test_%s_%1.1fmab_fr%1.2f.pdf"%(TH["date"][i].strftime("%Y%m%d_%H%M"), TH["mab"][j],0.0),PLOT = True)
calculates_Xi(freq, spec[j,i,:], spec_noise[j,:], TH["U"][j,i], TH["epsilon"][j,i], LO = LO[j,i],T = TH["mTemp"][j,i], fmax0 = 0.1, theos = THEOS, figname = "test_%s_%1.1fmab_fr%1.2f.pdf"%(TH["date"][i].strftime("%Y%m%d_%H%M"), TH["mab"][j],0.1),PLOT = True)
calculates_Xi(freq, spec[j,i,:], spec_noise[j,:], TH["U"][j,i], TH["epsilon"][j,i], LO = LO[j,i],T = TH["mTemp"][j,i], fmax0 = 0.05, theos = THEOS, figname = "test_%s_%1.1fmab_fr%1.2f.pdf"%(TH["date"][i].strftime("%Y%m%d_%H%M"), TH["mab"][j],0.05),PLOT = True)
calculates_Xi(freq, spec[j,i,:], spec_noise[j,:], TH["U"][j,i], TH["epsilon"][j,i], LO = LO[j,i],T = TH["mTemp"][j,i], fmax0 = 0.15, theos = THEOS, figname = "test_%s_%1.1fmab_fr%1.2f.pdf"%(TH["date"][i].strftime("%Y%m%d_%H%M"), TH["mab"][j],0.15),PLOT = True)


Xi = np.full((TH["mab"].size, TH["time"].size), np.nan)
XiK = np.full((TH["mab"].size, TH["time"].size), np.nan)
Xi0 = np.full((TH["mab"].size, TH["time"].size), np.nan)
Xi_flag = np.full((TH["mab"].size, TH["time"].size), False)
for i in range(TH["time"].size):
     print "\nCalculating Xi %s"%(TH["date"][i])
     for j in range(TH["mab"].size):
          Xi[j,i],XiK[j,i], Xi0[j,i] = calculates_Xi(fr = freq, S = spec[j,i,:], Sn = spec_noise[j,:], U = TH["U"][j,i], epsilon = TH["epsilon"][j,i], LO = LO[j,i], T = TH["mTemp"][j,i], theos = THEOS, fmax0 = FRMAX)
          if TH["U"][j,i]>5*TH["q"][j,i]:
               Xi_flag[j,i] = True

Xi_clean = np.copy(Xi) #takes inertial
#Xi_clean = np.copy(XiK) #takes kraichnan
Xi_clean[Xi_flag == False] = np.nan
#Xi_clean[0,:] = np.nan #eliminates first thermistor
KT = 0.5*Xi_clean/grT**2
KT[grT<0] = np.nan
BF = KT*N2
Rf = BF/(TH["epsilon"]+BF)
Gamma =BF/TH["epsilon"]
Reb = TH["epsilon"]/viscosity(np.nanmean(TH["Temp"]))/N2
Reb[N2<0] = np.nan

#Ellison scale
LE = np.nanvar(TH["Temp"], axis = 2)**0.5/grT
LE[N2<0] = np.nan
LO[N2<0] = np.nan
LK = (viscosity(np.nanmean(TH["Temp"]))**3/ TH["epsilon"])**0.25
FrT = (LO/LE)**(2./3.)
ReT = ( LE/LK )**(4./3.)


mRf = np.nanmean(Rf[Rf>0])
ciRf = bs.bootstrap(Rf[Rf>0], confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean)
med_Rf = np.nanmedian(Rf[Rf>0])
mGamma = mRf/(1-mRf)

mFrT = np.nanmean(FrT[FrT>0])
ciFrT = bs.bootstrap(FrT[FrT>0], confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean)
med_FrT = np.nanmedian(FrT[FrT>0])

mepsilon = np.nanmean(epsilon[epsilon>0])
ciepsilon = bs.bootstrap(epsilon[epsilon>0], confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean)
med_epsilon= np.nanmedian(epsilon[epsilon>0])

mXi = np.nanmean(Xi_clean[Xi_clean>0])
ciXi = bs.bootstrap(Xi_clean[Xi_clean>0], confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean)
med_Xi= np.nanmedian(Xi_clean[Xi_clean>0])

mN2 = np.nanmean(N2[np.isfinite(N2)])
ciN2 = bs.bootstrap(N2[np.isfinite(N2)], confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean)
med_N2= np.nanmedian(N2[np.isfinite(N2)])

mKT = np.nanmean(KT[KT>0])
ciKT = bs.bootstrap(KT[KT>0], confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean)
med_KT= np.nanmedian(KT[KT>0])

mReb = np.nanmean(Reb[Reb>0])
ciReb = bs.bootstrap(Reb[Reb>0], confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean)
med_Reb= np.nanmedian(Reb[Reb>0])

mBF = np.nanmean(BF[BF>0])
ciBF = bs.bootstrap(BF[BF>0], confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean)
med_BF= np.nanmedian(BF[BF>0])

eps_x = epsilon[BF>0]
BF_y = BF[BF>0]

x_bin = np.arange(-11,-5.25,0.25)
y_bin = np.arange(-12,-6.25,0.25)
hist2d = np.full( (x_bin.size, y_bin.size), np.nan )
for i, x0 in enumerate(x_bin):
    for j, y0 in enumerate(y_bin):
        ii = (np.abs(np.log10(eps_x)-x0)<0.5) & (np.abs(np.log10(BF_y)-y0)<0.5)
        hist2d[i,j] = np.nansum(ii)

        

fig, ax = plt.subplots(2,3, figsize = (14,8))
ax[0,0].hist( np.log10(epsilon[epsilon>0]),20, ec = "k", color = "slateblue" )
ax[0,0].set_xlabel("$log_{10}\\varepsilon$ [W kg$^{-1}$]")
ax[0,0].set_ylabel("Counts")
ax[0,0].plot( np.log10( np.nanmean(epsilon) ), 130, "v", color = "k", mfc = "k", ms = 10 )
ax[0,0].plot( np.log10( np.nanmedian(epsilon) ), 130, marker = "v", color = "k", mfc = "w", ms = 10)
ax[0,0].annotate("mean:\n$%1.1f\\,[%1.1f-%1.1f]\\times 10^{-8}$\nmedian:\n$%1.2f\\times 10^{-8}$"%(mepsilon*1e8,ciepsilon[0]*1e8,ciepsilon[1]*1e8,med_epsilon*1e8), xy = (0.75,0.88), xycoords = "axes fraction", ha = "center", va = "center", fontsize = 9)

ax[0,1].hist( np.log10(Xi_clean[Xi_clean>0]),20, ec = "k" , color = "slateblue")
ax[0,1].plot( np.log10( np.nanmean(Xi_clean) ), 55, "v", color = "k", mfc = "k", ms = 10 )
ax[0,1].plot( np.log10( np.nanmedian(Xi_clean) ), 55, marker = "v", color = "k", mfc = "w", ms = 10)
ax[0,1].set_xlabel("$log_{10}\\chi$ [W kg$^{-1}$]")
ax[0,1].annotate("mean:\n$%1.1f\\,[%1.1f-%1.1f]\\times 10^{-7}$\nmedian:\n$%1.2f\\times 10^{-7}$"%(mXi*1e7,ciXi[0]*1e7,ciXi[1]*1e7,med_Xi*1e7), xy = (0.23,0.88), xycoords = "axes fraction", ha = "center", va = "center", fontsize = 9)
ax[0,1].set_xlim((-13,-3))

ax[0,2].hist( np.log10(N2[N2>0]),20, ec = "k", fc = "darkgoldenrod" )
ax[0,2].hist( np.log10(np.abs(N2[N2<0])),20, ec = "k", fc = "darkgreen" )
ax[0,2].legend( ["$N^2>0$","$N^2<0$"], loc = "center left")
ax[0,2].plot( np.log10( np.nanmean(N2) ), 300, "v", color = "k", mfc = "k", ms = 10 )
ax[0,2].plot( np.log10( np.nanmedian(N2) ), 300, marker = "v", color = "k", mfc = "w", ms = 10)
ax[0,2].set_xlabel("$log_{10}N^2$ [s$^{-2}$]")
ax[0,2].annotate("mean:\n$%1.2f\\,[%1.2f-%1.2f]\\times 10^{-5}$\nmedian:\n$%1.2f\\times 10^{-5}$"%(mN2*1e5,ciN2[0]*1e5,ciN2[1]*1e5,med_N2*1e5), xy = (0.26,0.88), xycoords = "axes fraction", ha = "center", va = "center", fontsize = 9)

ax[1,0].hist( np.log10(KT[KT>0]),20, ec = "k" , color = "slateblue")
ax[1,0].set_xlabel("$log_{10}K_T$ [m$^2$ s$^{-1}$]")
ax[1,0].set_ylabel("Counts")
ax[1,0].plot( np.log10( np.nanmean(KT) ), 60, "v", color = "k", mfc = "k", ms = 10 )
ax[1,0].plot( np.log10( np.nanmedian(KT) ), 60, marker = "v", color = "k", mfc = "w", ms = 10)
ax[1,0].annotate("mean:\n$%1.1f\\,[%1.1f-%1.1f]\\times 10^{-2}$\nmedian:\n$%1.2f\\times 10^{-4}$"%(mKT*1e2,ciKT[0]*1e2,ciKT[1]*1e2,med_KT*1e4), xy = (0.75,0.88), xycoords = "axes fraction", ha = "center", va = "center", fontsize = 9)

ax[1,1].hist( np.log10(Reb[Reb>0]), 20,ec = "k" , color = "slateblue")
ax[1,1].set_xlabel("$log_{10}Re_b$")
ax[1,1].plot( np.log10( np.nanmean(Reb) ), 120, "v", color = "k", mfc = "k", ms = 10 )
ax[1,1].plot( np.log10( np.nanmedian(Reb) ), 120, marker = "v", color = "k", mfc = "w", ms = 10)
ax[1,1].annotate("mean:\n$%1.1f\\,[%1.1f-%1.1f]\\times 10^{4}$\nmedian:\n$%1.1f\\times 10^{2}$"%(mReb*1e-4,ciReb[0]*1e-4,ciReb[1]*1e-4,med_Reb*1e-2), xy = (0.72,0.88), xycoords = "axes fraction", ha = "center", va = "center", fontsize = 9)

ax[1,2].hist( np.log10(Rf[Rf>0]), 20,ec = "k" , color = "slateblue")
ax[1,2].set_xlabel("$log_{10}R_f$")
ax[1,2].plot( np.log10( np.nanmean(Rf) ), 50, "v", color = "k", mfc = "k", ms = 10 )
ax[1,2].plot( np.log10( np.nanmedian(Rf) ), 50, marker = "v", color = "k", mfc = "w", ms = 10)
ax[1,2].annotate("mean:\n$%1.3f\\,[%1.3f-%1.3f]$\nmedian:\n$%1.3f$"%(mRf,ciRf[0],ciRf[1],med_Rf), xy = (0.25,0.3), xycoords = "axes fraction", ha = "center", va = "center", fontsize = 9)

ax2 = fig.add_axes([0.725,0.33,0.09,0.12])
ax2.contourf( 10**x_bin, 10**y_bin , hist2d.T, cmap = cm.hot_r)
ax2.plot(np.array([1e-10, 3e-7]), mGamma*np.array([1e-10, 3e-7]) )
ax2.annotate("$\\Gamma = %1.2f$"%(mGamma), xy = (0.05,0.8), xycoords = "axes fraction", color = cm.tab10(0))
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_ylabel("$\\mathcal{B} = K_T N^2$")
ax2.set_xlabel("$ \\varepsilon$")

labels = ["a.","b.","c.","d.","e.","f."]
for l,ax00 in enumerate(ax.ravel()):
    ax00.annotate(labels[l], xy = (0.02,0.02), xycoords = "axes fraction", fontweight = "bold")

fig.savefig("Fluxes_BBL_large.pdf", dpi = 300, bbox_inches = "tight")
plt.show()


fig, ax = plt.subplots(1,3, figsize = (16,4))
ax[0].hist( np.log10(epsilon[epsilon>0]),20, ec = "k", color = "slateblue" )
ax[0].set_xlabel("$log_{10}\\varepsilon$ [W kg$^{-1}$]")
ax[0].set_ylabel("Counts")
ax[0].plot( np.log10( np.nanmean(epsilon) ), 130, "v", color = "k", mfc = "k", ms = 10 )
ax[0].plot( np.log10( np.nanmedian(epsilon) ), 130, marker = "v", color = "k", mfc = "w", ms = 10)
ax[0].annotate("mean:\n$%1.1f\\,[%1.1f-%1.1f]\\times 10^{-8}$\nmedian:\n$%1.2f\\times 10^{-8}$"%(mepsilon*1e8,ciepsilon[0]*1e8,ciepsilon[1]*1e8,med_epsilon*1e8), xy = (0.75,0.86), xycoords = "axes fraction", ha = "center", va = "center", fontsize = 9)
ax[0].set_xticks(np.arange(-11,-4))

ax[1].hist( np.log10(Xi_clean[Xi_clean>0]),20, ec = "k" , color = "slateblue")
ax[1].plot( np.log10( np.nanmean(Xi_clean) ), 55, "v", color = "k", mfc = "k", ms = 10 )
ax[1].plot( np.log10( np.nanmedian(Xi_clean) ), 55, marker = "v", color = "k", mfc = "w", ms = 10)
ax[1].set_xlabel("$log_{10}\\chi$ [K$^2$ s$^{-1}$]")
ax[1].annotate("mean:\n$%1.1f\\,[%1.1f-%1.1f]\\times 10^{-7}$\nmedian:\n$%1.2f\\times 10^{-7}$"%(mXi*1e7,ciXi[0]*1e7,ciXi[1]*1e7,med_Xi*1e7), xy = (0.23,0.86), xycoords = "axes fraction", ha = "center", va = "center", fontsize = 9)
ax[1].set_xlim((-13,-3))
ax[1].set_xticks(np.arange(-12,-3))

"""
ax[1,0].hist( np.log10(N2[N2>0]),20, ec = "k", fc = "darkgoldenrod" )
ax[1,0].hist( np.log10(np.abs(N2[N2<0])),20, ec = "k", fc = "darkgreen" )
ax[1,0].legend( ["$N^2>0$","$N^2<0$"], loc = "center left")
ax[1,0].plot( np.log10( np.nanmean(N2) ), 300, "v", color = "k", mfc = "k", ms = 10 )
ax[1,0].plot( np.log10( np.nanmedian(N2) ), 300, marker = "v", color = "k", mfc = "w", ms = 10)
ax[1,0].set_xlabel("$log_{10}N^2$ [s$^{-2}$]")
ax[1,0].annotate("mean:\n$%1.2f\\,[%1.2f-%1.2f]\\times 10^{-5}$\nmedian:\n$%1.2f\\times 10^{-5}$"%(mN2*1e5,ciN2[0]*1e5,ciN2[1]*1e5,med_N2*1e5), xy = (0.26,0.86), xycoords = "axes fraction", ha = "center", va = "center", fontsize = 9)
ax[1,0].set_xticks(np.arange(-10,-1))
"""

ax[2].hist( np.log10(KT[KT>0]),20, ec = "k" , color = "slateblue")
ax[2].set_xlabel("$log_{10}K_T$ [m$^2$ s$^{-1}$]")
ax[2].set_ylabel("Counts")
ax[2].plot( np.log10( np.nanmean(KT) ), 60, "v", color = "k", mfc = "k", ms = 10 )
ax[2].plot( np.log10( np.nanmedian(KT) ), 60, marker = "v", color = "k", mfc = "w", ms = 10)
ax[2].annotate("mean:\n$%1.1f\\,[%1.1f-%1.1f]\\times 10^{-2}$\nmedian:\n$%1.2f\\times 10^{-4}$"%(mKT*1e2,ciKT[0]*1e2,ciKT[1]*1e2,med_KT*1e4), xy = (0.75,0.86), xycoords = "axes fraction", ha = "center", va = "center", fontsize = 9)
ax[2].set_xticks(np.arange(-8,3))


labels = ["a.","b.","c.","d."]
for l,ax00 in enumerate(ax.ravel()):
    ax00.annotate(labels[l], xy = (0.02,0.02), xycoords = "axes fraction", fontweight = "bold")

fig.savefig("Fluxes_BBL.pdf", dpi = 300, bbox_inches = "tight")
plt.show()


fig, ax = plt.subplots(1,2, figsize = (8,3.5))

ax[0].hist( np.log10(N2[(N2>0) & (np.isfinite(Xi))]),20, ec = "k", fc = "darkgoldenrod" )
ax[0].hist( np.log10(np.abs(N2[(N2<0) & (np.isfinite(Xi))])),20, ec = "k", fc = "darkgreen" )
ax[0].legend( ["$N^2>0$","$N^2<0$"], loc = "center left")
ax[0].plot( np.log10( np.nanmean(N2) ), 110, "v", color = "k", mfc = "k", ms = 10 )
ax[0].plot( np.log10( np.nanmedian(N2) ), 110, marker = "v", color = "k", mfc = "w", ms = 10)
ax[0].set_xlabel("$log_{10}N^2$ [s$^{-2}$]")
ax[0].annotate("mean:\n$%1.2f\\,[%1.2f-%1.2f]\\times 10^{-5}$\nmedian:\n$%1.2f\\times 10^{-5}$"%(mN2*1e5,ciN2[0]*1e5,ciN2[1]*1e5,med_N2*1e5), xy = (0.30,0.87), xycoords = "axes fraction", ha = "center", va = "center", fontsize = 9)
ax[0].set_ylabel("Counts")

ax[1].hist( np.log10(Rf[Rf>0]), 20,ec = "k" , color = "slateblue")
ax[1].set_xlabel("$log_{10}R_f$")
ax[1].plot( np.log10( np.nanmean(Rf) ), 50, "v", color = "k", mfc = "k", ms = 10 )
ax[1].plot( np.log10( np.nanmedian(Rf) ), 50, marker = "v", color = "k", mfc = "w", ms = 10)
ax[1].annotate("mean:\n$%1.3f\\,[%1.3f-%1.3f]$\nmedian:\n$%1.3f$"%(mRf,ciRf[0],ciRf[1],med_Rf), xy = (0.26,0.3), xycoords = "axes fraction", ha = "center", va = "center", fontsize = 9)

ax2 = fig.add_axes([0.645,0.64,0.13,0.22])
ax2.contourf( 10**x_bin, 10**y_bin , hist2d.T, cmap = cm.hot_r)
ax2.plot(np.array([1e-10, 3e-7]), mGamma*np.array([1e-10, 3e-7]) )
ax2.annotate("$\\Gamma = %1.2f$"%(mGamma), xy = (0.05,0.8), xycoords = "axes fraction", color = cm.tab10(0))
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_ylabel("$b = K_T N^2$")
ax2.set_xlabel("$ \\varepsilon$")

ax[0].annotate("a.", xy = (0.02,0.02), xycoords = "axes fraction", fontweight = "bold")
ax[1].annotate("b.", xy = (0.02,0.02), xycoords = "axes fraction", fontweight = "bold")

fig.savefig("Rf_BBL.pdf", dpi = 300, bbox_inches = "tight")
fig.savefig("Rf_BBL.png", dpi = 300, bbox_inches = "tight")
plt.show()



x_FrT = FrT.ravel()
y_Rf = Rf.ravel()

xxb = np.logspace(-1,1,21)
xx = 0.5*xxb[1:]+0.5*xxb[:-1]
yy = np.full(xx.size, np.nan)
nn = np.full(xx.size, np.nan)
for i in range(xx.size):
    ii = (x_FrT>=xxb[i-1]) & (x_FrT<=xxb[i])
    yy[i] = np.nanmedian(y_Rf[ii])
    nn[i] = np.nansum(np.isfinite(y_Rf[ii]))
    


 
fig = plt.figure(figsize = (10,5))

ax0 = fig.add_axes([0.08,0.12,0.34,0.6])
ax0.hist( np.log10(N2[N2>0]),20, ec = "k", fc = "darkgoldenrod" )
ax0.hist( np.log10(np.abs(N2[N2<0])),20, ec = "k", fc = "darkgreen" )
ax0.legend( ["$N^2>0$","$N^2<0$"], loc = "center left")
ax0.plot( np.log10( np.nanmean(N2) ), 300, "v", color = "k", mfc = "k", ms = 10 )
ax0.plot( np.log10( np.nanmedian(N2) ), 300, marker = "v", color = "k", mfc = "w", ms = 10)
ax0.set_xlabel("$log_{10}N^2$ [s$^{-2}$]")
ax0.annotate("mean:\n$%1.2f\\,[%1.2f-%1.2f]\\times 10^{-5}$\nmedian:\n$%1.2f\\times 10^{-5}$"%(mN2*1e5,ciN2[0]*1e5,ciN2[1]*1e5,med_N2*1e5), xy = (0.26,0.86), xycoords = "axes fraction", ha = "center", va = "center", fontsize = 9)
ax0.set_xticks(np.arange(-10,-1))
ax0.set_ylabel("Counts")

ax = fig.add_axes([0.51,0.12,0.34,0.6])
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

ax1 = fig.add_axes([0.86,0.12,0.13,0.6])
ax1.hist( np.log10(Rf[Rf>0]), 20,ec = "k" , color = "slateblue",orientation = "horizontal")
ax1.plot(90, np.log10(mRf), "<", mfc = "k", mec = "k", ms = 8)
ax1.plot(90, np.log10(med_Rf), "<", mfc = "w", mec = "k", ms = 8)
ax1.set_ylim(( np.log10(yl[0]),np.log10(yl[1]) ))
ax1.set_yticklabels([])
ax1.annotate("$R_f$ mean:\n%1.3f\n[%1.3f-%1.3f]\nmedian:\n$%1.3f$"%(mRf,ciRf[0],ciRf[1],med_Rf), xy = (0.5,0.2), xycoords = "axes fraction", ha = "center", va = "center", fontsize = 9)

ax2 = fig.add_axes([0.51,0.74,0.34,0.20])
ax2.bar( np.log10(xx),nn, width = 0.1, color = "slateblue", ec = "k" )
ax2.plot( np.log10(mFrT), 120,"v", mfc = "k", mec = "k", ms = 8)
ax2.plot( np.log10(med_FrT),120, "v", mfc = "w", mec = "k", ms = 8)
#ax2.set_ylim((0,4000))
ax2.set_xlim((-1, np.log10(25.)))
ax2.yaxis.tick_right()
ax2.set_ylabel("Counts")
ax2.yaxis.set_label_position("right")
ax2.set_xticks(np.log10([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,20]))
ax2.set_xticklabels([])
ax2.annotate("$Fr_T$ mean:\n$%1.1f\\,[%1.1f-%1.1f]$\nmedian:\n$%1.1f$"%(mFrT,ciFrT[0],ciFrT[1],med_FrT), xy = (0.2,0.60), xycoords = "axes fraction", ha = "center", va = "center", fontsize = 9)

ax3 = fig.add_axes([0.59,0.22,0.12,0.18])
ax3.contourf( 10**x_bin, 10**y_bin , hist2d.T, cmap = cm.hot_r)
ax3.plot(np.array([1e-10, 3e-7]), mGamma*np.array([1e-10, 3e-7]) )
ax3.annotate("$\\Gamma = %1.2f$"%(mGamma), xy = (0.05,0.8), xycoords = "axes fraction", color = cm.tab10(0))
ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_ylabel("$\\mathcal{b} = K_T N^2$")
ax3.set_xlabel("$ \\varepsilon$")

ax0.annotate("a.", xy = (0.02,0.02), xycoords = "axes fraction", fontweight = "bold")
ax.annotate("b.", xy = (0.02,0.02), xycoords = "axes fraction", fontweight = "bold")

fig.savefig("Rf_BBL_02.pdf", dpi = 300)
fig.savefig("Rf_BBL_02.png", dpi = 300)
plt.show()

MET= pickle.load(open("../../meteo/Meteo_Leman_all_MO.pi"))

date1 = DT.datetime(2020,1,12,0,0,0)
date2 = DT.datetime(2020,2,20,0,0,0)

time00 = convert_date_to_time([date1,date2])
ij = np.where( (TH["time"]>time00[0])&(TH["time"]<time00[-1]))[0]
time0 = TH["time"][ij]
date0 = convert_time_to_date(time0)
KT0 = 0.5*Xi/grT**2
KT0[grT<0] = np.nan
BF0 = KT0*N2
Rf0 = BF0/(TH["epsilon"]+BF0)
KOsb = TH["epsilon"][2,:]/N2[2,:]*0.15
KOsb[KOsb<0] = np.nan
N20 = np.copy(N2[0,:])
N20[N20<0] = np.nan

ijmet = np.where( (MET["datenum"]>time00[0])&(MET["datenum"]<time00[-1]))[0]
timemet = MET["datenum"][ijmet]
datemet = convert_time_to_date(timemet)

print np.nanmean(MET["P10"][ijmet])

eps1m = TH["epsilon"][2,ij]
us = (0.41*eps1m)**(1./3.)
delta = 11*1.4e-6/us
inteps = eps1m*np.log(9./delta)*1e6

us2 = (0.00368*TH["U"][2,ij]**2)**(1./2.)
epsLOW = us2**3/0.41
delta2 = 11*1.4e-6/us2
intepsLOW = epsLOW*np.log(9./delta2)*1e6

fig, ax = plt.subplots(4,1, sharex = True, figsize = (8,8))
l1,= ax[0].plot(date0, TH["U"][2,ij]*100,"k")
l2,=ax[0].plot(datemet,MET["WS_10m"][ijmet], zorder = -1, color = cm.Accent(4), lw = 1)
ax[0].legend((l2,l1),["Wind speed\n(m s$^{-1}$)","$u_{1\\,\\mathrm{mab}}$\n(cm s$^{-1}$)"], bbox_to_anchor = (1,0.8))
ax[0].set_ylabel("Flow / wind\nspeed")
#ax[0].set_ylim((0,22))
#ax0 = ax[0].twinx()
#ax0.plot(datemet,MET["sHFnet"][ijmet])
ax[1].semilogy(date0, N20[ij],"k")
ax[1].semilogy(date0, np.isnan(N20[ij])*0.85e-4,"|", color = cm.Dark2(1))
ax[1].set_ylim((1e-7,1e-4))
ax[1].set_ylabel("$N^2$ (s$^{-2}$)")

l1,=ax[2].semilogy(date0, TH["epsilon"][2,ij], color = "k")
#ax2 = ax[2].twinx()
#ax[2].semilogy(date0, Xi[2,ij],  ls = "-", color = cm.tab10(0), zorder = -1)
l2,=ax[2].semilogy(date0, (0.00368*TH["U"][2,ij]**2)**(3./2.)/0.41,ls ="-", lw = 1, color =  cm.Accent(4), alpha = 0.9)
ax[2].legend((l1,l2),["SF","LOW"], ncol = 2)
ax[2].set_ylabel("")
ax[2].set_ylim((1e-11,5e-5))
ax[2].semilogy(date0, np.isnan(N20[ij])*3.5e-5,"|", color = cm.Dark2(1))
ax[2].set_yticks(10**np.arange(-11.,-4))
ax[2].set_ylabel("$\\varepsilon_{1\\mathrm{mab}}$ (W kg$^{-1}$)")
ax[2].legend((l1,l2),["SF","LOW"], bbox_to_anchor = (1,0.66))

l1,=ax[3].semilogy(date0, KT0[2,ij],"k")
l2,=ax[3].semilogy(date0, KOsb[ij],ls = "-", lw = 1, color =  cm.Accent(4), alpha = 0.9)
l3,=ax[3].semilogy(date0, (0.41*0.00368*TH["U"][2,ij]**2)**0.5,lw = 1, color = cm.Dark2(6), alpha = 0.8)
ax[3].legend((l1,l2,l3),["$K_{z} = \\chi/(2\\,(\\partial_z \\theta)^2)$","$K_{z} = 0.15 \\varepsilon/N^2$","$K_{LOW} =\\kappa u_{*}\\,z$"],  bbox_to_anchor = (1,0.7))
ax[3].set_ylim((1e-7,1e0))
ax[3].semilogy(date0, np.isnan(N20[ij])*0.7e-0,"|", color = cm.Dark2(1))
ax[3].set_ylabel("$K_z$ (m$^2$ s$^{-1}$)")
ax[0].set_xlim((date1,date2)  )

labels = ["(a)","(b)","(c)","(d)"]
for k,ax0 in enumerate(ax):
    ax0.grid(True)
    ax0.annotate(labels[k], xy = (0.01,0.9), xycoords ="axes fraction")
fig.subplots_adjust(hspace=0.05)
fig.savefig("encyclopedia.png", dpi = 300, bbox_inches = "tight")
plt.close(fig)
plt.show()


micro = pickle.load( open("../../CTD_micro/microstructure_down_Lac_Leman_merged.pi") )

N2_1m = N2[2,ij]
BF_1m = 0.5*Xi[2,ij]/grT[2,ij]**2*N2[2,ij]
