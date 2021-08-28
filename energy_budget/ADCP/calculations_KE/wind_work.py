
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
from useful_functions import first_centered_differences
import vertical_modes as VM
import scipy.interpolate as intrp
from scipy import optimize
import glob
import seawater as sw
import datetime as DT
from windrose import WindroseAxes
from calc_wavelet import *
sys.path.insert(0, "/home/bieito/Documents/SCIENCE/EPFL/python_functions/")
from useful_functions import *
import HB_functions as hbf
from wgs84_ch1903 import *
import cmocean
ADCP = pickle.load(open("ADCP_Geneva_all_2019_2020_smooth_2h_4m_NM.pi"))
#ADCP = pickle.load(open("ADCP_Geneva_all_2019_2020_NM.pi"))

MET= pickle.load(open("meteo/Meteo_Leman_all_MO.pi"))
#MET= pickle.load(open("meteo/COSMO_Leman_all_MO.pi"))
#MET["tau_x"] = MET["rho_air"]*MET["Cd"]*MET["WS_ms_S_WVT"]*MET["u"]
#MET["tau_y"] = MET["rho_air"]*MET["Cd"]*MET["WS_ms_S_WVT"]*MET["v"]

nt = len(ADCP["date"])
usurf0 = np.sum(ADCP["NM"]["u"], axis = 0)[1,:]
vsurf0 = np.sum(ADCP["NM"]["v"], axis = 0)[1,:]
#usurf0 = np.nanmean(ADCP["smooth"]["u"][0:3,:], axis = 0)
#vsurf0 = np.nanmean(ADCP["smooth"]["v"][0:3,:], axis = 0)
#usurf0 = np.nanmean(ADCP["u"][0:10,:], axis = 0)
#vsurf0 = np.nanmean(ADCP["v"][0:10,:], axis = 0)
#usurf0 = ADCP["smooth"]["u"][3,:]
#vsurf0 = ADCP["smooth"]["v"][3,:]
#usurf0 = ADCP["u"][3,:]
#vsurf0 = ADCP["v"][3,:]

#barotropic mode
#usurf0 = ADCP["NM"]["u"][0,1,:]
#vsurf0 = ADCP["NM"]["v"][0,1,:]

#1st baroclinic
#usurf0 = ADCP["NM"]["u"][1,1,:]
#vsurf0 = ADCP["NM"]["v"][1,1,:]

#usurf0[np.isnan(usurf0)] = 0.
#vsurf0[np.isnan(vsurf0)] = 0.


#good one for all
usurf0 = np.full(nt,np.nan)
vsurf0 = np.full(nt,np.nan)
for i in range(nt):
    u0 = ADCP["u"][:,i]
    v0 = ADCP["v"][:,i]
    iu = np.where(np.isfinite(u0))[0]
    if iu.size>0:
        if iu[0]<5:
            usurf0[i] = u0[iu][0]
    iv = np.where(np.isfinite(v0))[0]
    if iv.size>0:
        if iv[0]<5:
            vsurf0[i] = v0[iv][0]

#usurf0[np.isnan(usurf0)] = 0.
#vsurf0[np.isnan(vsurf0)] = 0.


dt = ADCP["time"][1] - ADCP["time"][0]
wx = np.full(nt, np.nan)
wy = np.full(nt, np.nan) 
tau_x = np.full(nt, np.nan)
tau_y = np.full(nt, np.nan)
P10 = np.full(nt, np.nan)
for i in range(nt):
    ii = np.where( np.abs(MET["datenum"]-ADCP["time"][i])<dt)[0]
    tau_x[i] = np.nanmean(MET["tau_x"][ii])
    tau_y[i] = np.nanmean(MET["tau_y"][ii])
    wx[i] = np.nanmean(MET["u"][ii])
    wy[i] = np.nanmean(MET["v"][ii])
    #wx[i] = np.nanmean(MET["U"][ii]) #cosmo
    #wy[i] = np.nanmean(MET["V"][ii])
    P10[i] = np.nanmean(MET["P10"][ii])
#tau_x[np.isnan(tau_x)] = 0.
#tau_y[np.isnan(tau_y)] = 0.
#usurf0[np.isnan(usurf0)] = 0.
#vsurf0[np.isnan(vsurf0)] = 0.
RWx = usurf0*tau_x
RWy = vsurf0*tau_y
RW = RWx+RWy



fig = plt.figure()

plt.plot(ADCP["date"], np.nancumsum(RWx*dt*60*60))
plt.plot(ADCP["date"], np.nancumsum(RWy*dt*60*60))
plt.plot(ADCP["date"], np.nancumsum((RW)*dt*60*60))
plt.plot(ADCP["date"], np.nancumsum((0.0035*P10)*dt*60*60))

#plt.xlim((DT.datetime(2019,7,15,0,0,0),DT.datetime(2019,8,16,0,0,0)))


iPOS = RW>0
iNEG = RW<0
iFIN = np.isfinite(RW)

x = P10[iFIN]
y = RW[iFIN]

pp,covpp = np.polyfit( x,y,1, cov = True )
gamma_fit = pp[0]
#sgamma = covpp[0,0]**0.5

#bootstrap error estimate
NM = 5000
gamma00 = np.full(NM, np.nan)
for i in range(NM):
    nn = np.random.random()
    if nn<0.1:
        nn = 0.1
    nn *= len(x)
    nn = int(nn)
    ix = np.random.randint(len(x), size=nn)
    #print(ix.size)
    gamma00[i] = np.polyfit( x[ix],y[ix],1)[0]
sgamma_fit= np.nanstd(gamma00)
print(gamma_fit*100)
print(sgamma_fit*100)

binP10 = np.arange(0,5000,500)
cbinP10 = 0.5*binP10[1:]+0.5*binP10[:-1]
pP10,_ = np.histogram(P10[iFIN]*1e3,binP10)
#pP10 = 100./np.sum(pP10)*pP10

binRW = np.arange(0,50,5)
cbinRW = 0.5*binRW[1:]+0.5*binRW[:-1]
pRWp,_ = np.histogram(RW[iPOS]*1000,binRW)
pRWn,_ = np.histogram(-RW[iNEG]*1000,binRW)
#pRWp = 100./(np.sum(pRWp))*pRWp
#pRWn = 100./(np.sum(pRWn))*pRWn

fig = plt.figure()
ax = fig.add_axes([0.12,0.12,0.6,0.6])
l1,=ax.plot( P10[iNEG]*1e3, RW[iNEG]*1e3,".", ms = 3, color = "royalblue")
l2,=ax.plot( P10[iPOS]*1e3, RW[iPOS]*1e3,".", ms = 3, color = "crimson" )
l3,=ax.plot( ax.get_xlim(), np.array(ax.get_xlim())*pp[0] + pp[1], color = "k" )
ax.legend((l2,l1,l3),["$RW>0$","$RW<0$","$\\mathrm{RW} \\propto (%1.2f \\pm %1.2f)\\%% \\times P_{10}$"%(gamma_fit*100,sgamma_fit*100)], markerscale = 3)
#ax.set_yscale("symlog")
#ax.set_xscale("log")
ax.grid(True)
ax.set_ylim((-30,55))
ax.set_xlim((0,6500))
ax.set_ylabel("$RW$ [mW m$^{-2}$]")
ax.set_xlabel("$P_{10}$ [mW m$^{-2}$]")

ax1 = fig.add_axes([0.73,0.12,0.2,0.6])
ax1.barh(-cbinRW,pRWn, height = 4, color = "royalblue", ec = "k")
#ax1.barh(-cbinRW,pRWp, color = "crimson", height = 2, alpha = 0.5)
ax1.plot(pRWp, -cbinRW,color = "crimson")
ax1.barh(cbinRW,pRWp, height = 4, color = "crimson", ec = "k")
ax1.plot(pRWn, cbinRW,color = "royalblue")
#ax1.set_ylim((-4,2))
ax1.set_yticklabels([])
ax1.yaxis.tick_right()
#ax1.set_xticks([5,15,25])
ax1.set_xlabel("Counts")
ax1.set_xscale("log")
ax1.grid(True)
ax1.set_ylim((-30,55))
ax1.set_xticks([1e1,1e2,1e3,1e4])

ax2 = fig.add_axes([0.12,0.73,0.6,0.2])
ax2.bar(cbinP10,pP10, fc = "gray", ec = "k", width = 300)
#ax2.set_xlim((-1,4))
ax2.set_xticklabels([])
ax2.yaxis.tick_right()
#ax2.set_yticks([5,15,25])
ax2.set_ylabel("Counts")
ax2.set_yscale("log")
ax2.yaxis.set_label_position("right")
ax2.grid(True)
ax2.set_xlim((0,6500))
ax2.set_yticks([1e1,1e2,1e3,1e4])
#ax2.set_ylim((1e-1,1e3))

fig.savefig("RW_P10_correlation.png", bbox_inches = "tight", dpi = 300)
fig.savefig("RW_P10_correlation.pdf", bbox_inches = "tight", dpi = 300)
plt.show()



gamma =  np.nanmean(RW)/np.nanmean(MET["P10"])
print "RW %1.3f mW/m2"%(np.nanmean(RW)*1000)
print "P10 %1.1f mW/m2"%(np.nanmean(MET["P10"])*1000)
print "Efficiency %1.3f %%, fit: %1.3f %%"%(gamma*100,gamma_fit*100)
print "RW fit %1.3f mW/m2"%(np.nanmean(MET["P10"])*gamma_fit*1000)


#ntm = len(MET["datenum"])
Nd = 60
nn = int(Nd/dt)+1
cRW = moving_average(RW, nn, window = "gauss")

#dt_met = MET["datenum"][1] - MET["datenum"][0]
#nn = int(Nd/dt_met)+1
#cP10 = moving_average( MET["P10"], nn, window = "gauss")
cP10 = moving_average( P10, nn, window = "gauss")

xx = np.copy(usurf0)
xx[np.isnan(xx)] = 0.
yy = np.copy(tau_x)
yy[np.isnan(yy)] = 0.
XWLx = calc_crosswavelet(ADCP["date"],xx,yy, steps = 32, PLOT = False, figname = "crosswavelet_wx.png")

yy1 = np.copy(tau_y)
yy1[np.isnan(yy1)] = 0.
XWLy = calc_crosswavelet(ADCP["date"],xx,yy1, steps = 32, PLOT = False, figname = "crosswavelet_wy.png")


#preparation for the plot
PW = np.log10(XWLx["PW"])
PW=np.clip(PW,-5,8)
N = XWLx["date"].size
nquiverx = 120
nquiverz = 32
iR = np.arange(0,N,N//nquiverx)
dateR = [XWLx["date"][i] for i in iR]
Nper = XWLx["period"].size
Nper = Nper//nquiverz

#gamma = 0.0036

fig = plt.figure( figsize = (10,7) )
ax0 = fig.add_axes([0.12,0.75,0.735,0.2])
l1,=ax0.plot(ADCP["date"], cRW*1000)
#l2,=ax0.plot(MET["date_dt"],cP10*1000.*gamma)
l2,=ax0.plot(ADCP["date"],cP10*1000.*gamma)
#ax0.plot(MET["date_dt"],cP10*1000.*gamma_fit)
ax0.fill_between( ADCP["date"],np.isnan(usurf0).astype(float)*-1,np.isnan(usurf0).astype(float)*3, color = "gray", alpha = 0.5)
ax0.axhline(0, lw = 1, color = "k")
ax0.set_ylim((-1,3))
ax0.legend((l1,l2), ["$RW$","$%1.4f \\times P_{10}$"%(gamma)], loc = "upper center", framealpha = 1, ncol = 2)
ax0.set_ylabel("Wind Work [mW m$^{-2}$]")
#ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b")

ax1 = fig.add_axes([0.12,0.07,0.92,0.65])
cc=ax1.contourf(XWLx["date"], XWLx["period"], PW,int(XWLx["steps"]),cmap = cm.jet) #
cb=plt.colorbar(cc)
cb.set_label("Cross-Wavelet Power")
coicoi=np.clip(XWLx["coi"],0,XWLx["coi"].max())
ax1.contour(XWLx["date"], XWLx["period"],XWLx["sig"], np.arange(0,2,1), linewidths = 0.5, colors = "k")
ax1.quiver(dateR, XWLx["period"][::Nper], XWLx["u"][::Nper, iR], XWLx["v"][::Nper, iR], units='height',
          angles='uv', pivot='mid', linewidth=0.5, edgecolor='k', color = "k",
          headwidth=3, headlength=3, headaxislength=3, minshaft=2,
          minlength=0, scale =40)
ax1.fill_between(XWLx["date"],coicoi,XWLx["period"].max(),alpha=0.8, color='w', hatch='x')
yt = 2**np.arange(np.ceil(np.log2(XWLx["period"].min())), np.ceil(np.log2(XWLx["period"].max())))
#ax1.set_xlim(xl)
ax1.set_yscale('log')
ax1.set_yticks(yt)
ax1.set_ylim((np.min(XWLx["period"]), np.max(XWLx["period"])))
ax1.tick_params(axis='y', which='minor', left=False)
ax1.set_yticklabels(yt)
ax1.set_ylabel("time [days]")
ax1.grid(b=None, which='major', axis='y', alpha=0.2, antialiased=True, c='k', linestyle='-.')
ax1.fill_between( ADCP["date"],np.isnan(usurf0).astype(float)*XWLx["period"][0],np.isnan(usurf0).astype(float)*XWLx["period"][-1], alpha=0.8, color='gray')
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
ax1.annotate("$RW_x \sim \\tau_x$", xy = (0.02,0.05), xycoords = "axes fraction", fontsize = 12,bbox = dict(color = "w", lw = 1))
xl = ax1.get_xlim()
xtk = ax1.get_xticks()
ax0.set_xlim(xl)
ax0.set_xticks(xtk)
ax0.set_xticklabels([])
fig.savefig("cross_wavelet_wind_work.png", dpi = 300)
plt.close(fig)


##Saves the data
ADCP["wind_work"] = dict()
ADCP["wind_work"]["tau_x"] = np.copy(tau_x) 
ADCP["wind_work"]["tau_y"] = np.copy(tau_y)
ADCP["wind_work"]["wx"] = np.copy(wx)
ADCP["wind_work"]["wy"] = np.copy(wy)
ADCP["wind_work"]["u0"] = np.copy(usurf0)
ADCP["wind_work"]["v0"] = np.copy(vsurf0)
ADCP["wind_work"]["RWx"] = np.copy(RWx)
ADCP["wind_work"]["RWy"] = np.copy(RWy)
ADCP["wind_work"]["RW"] = np.copy(RW)
ADCP["wind_work"]["P10"] = np.copy(P10)
ADCP["wind_work"]["smooth_RW"] = np.copy(cRW)
ADCP["wind_work"]["smooth_P10"] = np.copy(cP10)
ADCP["wind_work"]["gamma"] = np.copy(gamma)
ADCP["wind_work"]["gamma_fit"] = np.copy(gamma_fit)


pickle.dump(ADCP,open("ADCP_Geneva_all_2019_2020_smooth_2h_4m_NM_WW.pi","w+"))


um = np.nanmean(np.nansum(ADCP["NM"]["u"], axis = 0), axis = 0)
vm = np.nanmean(np.nansum(ADCP["NM"]["v"], axis = 0), axis = 0)
#um = np.nanmean(ADCP["u"], axis = 0)
#vm = np.nanmean(ADCP["v"], axis = 0)

"""
cov = np.cov(um,vm)
eig_val, eig_vec = np.linalg.eig(cov)
UmVae = eig_val[0]/np.sum(eig_val) 
Um_angle = np.arccos(eig_vec[0,0])*180/np.pi
"""
cum_x = np.nancumsum(um*60*60)/1000.
cum_y = np.nancumsum(vm*60*60)/1000.
cum_Wx = np.nancumsum(wx*60*60)/1000./10.
cum_Wy = np.nancumsum(wy*60*60)/1000./10.


lon = 6+39./60
lat = 46+30/60.

convert = GPSConverter()
Xsamp = 540208#convert.WGStoCHy(lat,lon)#+2e6
Ysamp = 150269#convert.WGStoCHx(lat,lon)#+1e6

print(convert.WGStoCHy(lat,lon))#+2e6)
print(convert.WGStoCHx(lat,lon))

lon_SHL2 = 6.55114
lat_SHL2 = 46.43486

print(convert.WGStoCHy(lat_SHL2,lon_SHL2))#+2e6)
print(convert.WGStoCHx(lat_SHL2,lon_SHL2))

Xshl2 = (convert.WGStoCHy(lat_SHL2,lon_SHL2) - Xsamp)/1000.
Yshl2 = (convert.WGStoCHx(lat_SHL2,lon_SHL2) - Ysamp)/1000.
#Xshl2 = (535369-Xsamp)/1000.
#Yshl2 = (144588-Ysamp)/1000.

BAT = io.loadmat("bath/BathGeneva.mat")
bat = -BAT["Bathymetry"]
#topo = BAT["OveralTopography"]
x_bat = (BAT['LongitudeBath'][0,:]-Xsamp)/1000.
y_bat = (BAT['LatitudeBath'][0,:]-Ysamp)/1000.

ndays = ADCP["time"][-1] - ADCP["time"][0]
nn = 4
dint = ndays//nn
time_avg = np.arange( ADCP["time"][0],ADCP["time"][-1],dint)
date_avg = convert_time_to_date(time_avg)

time_avgC = 0.5*(time_avg[1:]+time_avg[:-1])
date_avgC = convert_time_to_date(time_avgC)
Navg = time_avgC.size

fig = plt.figure(figsize = (12,4))
for i in range(Navg):
    #imet = (MET["datenum"]>=time_avg[i]) & (MET["datenum"]>=time_avg[i+1])
    #dirW = MET["WindDir_D1_WVT"][imet]
    #magW = MET["WS_10m"][imet]

    iadcp = (ADCP["time"]>=time_avg[i]) & (ADCP["time"]<=time_avg[i+1])
    #wx00 = um[iadcp]*100
    #wy00 = vm[iadcp]*100
    wx00 = wx[iadcp]
    wy00 = wy[iadcp]
    #magW = (wx00**2+wy00**2)**0.5
    magW, dirW = hbf.uv_to_wdir(wx00, wy00)
    igood = np.isfinite(magW+dirW)
    dirW = dirW[igood]
    magW = magW[igood]
    cc = 141+i
    ax0 = fig.add_subplot(cc, projection = "windrose" )
    ax0.bar( dirW, magW, bins = np.arange(0,20.,2.5),normed=True, opening=0.8, edgecolor='white')
ax0.set_legend(loc = "bottom", ncol = 10, bbox_to_anchor = ( 1.2,-0.2))

plt.show()


intcx = intrp.interp1d(ADCP["time"], cum_x)
cx_sea = intcx(time_avg)
intcy = intrp.interp1d(ADCP["time"], cum_y)
cy_sea = intcy(time_avg)

intcwx = intrp.interp1d(ADCP["time"], cum_Wx)
cwx_sea = intcwx(time_avg)
intcwy = intrp.interp1d(ADCP["time"], cum_Wy)
cwy_sea = intcwy(time_avg)


colors_st = ["seagreen", "gold", "saddlebrown", "royalblue" ]

levels = np.arange(0.,350,25)
levels[0] = 5.
fig, ax = plt.subplots()
#ax.contour(x_bat, y_bat, bat,[10],color = "k", lw = 1)
ax.contourf(x_bat, y_bat, bat,levels,cmap = cmocean.cm.deep)
c=ax.contour(x_bat, y_bat, bat,levels,colors = "k", linewidths = 0.5)
plt.clabel(c, fmt = "%d", colors = "w", fontsize = 6)
for i in range(Navg):
    #imet = (MET["datenum"]>=time_avg[i]) & (MET["datenum"]>=time_avg[i+1])
    #dirW = MET["WindDir_D1_WVT"][imet]
    #magW = MET["WS_10m"][imet]

    iadcp = (ADCP["time"]>=time_avg[i]) & (ADCP["time"]<=time_avg[i+1])
    ax.plot(cum_x[iadcp]/5., cum_y[iadcp]/5., color = "k", lw = 4)
    ax.plot(cum_x[iadcp]/5., cum_y[iadcp]/5., color = colors_st[i], lw = 2)
    #ax.plot(cum_x, cum_y,"k--")
    ax.plot(cum_Wx[iadcp]/50., cum_Wy[iadcp]/50.,"-", color = "k", lw = 4)
    ax.plot(cum_Wx[iadcp]/50., cum_Wy[iadcp]/50., ls ="dotted", color = colors_st[i], lw = 3)
    ax.annotate( date_avg[i].strftime("%d-%b"), xy = (cx_sea[i]/5., cy_sea[i]/5.+0.5) ,va = "bottom", ha = "left", rotation = 30, fontsize = 8  )
ax.annotate( date_avg[-1].strftime("%d-%b"), xy = (cx_sea[-1]/5., cy_sea[-1]/5.+0.5) ,va = "bottom", ha = "left", rotation = 30, fontsize = 8 )
ax.plot(0, 0,"*", ms = 15, mec = "k", mfc = "magenta")
ax.plot(Xshl2, Yshl2,"s", ms = 7, mec = "k", mfc = "yellow")
ax.annotate("SHL2", xy = (Xshl2*0.95, 0.95*Yshl2), ha = "left", va = "bottom", color = "w", fontsize = 9)
ax.plot(cx_sea[1:]/5., cy_sea[1:]/5., "o", mfc = "w", mec = "k")
ax.plot(cwx_sea[1:]/50., cwy_sea[1:]/50., "o", mfc = "w", mec = "k")
ax.annotate("Depth\naveraged\ncurrents/5", xy = (0.13,0.8), xycoords = "axes fraction", fontweight = "bold", ha = "center")
ax.annotate("Wind/50", xy = (0.65,0.4), xycoords = "axes fraction", fontweight = "bold")
ax.annotate("$Geneva$", xy = (0.02,0.10), xycoords = "axes fraction")
ax.annotate("$Lausanne$", xy = (0.45,0.85), xycoords = "axes fraction")
ax.annotate("$Evian$", xy = (0.57,0.54), xycoords = "axes fraction")
ax.annotate("$FRANCE$", xy = (0.98,0.03), xycoords = "axes fraction", ha = "right")
ax.annotate("$SWITZERLAND$", xy = (0.98,0.95), xycoords = "axes fraction", ha = "right")
ax.set_xlabel("Zonal distance [km]")
ax.set_ylabel("Meridional distance [km]")

ax2 = fig.add_axes([0.45,0.15,0.1,0.1])
im = plt.imread('arrow.png')
ax2.imshow(im)
ax2.annotate( "NORTH", xy = (0.5,0.99), ha = "center", xycoords = "axes fraction" )
ax2.axis("off")

fig.savefig("map.pdf", dpi = 300)
plt.show()



vbc = np.nansum(ADCP["NM"]["v"],axis = 0)
#vbc = np.copy(ADCP["v"])
vbc[vbc == 0] = np.nan
vbc-= np.nanmean(vbc, axis = 0)

Nd = 15
nn = int(Nd/dt)+1
vbcs = np.full(vbc.shape,np.nan)
for i in range(vbc.shape[0]):
    vbcs[i,:] = moving_average(vbc[i,:], nn, window = "gauss")
cum_vbc = np.nancumsum(vbc, axis = 1)
int_vbcs = np.nancumsum(vbcs[::-1,:]*2,axis = 0)
int_vbcs = int_vbcs[::-1,:]
imax = np.argmax( np.abs(int_vbcs), axis = 0)
max_vbcs = np.full(imax.size, np.nan)
for i in range(imax.size):
    max_vbcs[i] = int_vbcs[imax[i],i]
Ekmany = - tau_x/sw.f(46.)/1000
sEkmany = moving_average(Ekmany, nn, window = "gauss")

per = 97.*1e3
area = (570-340)*1e6

w = np.nanmean(int_vbcs, axis = 1)*per/area

ST = pickle.load(open("smooth_temperature_evolution_heat_budget.pi"))
dens0 = np.nanmean(ST["denGRs"], axis = 1)
z0 = ST["depth"]
intW = intrp.interp1d( ADCP["z"], w, bounds_error = False )
w0 = intW(z0)
intV = intrp.interp1d( ADCP["z"], np.nanmean(vbc, axis = 1), bounds_error = False )
V0 = intV(z0)


b = -9.81*(dens0-np.nanmean(dens0))/np.nanmean(dens0)
#b = -9.81*(dens0)/np.nanmean(dens0)
btr = V0*b*per/area
cum_sum_btr = np.nancumsum(btr*0.5*1e6)
print np.nansum(btr*0.5)*1e6
#b = -9.81*(dens0)/np.nanmean(dens0)
JB = w0*b
print np.nansum(JB*0.5)*1e6

z00 = np.copy(z0)
z00[0] = -550.
densAD = np.full(vbc.shape, np.nan)
for i in range(vbc.shape[0]):
    ii = np.nanargmin( np.abs(ADCP["z"][i]-z00) )
    dd = ST["denGRs"][ii,:]
    jj = np.isfinite(dd)
    if np.sum(jj)==0:
        continue
    xx = ST["timeGRs"][jj]
    dd = dd[jj]
    int0 = intrp.interp1d(xx,dd, bounds_error = False)
    densAD[i,:] = int0(ADCP["time"])

mas_tr = -9.81*vbc*densAD*per/area/np.nanmean(densAD)
cum_mas_tr = np.nancumsum( mas_tr*60*60, axis = 1 )
btrmean = np.nanmean(mas_tr, axis = 1)
print np.nansum(btrmean*0.5*1e6)
