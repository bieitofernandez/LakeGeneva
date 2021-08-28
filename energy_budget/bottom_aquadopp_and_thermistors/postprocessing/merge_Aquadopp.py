import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import glob
import seawater as sw
import scipy.stats as stats
import scipy.interpolate as intrp
import copy
from scipy import io
import pybootstrap as bs
ADfiles = sorted( glob.glob("*.pi"))

zref = 1
B = np.arange(3)

AD = dict()
AD["date"] = []
AD["time"] = []
j= 0
for ADf0 in ADfiles[0:11]:
    print ADf0
    AD0 = pickle.load(open(ADf0))
    dataEXP = copy.deepcopy(AD0["mDATA"])
    dataEXP.pop("dates")
    ADf1 = ADf0[0:ADf0.rfind(".")]+".mat"
    io.savemat(ADf1,dataEXP)
    dt, dz = AD0["mDATA"]["u"].shape
    for i in range(dt):
        if np.isnan(AD0["mDATA"]["mab"][i,:]).all():
            u_ = np.full((3,1),np.nan)
            upc_ = np.full((3,1),np.nan)
            eps1m_ = np.nan
        else:
            izref = np.nanargmin( np.abs(AD0["mDATA"]["mab"][i,:]-zref))
            u_ = np.array([ AD0["mDATA"]["u"][i,izref],AD0["mDATA"]["v"][i,izref],AD0["mDATA"]["w"][i,izref]  ])
            upc_ = np.array([ AD0["mDATA"]["v_pc1"][i,izref],AD0["mDATA"]["v_pc2"][i,izref],AD0["mDATA"]["v_pc3"][i,izref]  ])
            eps1m_ = np.array([])
            for k in range(3):
                izref = np.nanargmin( np.abs(AD0["mDATA"]["mab%d"%(k+1)][i,:]-zref))
                eps1m_ = np.r_[ eps1m_, np.nanmean(AD0["mDATA"]["SF"]["epsilon"][k,izref-5:izref+6,i])]
            eps1m_ = np.nanmean(eps1m_)
                
        AD["date"].append(AD0["mDATA"]["dates"][i])
        AD["time"].append(AD0["mDATA"]["time"][i])   
        if j==0:
            AD["v_1m"] = np.copy(u_)
            AD["vpc_1m"] = np.copy(upc_)
            AD["epsilon_SF_1m"] = eps1m_
            AD["mepsilon_SF"] = np.nanmean(AD0["mDATA"]["SF"]["epsilon"][:,:,i])
            #AD["mepsilon_SF"] = AD0["mDATA"]["SF"]["mepsilon"][i]
            AD["ustar"] = AD0["mDATA"]["LOW"]["us"][i]
            AD["z0"] = AD0["mDATA"]["LOW"]["z0"][i]
            AD["R2_LOW"] = AD0["mDATA"]["LOW"]["R2"][i]
            j+=1
        else:
            AD["v_1m"] = np.c_[AD["v_1m"],u_]
            AD["vpc_1m"] = np.c_[AD["vpc_1m"],upc_]
            AD["epsilon_SF_1m"] = np.c_[AD["epsilon_SF_1m"],eps1m_]
            AD["mepsilon_SF"] = np.c_[AD["mepsilon_SF"],np.nanmean(AD0["mDATA"]["SF"]["epsilon"][:,:,i])]
            #AD["mepsilon_SF"] = np.c_[AD["mepsilon_SF"],AD0["mDATA"]["SF"]["mepsilon"][i]]
            AD["ustar"] = np.c_[AD["ustar"],AD0["mDATA"]["LOW"]["us"][i]]
            AD["z0"] = np.c_[AD["z0"],AD0["mDATA"]["LOW"]["z0"][i]]
            AD["R2_LOW"] = np.c_[AD["R2_LOW"],AD0["mDATA"]["LOW"]["R2"][i]]
            j +=1

AD["time"] = np.array(AD["time"])


#FITS Cd LOW
u2 = np.nansum(AD["vpc_1m"][0:2,:]**2, axis = 0)
AD["h_speed"] = u2**0.5

#us2 = (AD["epsilon_SF_1m"][0,:]*0.41)**(2./3.)
us2 = (AD["ustar"][0,:])**(2.)
us2[AD["R2_LOW"][0,:]<0.7] = np.nan

ii = np.isfinite(us2)
x = u2[ii]
y = us2[ii]
x = x[:,np.newaxis]
y = y[:,np.newaxis]
Cd = np.linalg.lstsq(x, y)[0][0,0]

alpha = 0.05
N = x.size
ymod = x*Cd
yres = y-ymod
R2f = 1- np.var(yres)/np.var(y)
tval = stats.t.ppf(1-alpha,N-2)
e = tval*(((1-R2f)/R2f)/(N-2))**0.5

AD["Cd"] = np.copy(Cd)
z0 = np.exp( 0.41/Cd**0.5)**-1
print "z0 = %1.1f mm"%(z0*1000)
AD["mz0"] = np.copy(z0)
AD["ustar_Cd"] = (AD["Cd"]*u2)**0.5
AD["epsilon_LOW_1m"] = AD["ustar"]**3/0.41

#FITS CD structure functions
x = np.copy(u2)**0.5
y = (AD["epsilon_SF_1m"][0,:]*0.41)**(1./3.)
ii = (np.isfinite(x) ) & (np.isfinite(y)) #& (x**0.5>0.01)
x = x[ii]
y = y[ii]
x = x[:,np.newaxis]**2
y = y[:,np.newaxis]**2
Cd_SF = np.linalg.lstsq(x, y)[0][0,0]
alpha = 0.05
N = x.size
ymod = x*Cd
yres = y-ymod
R2sf = 1- np.var(yres)/np.var(y)
tval = stats.t.ppf(1-alpha,N-2)
esf = tval*(((1-R2sf)/R2sf)/(N-2))**0.5

xx= x**0.5
yy = y**0.5
x_bin = np.linspace(-3.5,-0.5,30)
y_bin = np.linspace(-4,-2,31)
dx = 0.5*(x_bin[1]-x_bin[0])
dy = 0.5*(y_bin[1]-y_bin[0])
HH = np.full( (y_bin.size, x_bin.size), np.nan )
for i in range(x_bin.size):
    for j in range(y_bin.size):
        HH[j,i] = np.sum( (np.abs(np.log10(xx)-x_bin[i])<= dx) &  (np.abs(np.log10(yy)-y_bin[j])<= dy) )



print Cd_SF
AD["Cd_SF"] = np.copy(Cd_SF)
z0_SF = np.exp( 0.41/Cd_SF**0.5)**-1
AD["mz0_SF"] = np.copy(z0_SF)

epsilon_1m = AD["epsilon_SF_1m"][0,:]
mepsilon = np.nanmean(epsilon_1m[epsilon_1m>0])
ciepsilon = bs.bootstrap(epsilon_1m[epsilon_1m>0], confidence = 0.99, iterations = 200, sample_size = 1., statistic = np.nanmean)
med_epsilon= np.nanmedian(epsilon_1m[epsilon_1m>0])


fig = plt.figure(figsize = (8,7))
ax0 = fig.add_axes([0.1,0.50,0.83,0.2])
#l1,= ax0.semilogy(AD["date"],AD["ustar_Cd"]**3/0.41,color = "gray", lw = 1)
l2, = ax0.semilogy(AD["date"],AD["epsilon_SF_1m"].T, lw = 1, color = "k")
ax0.set_ylabel("$\\varepsilon_{\\mathrm{1\\, mab}}$ [W kg$^{-1}$]")
ax0.set_ylim((1e-11,5e-5))
#ax.semilogy(AD["date"],AD["epsilon_LOW_1m"].T,"--")
#ax0.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
#ax0.legend(["$\\varepsilon_{LOW}$","$\\varepsilon_{SF}$"], ncol = 2)
ax0.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
ax0.grid(True)
xl = ax0.get_xlim()

ax0b = fig.add_axes([0.1,0.73,0.83,0.2])
ax0b.plot(AD["date"],AD["vpc_1m"][0,:]*100, lw = 1, color = "k")
ax0b.set_xlim(xl)
ax0b.axhline(0, color = "gray", ls = "dotted", lw = 1)
ax0b.set_xticklabels([])
ax0b.set_ylabel("$u_{\\mathrm{1\,mab}}^{pc1}$ [cm s$^{-1}$]")
ax0b.grid(True)

ax = fig.add_axes([0.1,0.08,0.34,0.36])
ax.hist( np.log10(epsilon_1m[epsilon_1m>0]),20, ec = "k", color = "slateblue" )
ax.set_xlabel("$log_{10}\\varepsilon_{\\mathrm{1\, mab}}$ [W kg$^{-1}$]")
ax.set_ylabel("Counts")
ax.plot( np.log10( np.nanmean(epsilon_1m) ), 50, "v", color = "k", mfc = "k", ms = 10 )
ax.plot( np.log10( np.nanmedian(epsilon_1m) ), 50, marker = "v", color = "k", mfc = "w", ms = 10)
ax.annotate("mean:\n$%1.1f\\,[%1.1f-%1.1f]\\times 10^{-8}$\nmedian:\n$%1.2f\\times 10^{-8}$"%(mepsilon*1e8,ciepsilon[0]*1e8,ciepsilon[1]*1e8,med_epsilon*1e8), xy = (0.75,0.95), xycoords = "axes fraction", ha = "center", va = "top", fontsize = 9)
#ax.grid(True)

#fig, ax = plt.subplots()
ax1 = fig.add_axes([0.53,0.08,0.42,0.36])
#ax1.loglog(x**0.5,y**0.5,".", ms =1, color = "gray")
cc = ax1.pcolor( 10**x_bin, 10**y_bin, HH, cmap = cm.hot_r )
cb = plt.colorbar(cc, ax = ax1)
cb.set_label("Counts")
ax1.loglog(np.sort(x[xx>5e-3])**0.5,(Cd_SF*np.sort(x[xx>5e-3]))**0.5, color = "slateblue")
#plt.loglog(np.sort(x),(Cd**0.5*np.sort(x)))
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_ylabel("$u_{*}$ [m s$^{-1}$]")
ax1.set_xlabel("$u_{\\mathrm{1\, mab}}$ [m s$^{-1}$]")
text0 = "Structure functions\n$C_d = %1.5f \pm %1.5f$ "%(AD["Cd_SF"],Cd*esf)
ax1.annotate(text0, xy = (0.05,0.95), xycoords= "axes fraction", verticalalignment = "top")
ax1.annotate("$u_{*} = (\kappa \\varepsilon^{\\mathrm{1\, mab}})^{1/3}$",xy = (0.95,0.15), xycoords= "axes fraction", verticalalignment = "top",ha = "right",fontsize = 14)
ax1.set_xlim((5e-4,2e-1))
ax1.set_ylim((1e-4,2e-2))
#ax1.grid(True)
#fig.savefig("drag_coefficient_fit_SF.pdf", dpi = 300, bbox_inches = "tight")
ax0.annotate("b.", xy = (0.02,0.02), xycoords = "axes fraction", fontweight = "bold")
ax0b.annotate("a.", xy = (0.02,0.02), xycoords = "axes fraction", fontweight = "bold")
ax.annotate("c.", xy = (0.02,0.02), xycoords = "axes fraction", fontweight = "bold")
ax1.annotate("d.", xy = (0.02,0.02), xycoords = "axes fraction", fontweight = "bold")
fig.savefig("Aquadopp_overall.pdf", dpi = 300, bbox_inches = "tight")
plt.show()


fig = plt.figure(figsize = (10,7))
ax0 = fig.add_axes([0.1,0.62,0.83,0.30])
l1,= ax0.semilogy(AD["date"],AD["ustar_Cd"]**3/0.41,color = "gray", lw = 1)
l2, = ax0.semilogy(AD["date"],AD["epsilon_SF_1m"].T, lw = 1, color = "k")
ax0.set_ylabel("$\\varepsilon_{\\mathrm{1\, mab}}$ [W kg$^{-1}$]")
ax0.set_ylim((1e-12,5e-5))
#ax.semilogy(AD["date"],AD["epsilon_LOW_1m"].T,"--")
ax0.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
ax0.legend(["$\\varepsilon_{LOW}$","$\\varepsilon_{SF}$"], ncol = 2)

ax = fig.add_axes([0.1,0.1,0.39,0.44])
cc = ax.scatter(u2**0.5, us2**0.5,5, AD["R2_LOW"][0,:], vmin = 0, vmax = 1)
ax.plot(np.sort(u2)**0.5, (Cd*np.sort(u2))**0.5, 'k-')
cb = plt.colorbar(cc)
cb.set_label("$R^2$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel("$u^{*}$ [m s$^{-1}$]")
ax.set_xlabel("$u_{\\mathrm{1\,mab}}$ [m s$^{-1}$]")
text0 = "LOW\n$C_d = %1.5f \pm %1.5f$\n$R^2 = %1.4f$"%(AD["Cd"],Cd*e,R2f)
ax.annotate(text0, xy = (0.05,0.95), xycoords= "axes fraction", verticalalignment = "top")
ax.set_xlim((2e-4,2e-1))
ax.set_ylim((6e-5,2e-2))


#fig, ax = plt.subplots()
ax1 = fig.add_axes([0.58,0.1,0.35,0.44])
ax1.loglog(x**0.5,y**0.5,".", ms =1, color = "gray")
ax1.loglog(np.sort(x)**0.5,(Cd_SF*np.sort(x))**0.5, color = "k")
#plt.loglog(np.sort(x),(Cd**0.5*np.sort(x)))
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_ylabel("$u_{\\mathrm{SF}}^{*}$ [m s$^{-1}$]")
ax1.set_xlabel("$u_{\\mathrm{1\,mab}}$ [m s$^{-1}$]")
text0 = "Structure functions\n$C_d = %1.5f \pm %1.5f$\n$R^2 = %1.4f$ "%(AD["Cd_SF"],Cd*esf,R2sf)
ax1.annotate(text0, xy = (0.05,0.95), xycoords= "axes fraction", verticalalignment = "top")
ax1.annotate("$u_{\\mathrm{SF}}^{*} = (\kappa \\varepsilon^{\\mathrm{1 m}}_{\\mathrm{SF}})^{1/3}$",xy = (0.95,0.10), xycoords= "axes fraction", verticalalignment = "top",ha = "right",fontsize = 14)
ax1.set_xlim((2e-4,2e-1))
ax1.set_ylim((6e-5,2e-2))
#fig.savefig("drag_coefficient_fit_SF.pdf", dpi = 300, bbox_inches = "tight")
fig.savefig("Aquadopp_overall_hist.pdf", dpi = 300, bbox_inches = "tight")
plt.show()


pickle.dump(AD,open("merged_aquadop.pi","w+"))

x = np.copy(u2)**0.5
y = AD["mepsilon_SF"][0,:]
ii = (np.isfinite(x) ) & (np.isfinite(y)) & (x>0.01)
x = np.log10(x[ii])
y = np.log10(y[ii])
p = np.polyfit(x,y,1)
print p
pp = np.poly1d(p)


fig, ax = plt.subplots()
ax.loglog(u2**0.5,AD["mepsilon_SF"].T,".", ms = 0.5)
ax.loglog( 10**np.sort(x), 10**pp(np.sort(x)))
plt.show()
#ax.semilogy(AD["date"],AD["epsilon_LOW_1m"].T,"--")

