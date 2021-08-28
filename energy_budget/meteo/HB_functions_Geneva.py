import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.dates as mdates
import matplotlib.patches as patches
import datetime as dt
import os
import scipy.interpolate as intrp
import scipy.signal as signal
import seawater as sw
import pickle
from HB_functions import *
from time_conversion import *

#extracts a period of data
def extract_period_from_dict(DATA, start_date, end_date):
    start_time, end_time = convert_date_to_time([start_date,end_date])
    index = np.where( (DATA["datenum"]>= start_time) & (DATA["datenum"]<= end_time))[0]
    L = len(DATA["datenum"])
    for var0 in DATA.keys():
        if type(DATA[var0]) == np.ndarray or type(DATA[var0]) == np.ma.core.MaskedArray:
            if DATA[var0].ndim == 1 and DATA[var0].size == L and  (type(DATA[var0][0]) ==np.float64 or type(DATA[var0][0]) ==np.float32):
                DATA[var0] = DATA[var0][index]
            elif DATA[var0].ndim == 2 and (type(DATA[var0][0,0]) ==np.float64 or type(DATA[var0][0,0]) ==np.float32):
                DATA[var0] = DATA[var0][:,index]
        elif type(DATA[var0]) ==  list and len(DATA[var0]) == L:
            DATA[var0] = [DATA[var0][j] for j in index]
    return DATA
                
####### IMPORT T MOORING DATA
def import_T_mooring(filename, mvar = "T", verbose = True):
    print "Importing data from T mooring"
    Tmoor = dict()
    Tmoor["date_str"] = []
    Tmoor["date_dt"] = []
    Tmoor["datenum"] = []
    Tmoor[mvar] = []
    Tmoor["z"] = []
    Tmoorf = open(filename,"r")
    NH = 0

    name_start = filename.rfind("/")
    name_start += 1
    code = filename[ name_start:name_start+3]
    print code
    print "Reading mooring data"
    for line in Tmoorf:
        line_s = line.split()
        if NH == 0:
            for value0 in line_s:
                if value0[0:3] == code or value0[0:3] == "wtr":
                    Tmoor["z"].append(float(value0[4:]))
        else:
            colss = np.arange(2,len(line_s))
            date_str0 = line_s[0]+" "+line_s[1]
            date_str0 = date_str0.rstrip()
            #it only reads data with 00 seconds to avoid to much data in PAR files
            if date_str0[-2:] != "00":
                continue
            if verbose:
                print "Data line %d"%(NH)
            Tmoor["date_str"].append( date_str0 )
            Tmoor["date_dt"].append( dt.datetime.strptime(date_str0, "%Y-%m-%d %H:%M:%S") )
            date_dt0 = Tmoor["date_dt"][-1]
            time0 = date_dt0.toordinal() + date_dt0.hour/24. + date_dt0.minute/24./60. + date_dt0.second/24./60./60.
            Tmoor["datenum"].append(time0)
            
            #data0 = [float(line_s[c].rstrip()) for c in colss]
            data0 = []
            for c in colss:
                try: 
                    data00 = float(line_s[c].rstrip())
                except ValueError:
                    data00 = np.nan
                data0.append(data00)
            Tmoor[mvar].append(data0)
            

        NH += 1
    Tmoor["z"] = np.array(Tmoor["z"])
    Tmoor[mvar] = np.array(Tmoor[mvar]).T
    Tmoor["datenum"] = np.array(Tmoor["datenum"])
    i1 = filename.rfind("/")
    i2 = filename.rfind(".")
    fileout = filename[i1+1:i2] + ".pi"
    with open(fileout,"w+") as f:
        pickle.dump(Tmoor,f)
    print "%s Mooring data writen to %s"%(mvar,fileout)
    
    return Tmoor

##################################
#### HEAT BUDGET CALCULATIONS ###
#################################
class heat_budget_calculator():
    def __init__(self,dates = "201902-201904",re_load = False, **params):
        self.datesR = dates
        self.params = params
        if "reload" not in self.params.keys():
            self.params["reload"] = False
        if "Nfilt0" not in params:
            self.params["Nfilt0"] = 0
        if "file_ext" not in params:
            self.params["file_ext"] = ".pdf"

        #file names with imported data
        fileT = "wtr_%s_raw_cut.pi"%(dates)
        filePAR = "par_%s_raw_cut.pi"%(dates)
        fileCOSMO = "COSMOdata.pi"

        ffolder = os.listdir(".")
        if fileT not in ffolder or re_load:
            import_T_mooring("data/wtr_%s_raw_cut.txt"%(self.datesR))
        if filePAR not in ffolder or re_load:
            import_T_mooring("data/par_%s_raw_cut.txt"%(self.datesR), mvar = "PAR")
        if fileCOSMO not in ffolder or re_load:
            COSMO = import_cosmo_data("data/COSMO_DATA_LExplore_B.dat")
            
        with open(fileT) as f:
            self.Tmoor = pickle.load(f)
        with open(filePAR) as f:
            self.PARmoor = pickle.load(f)
        with open(fileCOSMO) as f:
            self.COSMO = pickle.load(f)

        if np.sum(np.isnan(self.Tmoor["T"]))>0:
            nc = np.where( np.isnan(np.sum(self.Tmoor["T"], axis = 0)))[0]
            for i in nc:
                y = self.Tmoor["T"][:,i]
                x = self.Tmoor["z"]
                ii = np.isfinite( y )
                if not ii[0]:
                    self.Tmoor["T"][0,i] = self.Tmoor["T"][1,i]
                if not ii[-1]:
                    self.Tmoor["T"][-1,i] = self.Tmoor["T"][-2,i]
                ii = np.where(ii)[0]
                y = np.log(y[ii])
                x = x[ii]
                fin = intrp.interp1d(x,y)
                self.Tmoor["T"][:,i] = np.exp(fin(self.Tmoor["z"]))

        if np.sum(np.isnan(self.PARmoor["PAR"]))>0:
            nc = np.where( np.isnan(np.sum(self.PARmoor["PAR"], axis = 0)))[0]
            for i in nc:
                y = self.PARmoor["PAR"][:,i]
                x = self.PARmoor["z"]
                ii = np.isfinite( y )
                if not ii[0]:
                    self.PARmoor["PAR"][0,i] = self.PARmoor["PAR"][1,i]
                if not ii[-1]:
                    self.PARmoor["PAR"][-1,i] = self.PARmoor["PAR"][-2,i]
                ii = np.where(ii)[0]
                y = y[ii]
                x = x[ii]
                fin = intrp.interp1d(x,y)
                self.PARmoor["PAR"][:,i] = fin(self.PARmoor["z"])

        self.period_extracted = False
            

    def delete_T_period(self,start_date, end_date,depth_index = []):
        #start_date, end_date are datetime types
        f,c = self.Tmoor["T"].shape
        if len(depth_index)==0:
            depth_index = range(self.Tmoor["z"].size)

        for j in range(len(self.Tmoor["date_dt"])):
            if (self.Tmoor["date_dt"][j]>= start_date) and (self.Tmoor["date_dt"][j]<= end_date):
                self.Tmoor["T"][:,j] = np.nan

        x = np.arange(c)
        for k in depth_index:
            ii = np.where( np.isfinite(self.Tmoor["T"][k,:]) )[0]
            fint = intrp.interp1d(x[ii],self.Tmoor["T"][k,ii], bounds_error = False)
            self.Tmoor["T"][k,:] = fint(x)

            
    def extracts_period(self, **kargs):
        #OPTIONAL ARGUMENTS
        #time_min, time_max: datetime for selecting a period
        #otherwise the full dataseries it is used

        #needs to fix this
        self.COSMO["datenum"] = np.array(self.COSMO["datenum"])
        self.Tmoor["datenum"] = np.array(self.Tmoor["datenum"])
        self.PARmoor["datenum"] = np.array(self.PARmoor["datenum"])
        
        #puts all the variables in the same grid
        f,c = self.Tmoor["T"].shape
        
        time_min = np.max( [np.min(self.COSMO["datenum"]),np.min(self.Tmoor["datenum"]),np.min(self.PARmoor["datenum"])] )    
        time_max = np.min( [np.max(self.COSMO["datenum"]),np.max(self.Tmoor["datenum"]),np.max(self.PARmoor["datenum"])] )

        #selects a period if given
        if "time_min" in kargs:
            time_min0 = kargs["time_min"].toordinal() + kargs["time_min"].hour/24. + kargs["time_min"].minute/24./60. + kargs["time_min"].second/24./60./60.
            time_min = np.max([time_min0, time_min])
        if "time_max" in kargs:
            time_max0 = kargs["time_max"].toordinal() + kargs["time_max"].hour/24. + kargs["time_max"].minute/24./60. + kargs["time_max"].second/24./60./60.
            time_max = np.min([time_max0, time_max])

        index_cosmo_time= np.where( (self.COSMO["datenum"]>= time_min) & (self.COSMO["datenum"]<= time_max))[0]    
        date_min, date_max = convert_time_to_date([time_min,time_max])
        self.METEO = extract_period_from_dict(self.COSMO, date_min, date_max)
        """
        self.METEO = dict()
        for var0 in self.COSMO.keys():
            if type(self.COSMO[var0]) == np.ndarray and type(self.COSMO[var0][0]) ==np.float64:
                if self.COSMO[var0].ndim == 1:
                    self.METEO[var0] = self.COSMO[var0][index_cosmo_time]
                elif self.COSMO[var0].ndim == 2:
                    self.METEO[var0] = self.COSMO[var0][:,index_cosmo_time]
            elif type(self.COSMO[var0]) ==  list and len(self.COSMO[var0]) == len(self.COSMO["datenum"]):
                self.METEO[var0] = [self.COSMO[var0][j] for j in index_cosmo_time]
        """
        first_index_mooring =  np.where( self.Tmoor["datenum"]== time_min)[0][0]
        last_index_mooring =  np.where( self.Tmoor["datenum"]== time_max)[0][0]+1
        DT_moor = self.Tmoor["datenum"][1] - self.Tmoor["datenum"][0]
        DT_met = self.METEO["datenum"][1] - self.METEO["datenum"][0]
        index_T = np.arange(first_index_mooring,last_index_mooring, int(DT_met/DT_moor))
        if index_T.size==index_cosmo_time.size+1:
            index_T = index_T[:-1]
        
        self.Ntime = len(index_T)
        self.datenum = self.Tmoor["datenum"][index_T]
        self.dates = [self.Tmoor["date_dt"][first_index_mooring]]
        for i in range(1,self.Ntime):
            self.dates.append(self.dates[-1]+dt.timedelta(days = DT_met))
        self.dates = self.METEO["date_dt"]
        self.datenum = np.array(self.METEO["datenum"])

        """
        Nfilt = max([int(DT_met/DT_moor),self.params["Nfilt0"]])
        if Nfilt%2 == 0:
            Nfilt += 1
        Tfilt = np.full(self.Tmoor["T"].shape,np.nan)
        filt = np.hanning(Nfilt)
        filt/=np.sum(filt)
        iifilt = np.arange(Nfilt//2,c-Nfilt//2)

        for j in range(f):
            Tfilt[j,iifilt] = np.convolve( self.Tmoor["T"][j,:], filt, mode = "valid")
        """
        Nfilt = max([int(DT_met/DT_moor),self.params["Nfilt0"]])
        DT_filt = Nfilt*DT_moor
        fl = 1/float(DT_filt)
        fs = 1/float(DT_moor)
        fN = fs/2
        b, a = signal.butter(4, fl/fN, btype = "highpass")
        Thigh = np.full(self.Tmoor["T"].shape,np.nan)
        for j in range(f):
            Thigh[j,:] = signal.filtfilt(b,a, self.Tmoor["T"][j,:], padlen = self.Tmoor["T"][j,:].size/2)
        Tfilt = self.Tmoor["T"] - Thigh
        
        self.z = self.Tmoor["z"]
        self.Ndepth = self.z.size
        self.delta_time = float(( self.dates[1] - self.dates[0]).seconds)
        self.delta_z = np.diff(self.z)
        
        
        self.T = Tfilt[:,index_T]

        #reads PAR data
        first_index_mooring_PAR =  np.where( self.PARmoor["datenum"]== time_min)[0][0]
        last_index_mooring_PAR =  np.where( self.PARmoor["datenum"]== time_max)[0][0]+1
        index_PAR = np.arange(first_index_mooring_PAR, last_index_mooring_PAR, int(DT_met/DT_moor))
        if index_PAR.size==index_cosmo_time.size+1:
            index_PAR = index_PAR[:-1]

        self.PAR = self.PARmoor["PAR"][:,index_PAR]
        self.zPAR = self.PARmoor["z"]
        #sometimes there are negative values
        self.PAR = np.abs(self.PAR)

        #gets heat fluxes    
        self.METEO["Wsp"] = (self.METEO["U"]**2 + self.METEO["V"]**2)**0.5
        self.Hnet, self.Hturb0, self.H, self.flux_names = heat_fluxes(self.METEO["date_dt"], self.METEO["T_2M"]-273.16, self.T[0,:],self.METEO["Wsp"], self.METEO["RELHUM_2M"], self.METEO["PS"]/100., self.METEO["CLCT"]/100., self.METEO["GLOB"])

        #attenuation coefficient
        

        #Calculates Heat content
        istart = np.where(np.isfinite(self.T[0,:]))[0][0]
        Cp = sw.cp(0,self.T,np.tile(self.z,(self.Ntime,1)).T)
        rho  = sw.dens(0,self.T,np.tile(self.z,(self.Ntime,1)).T)
        self.heat_content = Cp*rho*(self.T - np.tile(self.T[:,istart], (self.Ntime,1)).T)
        self.int_heat_content = np.trapz(self.heat_content, x = self.z, axis = 0)

        
        #plots fluxes
        fig, ax = plt.subplots(3,1, sharex = True, figsize = (8,8))
        ax[0].plot( self.METEO["date_dt"], self.H["HC"], label = "Flink")
        ax[0].plot( self.METEO["date_dt"], -self.METEO["ASHFL_S"], label = "COSMO")
        ax[0].set_ylabel("$H_{sen}$ (W m$^{-2}$)")
        ax[0].legend()
        ax[0].grid(True)
        
        ax[1].plot( self.METEO["date_dt"], self.H["HA"], label = "Flink")
        ax[1].plot( self.METEO["date_dt"], -self.METEO["LW_IN_TG"], label = "Flink")
        ax[1].set_ylabel("$H_{LWdown}$ (W m$^{-2}$)")
        ax[1].grid(True)
        
        ax[2].plot( self.METEO["date_dt"], self.H["HE"], label = "Flink")
        ax[2].plot( self.METEO["date_dt"], -self.METEO["ALHFL_S"], label = "Flink")
        ax[2].set_ylabel("$H_{lat}$ (W m$^{-2}$)")
        ax[2].grid(True)
        
        fig.tight_layout()
        date1 = self.dates[0].strftime("%Y%m%d_%H%M")
        date2 = self.dates[-1].strftime("%Y%m%d_%H%M")
        fig.savefig("Heat_fluxes_comparison_%s-%s.%s"%(date1,date2,self.params["file_ext"]),dpi = 300, bbox_inches="tight")
        plt.close(fig)

        #remplaces fluxes with COSMO
        """
        H["HC"] = METEO["ASHFL_S"]
        H["HA"] = METEO["LW_IN_TG"]
        H["HE"] = METEO["ALHFL_S"]
        Hturb0 = H["HW"]+H["HA"]+H["HE"]+H["HC"]
        Hnet = Hturb0+H["HS"]
        """

        self.period_extracted = True

        

    def plot_timeseries(self):
        #Plots the time series
        if not self.period_extracted:
            print "Period not previously extracted. Extracting the whole dataset"
            self.extracts_period()
            
        fig,ax = plt.subplots(3,1, figsize = (10,9), sharex = True)
        ax[0].plot(self.dates,self.Hnet, color = "k", label = "net")
        for he in self.H.keys():
            ax[0].plot(self.dates, self.H[he], label = self.flux_names[he],lw = 1)

        ax[0].legend(ncol = 6)
        ax[0].set_ylabel("Heat fluxes [W m$^{-2}$]")
        ax[0].grid(True)
        cc=ax[1].contourf(self.dates,self.z,self.T, cmap = cm.inferno, zorder = -2)
        cc2=ax[1].contour(self.dates,self.z,self.T, colors = "gray",linewidths = 0.1, zorder = -1)
        #rect = patches.Rectangle((dt.datetime(2019,3,9,0,0),20),dt.timedelta(days = 23.4),8,linewidth=1,facecolor='white', edgecolor = "k", zorder = 0, alpha = 0.5)
        #ax[1].add_patch(rect)
        ax1 = fig.add_axes([0.99,0.395,0.015,0.22])
        cb = fig.colorbar(cc,cax = ax1)
        cb.set_label("T [$^{\circ}$C]")
        
        ax[1].clabel(cc2,fmt = "%1.1f")
        ax[1].set_ylim((30,0))
        ax[1].set_ylabel("Depth (m)")

        #ax[1].plot(self.dates[1:],np.nancumsum(int_Delta_heat*delta_time))
        l1,=ax[2].plot(self.dates,self.int_heat_content/1e8,'k')
        l2,=ax[2].plot(self.dates, -np.nancumsum(self.Hnet*self.delta_time)/1e8,ls='-', color = "gray")
        ax[2].set_ylabel("Heat content [10$^8$ J m$^{-2}$]")
        ax[2].grid(True)
        ax[2].legend((l1,l2),["Integrated heat content anomaly","Accumulated atmospheric flux"])
        fig.tight_layout()
        date1 = self.dates[0].strftime("%Y%m%d_%H%M")
        date2 = self.dates[-1].strftime("%Y%m%d_%H%M")
        fig.savefig("Temperature_%s-%s_Lexplore.%s"%(date1,date2,self.params["file_ext"]),dpi = 300, bbox_inches = "tight")
        plt.close("all")


    def get_attenuation_coefficient(self, minPAR = 50, PLOT = False):
        if not self.period_extracted:
            print "Period not previously extracted. Extracting the whole dataset"
            self.extracts_period()
            
        #gets attenuation constant from PAR measurements
        kd0 = np.full( self.Ntime, np.nan )
        R2 = np.full( self.Ntime, np.nan )

        for i in range(self.Ntime):
            PAR0 = self.PAR[:,i]
            if PAR0[0]<minPAR:
                continue
            y = np.log(PAR0)
            p = np.polyfit(self.zPAR,y,1)
            fp = np.poly1d(p)
            yp = fp(self.zPAR)
            R2[i] = np.abs(np.var(yp)/np.var(y))
            if p[0]<0 and R2[i]>0.7:
                kd0[i] = p[0]

        days = np.arange(int(np.min(self.datenum)), int(np.max(self.datenum))+1)
        self.kd = np.full( self.Ntime, np.nan )
        for day in days:
            ii = np.where( np.floor(self.datenum) == day  )
            self.kd[ii] = np.nanmean(kd0[ii])
        
        if PLOT:
            fig, ax = plt.subplots(2,1)
            ax[0].plot(self.dates, self.kd)
            ax[0].plot(self.dates, kd0)
            ax[0].set_ylabel("$k_z$ [1 m$^{-1}$]")
            ax[1].plot(self.dates, R2)
            ax[1].set_ylabel("$R^2$")
            plt.show()
            
    def overall_heat_budget(self, **kargs):
        if not self.period_extracted:
            print "Period not previously extracted. Extracting the whole dataset"
            self.extracts_period()
        if not hasattr(self,"kd"):
            print "Attenuation coeficient was not available. Calculating..."
            self.get_attenuation_coefficient()
        elif self.kd.size>self.Ntime:
            print "Updating attenuation coefficient"
            self.get_attenuation_coefficient()

            
        z = np.copy(self.z)
        z[0] = 0.
        zc = 0.5*z[1:]+0.5*z[:-1]
        index = np.where(np.isfinite(self.T[0,:]))[0]

        #selects a period if given
        if "time_min" in kargs:
            time_min0 = kargs["time_min"].toordinal() + kargs["time_min"].hour/24. + kargs["time_min"].minute/24./60. + kargs["time_min"].second/24./60./60.
            index2 = np.where(self.datenum[index]>= time_min0)[0]
            index = index[index2]
        if "time_max" in kargs:
            time_max0 = kargs["time_max"].toordinal() + kargs["time_max"].hour/24. + kargs["time_max"].minute/24./60. + kargs["time_max"].second/24./60./60.
            index2 = np.where(self.datenum[index]<= time_max0)[0]
            index = index[index2]


        time_spam = self.dates[index[-1]] - self.dates[index[0]]
        if time_spam>=dt.timedelta(days=7):
            Tin = np.nanmean(self.T[:,index[0]:index[0]+25],axis =1)
            Tf = np.nanmean(self.T[:, index[-1]-24:index[-1]+1], axis = 1)
        else:
            Tin = self.T[:,index[0]]
            Tf = self.T[:,index[-1]]

        Tin = 0.5*Tin[1:]+0.5*Tin[0:-1]
        Tf = 0.5*Tf[1:]+0.5*Tf[0:-1]
        Cpin = sw.cp(0,Tin,zc)
        rhoin  = sw.dens(0,Tin,zc)
        Cpf = sw.cp(0,Tf,zc)
        rhof  = sw.dens(0,Tf,zc)

        mT = np.nanmean(self.T[:,index], axis = 1)
        mTc = 0.5*mT[1:]+0.5*mT[:-1]
        dz2 = z[2:]-z[:-2]
        dTdz = np.full(z.size,np.nan)
        dTdz[1:-1] = -(mT[2:] - mT[:-2])/dz2
        mCp = sw.cp(0,mT,z)
        mrho  = sw.dens(0,mT,z)
        dTdz[0] = dTdz[1]
        dTdz[-1] = dTdz[-2]

        Heat_in = Tin*Cpin*rhoin
        Heat_f = Tf*Cpf*rhof
        time_int = (self.datenum[index[-1]] - self.datenum[index[0]])*24.*60*60
        delta_Heat = (Heat_f-Heat_in)/time_int

        kdm = np.tile(self.kd,(self.Ndepth,1))
        I = self.H["HS"]*np.exp( np.tile(z,(self.Ntime,1)).T*kdm)
        intI = np.nansum(I*self.delta_time,axis = 1)/time_int
        intdIdz = np.diff(intI)/self.delta_z

        intHturb0 = np.nansum(self.Hturb0*self.delta_time)/time_int 
        intHturb = np.full(intI.shape,np.nan)
        intHturb[0] =intHturb0
        Nz = intHturb.size
        for j in range(1,Nz):
            intHturb[j] = delta_Heat[j-1]*self.delta_z[j-1] + intHturb[j-1] + intI[j-1] - intI[j]

        intdHtdz = np.diff(intHturb)/self.delta_z
        Kz = -intHturb/dTdz/mCp/mrho
        Kz[Kz<=0] = np.nan

        time_start = self.dates[index[0]].strftime("%d %b %H:%M")
        time_end = self.dates[index[-1]].strftime("%d %b %H:%M")

        fig, ax = plt.subplots(1,4, sharey = True, figsize = (12,5))
        ax[0].plot(Tin,zc,'o-',label="$T_{start}$")
        ax[0].plot(Tf,zc,'o-',label="$T_{end}$")
        ax[0].plot(mTc,zc,'o-', label = "$\langle T \\rangle$")
        ax[0].set_ylim((30,0))
        ax[0].set_ylabel("Depth [m]")
        ax[0].legend(loc = "lower right")
        ax[0].set_xlabel("$T$ [$^{\circ}$C]")
        ax[0].grid(True)
        
        ax[1].plot(intI,z,"o-", label = "$H_{SW}$")
        ax[1].plot(intHturb,z,"o-", label = "$H_{Turb}$")
        ax[1].plot(intHturb+intI,z,"o-", color = "k", label = "$H_{SW} + H_{Turb}$")
        ax[1].legend(loc = "lower right")
        xlim0 = np.array(ax[1].get_xlim())
        xlim1 = np.max(np.abs(xlim0))
        ax[1].set_xlim((-xlim1,xlim1))
        ax[1].grid(True)
        ax[1].set_xlabel("Mean Heat fluxes [W m$^{-2}$]")
        
        ax[2].plot(intdIdz,zc, "o-", label = "-$\\nabla H_{SW}$")
        ax[2].plot(intdHtdz,zc, "o-", label = "-$\\nabla H_{Turb}$")
        ax[2].plot(delta_Heat,zc,  "o-", color = "k",label = "-$\\nabla (H_{SW} + H_{Turb})$")
        #ax[2].plot(intdIdz+intdHtdz,zc,  ".", color = "r",label = "-$\\nabla (H_{SW} + H_{Turb})$")
        ax[2].legend(loc = "lower right")
        xlim0 = np.array(ax[2].get_xlim())
        xlim1 = np.max(np.abs(xlim0))
        ax[2].set_xlim((-xlim1,xlim1))
        ax[2].grid(True)
        ax[2].set_xlabel("Mean Heat changes [W m$^{-3}$]")
        
        
        ax[3].semilogx(Kz, z,"o-")
        ax[3].set_xlabel("$K_z$ [m$^2$ s$^{-1}$]")
        ax[3].set_xlim((1e-8,1e-1))
        ax[3].grid(True)

        fig.suptitle("Heat budget %s to %s"%(time_start,time_end))

        date1 = self.dates[index[0]].strftime("%Y%m%d_%H%M")
        date2 = self.dates[index[-1]].strftime("%Y%m%d_%H%M")
        #fig.tight_layout()
        fig.savefig("Overall_heat_budget_%s-%s.%s"%(date1,date2,self.params["file_ext"]),dpi = 300, bbox_inches = "tight")
        plt.close(fig)

        ovHB = dict()
        variables = ["time_start","time_end","z","zc","Tin","Tf","mT","dTdz",\
                     "intI","intdIdz","intHturb","intdHtdz","Kz"]
        for var0 in variables:
            exec("ovHB[\"%s\"] = %s"%(var0,var0))

        return ovHB


    def time_resolved_heat_budget(self,Nfilt = 6):
        if not self.period_extracted:
            print "Period not previously extracted. Extracting the whole dataset"
            self.extracts_period()
        if not hasattr(self,"kd"):
            print "Attenuation coeficient was not available. Calculating..."
            self.get_attenuation_coefficient()
        elif self.kd.size>self.Ntime:
            print "Updating attenuation coefficient"
            self.get_attenuation_coefficient()
        z_b = np.copy(self.z)
        z_b[0] = 0.
        delta_z = np.diff(z_b)
        z_c = 0.5*z_b[1:] + 0.5*z_b[0:-1]
        
        #Some more smoothing
        #datapoints (1 datapoint = 1 hour)
        print "\n %d"%(Nfilt)
        
        if Nfilt > 0:
            if Nfilt%2 == 0:
                Nfilt += 1
            print "Smoothing the signals"
            filt = np.hanning(Nfilt)
            filt/=np.sum(filt)
            I0 = np.full(self.H["HS"].shape,np.nan)
            Ht0= np.full(self.Hturb0.shape,np.nan)
            kdf = np.full(self.kd.shape,np.nan)
            iifilt = np.arange(Nfilt//2,self.Ntime-Nfilt//2)
            I0[iifilt] = np.convolve( self.H["HS"], filt, mode = "valid")
            Ht0[iifilt] = np.convolve( self.Hturb0, filt, mode = "valid")
            kdf[iifilt] = np.convolve( self.kd, filt, mode = "valid")
            
            T_b = np.full(self.T.shape, np.nan)
            [f,c] = T_b.shape
            for j in range(f):
                T_b[j,iifilt] = np.convolve( self.T[j,::], filt, mode = "valid")
    
        else:
            I0 = np.copy(H["HS"])
            Ht0 = np.copy(Hturb0)
            kdf = np.copy(kd)
            T_b = np.copy(T)
            
        dTdz_b = np.full(self.T.shape, np.nan)
        dz2 = z_b[2:]-z_b[:-2]
        dz2 = np.tile(dz2,(self.Ntime,1)).T
        dTdz_b[1:-1,:] = -(T_b[2:,:] - T_b[:-2,:])/dz2
        dTdz_c = 0.5*dTdz_b[:,1:]+0.5*dTdz_b[:,0:-1]
        
        T_c = 0.5*T_b[1:,:] +0.5*T_b[0:-1,:]
        delta_T = (T_c[:,1:] - T_c[:,0:-1])/self.delta_time
        T_c = 0.5*T_c[:,1:]+0.5*T_c[:,0:-1]

        #Heat content change
        Cp_c = sw.cp(0,T_c,np.tile(z_c,(self.Ntime-1,1)).T)
        rho_c  = sw.dens(0,T_c,np.tile(z_c,(self.Ntime-1,1)).T)
        Delta_heat = Cp_c*rho_c*delta_T #[W m-3]
        int_Delta_heat = np.sum(Delta_heat*np.tile(self.delta_z,(self.Ntime-1,1)).T, axis = 0)


    
        kdm = np.tile(kdf,(z_b.size,1))
        I = I0*np.exp( np.tile(z_b,(self.Ntime,1)).T*kdm)
        I = 0.5*I[:,1:]+0.5*I[:,0:-1]
        Hturb = np.full( (z_b.size,self.Ntime-1), np.nan )
        Hturb[0,:] = ( 0.5*Ht0[1:] + 0.5*Ht0[0:-1])
        
        for i in range(self.Ntime-1):
            for j in range(1,self.Ndepth):
                Hturb[j,i] = Delta_heat[j-1,i]*self.delta_z[j-1] + I[j-1,i] - I[j,i] + Hturb[j-1,i]
    
        Cp_cb = np.full(Hturb.shape, np.nan)
        Cp_cb[1:-1,:] = 0.5*(Cp_c[1:,:]+Cp_c[:-1,:])
        rho_cb = np.full(Hturb.shape, np.nan)
        rho_cb[1:-1,:] = 0.5*(rho_c[1:,:]+rho_c[:-1,:])
        Kz = -Hturb/dTdz_c/Cp_cb/rho_cb
        #Kz[Kz<=0] = np.nan

        datesc = []
        for date in self.dates[1:]:
            datesc.append(date - dt.timedelta(seconds = int(self.delta_time)))
        
        fig,ax = plt.subplots(3,1, sharex = True, sharey = True, figsize =(8,9))
        cc=ax[0].contourf( datesc, z_b, Hturb , cmap = cm.RdBu_r)
        ax[0].contour(datesc, z_b, Hturb , 0, colors = "k", linewidths = 0.5)
        plt.colorbar(cc,ax = ax[0])
        ax[0].set_ylim((30,0))
        ax[0].set_ylabel("Depth (m)")
        ax[0].set_title("Turbulent heat flux [W m$^{-2}$]")
        
        cc2=ax[1].contourf(datesc, z_b, dTdz_c )
        plt.colorbar(cc2,ax = ax[1])
        ax[1].set_ylabel("Depth (m)")
        ax[1].set_title("dTdz [$^{\circ}$C m$^{-1}$]")
        
        cc3=ax[2].pcolormesh(datesc, z_b, np.log10(Kz), vmin = -5, vmax = -1)
        plt.colorbar(cc3,ax = ax[2], extend = "both")
        ax[2].set_title("$K_z$ [m$^{2}$ s$^{-1}$]")
        ax[2].set_ylabel("Depth (m)")
        fig.tight_layout()
        date1 = self.dates[0].strftime("%Y%m%d_%H%M")
        date2 = self.dates[-1].strftime("%Y%m%d_%H%M")
        fig.savefig("Heat_budget_time_resolved_%s-%s.%s"%(date1,date2,self.params["file_ext"]),dpi = 300)
        plt.close(fig)


        trHB = dict()
        variables = ["z_b","z_c","Hturb","I","dTdz_c","Kz"]
        for var0 in variables:
            exec("trHB[\"%s\"] = %s"%(var0,var0))

        return trHB
