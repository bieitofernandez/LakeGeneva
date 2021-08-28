##FUNCTIONS FOR READING METEO DATA AND CALCULATING HEAT FLUXES
# B. Fernandez Castro 2020.04.22

import numpy as np
import datetime as dt
import pickle
#from Pysolar import solar # Pysolar package version 0.6
import pysolar_BFC as solar # Pysolar package version 0.6 
import seawater as sw
import copy


#### IMPORT DATA FROM WEATHER STATION
def import_weather_station(filename):
    print "Importing data from weather station"
    MET = dict()
    metf = open(filename,"r")
    NH = 0

    for line in metf:
        print "Importing data-line %d"%(NH)
        if NH == 1:
            variables0 = line.split(",")
            print variables0
            variables = []
            for var0 in variables0:
                var0 = var0.replace("\"","")
                var0 = var0.replace(" ","")
                var0 = var0.rstrip() #deletes new line character if some
                variables.append(var0)

            MET["variables"] = np.copy(variables)
            for var0 in variables:
                MET[var0] = []
            MET["date_dt"] = []
            MET["datenum"] = []
            
        elif NH == 2:
            units0 = line.split(",")
            MET["units"] = []
            for un0 in units0:
                un0 = un0.replace("\"","")
                un0 = un0.replace(" ","")
                un0 = un0.rstrip()
                MET["units"].append(un0)
                
        elif NH>3:
            line_s = line.split(",")
            iv = 0
            for var0 in variables:
                value = line_s[iv]
                if value == "NAN" or value =="\"NAN\"":
                    value = -999999.
                elif value[0] != "\"":
                    value = float(value)
                else:
                    value = value.replace("\"","")
                     #print value
                    
                MET[var0].append(value)
                iv+=1
            MET["date_dt"].append( dt.datetime.strptime(MET["TIMESTAMP"][-1], "%Y-%m-%d %H:%M:%S") )
            date_dt0 = MET["date_dt"][-1]
            time0 = date_dt0.toordinal() + date_dt0.hour/24. + date_dt0.minute/24./60. + date_dt0.second/24./60./60.
            MET["datenum"].append(time0)
        NH+=1
    #converts to numpy arrays
    for var0 in variables:
        if type(MET[var0][0]) == float:
            MET[var0] = np.array(MET[var0])
            MET[var0][MET[var0]==-999999.] = np.nan
    print "Import succesful"

    with open("METEOdata.pi","w+") as f:
        pickle.dump(MET,f)
    print "Meteo data saved in METEOdata.pi"
    
    return MET


#### IMPORT COSMO DATA
def import_cosmo_data(filename):
    print "Importing data from COSMO"
    COSMO = dict()
    cmsf = open(filename,"r")
    NH = 0
    for line in cmsf:
        print "Data line %d"%(NH)
        if NH == 0:
            variables = line.split()
            COSMO["variables"] = np.copy(variables)
            for var0 in variables:
                COSMO[var0] = []
                COSMO["date_str"] = []
                COSMO["date_dt"] = []
                COSMO["datenum"] = []
        elif NH == 1:
            COSMO["units"] = line.split("  ")
            COSMO["units"][-1] = COSMO["units"][-1].rstrip()
        elif NH>1:
            line_s = line.split();
            iv = 0
            if line_s[1] == "NA":
                continue
            for var0 in variables:
                if iv>3:
                    if line_s[iv] != "NA":
                        COSMO[var0].append(float(line_s[iv]))
                    else:
                        COSMO[var0].append(-999999.)
                else:
                    COSMO[var0].append(line_s[iv])
                iv+=1
            COSMO["date_str"].append(COSMO["date"][-1] + " " + COSMO["time"][-1])
            print COSMO["date_str"][-1]
            COSMO["date_dt"].append( dt.datetime.strptime(COSMO["date_str"][-1], "%Y-%m-%d %H:%M:%S") )
            date_dt0 = COSMO["date_dt"][-1]
            time0 = date_dt0.toordinal() + date_dt0.hour/24.  + date_dt0.minute/24./60. + date_dt0.second/24./60./60.
            COSMO["datenum"].append(time0)
        NH+=1
        
    #converts to numpy arrays
    allvars = variables + ["datenum"]
    for var0 in allvars:
        if type(COSMO[var0][0]) == float:
            COSMO[var0] = np.array(COSMO[var0])
            COSMO[var0][COSMO[var0]==-999999.] = np.nan
    print "Import succesful"

    #save to pickle
    with open("COSMOdata.pi","w+") as f:
        pickle.dump(COSMO,f)
    print "Cosmo data saved as picklein COSMOSdata.pi"
    
    return COSMO

########################
#cloud cover calculation
########################
    
def cloud_cover(date,LON,LAT,ELEV,Q,T,Rh,P):
    #Woolway 2015 procedure to calculate DAILY cloud cover using the Pysolar package version 0.6 to obtain Zenith
    Zenit = np.full(len(date), np.nan)
    DOY = np.full(len(date), np.nan)
    for i in range(len(date)):
        Zenit[i] = 90- solar.GetAltitude(LAT,LON,date[i]) #, ELEV, T[i], P[i])
        DOY[i] = (date[i]-dt.datetime(date[i].year,1,1)  ).total_seconds()/24./60./60.
        #CSRad[i] = solar.radiation.GetRadiationDirect(date[i],altitude)
    Ieff = 1353*(1+0.034*np.cos( 2*np.pi*DOY/365 ))
    Ieff[Ieff<0]=0.
    cosZ = np.cos(np.pi/180.*Zenit)
    cosZ[cosZ<0] = 0.

    #Rayleigh scattering
    p = (101325.*(1-ELEV*2.25577e-5)**5.25588)/100. #surface pressure
    m = 35.*cosZ*(1244.*cosZ**2+1)**-0.5 #air-mass thickness
    TrTpg = 1.021-0.084*( m*(0.000949*p + 0.051) )**0.5

    #Water vapour absortion
    es = saturation_vapour_pressure(T)
    ez = Rh*es/100.
    Td = (243.5*np.log(ez/6.112))/(17.67-np.log(ez/6.112)) + 33.8
    G = G_constant(DOY,LAT)
    pw = np.exp( (0.1133-np.log(G+1)) +0.0393*Td )
    Tw = 1. - 0.077*(pw*m)**0.3

    #Aerosol attenuation
    Ta = 0.95**m

    #clear-sky irradiance
    Ic = Ieff*cosZ*TrTpg*Tw*Ta

    #calculates daily cloud cover
    deltat = date[1]-date[0]
    deltat = deltat.total_seconds()/24./60./60.
    ndp = int(1/deltat)
    if ndp%2 == 0:
        ndp+=1

    iw = int(ndp//2)
    hw = np.ones(int(ndp))
    hw /= np.sum(ndp)
    mIc = np.full( Ic.size, np.nan)
    mQ = np.full( Ic.size, np.nan)
    mIc[iw:-iw] = np.convolve(  Ic, hw, mode = "valid")
    mQ[iw:-iw] = np.convolve(  Q, hw, mode = "valid")
    #percentage of clear sky radiation
    csf = mQ/mIc
    csf[0:iw] = csf[iw]
    csf[-iw+1:] = csf[-iw]
    #corrects for inaccuracies of maximum irradiance (which seems to be too high)
    maxcsf = np.percentile(csf[np.isfinite(csf)],99.)
    csf = csf/maxcsf
    csf[csf>1] = 1.
    clf = 1 - csf

    return clf, Ic

def G_constant(DOY,LAT):
    G = np.full(len(DOY), np.nan)

    G0 = np.array( [ [3.37,2.85,2.80,2.64],
           [2.99,3.02,2.70,2.93],
           [3.60,3.00,2.98,2.93],
           [3.04,3.11,2.92,2.94],
           [2.70,2.95,2.77,2.71],
           [2.52,3.07,2.67,2.93],
           [1.76,2.69,2.61,2.61],
           [1.60,1.67,2.24,2.63],
           [1.11,1.44,1.94,2.02]])
    iseason = np.floor(DOY/366.*4).astype(int)
    ilat = np.floor(LAT/91.*9).astype(int)
    for k in range(len(DOY)):
        G[k] = G0[ilat, iseason[k]]
    return G

    
###########################
### Wind transformations ##
##########################
def wdir_to_uv(w,alpha):
    alpha = 270.-alpha
    alpha *=np.pi/180
    
    u = w*np.cos(alpha)
    v = w*np.sin(alpha)
    return u,v

def uv_to_wdir(u,v):
    w = (u**2+v**2)**0.5
    alpha = 180/np.pi*np.arctan2(v,u)
    alpha = 270.-alpha
    alpha[alpha>360]-=360
    #alpha[alpha>180] = 360 - alpha[alpha>180]
    return w, alpha

#######################################
########### Required functions ########
#######################################
def latent_heat_vap(T):
    #latent heat of vaporiyation.
    #Inputs
    #T [degC]
    #Output:
    #Lv in J/kg
    Lv = 2.501e6 - 2370.*T
    return Lv

def air_density(T, RH, p):
    #Calculates air density kg/m3
    #inputs
    # T: air temperature in [degC]
    # RH: relative humidity in [%]
    # p: pressure in [hPa]
    """
    p = 100*p
    Rd = 287.058 #[J/Kg/K]
    Rv = 461.495 #[J/Kg/K]
    psat = saturation_vapour_pressure(T)*100 #[Pa]
    pv = RH/100.*psat
    pd = p - pv
    rho = pd/Rd/(T+273.16) + pv/Rv/(T+273.16)
    """
    e_s = saturation_vapour_pressure(T)
    e_a = RH*e_s/100. # vapour pressure, mb
    q_z = 0.622*e_a/p
    R_a = 287*(1 + 0.608*q_z)
    rho = 100*p/(R_a*(T + 273.16))
    
    #print rho
    return rho

def air_kin_visco(rho,T):
    # air viscosity
    # [m2/s]
    KinV = (1/rho)*(4.94e-8*T + 1.7184e-5)
    return KinV

def saturation_vapour_pressure(T):
    #Saturation vapour pressure (hPa)
    #Input:
    #T: air temperature in degree C
    #Outout
    #es: saturation vapour pressure (mbar/hPa)
    es = 6.11*np.exp( (17.27*T)/(237.3+T) )
    return es

##################################################
############ Heat flux calculator ################
#################################################                
class HeatFluxCalculator():
    def __init__(self, **kargs):
        #required variables
        #z0: height of the observation
        #Tw: Water temperature in [degC]
        #Ta: Air temperature in [degC]
        #Wsp: wind speed in [m/s]
        ##RH: relative humidity in [%]
        #P: atmospheric pressure [hPa/mbar]
        #C: cloud cover fraction [0...1]
        #SRad: downward solar radiation [W m-2]
        
        
        ############
        ##Parameters
        ############
        default_c = dict()

        #default system
        default_c["altitude"] = 372.
        default_c["latitude"] = 46.
        
        #constants for Monin-Obukhov
        default_c["kappa"] = 0.41
        default_c["Cpa"] = 1006.
        default_c["alpha1"] = 0.013
        default_c["alpha2"] = 0.11
        default_c["b1"] = 2.67
        default_c["b2"] = -2.57
        default_c["eta_thres"] = 15.
        default_c["etam"] = -1.574
        default_c["etah"] = -0.465

        #constants for absorbed short-wave from Fink
        default_c["Adiff"] = 0.066
        default_c["Adir"] = np.array([0.220,0.161,0.108,0.084,0.075,0.073,0.074,0.080,0.099,0.144,0.210,0.241])

        #constants for longwave
        default_c["sigma"] = 5.67e-8 #W m-2 K-4
        default_c["AL"] = 0.03
        default_c["a"] = 1.0592
        default_c["Cc"] = 0.17
        
        def_c_names = default_c.keys()
        for const in def_c_names:
            if const in kargs:
                exec("self.%s = kargs[\"%s\"]"%(const, const))
            else:
                exec("self.%s = default_c[\"%s\"]"%(const, const))

        
        ##Reads variables
        inputvars = ["zu","zt","zq","Tw","Ta","Wsp","RH","P","C","SRad","date"]
        if not "P" in kargs:
            #pressure as in Woolway
            kargs["P"] = 101325*(1. - 2.25577e-5*self.altitude)**5.25588/100.*np.ones( len(kargs["date"]))
        for vv in inputvars:
            exec( "self.%s = kargs[\"%s\"]"%(vv,vv) )

        #calculates density of air and water
        self.g = 9.780310*(1+(0.00530239*(np.sin(np.abs(self.latitude))**2) - 0.00000587*(np.sin(np.abs(2*self.latitude))**2) - (31.55e-8)*self.altitude));
        self.rhoa = air_density(self.Ta, self.RH, self.P)
        self.rhow = sw.dens0(0.2, self.Tw)
        
    def shortwave_Fink(self):
        #absorbed shortwave radiation according to Fink 2014
        print "Absorbed shortwave radiation according to Fink 2014"
        Fdir = (1.-self.C)/( (1.-self.C) + 0.5*self.C )
        Fdiff = 0.5*self.C/( (1.-self.C) + 0.5*self.C )
        month = np.zeros(len(self.date))
        self.SRad[self.SRad<0.] = 0.
        for (i,date) in zip(range(len(self.date)),self.date):
            month[i] = date.month
        month = month.astype(int)
        self.Qsw =-( self.SRad*Fdir*(1.-self.Adir[month-1]) + self.SRad*Fdiff*(1.-self.Adiff))
        self.month = month

    def longwave_Fink(self):
        #calculates absorbed and emitted long-wave radiation according to Fink
        print "Long-wave radiation according to Fink"
        #absorbed
        es = saturation_vapour_pressure( self.Ta )
        ea = self.RH*es/100.
        Ea = self.a*(1+self.Cc*self.C**2)*1.24*(ea/(self.Ta+273.16))**(1/7.) #atmospheric emisivity
        self.Qlw_in = - (1-self.AL)*Ea*self.sigma*(self.Ta+273.16)**4
        #emited
        self.Qlw_out =   0.972*self.sigma*(self.Tw+273.16)**4
        
    def latent_sensible_Fink(self):
        #latent and sensible heat flux according to Fink
        print "Latent and sensible heat flux according to Fink"
        #latent
        es = saturation_vapour_pressure( self.Ta )
        ea = self.RH*es/100.
        esw  = saturation_vapour_pressure( self.Tw ) #attention: it is with water temperature
        f = 4.8 +1.98*self.Wsp + 0.28*(self.Tw-self.Ta)
        self.Qlat = f*(esw-ea)

        #sensible
        Lv = latent_heat_vap(self.Tw)
        gamma = (self.Cpa*self.P)/(0.622*Lv)
        self.Qsen = gamma*f*(self.Tw-self.Ta)

    def windstress_wuest(self):
        #calculates wind stress following Wuest 2003
        print "Wind stress following Wuest 2003"
        nt = len(self.date)
        self.tau = np.full(nt, np.nan)
        self.u10 = np.full(nt, np.nan)
        self.Cd10 = np.full(nt, np.nan)
        self.Cd10, self.u10 = self.__drag_coefficient_wuest__(self.Wsp, self.zu)
        self.tau = self.rhoa*self.Cd10*self.u10**2
        #friction velocity on air
        self.us = (self.tau/self.rhoa)**0.5
        #friction velocity on water
        self.us_water = (self.tau/self.rhoa)**0.5
        
    def latent_sensible_windstress_MO(self, verbose = 0, mit = 20):
        print "Latent and sensible heat flux following similarity theory"
        nt = len(self.date)
        #calculates the time series of turbulent heat fluxes and wind stress following similarity theory
        self.tau = np.full(nt, np.nan)
        self.us = np.full(nt, np.nan)
        self.u10 = np.full(nt, np.nan)
        self.Cd10 = np.full(nt, np.nan)
        self.Qsen = np.full(nt, np.nan)
        self.Qlat = np.full(nt, np.nan)
        self.Lw = np.full(nt, np.nan)
        self.Ch10 = np.full(nt, np.nan)
        self.Ce10 = np.full(nt, np.nan)
        for i in range(nt):
            A = self.MO_flux_calculation_1step(i, verbose)
            self.tau[i] = A[0]
            self.us[i] = A[1]
            self.u10[i] = A[2]
            self.Cd10[i] = A[3]
            self.Qsen[i] = A[4]
            self.Qlat[i] = A[5]
            self.Lw[i] = A[6]
            self.Ch10[i] = A[7]
            self.Ce10[i] = A[8]
        #calculates friction velocity on water
        self.us_water = (self.tau/self.rhow)**0.5
        
    def MO_flux_calculation_1step(self, i, verbose = 0, maxiter = 20):
        #calculates transfer coefficients and fluxes
        if verbose>0:
            print "Monin Obukhov fluxes, index: %d, date: %s"%(i,self.date[i].strftime("%Y%m%d %H:%M"))
        
        #constants
        const_gas = 287.1

        #data
        T0 = np.copy(self.Tw[i])
        Tz = np.copy(self.Ta[i])
        uz = np.copy(self.Wsp[i])
        RH = np.copy(self.RH[i])
        p = np.copy(self.P[i])
        rhoz = np.copy(self.rhoa[i])

        #avoids very low wind speeds
        if uz<0.2:
            uz = 0.2
        
        es = saturation_vapour_pressure(Tz) #saturated vapour pressure at Ts, mb
        ez = RH*es/100.#vapour pressure mb
        qz = 0.622*ez/p#specific humidiy
        esat = saturation_vapour_pressure(T0) #saturated vapour pressure at water t, mb
        q0 = 0.622*esat/p #humidity at saturation
        Lv = latent_heat_vap(T0)
        
        #air viscosity
        nua = air_kin_visco(rhoz,Tz)

        #virtual temperature
        Tv = (Tz + 273.16)*(1+0.61*qz)
        
        #starts the iteration
        #if T0 == Tz:
        #    Tz += 1e-15
        #if q0 == qz:
        #    qz += 1e-15
        iterates = True

        #my inicialisation
        Cdz = 2.e-3
        Chz = 2.e-3
        Cez = 2.e-3
        us = (Cdz*uz**2)**0.5
        Ts = -  Chz * uz * (T0 - Tz)/us
        qs = - Cez*uz *(q0-qz)/us
        tau = rhoz * us**2
        Qsen = - rhoz * self.Cpa *  us * Ts
        Qlat = - rhoz * Lv * us * qs
        
        """
        #woolway inicialisation
        us = uz*np.sqrt(0.00104+0.0015/(1+np.exp((-uz+12.5)/1.56)))
        z0u = self.alpha1*us**2/self.g + self.alpha2*nua/us
        while True:
            us = self.kappa*uz/np.log(self.zu/z0u)
            z0u_prev = np.copy(z0u)
            z0u = self.alpha1*us**2/self.g + self.alpha2*nua/us
            if np.abs(z0u-z0u_prev)/np.abs(z0u_prev)<1e-5:
                break

        Cdz = (us**2)/(uz**2)
        Re = us*z0u/nua
        xq = self.b1*Re**0.25 +self.b2
        z0tq = np.real(z0u/np.exp(xq))
        Chz = self.kappa*np.sqrt(Cdz)/(np.log(self.zu/z0tq))
        Cez = np.copy(Chz)
        Ts = -  Chz * uz * (T0 - Tz)/us
        qs = - Cez*uz *(q0-qz)/us
        tau = rhoz * us**2
        Qsen = - rhoz * self.Cpa *  us * Ts
        Qlat = - rhoz * Lv * us * qs
        """
        
        #Monin Obukhov
        Lw = (-rhoz * us**3 * Tv) / ( self.kappa*self.g*( Qsen/self.Cpa + 0.61*(Tz + 273.16)*Qlat/Lv ) )
        uz_ = np.copy(uz)
        
        j=0
        while iterates:
            j+=1

            #roughness length scales
            z0u = self.alpha1*us**2/self.g + self.alpha2*nua/us
            Re = us*z0u/nua
            xq = self.b1*Re**0.25 +self.b2
            if xq<0:
                xq = 0.
            z0tq = z0u/np.exp(xq)

            #recalculate us, Ts, qs, and Cx
            Fu = self.__sim_fun_momentum__( self.zu, z0u, Lw)
            us_1 = uz_*self.kappa*(Fu)**-1
            Cdz_1 = us_1**2/uz_**2


            Fh = self.__sim_fun_heat_moisture__(self.zt, z0tq, Lw)
            Ts_1 = (Tz-T0)*self.kappa*(Fh)**-1
            Chz_1 = -Ts*us_1/(uz_*(T0-Tz))
       
            qs_1 = (qz-q0)*self.kappa*(Fh)**-1
            Cez_1 = -qs*us_1/(uz_*(q0-qz))
           
            #displais the results
            if verbose == 2:
                print "Date: %s, Iteration %d"%(self.date[i].strftime("%Y%m%d %H:%M"), j)
                print "Cd0 = %1.3e, Cd1 = %1.3e"%(Cdz,Cdz_1)
                print "Ch0 = %1.3e, Ch1 = %1.3e"%(Chz,Chz_1)
                print "Ce0 = %1.3e, Ce1 = %1.3e"%(Cez,Cez_1)
                print "us = %1.3e, us1 = %1.3e\n"%(us,us_1)

            #checks if convergence
            if np.abs((us_1-us)/us)<1e-4 and np.abs((qs_1-qs)/qs)<1e-4 and np.abs((Ts_1-Ts)/Ts)<1e-4:
                if verbose > 0:
                    print "Converged after %d iterations"%(j)
                iterates = False
            if j>=maxiter:
                if verbose > 0:
                    print "Did not converge after %d iterations"%(maxiter)
                iterates = False

            #copies
            Cdz = np.copy(Cdz_1)
            Chz = np.copy(Chz_1)
            Cez = np.copy(Cez_1)
            us = np.copy(us_1)
            Ts = np.copy(Ts_1)
            qs = np.copy(qs_1)

            #FLuxes
            tau = rhoz * us**2
            Qsen = - rhoz * self.Cpa * us * Ts
            Qlat = -rhoz * Lv * us * qs

            Fu = self.__sim_fun_momentum__( 10., z0u, Lw)
            u10 = us/self.kappa*Fu
            Cd10 = us**2/u10**2

            Fh = self.__sim_fun_heat_moisture__(10., z0tq, Lw)
            T10 = Ts/self.kappa*Fh+T0
            Ch10 = -us*Ts/(u10*(T0-T10))
            
            q10 = qs/self.kappa*Fh+q0
            Ce10 = -us*qs/(u10*(q0-q10))

            #print "%d\n"%(j)
            #print Qsen
            #print Qlat
            #print us
            #print Cd10

            if iterates:
                #Monin Obukov
                Lw = (-rhoz * us**3 * Tv) / ( self.kappa*self.g*( Qsen/self.Cpa + 0.61*(Tz + 273.16)*Qlat/Lv ) )
                zeta = self.zu/Lw
                #for very stable
                if zeta>1:
                    uz_ = np.max([0.1,uz_])
                elif zeta<0.:
                    #avoid singularity at um = 0 for unstable conditions    
                    Th = (Tz +273.13)*(1000./p)**(const_gas/1004.67)#potential temperature
                    Thvstar = Ts*(1+0.61*qz/1000.) + 0.61*Th*qs
                    Thv = Th*(1+0.61*qz/1000.)
                    wc = (-self.g*us*Thvstar/Thv)**(1./3.)
                    uz_ = (uz_**2 + wc**2)**0.5


        
        return tau, us, u10, Cd10, Qsen, Qlat, Lw, Ch10, Ce10


        
    def __phim__(self,eta):
        chi = (1.-16.*eta)**0.25
        phi = 2*np.log( (1.+chi)/2. ) + np.log( (1.+chi**2)/2. ) - 2*np.arctan(chi) + np.pi/2
        return phi

    def __phih__(self,eta):
        chi = (1.-16.*eta)**0.25
        phi = 2*np.log( (1.+chi**2)/2. )
        return phi
        
    def __sim_fun_momentum__(self, z, z0, Lw):
        #Integrals of similarity functions for momentum
        #sets limits to eta
        eta = z/Lw
        if eta<-self.eta_thres:
            eta  = -self.eta_thres
        elif eta>self.eta_thres:
            eta = self.eta_thres

        if eta>1:
            F =  ( np.log(Lw/z0) + 5. ) + ( 5.*np.log(eta) + (eta) - 1.  )
        elif eta>= 0:
            F = np.log(z/z0) + 5.*(eta)
        elif eta>=self.etam:
            F = np.log(z/z0) - self.__phim__(eta)
        else:
            F = ( np.log( self.etam*Lw/z0 ) - self.__phim__(self.etam) ) + \
                1.14*( (-eta)**(1./3.) - (-self.etam)**(1./3.) )

        return F
                
    def __sim_fun_heat_moisture__(self,  z, z0, Lw):
        #Integrals of similarity functions for heat and moisture
        #sets limits to eta
        eta = z/Lw
        if eta<-self.eta_thres:
            eta  = -self.eta_thres
        elif eta>self.eta_thres:
            eta = self.eta_thres
            
        if eta>1:
            F =  ( np.log(Lw/z0) + 5. ) + ( 5.*np.log(eta) + (eta) - 1.  )
        elif eta>= 0:
            F =  np.log(z/z0) + 5.*(eta)
        elif eta>=self.etah:
            F =  np.log(z/z0) - self.__phih__(eta)
        else:
            F =  ( np.log( (self.etah*Lw/z0) ) - self.__phih__(self.etah) ) + \
                0.8*(  (-self.etah)**(-1/3.) - (-eta)**(-1./3.)  )
            
        return F

    def __drag_coefficient_wuest__(self,U, z = 10.):
        #Calculates wind drag according to Wuest for wind at arbitrary z,
        #also calculates wind at 10 m
        print "Drag coefficient calculation"
        k = 0.41
        K = 11.3
        g = 9.81
        a=np.log(z/10.)/k
        Cd = np.full(U.size, 0.)
        U10 = np.full(U.size, 0.)
        for i in range(U.size):
            if U[i]==0:
                continue
            #gets value for low winds
            u = np.copy(U[i])
            if u<0.2:
                U10[i] = np.copy(U[i])
                Cd[i] = 0.
            elif u<=3:
                u10_1 = np.copy(U[i])
                flag = True
                while flag:
                    u10_0 = np.copy(u10_1)
                    Cd1 = 0.0044*u10_0**(-1.15)
                    #print CdS
                    u10_1=u/(1+a*np.sqrt(Cd1))
                    if np.abs(u10_0-u10_1)<1e-4:
                        flag = False
            else:        
                #for high winds
                flag = True
                u = np.copy(U[i])
                Cd1 = 0.001
                u10_1 = u/(1+a*np.sqrt(Cd1))
                while flag:
                    Cd0 = np.copy(Cd1)
                    u10_0 = np.copy(u10_1) 
                    Cd1 = (k**(-1)* np.log( g*10/Cd0/u10_0**2) + K)**(-2)
                    u10_1 = u/(1+a*np.sqrt(Cd1))
                    if np.abs(Cd0-Cd1)<1e-6 and np.abs(u10_0-u10_1)<1e-4:
                        flag = False
                    
            Cd[i] = Cd1
            U10[i] = u/(1+a*np.sqrt(Cd[i]))
        
        if z == 10:    
            U10 = np.copy(U)
        
        return Cd, U10
    
