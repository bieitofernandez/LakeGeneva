#Water properties calculations
#Bieito Fernandez Castro April 2020
import numpy as np
import seawater as sw
#salinity functions for Lake Geneva, and Zurich but can be extended

def salinity_density(T,C,p,lake = "Leman", lat = 46.2):
    #calculates both salinity, density and depth from T, C, p
    S, C20 =  salinity(T, C, lake)
    rho, depth = density_depth(T,S,p,lake,lat)
    return S, rho, depth

def density_depth(T,S,p,lake, lat = 46.2):
    if lake == "Zurich":
        beta = 0.807e-3
    elif lake == "Leman":
        beta = 0.807e-3
    rho = density(T,S,lake)
    mrho = np.cumsum(rho)/np.arange(1,rho.size+1)
    depth = 1e4*p/mrho/sw.g(lat)
    return rho, depth

def density(T,S,lake):
    if lake == "Zurich":
        beta = 0.807e-3
    elif lake == "Leman":
        beta = 0.807e-3
    #same as Millero up to ~30degC
    rho0 = 999.84298 + 1e-3*(65.4891*T - 8.56272*T**2 + 0.059385*T**3)
    rho = rho0*(1+beta*S)
    return rho


def salinity(T, C, lake = "Leman"):
    if lake == "Zurich":
        fT = 1.8626 - 0.052908*T + 0.00093057*T**2 - 6.78e-6*T**3
        C20 = fT*C
    elif lake == "Leman":
        fT = 1.684 - 0.04645*T + 0.000602*T**2
        C20 = fT*C
    S = salinity_from_C20(C20, lake)

    return S, C20
    
def salinity_from_C20(C20, lake = "Leman"):
    if lake == "Zurich":
        S = 0.7999e-3*C20
    elif lake == "Leman":
        S = 0.874e-3*C20
    return S

def Gamma_adiabatic(T,S,p, lat = 46.):
    alpha = sw.alpha(S, T, p)
    cp = sw.cp(S, T, p)
    Gamma = sw.g(lat)*alpha*(T-273.15)/cp
    return Gamma


def potential_temperature(T, S, p, z,  lat = 46.2):
    iif = np.where(np.isfinite(T))
    PT = np.full(T.size, np.nan)
    T = T[iif]
    p = p[iif]
    z = z[iif]
    S = S[iif]
    pt0 = np.copy(T)
    n = pt0.size
    pt1 = np.full(n, np.nan)

    iterate = True
    j=0
    while iterate:
        intGamma = np.zeros(n)
        for i in range(1,n):
            Gamma0 = Gamma_adiabatic(pt0, S[i],p, lat)
            intGamma[i] = np.trapz(Gamma0[0:i+1], x=z[0:i+1])
        pt1 = T + intGamma
        j+=1
        print j
        if j>100 or np.max(np.abs(pt1-pt0))<1e-4:
            iterate = False
        else:
            pt0 = np.copy(pt1)


    PT[iif] = pt1
    return PT
        

def oxygen_saturation(T, S,  altitude = 372., lat = 46.2, units = "mgl"):
    #calculates oxygen saturation in mg/l according to Garcia-Benson
    #to be coherent with Hannah
    if units != "mgl" and units != "mll":
        units = "mgl"
    mgL_mlL = 1.42905
    mmHg_mb = 0.750061683
    mmHg_inHg = 25.3970886
    standard_pressure_sea_level = 29.92126
    standard_temperature_sea_level = 15 + 273.15
    gravitational_acceleration = gr = sw.g(lat)
    air_molar_mass = 0.0289644
    universal_gas_constant = 8.31447
    baro = (1./mmHg_mb) * mmHg_inHg * standard_pressure_sea_level * np.exp((-gravitational_acceleration * air_molar_mass * altitude)/(universal_gas_constant * standard_temperature_sea_level))
    u = 10**(8.10765 - 1750.286/(235 + T))
    press_corr = (baro * mmHg_mb - u)/(760 - u)

    Ts = np.log((298.15 - T)/(273.15 + T))
    lnC = 2.00907 + 3.22014 * Ts + 4.0501 * Ts**2 + 4.94457 *  Ts**3 + -0.256847 * Ts**4 + 3.88767 * Ts**5 - S * (0.00624523 + 0.00737614 * Ts + 0.010341 * Ts**2 + 0.00817083 * Ts**3) - 4.88682e-07 * S**2
    O2sat = np.exp(lnC)
    if units == "mll":
        O2sat = O2sat*press_corr
    elif units == "mgl":
        O2sat = O2sat*mgL_mlL*press_corr

        
    return O2sat
    
        
    
