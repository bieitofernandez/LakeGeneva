#Extraction and processing of Sea and Sun CTD data
#Bieito Fernandez Castro April 2020
import numpy as np
import sys
import matplotlib.pyplot as plt
import datetime as dt
import salinity_density as SD
from glob import glob
from time_conversion import *
import os
from time import sleep
import seawater as sw


def first_centered_differences(x, y, fill = False):
    if x.size != y.size:
        print "first-centered differences: vectors do not have the same size"
    dy = np.full( x.size, np.nan )
    iif = np.where( (np.isfinite(x)) & (np.isfinite(y))) [0]
    if iif.size == 0:
        return dy
    x0 = x[iif]
    y0 = y[iif]
    dy0 = np.full( x0.size, np.nan )
    #calculates differences
    dy0[0] = (y0[1] - y0[0])/(x0[1]-x0[0])
    dy0[-1] = (y0[-1] - y0[-2])/(x0[-1]-x0[-2])
    dy0[1:-1] = (y0[2:] - y0[0:-2])/(x0[2:]- x0[0:-2])
    
    dy[iif] = dy0

    if fill:
        dy[0:iif[0]] = dy[iif[0]]
        dy[iif[-1]+1:] = dy[iif[-1]]
    return dy


def extract_SandS_file(filename,lake="Leman", Nstart=60, deltaP=0.5, minC = 0.1, minV = 0.05, lat = 46.2, altitude = 372.,PLOT = False):
    #This function extracts data from one file, cuts the profile and calculates derived variables (salinity, density, potential temperature, oxygen saturation)
    #Nstart, deltaP, minC (minimun conductiviy), minV (minimum velocity) are paramaters used to cut the profiles.
    #lake, altitude, lat (latitude) are lake dependent parameters used for several calculations
    if lake == "Leman":
        lat = 46.2
        altitude = 372.

    #Reads the data
    cnvf = open(filename)
    print filename
    variables = []
    vardescription = []
    record = False
    count0 = 0
    for line in cnvf.readlines():
        if count0 == 1:
            lineS = line.split()
            OUT = dict()
            variables = [x0.rstrip() for x0 in lineS[1:]]
            for var0 in variables:
                OUT[var0] = []
            print variables
        if count0 == 4:
            record = True

        if line[0]==";":
            count0+=1

        if record:
            lineS = line.split()
            for (value,var0) in zip(lineS,variables):
                if var0 == "Datasets":
                    OUT[var0].append(float(value))
                elif var0 == "IntD" or var0 == "IntT":
                    if len(lineS)>len(variables) and var0 == "IntT":
                        OUT[var0].append(value +" "+ lineS[-1])
                    else:
                        OUT[var0].append(value)
                else:
                    OUT[var0].append(float(value))
            
    for var0 in variables:
        if type(OUT[var0][0]) == float:
            OUT[var0] = np.array(OUT[var0])
    OUT["time"] = []

    #Deals with different formats for the date (for lake Geneva this is a mess, but this fixes it for all the possible cases)
    D0 = OUT["IntD"][0] +" "+ OUT["IntT"][0]
    print D0
    if "." in D0:
        dateformat = "%d.%m.%Y %H:%M:%S"
    elif "/" in D0:
        if "PM" in D0 or "AM" in D0:
            dateformat = "%m/%d/%Y %I:%M:%S %p"
        else:
            dateformat = "%d/%m/%Y %H:%M:%S"

    #converts data format
    for (D,T) in zip(OUT["IntD"],OUT["IntT"]):
        OUT["time"].append(dt.datetime.strptime(D+" "+T,dateformat))

    #profile duration and sampling interval
    duration = (OUT["time"][-1] - OUT["time"][0]).total_seconds()
    samp_int = duration/float(len(OUT["time"]))

    #low-passes the pressure signal to calculate falling speed
    sPress = np.full(OUT["Press"].size,0.)
    Nc = int(1/samp_int)
    if Nc%2 == 0:
        Nc+=1
    conv = np.ones(Nc)/float(Nc)
    sPress = np.convolve(OUT["Press"],conv, mode = "same")

    #gets falling speed
    OUT["w"] = np.zeros(sPress.size)
    OUT["w"][1:-1] = (sPress[2:]-sPress[:-2])/(2*samp_int)
    OUT["w"][0] = OUT["w"][1]
    OUT["w"][-1] = OUT["w"][-2]

    #calculates the conductivity gradient to detect when the instrument is on the water
    diffC = first_centered_differences(np.arange(OUT["Cond"].size), OUT["Cond"])
    i0w = np.where(diffC>np.percentile(diffC,90))[0][0]
    if i0w>0:
        minC = max([np.nanmax(OUT["Cond"][0:i0w]),minC])

    #looks which values correspond to water, looking at the values of the conductivity and its gradient
    
    #iwater = np.where( (OUT["Cond"]>minC) & (diffC<0.005*np.max(diffC)))[0]
    #or:
    iwater = np.where( (OUT["Cond"]>minC) & (diffC<np.percentile(diffC,90)))[0]

    #Exists if the instrument was never on the water
    if len(iwater) == 0:
        iread = np.array([])
        for var0 in variables:
            if type(OUT[var0]) == np.ndarray:
                OUT[var0] = iread
            else:
                OUT[var0] = [OUT[var0][i] for i in iread]
        OUT["filename"] = filename
        print "No profile detected"
        return OUT

    #corrects for atmospheric pressure
    minPr = np.min(OUT["Press"])#np.percentile(OUT["Press"],0.1)#np.percentile(OUT["Press"][ 0:iwater[0]],10)
    print "Pressure correction %1.3f "%(-minPr)
    OUT["Press"] -= minPr

    
    ###Looks for the data corresponding to the downward profile
    posV = -np.ones(OUT["w"].shape)
    posV[OUT["w"]>0] = 1
    
    posP = -np.ones(OUT["Press"].shape)
    posP[OUT["Press"]>0] = 1

    posC = -np.ones(OUT["Cond"].shape)
    posC[OUT["Cond"]>minC] = 1

    maxP = np.nanmax(OUT["Press"])
    
    #cuts at the end and removes three seconds because sometimes there are peaks...
    iend = np.where( (OUT["Press"]>(maxP-deltaP)) & (OUT["w"]<=minV) )[0][0]-int(3/samp_int)

    #Looks for the start of the profile. I don't remember exactly how I did this. This is the result of many adjustments to different particular cases
    #It doesn't mean it will work always
    istart = -9999
    for i in range(iwater[0],iwater[-1]-Nstart):
        jj = np.arange(i,i+Nstart)
        jj2 = np.arange(i+Nstart, iend)
        if np.sum(posV[jj])==jj.size and np.sum(posP[jj])==jj.size and np.sum(posC[jj])==jj.size and np.sum(posV[jj2])>=0.5*jj2.size:
            istart = i 
            break
    #Exists if no profile detected
    if istart < 0:
        iread = np.array([])
        for var0 in variables:
            if type(OUT[var0]) == np.ndarray:
                OUT[var0] = OUT[var0][iread]
            else:
                OUT[var0] = [OUT[var0][i] for i in iread]
        OUT["filename"] = filename
        print "No profile detected"
        return OUT
    
    #plots the profile detection parameters
    if PLOT:
        fig, ax = plt.subplots(3,1)
        ax[0].plot(OUT["w"])
        ax[0].axvline(istart, color = "r")
        ax[0].axvline(iend, color = "r")
        ax[1].plot(OUT["Press"])
        ax[1].axvline(istart, color = "r")
        ax[1].axvline(iend, color = "r")
        ax[2].plot(OUT["Cond"])
        ax[2].axvline(istart, color = "r")
        ax[2].axvline(iend, color = "r")
        ax[0].set_title(filename)
        plt.show(block = True)

    #Gets the data for the profile and removes the rest
    iread = np.arange(istart,iend+1)
    for var0 in variables:
        if type(OUT[var0]) == np.ndarray:
            OUT[var0] = OUT[var0][iread]
        else:
            OUT[var0] = [OUT[var0][i] for i in iread]

    #calculates the derived variables
    OUT["S"], OUT["den"], OUT["depth"] = SD.salinity_density(OUT["Temp"],OUT["Cond"]*1000,OUT["Press"],lake, lat)    #salinity
    OUT["filename"] = filename

    OUT["PT"] = SD.potential_temperature(OUT["Temp"], OUT["S"],OUT["Press"], OUT["depth"], lat) #potential temperature
    OUT["pden"] = SD.density(OUT["PT"], OUT["S"],lake) #potential density
    
    isort = np.argsort( OUT["pden"])
    OUT["pden_sorted"] = OUT["pden"][isort] #sorted potential density for stability calculations

    #oxygen saturation
    OUT.pop("sat") #Deletes the instrument calculation of DO sat (this is wrong for non sea level)
    OUT["DOsat"] = SD.oxygen_saturation(OUT["PT"], OUT["S"], altitude, lat)
    OUT["pctDOsat"] = OUT["DO_mg"]/OUT["DOsat"]*100

    #thorpe displacements
    isortT = np.argsort(OUT["PT"])[::-1]
    OUT["thorpe_disp"] = -(OUT["depth"]-OUT["depth"][isortT])

    OUT["lat"] = lat
    OUT["lake"] = lake
    OUT["altitude"] = altitude

    #plots some results to check
    if PLOT:
        fig, ax = plt.subplots(1,3, sharey = True)
        ax[0].plot( OUT["Temp"], -OUT["depth"])
        ax[1].plot( OUT["Cond"], -OUT["depth"])
        #ax[2].plot( OUT["sbeox0PS"], -OUT["depFM"])
        ax[2].plot( OUT["Chl_A"], -OUT["depth"])
        ax[1].set_title(filename)
        plt.show(block = False)
        
    print "File %s extracted"%(filename)


    #how to rename dict key?
    #mydict[new_key] = mydict.pop(old_key)
    
    return OUT

def extract_all_files_folder(folder,minC = 0.1,  minV=0.05,recursive = False, PLOT = False):
    #This function extracts all files in a given folder
    files = [y for x in os.walk(folder) for y in glob(os.path.join(x[0], '*.TOB'))]
    files = sorted(files)
    CTD = []
    print files
    for fl in files:
        CTD0=extract_SandS_file(fl,  minC = minC, minV = minV, PLOT = PLOT)
        if len(CTD0["time"])>0:
            CTD.append(CTD0)

    return CTD    


def SandS_regular_matrix(CTD,dD=0.5, dDT = 5., alpha = 0.2):
    #This functions puts all the profiles in a regular grid with vertical spacing of dD = 0.5m.
    #For Turbulent quantities it smoothes the values with a window of dDT = 5. m
    #alpha is a constant for diffusivity calculations (related to mixing efficiency)
    CTDv = dict()
    datesCTD = [CTD0["time"][0] for CTD0 in CTD]
    datenumCTD = convert_date_to_time(datesCTD)+366
    isort = np.argsort(datenumCTD)
    CTDv["time"] = datenumCTD[isort]
    CTDv["date"] = [datesCTD[i] for i in isort]
    
    maxDepth = int(max([np.nanmax(CTD0["depth"]) for CTD0 in CTD]))
    
    CTDv["depth"] = np.arange(0,maxDepth+dD,dD)
    d1 = CTDv["depth"].size
    d2 = CTDv["time"].size
    variables = ['den', 'pden' ,'pden_sorted', 'Temp','PT', 'S', 'DO_mg', 'Chl_A', 'Cond', 'w', 'pH', 'Turb', 'DOsat','pctDOsat']
    for var0 in variables:
        CTDv[var0] = np.full( (d1,d2), np.nan  )
    CTDv["LT"] = np.full( (d1,d2), np.nan  )
    CTDv["N2"] = np.full( (d1,d2), np.nan  )
    CTDv["pden_smooth"] = np.full( (d1,d2), np.nan  )
    CTDv["N2_smooth"] = np.full( (d1,d2), np.nan  )

    for i in range(d2):
        isr = isort[i]
        for j in range(d1):
            jj = np.where( (CTD[isr]["depth"]>=CTDv["depth"][j]-dD/2) & (CTD[isr]["depth"]<=CTDv["depth"][j]+dD/2))[0]
            if jj.size == 0:
                continue
            for var0 in variables:
                CTDv[var0][j,i] = np.nanmean(CTD[isr][var0][jj])
            jjT = np.where( (CTD[isr]["depth"]>=CTDv["depth"][j]-dDT/2) & (CTD[isr]["depth"]<=CTDv["depth"][j]+dDT/2))[0]
            CTDv["LT"][j,i] = np.sqrt( np.nanmean( CTD[isr]["thorpe_disp"][jjT]**2 ))
            CTDv["pden_smooth"][j,i] = np.nanmean(CTD[isr]["pden_sorted"][jjT])
        CTDv["N2"][:,i] = - sw.g(CTD[isr]["lat"])/CTDv["den"][:,i]*first_centered_differences(-CTDv["depth"], CTDv["pden_sorted"][:,i])
        CTDv["N2_smooth"][:,i] = - sw.g(CTD[isr]["lat"])/CTDv["pden_smooth"][:,i]*first_centered_differences(-CTDv["depth"], CTDv["pden_smooth"][:,i])

    CTDv["KT"] = alpha*0.5*CTDv["N2_smooth"]**0.5*CTDv["LT"]**2+1.4e-7
    
    
    return CTDv



def plots_profile_from_regular_matrix(CTD, index, alpha = 0.2, shows = False):
    #Plots the (previously gridded) profile
    print "Plotting profile %d, %s"%(index+1, CTD["date"][index].strftime("%Y %b %d %H:%M"))

    maxdepth = np.max(CTD["depth"])
    fig, ax = plt.subplots(2,4, sharey = True, figsize = (12,8))
    l1,=ax[0,0].plot(CTD["Temp"][:,index],CTD["depth"], label = "T")
    l2,=ax[0,0].plot(CTD["PT"][:,index],CTD["depth"],lw = 0.5, label = "$\\theta$")
    ax[0,0].legend()
    ax[0,0].set_ylim((maxdepth,0))
    ax[0,0].set_ylabel("Depth [m]")
    ax[0,0].set_xlabel("T [$^{\circ}$C]")
    ax[0,0].grid(True)
    
    ax[0,1].plot(CTD["S"][:,index],CTD["depth"])
    ax[0,1].set_ylim((maxdepth,0))
    ax[0,1].set_xlabel("Sal. [g/kg]")
    ax[0,1].grid(True)
    
    ax[0,2].plot(CTD["DO_mg"][:,index],CTD["depth"])
    #ax[0,2].plot(CTD["DOsat"][:,index],CTD["depth"])
    ax[0,2].set_ylim((maxdepth,0))
    ax[0,2].set_xlabel("DO [mg/l]")
    ax[0,2].grid(True)
    
    ax02 = ax[0,2].twiny()
    ax02.plot(CTD["pctDOsat"][:,index],CTD["depth"],'-', color = l2.get_color(), alpha = 0.5)
    #ax02.plot(CTD["DO_mg"][:,index]/CTD["DOsat"][:,index]*100,CTD["depth"],'--',alpha = 0.5)
    ax02.set_ylim((maxdepth,0))
    ax02.set_xlabel("O2 sat [%] (orange)")
    
    ax[0,3].plot(CTD["pH"][:,index],CTD["depth"])
    ax[0,3].set_ylim((maxdepth,0))
    ax[0,3].grid(True)
    ax[0,3].set_xlabel("pH")
    
    ax[1,0].plot(CTD["Chl_A"][:,index],CTD["depth"])
    ax[1,0].set_ylim((maxdepth,0))
    ax[1,0].grid(True)
    ax[1,0].set_xlabel("Chlorophyll-a [ug/l]")
    ax[1,0].set_ylabel("Depth [m]")
    
    ax[1,1].plot(CTD["Turb"][:,index],CTD["depth"])
    ax[1,1].set_ylim((maxdepth,0))
    ax[1,1].set_xlabel("Turbidity [NTU]")
    ax[1,1].grid(True)

    ax[1,2].semilogx( CTD["LT"][:,index], CTD["depth"])
    ax[1,2].set_ylim((maxdepth,0))
    ax[1,2].set_xlabel("log10 $L_T$ [m]")
    ax[1,2].set_xlim([1e-3,100])
    ax[1,2].grid(True)

    ax[1,3].semilogx( CTD["KT"][:,index], CTD["depth"])
    ax[1,3].set_ylim((maxdepth,0))
    ax[1,3].set_xlabel("log10 $K_T$ [m$^2$ s$^{-1}$]")
    ax[1,3].set_xlim([1e-7,1e-1])
    ax[1,3].grid(True)
    
    fig.savefig("CTD_profile_%s.png"%(CTD["date"][index].strftime("%Y%m%d_%H%M")), dpi = 300, bbox_inches = "tight")
    if shows:
        plt.show()
    plt.close(fig)


