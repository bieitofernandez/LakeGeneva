import sys
sys.path.insert(0, "../")
from Aquadopp_processing_functions import *
from scipy import signal                
########################################
######################################
#folder = "/home/bieito/Documents/SCIENCE/EPFL/Leman_data/Aquadopp/Dep_20190909/data_converted/DEP204/"
folder = "/media/bieito/29A638795E5A52D3/EPFL_DATA/Aquadopp/Dep_20190909/data_converted/DEP203/"
##AD.gets_interval(min_pres = 9)???


## READS ALL THE DATA
AD = aquadopp_deployment(folder,min_pres = 80.)

AD.gets_mean_profiles()
AD.finds_bottom_earth_coords()
AD.rotates_into_principal_coordinates()

#AD = aquadopp_deployment(folder,pi_file = "Aquadopp_DEP0203.pi")

AD.plot_mean_timeseries()
#LOW fit
AD.fits_LOW_timeseries(save = True, startz = 4)
AD.drag_coefficient(corrected = False, SF = False)

#structure functions (they need a drag coefficient estimate for Jabbari correction)
AD.SF_timeseries(PLOT = False) #depth resolved

### median z0:
z00 = np.nanmedian(AD.mDATA["LOW"]["z0"])
#LOW fit corrected
AD.fits_LOW_timeseries(z00 = z00,corrected = True, save = True, startz = 4)
AD.drag_coefficient(corrected = True, SF = False)
#AD.SF_mean_timeseries(PLOT = False)


#AD = aquadopp_deployment(folder,pi_file = "Aquadopp_DEP0203.pi")
#AD.folder = folder
#AD.Cd = 0.007688
AD.SF_timeseries(PLOT = False, Jabbari = True) #depth resolved

AD.drag_coefficient(corrected = True, SF = True)
AD.drag_coefficient(corrected = False, SF = True)

AD.SF_vs_LOW()
AD.SF_vs_vref()

BN = 460
z00 = np.nanmedian(AD.mDATA["LOW"]["z0"])
AD.plot_mean_profiles(BN)
AD.fits_LOW_1prof(BN,startz = 4,z0 = z00,PLOT = True, corrected = True)
AD.fits_LOW_1prof(BN,startz = 4,z0 = z00,PLOT = True, corrected = False)
epsilon = AD.SF_one_burst(BN, Jabbari = True, PLOT = True, MEAN = False)[0]

#AD.inertial_one_burst(145)


