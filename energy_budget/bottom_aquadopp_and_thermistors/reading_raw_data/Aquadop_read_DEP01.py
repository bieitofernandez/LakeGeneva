import sys
sys.path.insert(0, "../")
from Aquadopp_processing_functions import *
from scipy import signal                
########################################
######################################
#folder = "/home/bieito/Documents/SCIENCE/EPFL/Leman_data/Aquadopp/Dep_20190816/data/"
folder = "/media/bieito/29A638795E5A52D3/EPFL_DATA/Aquadopp/Dep_20190816/data"
##AD.gets_interval(min_pres = 9)???


## READS ALL THE DATA
AD = aquadopp_deployment(folder,min_pres = 80.)
AD.gets_mean_profiles()


#AD = aquadopp_deployment(folder,pi_file = "Aquadopp_DEP0102.pi")
#AD.folder = folder #if change folder

#AD.finds_bottom_earth_coords()
AD.rotates_into_principal_coordinates()

AD.plot_mean_timeseries()
#LOW fit
AD.fits_LOW_timeseries(save = True, startz = 4)
AD.drag_coefficient(corrected = False, SF = False)

AD.SF_timeseries(PLOT = False)
#AD.SF_mean_timeseries(PLOT = False)

### median z0:
z00 = np.nanmedian(AD.mDATA["LOW"]["z0"])
#LOW fit corrected
AD.fits_LOW_timeseries(z00 = z00,startz = 4, corrected = True, save = True)
AD.drag_coefficient(corrected = True, SF = False)



z00 = np.nanmedian(AD.mDATA["LOW"]["z0"])
AD.drag_coefficient(corrected = True, SF = True)
AD.drag_coefficient(corrected = False, SF = True)

AD.SF_vs_LOW()
AD.SF_vs_vref()

BN = 212
AD.fits_LOW_1prof(BN,z0 = z00, startz = 4, PLOT = True, corrected = True)
AD.fits_LOW_1prof(BN,startz = 4, PLOT = True, corrected = False)
epsilon = AD.SF_one_burst(BN, Jabbari = True, PLOT = True, MEAN = False)[0]


"""
mu = np.nanmean(AD.mDATA["u"],axis = 1)
mv = np.nanmean(AD.mDATA["v"],axis = 1)
mw = np.nanmean(AD.mDATA["w"],axis = 1)

muu = np.c_[mu,mv,mw].T
#muu = np.c_[AD.mDATA["u"].ravel(),AD.mDATA["v"].ravel(),AD.mDATA["w"].ravel()].T
ii = np.isfinite(np.sum(muu, axis =0))
CoVar = np.cov(muu[:,ii])

LB, V = np.linalg.eig(CoVar)
inV = np.linalg.inv(V)
muuR = np.dot(inV,muu)
#http://planning.cs.uiuc.edu/node101.html
#yau
alpha = np.arctan2(inV[1,0],inV[0,0])*180/np.pi 
#pitch
beta = np.arctan2(-inV[2,1],np.sqrt( inV[2,1]**2 + inV[2,2]**2 ))*180/np.pi
#roll
gamma = np.arctan2(inV[2,1],inV[2,2])*180/np.pi

print alpha
print beta
print gamma
"""
