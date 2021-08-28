import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import glob
from scipy import io
import pickle
import copy
import scipy.interpolate as intrp
import scipy.stats as stats
import statsmodels.api as sm
from scipy import signal

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

def convert_time_to_date(time):
     date = [dt.datetime.fromordinal(int(time0)) + dt.timedelta(time0%1) for time0 in  time] 
     return date
def convert_date_to_time(date):
     N = len(date)
     time = np.full(N, np.nan)
     for i in range(N):
          time[i]=date[i].toordinal() + date[i].hour/24. + date[i].minute/24./60. + date[i].second/24./60./60. + date[i].microsecond/24./60./60./1e6
     return time

##rotation functions
def heading_matrix(head):
     hh = np.pi*(head-90.)/180.
     H = np.array([[np.cos(hh),np.sin(hh),0], [-np.sin(hh), np.cos(hh), 0],[0,0,1]])
     return H

def tilt_matrix(pitch, roll):
     pp = np.pi*pitch/180.
     rr = np.pi*roll/180.
     P = np.array([[np.cos(pp),-np.sin(pp)*np.sin(rr),-np.cos(rr)*np.sin(pp)],\
                   [0, np.cos(rr), -np.sin(rr)],\
                   [np.sin(pp), np.sin(rr)*np.cos(pp), np.cos(pp)*np.cos(rr)]])
     return P

def transform_vector(TM, x,y,z):
     v = np.array([[x,y,z]]).T
     vt = np.dot(TM,v)
     return vt[0,0], vt[1,0], vt[2,0]

def transform_v_profile(TM, v1,v2,v3):
     Nd = v1.size
     v1t = np.full(Nd, np.nan)
     v2t = np.full(Nd, np.nan)
     v3t = np.full(Nd, np.nan)
     for i in range(Nd):
          v1t[i], v2t[i], v3t[i] = transform_vector(TM, v1[i],v2[i],v3[i])
           
     return v1t, v2t, v3t

#finds bottom
def find_a_maximum_top(a, factor = 99.):
     """
     b = np.copy(a)
     b = b[0:np.argmax(b)+1] #this creates trouble 03 April 2020
     b = b-np.min(b)
     maxb = np.max(b)
     ib = np.where(b < maxb/factor)[0][-1]
     """
     #new version of 03 April 2020 to avoid trouble
     Da = np.diff(a)
     Damax = np.percentile(Da,factor)
     #plt.plot(Da)
     #plt.axhline(Damax)
     #plt.show()
     i = np.where( Da>Damax)[0][0]-1

     return i    

########################
### Structure function
#######################


##############################
#law-of-the-wall calculations#
###############################
class LOW_model():
     def __init__(self, params):
          self.kappa = 0.41
          self.us = params["us"]
          self.z0 = params["z0"]
          if not "L" in params.keys():
               self.corr = False
          else:
               self.corr = True
               self.L = params["L"]

     def eval_u(self, z):
          #evaluates the vertical profile of u
          u = self.us/self.kappa*np.log(z/self.z0)
          if self.corr:
               u+= self.us/self.kappa*z/self.L
          return u
     
     def eval_mepsilon(self,z):
          mepsilon = self.us**3/self.kappa*np.log( np.max(z)/self.z0)
          mepsilon /= (np.max(z)-self.z0)
          return mepsilon


##########################
class aquadopp_deployment():
    def __init__(self,folder, save = False, **kargs):
        if "minpres" in kargs:
             #minimum mean preassure to do calculations
             self.min_pres = kargs["min_pres"]
        else:
             self.min_pres = 0.
        if ("pi_file" not in kargs):
             self.folder = folder
             self.time_vars = ["time","head","pitch","roll","pres", "temp","v1","v2","v3",\
                          "a1","a2","a3","c1","c2","c3"]
             self.reads_deployment_info(kargs)
        else:
             pi_filename = kargs["pi_file"]
             data = pickle.load(open(pi_filename))
             for var in data.keys():
                  exec("self.%s = data[\"%s\"]"%(var,var))
             

    def save_pickle(self, filename = ""):
        out = dict()
        vars_to_save = self.__dict__.keys()
        for var0 in vars_to_save:
             exec("out[\""+var0+"\"] = self."+var0)

        if len(filename)==0:
             filename = "Aquadopp_%s.pi"%(self.deployname)
        print "Saving dataset in %s"%(filename)
        pickle.dump(out,open(filename,"w+"))

        
    def reads_deployment_info(self, args):
        #reads the basic deployment info 
        hdrfile = glob.glob(self.folder+"/*.hdr")[0]
        
        DPL = dict()
        read_v = dict()
        read_v["meas_int"] = "Measurement/Burst interval"
        read_v["cell_size"] = "Cell size"
        read_v["N_cells"]  = "Number of cells"
        read_v["blanking_distance"] = "Blanking distance"
        read_v["N_samples"] = "Samples per burst"
        read_v["fs"] = "Sampling rate"
        read_v["coords"] = "Coordinate system"
        read_v["orientation"] = "Orientation"
        read_v["Tmatrix"] = "Transformation matrix"
        
        
        #reads info from hdr file
        LF = open(hdrfile)
        i1 = hdrfile.rfind("DEP")
        i2 = hdrfile.rfind(".")
        print hdrfile[i1:i2]
        self.deployname = hdrfile[i1:i2]
        print "Reading hdr for deployment %s"%(self.deployname)
        reads_tmatrix = False
        DPL["Tmatrix"] = []
        countTM = 0
        for line in LF.readlines():
            lineS = line.split()
            for var in read_v.keys():
                if read_v[var] in line:
                
                    if var == "Tmatrix":
                         reads_tmatrix = True
                         
                    elif var == "orientation" or var == "coords":
                         #text vars
                         DPL[var] = lineS[-1]
                    elif var == "N_cells" or var== "N_samples":    
                        DPL[var] = float(lineS[-1])
                    else:
                        if "N/A" in lineS[-2]:
                            DPL[var] = np.nan
                        else:
                            DPL[var] = float(lineS[-2])
                    
            if countTM == 3:
                 reads_tmatrix = False
            if reads_tmatrix:
                 Tm0 = [float(char0) for char0 in lineS[-3:]]
                 DPL["Tmatrix"].append(Tm0)
                 countTM+=1
        DPL["blanking_distance"] /= np.cos(25.*np.pi/180) #converts the cell size to along beam           
        DPL["Dr"] = DPL["cell_size"]/1000./np.cos(25.*np.pi/180) #converts the cell size to along beam  
        DPL["r"] = (1+np.arange(DPL["N_cells"]))*DPL["Dr"] + DPL["blanking_distance"]
        DPL["Tmatrix"] = np.array(DPL["Tmatrix"])
        if DPL["orientation"] == "DOWNLOOKING":
             DPL["blanking_distance"] = -DPL["blanking_distance"]
             DPL["Tmatrix"][1:,:] = -DPL["Tmatrix"][1:,:] 

        ####
        
        #reads times
        print self.folder+"/"+self.deployname+"*.sen"
        sensfnames = sorted(glob.glob(self.folder+"/"+self.deployname+"*.sen"))
        DPL["Nfiles"] = len(sensfnames)
        DPL["dates"] = []
        DPL["time"] = []
        DPL["filenames"] = []
        DPL["burst_number"] = []
       
        #read_status = True  
        for i in range(DPL["Nfiles"]):
            dates = []
            head = []
            #open files
            fsen = open(sensfnames[i])
            i1 = sensfnames[i].rfind("/")+1
            i2 = sensfnames[i].rfind(".")
            DPL["filenames"].append(sensfnames[i][i1:i2])
            #i3 = sensfnames[i].rfind("DEP")+5
            DPL["burst_number"].append( int(sensfnames[i][i2-3:i2])  )
            #reads
            print "Reads File: %s"%(sensfnames[i])
            #print fsen.readlines() #line by line
            for line_s in fsen:
                lineS = line_s.split()
                month = int(lineS[0])
                day = int(lineS[1])
                year = int(lineS[2])
                hour = int(lineS[3])
                minute = int(lineS[4])
                second = float(lineS[5])
                #if read_status == True:
                #    status = int(lineS[9][0])
                #    print "Status bit %d"%(status)
                #    read_status = False
                date0 = dt.datetime(year, month, day, hour, minute, int(second),int((second%1)*1e6))
                if "date_start" in args:
                    if date0<args["date_start"]:
                        continue
                if "date_end" in args:
                    if date0>args["date_end"]:
                        continue
                dates.append(date0)    
            if len(dates)>1:        
                #converts to matrix and stores
                print "File ended"
                DPL["dates"].append(dates[0])
                
            else:
                print "File Skipped"
        DPL["N_bursts"] = len(DPL["time"])
        DPL["time"] = np.array(convert_date_to_time(DPL["dates"]))
        DPL["burst_number"] = np.array(DPL["burst_number"])
        #DPL["status"] = status
        #if status == 1:
        #    DPL["Tmatrix"][1:,:] = -DPL["Tmatrix"][1:,:] 
        for var in DPL.keys():
             exec("self.%s = DPL[\"%s\"]"%(var, var))


    def reads_burst_file(self, bn):
         #gets data for a burst
         ffname = self.folder+"/"+self.deployname+"%03d"%(bn)
         print "Reading file "+self.deployname+"%03d"%(bn)

         BR = dict()
         BR["dates"] = []
         for var in self.time_vars:
              BR[var] = []

         fsen = open(ffname+ ".sen")
         fv1 = open(ffname + ".v1")
         fv2 = open(ffname + ".v2")
         fv3 = open(ffname + ".v3") 
         fa1 =  open(ffname + ".a1")
         fa2 =  open(ffname + ".a2")
         fa3 =  open(ffname + ".a3")
         fc1 =  open(ffname + ".c1")
         fc2 =  open(ffname + ".c2")
         fc3 =  open(ffname + ".c3")

         head = []
         pitch = []
         roll = []
         pres = []
         temp = []
         v1 = []
         v2 = []
         v3 = []
         a1 = []
         a2 = []
         a3 = []
         c1 = []
         c2 = []
         c3 = []

         for (line_s,line_v1,line_v2,line_v3,line_a1,line_a2,line_a3,line_c1,line_c2,line_c3) in zip(fsen,fv1,fv2,fv3,fa1,fa2,fa3,fc1,fc2,fc3):
              lineS = line_s.split()
              month = int(lineS[0])
              day = int(lineS[1])
              year = int(lineS[2])
              hour = int(lineS[3])
              minute = int(lineS[4])
              second = float(lineS[5])
              date0 = dt.datetime(year, month, day, hour, minute, int(second),int((second%1)*1e6))

              BR["dates"].append(date0)
              head.append(float(lineS[12]))
              pitch.append(float(lineS[13]))
              roll.append(float(lineS[14]))
              pres.append(float(lineS[15]))
              temp.append(float(lineS[16]))
              
              lineS = line_v1.split()
              v10 = [float(lineS[j]) for j in range(2, len(lineS))]
              v1.append(v10)
              
              lineS = line_v2.split()
              v20 = [float(lineS[j]) for j in range(2, len(lineS))]
              v2.append(v20)
              
              lineS = line_v3.split()
              v30 = [float(lineS[j]) for j in range(2, len(lineS))]
              v3.append(v30)
              
              lineS = line_a1.split()
              a10 = [float(lineS[j]) for j in range(2, len(lineS))]
              a1.append(a10)
              
              lineS = line_a2.split()
              a20 = [float(lineS[j]) for j in range(2, len(lineS))]
              a2.append(a20)
              
              lineS = line_a3.split()
              a30 = [float(lineS[j]) for j in range(2, len(lineS))]
              a3.append(a30)
              
              lineS = line_c1.split()
              c10 = [float(lineS[j]) for j in range(2, len(lineS))]
              c1.append(c10)
              
              lineS = line_c2.split()
              c20 = [float(lineS[j]) for j in range(2, len(lineS))]
              c2.append(c20)
              
              lineS = line_c3.split()
              c30 = [float(lineS[j]) for j in range(2, len(lineS))]
              c3.append(c30)

         time = convert_date_to_time(BR["dates"]) 
         for var in self.time_vars:
              exec("BR[\"%s\"] = np.array(%s)"%(var,var))
              
         return BR

    

    def gets_mean_profiles(self,  save = True,  **kargs):
        print "\n------------------\nCalculates mean velocity profiles\n----------------\n" 
        if "index" in kargs:
            ind = np.array(kargs["index"])
        else:
            ind = np.arange( len(self.time))
             
        self.mDATA = dict()
        self.mDATA["indexes"] = ind
                
        for var in self.time_vars:
            self.mDATA[var] = []
            if var != "time":
                 self.mDATA["s"+var] = []
                 
        self.mDATA["inwater"] = []         
        for i in ind:
             BR = self.reads_burst_file(self.burst_number[i])
             if np.min(BR["pres"])> self.min_pres:
                  self.mDATA["inwater"].append(True)
             else:
                  self.mDATA["inwater"].append(False)
             for var in self.time_vars:
                self.mDATA[var].append(np.mean(BR[var], axis = 0))
                Ndata = float(BR[var].shape[0])
                if var != "time":
                     self.mDATA["s"+var].append(np.std(BR[var], axis = 0)) #/Ndata**0.5)

        for var in self.time_vars:              
             self.mDATA[var] = np.array(self.mDATA[var])
             if var != "time":
                  self.mDATA["s"+var] = np.array(self.mDATA["s"+var])

        if save:
             self.save_pickle()
             

    def finds_bottom_earth_coords(self, PLOT = False, save = True, **kargs):
        print "\n------------------\nFinds bottom and rotates \n----------------\n" 
        #rotates velocities and creates vertical coordinate
             
        if "bot_fac" not in kargs:
             bot_fac = 99.
        else:
             bot_fac = kargs["bot_fac"]
             
        NT = self.mDATA["time"].size

        self.mDATA["dates"] = convert_time_to_date(self.mDATA["time"])
        self.mDATA["u"] = np.full(self.mDATA["v1"].shape, np.nan)
        self.mDATA["v"] = np.full(self.mDATA["v1"].shape, np.nan)
        self.mDATA["w"] = np.full(self.mDATA["v1"].shape, np.nan)
        self.mDATA["shu"] = np.full(self.mDATA["v1"].shape, np.nan)
        self.mDATA["shv"] = np.full(self.mDATA["v1"].shape, np.nan)
        self.mDATA["sh2"] = np.full(self.mDATA["v1"].shape, np.nan)
        self.mDATA["mab1"] = np.full(self.mDATA["v1"].shape, np.nan)
        self.mDATA["mab2"] = np.full(self.mDATA["v1"].shape, np.nan)
        self.mDATA["mab3"] = np.full(self.mDATA["v1"].shape, np.nan)
        self.mDATA["mab"] = np.full(self.mDATA["v1"].shape, np.nan)
        self.mDATA["mzB"] = np.full((NT,3), np.nan)
        self.mDATA["mrB"] = np.full((NT,3), np.nan)
        
        
        BD0 = np.copy(self.blanking_distance)
        Dr0 = np.copy(self.Dr)
        Nc0 = np.copy(self.N_cells)
        for i in range(NT):
             print "Finds ground and rotate for file %d of %d (DEP%03s)"%(i+1,NT,self.burst_number[self.mDATA["indexes"][i]])
             if not self.mDATA["inwater"][i]:
                  print "Skip. Not in water"
                  continue
                  
             hM = heading_matrix( self.mDATA["head"][i])
             pM = tilt_matrix( self.mDATA["pitch"][i], self.mDATA["roll"][i])
             gM = np.dot(hM, pM)
             RM = np.dot(hM, np.dot(pM, self.Tmatrix))

             #vertical coordinate for each beam coordinates
             z = -np.cos(25.*np.pi/180)*self.r
             _,_,z1 = transform_v_profile(gM, 0*z, 0*z, z)
             _,_,z2 = transform_v_profile(gM, 0*z, 0*z, z)
             _,_,z3 = transform_v_profile(gM, 0*z, 0*z, z)
             
             #z = (z1+z2+z3)/3.             

             #find bottom
             iAmax = np.full(3,0)
             iAmax[0] = find_a_maximum_top( self.mDATA["a1"][i,:], factor = bot_fac )
             iAmax[1] = find_a_maximum_top( self.mDATA["a2"][i,:], factor = bot_fac )
             iAmax[2] = find_a_maximum_top( self.mDATA["a3"][i,:], factor = bot_fac )
             self.mDATA["mzB"][i,0] =  z1[ iAmax[0]]
             self.mDATA["mzB"][i,1] =  z2[iAmax[1]]
             self.mDATA["mzB"][i,2] =  z3[iAmax[2]] 
             self.mDATA["mrB"][i,:] =  self.r[ iAmax]


             self.mDATA["mab1"][i,:] = z1 - self.mDATA["mzB"][i,0]
             self.mDATA["mab2"][i,:] = z2 - self.mDATA["mzB"][i,1]
             self.mDATA["mab3"][i,:] = z3 - self.mDATA["mzB"][i,2]

             self.mDATA["mab"][i,:] = z - np.nanmean(self.mDATA["mzB"][i,:])

             #interpolates velocities to a grid referred to the bottom, not to the instrument (to account for different bottom distances experienced by the different beams)
             fv1 = intrp.interp1d( self.mDATA["mab1"][i,:], self.mDATA["v1"][i,:], bounds_error = False )
             fv2 = intrp.interp1d( self.mDATA["mab2"][i,:], self.mDATA["v2"][i,:], bounds_error = False )
             fv3 = intrp.interp1d( self.mDATA["mab3"][i,:], self.mDATA["v3"][i,:], bounds_error = False )
             v1B = fv1(self.mDATA["mab"][i,:])
             v2B = fv2(self.mDATA["mab"][i,:])
             v3B = fv3(self.mDATA["mab"][i,:])
             #transforms velocities
             self.mDATA["u"][i,:], self.mDATA["v"][i,:], self.mDATA["w"][i,:] = transform_v_profile(RM,v1B, v2B, v3B)

             self.mDATA["shu"][i,:] = first_centered_differences(self.mDATA["mab"][i,:], self.mDATA["u"][i,:])
             self.mDATA["shv"][i,:] = first_centered_differences(self.mDATA["mab"][i,:], self.mDATA["v"][i,:])
             self.mDATA["sh2"][i,:] = self.mDATA["shu"][i,:]**2 + self.mDATA["shv"][i,:]**2
             
            
             if PLOT:
                  self.plot_mean_profiles(i)
        if save:
            self.save_pickle()
                  

    def rotates_into_principal_coordinates(self):
        if "u" not in self.mDATA:
            print "Earth coordinate velocities not found"
            self.finds_bottom_earth_coords()

        u_mat = np.c_[self.mDATA["u"].ravel(),self.mDATA["v"].ravel(),self.mDATA["w"].ravel()].T
        ii = np.isfinite(np.sum(u_mat, axis =0))
        CoVar = np.cov(u_mat[:,ii])
        LB, V = np.linalg.eig(CoVar)
        idx = LB.argsort()[::-1]   
        LB = LB[idx]
        V = V[:,idx]
        inV = np.linalg.inv(V)

        #http://planning.cs.uiuc.edu/node101.html
        #yau
        alpha = np.arctan2(inV[1,0],inV[0,0])*180/np.pi
        #pitch
        beta = np.arctan2(-inV[2,1],np.sqrt( inV[2,1]**2 + inV[2,2]**2 ))*180/np.pi
        #roll
        gamma = np.arctan2(inV[2,1],inV[2,2])*180/np.pi

        print "Eigen values"
        print LB
        print "Yau: %2.1f deg, Pitch: %1.2f deg, Roll: %1.2f deg"%(alpha, beta, gamma)
        self.mDATA["pc_angle_Yau"] = alpha
        self.mDATA["pc_angle_Pitch"] = beta
        self.mDATA["pc_angle_Roll"] = gamma
        
        print "Velocities rotated to principal components"
        d1,d2 = self.mDATA["u"].shape
        self.mDATA["v_pc1"] = np.full((d1,d2), np.nan)
        self.mDATA["v_pc2"] = np.full((d1,d2), np.nan)
        self.mDATA["v_pc3"] = np.full((d1,d2), np.nan)
        for i in range(d1):
            u_mat = np.c_[self.mDATA["u"][i,:],self.mDATA["v"][i,:],self.mDATA["w"][i,:]].T
            uR_mat = np.dot(inV,u_mat)
            self.mDATA["v_pc1"][i,:] = uR_mat[0,:]
            self.mDATA["v_pc2"][i,:] = uR_mat[1,:]
            self.mDATA["v_pc3"][i,:] = uR_mat[2,:]
        
         
    def plot_mean_profiles(self,bn):
         i = np.where(self.burst_number == bn)[0][0]
         Ut = (self.mDATA["u"][i,:]**2+ self.mDATA["v"][i,:]**2 +self.mDATA["w"][i,:]**2)**0.5
         U = (self.mDATA["v1"][i,:]**2+ self.mDATA["v2"][i,:]**2 +self.mDATA["v3"][i,:]**2)**0.5
         mzB = np.mean(self.mDATA["mzB"][i,:])

         bn = self.burst_number[i]

         datestr = self.mDATA["dates"][i].strftime("%d %b %H:%M")
         #datestr_fname = self.mDATA["dates"][i].strftime("%y%m%d_%H%M")
         fig, ax = plt.subplots(1,3, figsize = (10,5))
         l1,=ax[0].plot(self.mDATA["v1"][i,:]*100, self.r,".-",lw = 1)
         l2,=ax[0].plot(self.mDATA["v2"][i,:]*100, self.r,".-",lw = 1)
         l3,=ax[0].plot(self.mDATA["v3"][i,:]*100,self.r,".-",lw = 1)
         ax[0].set_ylim(ax[0].get_ylim()[::-1])
         xl = ax[0].get_xlim()
         yl = ax[0].get_ylim()
         ax[0].set_xlim(xl)
         ax[0].set_ylim(yl)
         ax[0].axvline(0, color = "gray", lw = 0.5)
         ax[0].axhline(self.mDATA["mrB"][i,0],color = l1.get_color())
         ax[0].axhline(self.mDATA["mrB"][i,1],color = l2.get_color())
         ax[0].axhline(self.mDATA["mrB"][i,2],color = l3.get_color())
         ax[0].set_xlabel("v along-beam [cm/s]")
         ax[0].set_ylabel("r")

         ax0 = ax[0].twiny()
         ax0.plot(self.mDATA["a1"][i,:], self.r, ls = "--", alpha = 0.7)
         ax0.plot(self.mDATA["a2"][i,:], self.r, ls = "--",alpha = 0.7)
         ax0.plot(self.mDATA["a3"][i,:], self.r, ls = "--",alpha = 0.7)
         ax0.set_xlabel("Beam amplitude")

         
         l1,=ax[1].plot(self.mDATA["c1"][i,:]*100, self.r,".-",lw = 1)
         l2,=ax[1].plot(self.mDATA["c2"][i,:]*100, self.r,".-",lw = 1)
         l3,=ax[1].plot(self.mDATA["c3"][i,:]*100,self.r,".-",lw = 1)
         ax[1].set_ylim(ax[1].get_ylim()[::-1])
         xl = ax[1].get_xlim()
         yl = ax[1].get_ylim()
         ax[1].set_xlim(xl)
         ax[1].set_ylim(yl)
         ax[1].axvline(0, color = "gray", lw = 0.5)
         ax[1].axhline(self.mDATA["mrB"][i,0],color = l1.get_color())
         ax[1].axhline(self.mDATA["mrB"][i,1],color = l2.get_color())
         ax[1].axhline(self.mDATA["mrB"][i,2],color = l3.get_color())
         ax[1].set_xlabel("Correlation")
         ax[1].set_ylabel("r")
         ax[1].set_title(datestr)
         
         ax[2].plot(self.mDATA["u"][i,:]*100, self.mDATA["mab"][i,:],".-",lw = 1)
         ax[2].plot(self.mDATA["v"][i,:]*100, self.mDATA["mab"][i,:],".-",lw = 1)
         ax[2].plot(self.mDATA["w"][i,:]*100, self.mDATA["mab"][i,:],".-",lw = 1)
         ax[2].plot(Ut*100,self.mDATA["mab"][i,:],"k")
         xl = ax[2].get_xlim()
         yl = ax[2].get_ylim()
         ax[2].fill_between(xl,[np.min(yl),np.min(yl)],[0,0], color = "brown", alpha = 0.8)
         ax[2].set_xlim(xl)
         ax[2].set_ylim(yl)
         ax[2].axvline(0, color = "gray", lw = 0.5)
         ax[2].set_xlabel("u,v,w (ENU) [cm/s]")
         ax[2].set_ylabel("mab")
         
         ax[2].legend(["u, v1","v, v2","w, v3"])
         
         #ax[0].tick_params(axis='y', which='both', labelleft='off', labelright='on')
         ax[2].yaxis.set_label_position("right")
         ax[2].yaxis.tick_right()
         fig.tight_layout()
         fig.savefig("Aquadop_mean_prof_%s_%03d.png"%(self.deployname,bn), dpi = 300)
         
         plt.show() 

             
    def plot_mean_timeseries(self):
         z = np.copy(self.mDATA["mab"].T)

         #calculates limits for plots
         uu = np.r_[self.mDATA["u"],self.mDATA["v"]]
         uu = uu.ravel()
         uu *= 100
         maxu0 = max([np.abs(np.nanmax(uu)),np.abs(np.nanmin(uu))])
         minu = -maxu0
         maxu = maxu0
         du = (maxu-minu)/50
         maxu += du
         
         print maxu
         print minu
         print du

         
         ww = np.copy(self.mDATA["w"])
         ww = ww.ravel()
         ww *= 100
         maxw0 = max([np.abs(np.nanmax(ww)),np.abs(np.nanmin(ww))])
         minw = -maxw0
         maxw = maxw0
         dw = (maxw-minw)/50
         maxw += dw

         print maxw0
         
         u = np.copy(self.mDATA["u"].T)*100
         v = np.copy(self.mDATA["v"].T)*100
         w = np.copy(self.mDATA["w"].T)*100

         u[z<0] = np.nan
         v[z<0] = np.nan
         w[z<0] = np.nan

         mz = np.nanmean(z, axis = 1)

         mu = np.nanmean(u,axis = 0)
         mv = np.nanmean(v,axis = 0)
         mw = np.nanmean(w,axis = 0)
         
         fig = plt.figure(figsize=(7,7))
         
         ax2 = fig.add_subplot(412)
         ccu = ax2.contourf(self.mDATA["dates"],mz, u, np.arange(minu,maxu,du), cmap = cm.RdBu_r)
         cbu = plt.colorbar(ccu)
         cbu.set_label("u [cm s$^{-1}$]")
         ax2.set_ylim((0, np.max(mz)))
         ax2.set_xticklabels([])
         ax2.set_ylabel("mab")

         ax3 = fig.add_subplot(413)
         ccv = ax3.contourf(self.mDATA["dates"],mz, v, np.arange(minu,maxu,du), cmap = cm.RdBu_r)
         cbv = plt.colorbar(ccv)
         cbv.set_label("v [cm s$^{-1}$]")
         ax3.set_ylim((0, np.max(mz)))
         ax3.set_xticklabels([])
         ax3.set_ylabel("mab")

         ax4 = fig.add_subplot(414)
         ccw = ax4.contourf(self.mDATA["dates"],mz, w, np.arange(minw,maxw,dw), cmap = cm.RdBu_r)
         cbw = plt.colorbar(ccw)
         cbw.set_label("w [cm s$^{-1}$]")
         ax4.set_ylim((0, np.max(mz)))
         ax4.set_ylabel("mab")
         ax4.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
         xl = ax4.get_xlim()
         ax1 = fig.add_axes([0.095,0.77, 0.695,0.19])
         l1,=ax1.plot(self.mDATA["dates"], mu, lw = 1)
         l2,=ax1.plot(self.mDATA["dates"], mv, lw = 1)
         l3,=ax1.plot(self.mDATA["dates"], mw, lw = 1)
         ax1.legend((l1,l2,l3),["u","v","w"], ncol = 3)
         ax1.set_xticklabels([])
         ax1.set_xlim(xl)
         ax1.axhline(0, color = "gray")
         ax1.set_ylabel("$\\bar u$, $\\bar v$, $\\bar w$ [cm s$^{-1}$]")
         #print delta_y
         
         fig.tight_layout()

         date_str = self.mDATA["dates"][0].strftime("%Y%m%d_%H%M") + "_" + self.mDATA["dates"][-1].strftime("%Y%m%d_%H%M")
         
         fig.savefig("Aquadopp_time_series_%s_%s.png"%(self.deployname,date_str), dpi = 300)
         plt.close(fig)


    ########################################
    ########################################
    ############ LOW FITS ##################
    ########################################
    ########################################
    def fits_LOW_timeseries(self, corrected = False, minN = 3, startz = 0, minz = 0., maxz=0,z00 = np.nan, save = False):
         
         it,iz = self.mDATA["u"].shape
         us = np.full(it, np.nan)
         z0 = np.full(it, np.nan)
         mepsilon = np.full(it, np.nan)
         R2 = np.full(it, np.nan)
         err = np.full(it, np.nan)
         U1m = np.full(it, np.nan)
         if corrected:
              L = np.full(it, np.nan)

         for (i,bn) in enumerate(self.burst_number):
              if not self.mDATA["inwater"][i]:
                   continue
              print "\nLOW iteration %d of %d"%(i+1,it)
              params, R2[i], err[i], mepsilon[i], U1m[i] = self.fits_LOW_1prof(bn,startz,minN,minz,maxz,z00,corrected)
              us[i] = params["us"]
              z0[i] = params["z0"]
              if corrected:
                   L[i] = params["L"]
         if not corrected:
              Vname = "LOW"
              self.mDATA[Vname] = dict()
         else:
              Vname = "LOWc"
              self.mDATA[Vname] = dict()
              
              
         self.mDATA[Vname] = dict()
         self.mDATA[Vname]["us"] = us
         self.mDATA[Vname]["z0"] = z0
         self.mDATA[Vname]["mepsilon"] = mepsilon
         self.mDATA[Vname]["R2"] = R2
         self.mDATA[Vname]["err"] = err
         self.mDATA[Vname]["U1m"] = U1m
         if corrected:
             self.mDATA[Vname]["L"] = L
         if save:
            self.save_pickle()
        

          
    def low_fitting(self,z,y, z0 = np.nan,corr = False):
         #makes the fit given vectors of distance above bottom (z) and velocity (u)
         #if corr, takes into account the effect of stratification following Cannon 2018
         alpha = 0.05
         kappa = 0.41
         N = z.size
         if not corr:
              x = np.log(z)
              #p = np.polyfit( x,y,1 )
              if np.isnan(z0):
                  X = np.vstack([np.ones(N), x]).T
                  p  = np.linalg.lstsq(X, y)[0]
                  params = dict()
                  params["us"] = kappa*p[1]
                  params["z0"] = np.exp(-p[0]/p[1])
                  dof = N-2
              else:
                  x -= np.log(z0)
                  x = x[:,np.newaxis]
                  y = y[:,np.newaxis]
                  p  = np.linalg.lstsq(x, y)[0][0]
                  params = dict()
                  params["us"] = kappa*p[0]
                  params["z0"] = z0
                  dof = N-1
                  y = y[:,0]
              
         else:
              if np.isnan(z0):
                  x1 = np.log(z)
                  x2 = np.copy(z)
                  X = np.vstack([np.ones(N), x1, x2]).T
                  p  = np.linalg.lstsq(X, y)[0]
                  params = dict()
                  params["us"] = kappa*p[1]
                  params["z0"] = np.exp(-p[0]/p[1])
                  params["L"] = p[1]/p[2]
                  dof = N-3
              else:
                  x1 = np.log(z/z0)
                  x2 = np.copy(z)
                  X = np.vstack([x1, x2]).T
                  p  = np.linalg.lstsq(X, y)[0]
                  params = dict()
                  params["us"] = kappa*p[0]
                  params["z0"] = z0
                  params["L"] = p[0]/p[1]
                  dof = N-2
                  

         pp = LOW_model( params ) 
         ymod = pp.eval_u(z)
         N = z.size
         yres = y-ymod
         R2 = 1- np.var(yres)/np.var(y)
         if R2<0 or R2>1:
             e = np.nan
         else:    
             tval = stats.t.ppf(1-alpha,dof-2) #is this correct??
             e = tval*(((1-R2)/R2)/(dof-2))**0.5
         return e, R2, pp, params
              
         
    def fits_LOW_1prof(self,bn,minN = 3, startz = 0,  minz = 0., maxz = 0,z0 = np.nan, corrected = False, PLOT = False):
         if "v_pc1" not in self.mDATA:
              print "Principal components not found, calculating"
              self.rotates_into_principal_coordinates()
         i = np.where(self.burst_number == bn)[0][0]
         datestr = self.mDATA["dates"][i].strftime("%d %b %H:%M")
         #datestr_fname = self.mDATA["dates"][i].strftime("%y%m%d_%H%M")
         Ut = (self.mDATA["v_pc1"][i,::-1]**2+ self.mDATA["v_pc2"][i,::-1]**2)**0.5
         Z0 = self.mDATA["mab"][i,::-1]
         
         ii = (Z0>minz) & (np.isfinite(Ut))
         y = Ut[ii]
         Z = Z0[ii]

         if maxz == 0:
             maxz = len(Z)
         
         #looks for the best interval from Troy 2016, Saylor 1988
         if not corrected or PLOT:
              itest = np.arange(minN,maxz,1)
              e = np.full(itest.size, np.nan)
              R2 = np.full(itest.size, np.nan)
              us0 = np.full(itest.size, np.nan)
              for (i,j) in enumerate(itest):
                   jj = np.arange(0,j+1)
                   e[i], R2[i], pp, params = self.low_fitting( Z[jj],y[jj], z0 = z0, corr = corrected)
                   us0[i] = params["us"]
              e[us0<0] = np.nan
              if np.isnan(e).all():
                  params = dict()
                  params["us"] = np.nan
                  if not np.isnan(z0):
                      params["z0"] = z0
                  else:
                      params["z0"] = np.nan
                  if corrected:
                      params["L"] = np.nan
                  R2f = np.nan
                  ef = np.nan
                  mepsilon = np.nan
                  U1m = np.nan
                  return params, R2f, ef, mepsilon, U1m
              j = itest[np.nanargmin(e)]
         
         if not corrected:     
              jj = np.arange(startz,j)
         else:
              jj = np.arange(startz,Z.size)
              
         ef, R2f, pp, params = self.low_fitting( Z[jj],y[jj], z0 = z0, corr = corrected)
         us = params["us"]
         z0 = params["z0"]
         if corrected:
              L = params["L"]
         mepsilon = pp.eval_mepsilon(Z0)
         ymod2 = pp.eval_u(Z0)
         U1m = pp.eval_u(1.)
         if not corrected:
              annotation = "LOW\n$u_{\\ast} = %1.2f\\pm %1.2f$ mm/s\n$z_0 = %1.4f \\pm %1.4f$ mm\n$R^2 = %1.3f$\n$\epsilon = %1.1e$ W/kg\n$U_{1m} = %1.1f$ cm/s"%(us*1000,us*ef*1000,z0*1000,z0*ef*1000,R2f, mepsilon,U1m*100)
         else:
              annotation = "LOW+strat\n$u_{\\ast} = %1.2f\\pm %1.2f$ mm/s\n$z_0 = %1.4f \\pm %1.4f$ mm\n$L = %1.2f \\pm %1.2f$ cm\n$R^2 = %1.3f$\n$\epsilon = %1.1e$ W/kg\n$U_{1m} = %1.1f$ cm/s"%(us*1000,us*ef*1000,z0*1000,z0*ef*1000,L*100,ef*np.abs(L)*100,R2f,mepsilon,U1m*100)

         print "-------------\nLOW fit %s\n----------"%(datestr)
         print annotation
         print "-----------------\n"

         if PLOT:
              if not corrected:
                  filename = "LOWfit_%s_%03d.png"%(self.deployname, bn)
              else:
                  filename = "LOWfit_corr_%s_%03d.png"%(self.deployname, bn)
              fig,ax = plt.subplots(1,2, sharey = True)
              l1,=ax[0].plot( e,Z[itest]*100, label = "Error" )
              ax[0].set_ylabel("cmab")
              ax[0].set_xlabel("Error")
              ax[0].set_xlim((0,1))
              ax0  = ax[0].twiny()
              l2,=ax0.plot(R2,Z[itest]*100,color = "gray", label = "$R^2$")
              ax0.set_xlabel("$R^2$")
              ax0.set_xlim((0,1))
              lines = [l1,l2]
              ax[0].legend(lines, [l.get_label() for l in lines])
         
              ax[1].plot(Ut*100,Z0*100,'o-', color = "gray", mfc = "w", ms = 5)
              ax[1].plot(y[jj]*100,Z[jj]*100,'o', color= "gray", ms = 5)
              ax[1].plot(ymod2*100,Z0*100,color = "k",lw = 2)
              ax[1].set_xlabel("u [cm/s]")
              ax[1].set_ylim((0,150))
              ax[1].set_title(datestr)
              ax[1].annotate(annotation, xy = (0.05,0.95), xycoords = "axes fraction", verticalalignment = "top", fontsize = 9)
              plt.savefig(filename, dpi = 300, bbox_inches = "tight")
              
         return params, R2f, ef, mepsilon, U1m 
         
    def drag_coefficient(self,zref = 1., corrected = False, SF = False):
         it, iz = self.mDATA["u"].shape
         U2 = np.full(it, np.nan)
         u = np.full((3,it), np.nan)
         for i in range(it):
              ij = np.argmin(np.abs(self.mDATA["mab"][i,:]-zref))
              u[0,i] = self.mDATA["u"][i,ij]
              u[1,i] = self.mDATA["v"][i,ij]
              u[2,i] = self.mDATA["w"][i,ij]
              U2[i] = np.sum(u[:-1,i]**2)
         if corrected:
              us = np.copy(self.mDATA["LOWc"]["us"])
              R2 = np.copy(self.mDATA["LOWc"]["R2"])
              z0 = np.copy(self.mDATA["LOWc"]["z0"])
              L  = np.copy(self.mDATA["LOWc"]["L"])
              epsilon = np.copy(self.mDATA["LOWc"]["us"]**3/0.41/zref)
              #epsilon = np.copy(self.mDATA["LOWc"]["mepsilon"])
         else:
              us = np.copy(self.mDATA["LOW"]["us"])
              R2 = np.copy(self.mDATA["LOW"]["R2"])
              z0 = np.copy(self.mDATA["LOW"]["z0"])
              epsilon = np.copy(self.mDATA["LOW"]["us"]**3/0.41/zref)
              #epsilon = np.copy(self.mDATA["LOW"]["mepsilon"])

         #stores reference velocity
         self.mDATA["v_ref"] = np.copy(u)
         self.mDATA["z_ref"] = np.copy(zref)

         #epsilon from SF
         if SF:
             epsilonSF = np.full(it, np.nan)
             for i in range(it):
                 ij = np.argmin(np.abs(self.mDATA["mab"][i,:]-zref))
                 epsilonSF[i] = self.mDATA["SF"]["epsilonMP"][ij,i]
             #epsilonSF = np.nanmean(self.mDATA["SF"]["epsilonMP"], axis = 0)
             usSF = (epsilonSF*0.41*zref)**(1./3.) 

         #fits Cd using LOW results
         us2 = us**2 
         x = np.copy(U2) 
         y = np.copy(us2)
         ii = (R2>0.5) & (us>0) & (z0>1e-5) & (np.isfinite(x)) & (np.isfinite(y))
         x = x[ii]
         y = y[ii]
         x = x[:,np.newaxis]
         y = y[:,np.newaxis]

         alpha = 0.05
         N = x.size
         Cd = np.linalg.lstsq(x, y)[0][0,0]
         ymod = x*Cd
         yres = y-ymod
         R2f = 1- np.var(yres)/np.var(y)
         tval = stats.t.ppf(1-alpha,N-2)
         e = tval*(((1-R2f)/R2f)/(N-2))**0.5

         self.Cd = np.copy(Cd)

         date_str = self.mDATA["dates"][0].strftime("%Y%m%d_%H%M") + "_" + self.mDATA["dates"][-1].strftime("%Y%m%d_%H%M")
         if not corrected:
              title = "$C_d$ from LOW $u_{\\ast}$"
              figname = "drag_coefficient_%s_%s.png"%(self.deployname,date_str)
              figname2 = "LOW_timeseries_%s_%s.png"%(self.deployname,date_str)
         else:
              title = "$C_d$ from corrected-LOW $u_{\\ast}$"
              figname = "drag_coefficient_corrected_LOW_%s_%s.png"%(self.deployname,date_str)
              figname2 = "LOW_corrected_timeseries_%s_%s.png"%(self.deployname,date_str)
              
         text0 = "$C_d = %1.4f \pm %1.4f$\n$R^2 = %1.3f$"%(Cd,Cd*e,R2f)
         
         fig,ax = plt.subplots()
         cc = ax.scatter( U2*1e4, us2*1e4, 5,R2, vmin = 0, vmax = 1)
         ax.plot(np.sort(U2)*1e4, Cd*np.sort(U2)*1e4, 'k-')
         ax.set_xlabel(" $U_{%1.1f\mathrm{ m}}^2$ [cm$^2$ s$^{-2}$]"%zref)
         ax.set_ylabel(" $u_{\\ast}^2$ [cm$^2$ s$^{-2}$]")
         ax.set_xscale("log")
         ax.set_yscale("log")
         ax.annotate(text0, xy = (0.05,0.95), xycoords= "axes fraction", verticalalignment = "top")
         cb=plt.colorbar(cc)
         cb.set_label("$R^2$")
         ax.set_title(title)
         fig.savefig(figname, dpi = 300, bbox_inches = "tight")
         #plt.show()
         plt.close(fig)
             

         datesBAD = []
         jBAD = []
         for i in range(it):
              if not ii[i]:
                  datesBAD.append(self.mDATA["dates"][i])
                  jBAD.append(i)
         jBAD = np.array(jBAD)

         usB = np.copy(us)
         z0B = np.copy(z0)
         epsilonB = np.copy(epsilon)
         
         us[jBAD]= np.nan
         z0[jBAD]= np.nan
         epsilon[jBAD]= np.nan
         if corrected:
             LB = np.copy(L)
             LB[jBAD]= np.nan
         
         #plots the dissipation time series
         fig, ax = plt.subplots(4,1, sharex = True, figsize = (7,7))
         ax[0].plot( self.mDATA["dates"],u.T*100,lw = 0.5)
         ax[0].plot( self.mDATA["dates"],U2**0.5*100,lw = 1, color = "k")
         ax[0].axhline(0,color = "gray",lw = 0.5)
         ax[0].legend(["$u$","$v$","$w$","u$_{mod}$"], ncol = 4)
         ax[0].set_ylabel("u [cm s$^{-1}$]")
         ax[1].plot(self.mDATA["dates"], usB*1000, color = "gray")
         ax[1].plot(self.mDATA["dates"], us*1000,"k-")
         if SF:
             ax[1].plot(self.mDATA["dates"], usSF*1000,"-", lw = 1)
         ax[1].set_yscale("log")
         ax[1].set_ylabel("$u_{\\ast}$ [mm s$^{-1}$]")
         ax[1].set_ylim((0.1,10))
         if not corrected:
             ax[2].plot(self.mDATA["dates"], z0B*100,color = "gray")
             ax[2].plot(self.mDATA["dates"], z0*100,"k-")
             ax[2].set_ylabel("$z_0$ [mm]")
             ax[2].set_ylim((1e-3,10))
             ax[2].set_yscale("log")
         else:
             
             ax[2].plot(self.mDATA["dates"], L*100,color = "gray")
             ax[2].plot(self.mDATA["dates"], LB*100,"k-")
             ax[2].set_ylabel("$L$ [cm]")
             ax[2].set_ylim((-200,200))
             #ax[2].set_yscale("log")

         ax[3].plot(self.mDATA["dates"], epsilonB,color = "gray") #, width = 0.03)
         l1,=ax[3].plot(self.mDATA["dates"], epsilon,  color = "black") #width = 0.03
         ax[3].set_ylim((1e-11,1e-6))
         ax[3].set_yscale("log")
         ax[3].set_ylabel("$\\varepsilon$ [m$^2$ s$^{-3}$]")
         if SF:
             
             l2,=ax[3].plot(self.mDATA["dates"], epsilonSF) #width = 0.03
             ax[3].legend((l1,l2),["$\\varepsilon_{LOW}$","$\\varepsilon_{SF}$"],fontsize = 8)
         ax[3].xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
         fig.savefig(figname2, dpi = 300, bbox_inches = "tight")
         plt.close(fig)

         if SF:
             #fits Cd using LOW results
             
             x = np.copy(U2) 
             y = np.copy(usSF**2)
             ii = (np.isfinite(x)) & (np.isfinite(y))
             x = x[ii]
             y = y[ii]
             x = x[:,np.newaxis]
             y = y[:,np.newaxis]
             
             alpha = 0.05
             N = x.size
             CdSF = np.linalg.lstsq(x, y)[0][0,0]
             ymod = x*CdSF
             yres = y-ymod
             R2f = 1- np.var(yres)/np.var(y)
             tval = stats.t.ppf(1-alpha,N-2)
             e = tval*(((1-R2f)/R2f)/(N-2))**0.5
             
             text0 = "$C_d = %1.4f \pm %1.4f$\n$R^2 = %1.3f$"%(CdSF,CdSF*e,R2f)
             title = "$C_d$ from SF"
             figname3 = "drag_coefficient_SF_%s_%s.png"%(self.deployname,date_str)

             fig,ax = plt.subplots()
             ax.plot( U2*1e4, usSF**2*1e4,'k.')
             ax.plot(np.sort(U2)*1e4, CdSF*np.sort(U2)*1e4, 'k-')
             ax.set_xlabel(" $U_{%1.1f\mathrm{ m}}^2$ [cm$^2$ s$^{-2}$]"%zref)
             ax.set_ylabel(" $u_{\\ast}^2$ [cm$^2$ s$^{-2}$]")
             ax.set_xscale("log")
             ax.set_yscale("log")
             ax.annotate(text0, xy = (0.05,0.95), xycoords= "axes fraction", verticalalignment = "top")
             ax.set_title(title)
             fig.savefig(figname3, dpi = 300, bbox_inches = "tight")
             #plt.show()
             plt.close(fig)

    ########################
    ## STRUCTURE FUNCTIONS#
    ########################
    def SF_mean_timeseries(self,save = True, PLOT = False):
         #time series of mean SF dissipation
         if not hasattr(self,"mDATA"):
              self.gets_mean_profiles()
         it= self.mDATA["time"].size
         self.mDATA["mSF"] = dict()
         self.mDATA["mSF"]["epsilon"] = np.full(it, np.nan)
         self.mDATA["mSF"]["R2"] = np.full(it, np.nan)
         self.mDATA["mSF"]["sigmav"] = np.full(it, np.nan)

         for i in range(it):
              if not self.mDATA["inwater"][i]:
                   continue
              bn = self.burst_number[i]
              self.mDATA["mSF"]["epsilon"][i], self.mDATA["mSF"]["sigmav"][i],self.mDATA["mSF"]["R2"][i] = self.SF_one_burst(bn,PLOT = PLOT, MEAN = True)
         if save:
            self.save_pickle()

    def SF_timeseries(self,save = True, Jabbari = True, PLOT = False):
         #time series of SF dissipation profiles. Includes the mean
         if not hasattr(self,"mDATA"):
              self.gets_mean_profiles()
         it= self.mDATA["time"].size
         iz = self.r.size
         iz2 = self.mDATA["mab"][0,:].size
         
         self.mDATA["SF"] = dict()
         self.mDATA["SF"]["epsilon"] = np.full((3,iz,it), np.nan)
         self.mDATA["SF"]["epsilonMP"] = np.full((iz2,it), np.nan)
         self.mDATA["SF"]["R2"] = np.full((3,iz,it), np.nan)
         self.mDATA["SF"]["sigmav"] = np.full((3,iz,it), np.nan)
         self.mDATA["mSF"] = dict()
         self.mDATA["mSF"]["epsilon"] = np.full(it, np.nan)
         self.mDATA["mSF"]["R2"] = np.full(it, np.nan)
         self.mDATA["mSF"]["sigmav"] = np.full(it, np.nan)

         for i in range(it):
              if not self.mDATA["inwater"][i]:
                   continue
              bn = self.burst_number[i]
              self.mDATA["SF"]["epsilon"][:,:,i], self.mDATA["SF"]["sigmav"][:,:,i],self.mDATA["SF"]["R2"][:,:,i],self.mDATA["SF"]["epsilonMP"][:,i],self.mDATA["mSF"]["epsilon"][i], self.mDATA["mSF"]["sigmav"][i],self.mDATA["mSF"]["R2"][i] = self.SF_one_burst(bn,PLOT = PLOT, Jabbari = Jabbari, MEAN = False)
         if save:
            self.save_pickle()
              
              
    def second_order_sf(self,u,r):
         N = u.size
         #dr = r[1]-r[0]
         Nr = r.size
         D = np.full((N,Nr),np.nan)
         #i is the index for the displacement dr
         for i in range(1,Nr+1):
              j = 0
              while True:
                   if j-i<0:
                       j+=1
                       continue
                   if j+i>=N:
                       break
                   #D[j,i-1] = (u[j-1] - u[j+i])**2 #before May 13 2020, this is incorrect?!
                   D[j,i-1] = (u[j-i] - u[j+i])**2  #applies first centered differences
                   j+=1
         return D
    
    def Jabbari_C(self,yp):
        C = np.full(yp.size, np.nan)
        for (i,y) in enumerate(yp):
            if y<=0:
                C[i] = np.nan
            elif y<=160:
                C[i] = 0.048*y**0.65
            elif y<=500:
                C[i] = 0.42+0.45*np.log(y**0.4)
            elif y<2000:
                C[i] = 1.118 + (0.44)/(0.5+24*y**-0.61)
            elif y < 16200:
                C[i] = 0.699 + 0.455*np.log( y**0.295 )
            else:
                C[i] = 2.
        return C
                
                
    def SF_one_burst(self, bn, Jabbari = True, PLOT = False, MEAN = False, **kargs):
         print "------------------\nStructure functions %s %03d\n--------------"%(self.deployname, bn)
         if Jabbari:
             print "Jabbari correction"
         if "bot_fac" not in kargs:
             bot_fac = 99.
         else:
             bot_fac = kargs["bot_fac"]
         if "zref" not in kargs:
             zref = 1.
         else:
             zref = kargs["zref"]
         if "nbin" not in kargs:
             nbin = 7
         else:
             nbin = kargs["nbin"]
             
         if not hasattr(self,"Cd"):
             if Jabbari:
                 print "No Cd available, set to default 5e-3"
             Cd = 5e-3
         else:
             Cd = np.copy(self.Cd)
             
             
         ib = np.where(self.burst_number == bn)[0][0]
         BR = self.reads_burst_file(bn)
         NT = BR["time"].size
         NZ = self.r.size
         NZi = self.mDATA["mab"][ib,:].size
       
         beams = np.arange(1,4)
         mv = []
         ma = []
         mc = []
         for b in beams:
              mv.append( self.mDATA["v%d"%(b)][ib,:] )
              ma.append( self.mDATA["a%d"%(b)][ib,:])
              mc.append( self.mDATA["c%d"%(b)][ib,:])

         iAmax = np.full(3,0) 
         for (i,b) in enumerate(beams):     
              iAmax[i] = np.where(self.mDATA["mab%d"%(b)][ib,:]>0)[0][-1]#find_a_maximum_top( ma[i], factor = bot_fac )

         #Jabbari coefficients
         C = np.full((3,NZ),np.nan)
         if Jabbari:
             ij = np.argmin(np.abs(self.mDATA["mab"][ib,:]-zref))
             U2 = self.mDATA["u"][ib,ij]**2 + self.mDATA["v"][ib,ij]**2
             us = (Cd*U2)**0.5
             zv = 1e-6/us
             for (i,b) in enumerate(beams):
                 zp = self.mDATA["mab%d"%(b)][ib,:]/zv
                 C[i,:] = self.Jabbari_C(zp)
         else:
             C[:] = 2.1

         maxr  = (2*self.Dr)*nbin 
         r = np.arange(2*self.Dr,maxr,2*self.Dr)
         Nestim = 3*NT
         Nr = r.size
         D0 = np.full((3,NT,NZ,Nr), np.nan)
         for (i,b) in enumerate(beams):
              v = BR["v%d"%(b)]
              v[:,iAmax[i]:] = np.nan
              v = v - mv[i]
              for j in range(NT):
                   D0[i,j,:,:] = self.second_order_sf(v[j,:],r)
         #iifit1 = np.where(r<=maxr)[0][-1]
         #iifit = np.arange(iifit1+1)

         #mean fit
         D = np.nanmean(D0, axis = (0,1,2))
         y = np.copy(D)
         x = r**(2/3.)
         pol = np.polyfit(x,y,1)
         pol1 = np.poly1d(pol)
         intercept = pol[1] 
         mepsilon = (pol[0]/np.nanmean(C))**(3/2.)
         Df = pol1(r**(2/3.))
         msigmav = (np.abs(intercept)/2)**0.5
         mR2 = 1 - np.var( D - Df)/ np.var( D )
             

         print "epsilon = %1.2e m2/s3, sigmav = %1.3f mm/s, R2 = %1.4f"%(mepsilon, msigmav*1000,mR2)
         if PLOT:
              datestr = self.mDATA["dates"][ib].strftime("%d %b %H:%M")
              epsSF_lab = "$\epsilon$ = %1.2e m$^2$ s$^{-3}$\n$\sigma_v = %1.3f$ mm/s\nR$^2 = %1.4f$"%(mepsilon,msigmav*1000, mR2)
              fig2,ax2=plt.subplots(figsize = (5,4))
              
              #ax2.plot(r**(2/3.), D0.T,'.',alpha = 0.1, color = "gray")
              l1,=ax2.plot(r*100, D*1e4,"gray", lw = 1)
              #ax2.fill_between(r, (qLD)*1e4,(qHD)*1e4, color = "gray",alpha = 0.5, zorder = -1)
              #plt.show()
              l2,=ax2.plot(r*100, D*1e4,"o", markeredgecolor = "black", ms = 7)
              l3,=ax2.plot(r*100, Df*1e4)
              ax2.legend((l2,l3),["Data","Fit"], loc = "lower right")
              ax2.set_xlabel("$r$ [cm]")
              ax2.set_ylabel("D [10$^{-4}$ m$^2$ s$^{-2}$]")
              ax2.annotate(epsSF_lab, xy = (0.08,0.8), xycoords = "axes fraction")
              ax2.set_title(datestr)
              fig2.tight_layout()
              fig2.savefig("SF_%s%03d.png"%(self.deployname,bn),bbox_inches = "tight", dpi = 300)
              plt.close()

         #calculates for every depth     
         if not MEAN:
             print "Depth resolved calculation"
             #fit per depth
             epsilon = np.full((3,NZ), np.nan)
             epsilonI = np.full((3,NZi), np.nan)
             sigmav = np.full((3,NZ), np.nan)
             R2 = np.full((3,NZ), np.nan)
             for (i,b) in enumerate(beams):    
                 for j in range(NZ):
                     DD = np.nanmean(D0[i,:,j,:], axis = 0)
                     if np.isnan(DD).all():
                         continue
                     y = np.copy(DD)
                     x = r**(2/3.)
                     pol = np.polyfit(x,y,1)
                     pol1 = np.poly1d(pol)
                     intercept = pol[1] 
                     epsilon[i,j] = (pol[0]/C[i,j])**(3/2.)
                     DDf = pol1(r**(2/3.))
                     sigmav[i,j] = (np.abs(intercept)/2)**0.5
                     R2[i,j] = 1 - np.var( DD - DDf)/ np.var( DD )
                 iif = np.isfinite(epsilon[i,:])
                 inteps = intrp.interp1d(self.mDATA["mab%d"%(b)][ib,iif], epsilon[i,iif], bounds_error = False)
                 epsilonI[i,:] = inteps(self.mDATA["mab"][ib,:])
             epsilonMP = np.nanmean(epsilonI, axis = 0)
             return epsilon, sigmav, R2, epsilonMP, mepsilon, msigmav, mR2
         else:
             return mepsilon, msigmav, mR2


    def SF_vs_LOW(self, corrected = False, zref = 1.):
        if corrected:
            eps_LOW = self.mDATA["LOWc"]["us"]**3/0.41/zref
            R2_LOW = self.mDATA["LOWc"]["R2"]
        else:
            eps_LOW = self.mDATA["LOW"]["us"]**3/0.41/zref
            R2_LOW = self.mDATA["LOW"]["R2"]
        date_str = self.mDATA["dates"][0].strftime("%d %b %Y") + " to " + self.mDATA["dates"][-1].strftime("%d %b %Y")
        it, iz = self.mDATA["u"].shape
        epsilonSF = np.full(it, np.nan)
        for i in range(it):
            ij = np.argmin(np.abs(self.mDATA["mab"][i,:]-zref))
            epsilonSF[i] = self.mDATA["SF"]["epsilonMP"][ij,i]
            
        x = np.copy(eps_LOW)
        y = np.copy(epsilonSF)
        iigood = (R2_LOW>0.7) & (np.isfinite(eps_LOW))
        x = x[iigood]
        y = y[iigood]
        ii = np.argsort(x)
        x = x[ii]
        y = y[ii]
        x = np.log(x)
        y = np.log(y)
        jj = (np.isfinite(x)) & (np.isfinite(y))
        x = x[jj]
        y = y[jj]
        p = np.polyfit(x,y,1)
        pp = np.poly1d(p)
        DY = pp(x) - y
        R2 = 1 - np.var( DY)/ np.var( y )
        print R2
        ymod = np.exp(pp(x))
        x = np.exp(x)
        B = p[0]
        A = np.exp(p[1])

        eqq = "$\\varepsilon_{SF} = %1.2e \\times \\varepsilon_{LOW} ^{%1.2f}$\n$R^2 = %1.3f$"%(A,B,R2)
        
        fig, ax = plt.subplots()
        l1,=ax.plot(eps_LOW, self.mDATA["mSF"]["epsilon"],".", color = "gray")
        ax.set_xlim((1e-11,1e-6))
        ax.set_ylim((1e-11,1e-6))
        l2,=ax.plot( [1e-11,1e-6], [1e-11,1e-6],"k-")
        l3,=ax.plot(x, ymod, color = "darkblue")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.annotate(eqq, xy = (0.05,0.85), xycoords = "axes fraction", color = l3.get_color())
        ax.set_xlabel("$\\varepsilon_{LOW}$ [m$^2$ s$^{-3}$]")
        ax.set_ylabel("$\\varepsilon_{SF}$ [m$^2$ s$^{-3}$]")
        ax.set_title("%s %s"%(self.deployname, date_str))
        plt.savefig("epsilon_LOW_vs_SF_%s.png"%(self.deployname), dpi = 300)
        plt.close(fig)


    def SF_vs_vref(self, corrected = False):
        vref = np.nansum(self.mDATA["v_ref"][0:-1,:]**2, axis = 0)**0.5
        date_str = self.mDATA["dates"][0].strftime("%d %b %Y") + " to " + self.mDATA["dates"][-1].strftime("%d %b %Y")
        it, iz = self.mDATA["u"].shape
        epsilonSF = np.full(it, np.nan)
        for i in range(it):
            ij = np.argmin(np.abs(self.mDATA["mab"][i,:]-self.mDATA["z_ref"]))
            epsilonSF[i] = self.mDATA["SF"]["epsilonMP"][ij,i]
        
            
        x = np.copy(vref)
        y = np.copy(epsilonSF)
        ii = np.argsort(x)
        x = x[ii]
        y = y[ii]
        jj = (np.isfinite(x)) & (np.isfinite(y))
        x = x[jj]
        y = y[jj]
        x = np.log(x)
        y = np.log(y)
        p = np.polyfit(x,y,1)
        pp = np.poly1d(p)
        DY = pp(x) - y
        R2 = 1 - np.var( DY)/ np.var( y )
        print R2
        ymod = np.exp(pp(x))
        x = np.exp(x)
        B = p[0]
        A = np.exp(p[1])

        eqq = "$\\varepsilon_{SF} = %1.2e \\times v_{ref} ^{%1.2f}$\n$R^2 = %1.3f$"%(A,B,R2)
        
        fig, ax = plt.subplots()
        l1,=ax.plot(vref*100, self.mDATA["mSF"]["epsilon"],".", color = "gray")
        #ax.set_xlim((1e-11,1e-6))
        #ax.set_ylim((1e-11,1e-6))
        #l2,=ax.plot( [1e-11,1e-6], [1e-11,1e-6],"k-")
        l3,=ax.plot(x*100, ymod, color = "darkblue")
        ax.plot(x*100,1e-3*x**3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.annotate(eqq, xy = (0.05,0.85), xycoords = "axes fraction", color = l3.get_color())
        ax.set_xlabel("$v_{\mathrm{%1.1f m}}$ [cm s$^{-1}$]"%self.mDATA["z_ref"])
        ax.set_ylabel("$\\varepsilon_{SF}$ [m$^2$ s$^{-3}$]")
        ax.set_title("%s %s"%(self.deployname, date_str))
        plt.savefig("vref_vs_SF_%s.png"%(self.deployname), dpi = 300)
        plt.close(fig)
