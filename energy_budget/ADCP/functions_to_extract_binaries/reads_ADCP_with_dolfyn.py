import dolfyn as dlfn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import datetime as dt
import numpy as np
import pickle
from time_conversion import *
from scipy import io
from get_colormap import get_colormap
CMCV = get_colormap('/home/bieito/Documents/SCIENCE/EPFL/python_functions/WhiteBlueGreenYellowRed.ncmap') 

def rotation_matrix_2d(alpha):
     M = np.zeros((2,2))
     M[0,0] = np.cos(alpha)
     M[0,1] = -np.sin(alpha)
     M[1,0] = np.sin(alpha)
     M[1,1] = np.cos(alpha)
     return M
     
def rotate_velocity_2d(u,v,alpha):
     u2 = np.full(u.shape, np.nan)
     v2 = np.full(u.shape, np.nan)
     for index in np.ndindex(u.shape):
          M = rotation_matrix_2d(alpha[index]/180*np.pi)
          V = np.array([[u[index],v[index]]]).T
          V2 = np.dot(M,V)
          u2[index] = V2[0]
          v2[index] = V2[1]
     return u2, v2

#Class with all the functions to import and transform ADCP data
class adcp_data():
    def __init__(self,filename, transducer_depth, bottom_depth = 100., up = False, cabled = False,  **kargs):
        dat = dlfn.read(filename)
        print dat
        print dat.signal
        print dat.orient
        self.Er = np.nanmin(dat.signal.echo.astype(float))
        
        CC = np.nanmean(np.nanmean(dat.signal.corr/255.*100, axis = 0), axis = 0)
        igood = np.where(CC>20)[0]
        print igood[0]
        print igood[-1]
        print dat.mpltime.size
        time_good = dat.mpltime[igood[[0,-1]]]   
        date_good = convert_time_to_date(time_good)
        print date_good
        print "Extract desired/good subset of good data"
        date_subset = []
        if "start_date" in kargs:
            start_date = dt.datetime.strptime(kargs["start_date"], "%Y%m%d %H:%M")
            print start_date
            date_subset.append(max([start_date, date_good[0]]))
        else:
            date_subset.append(date_good[0])
    
        if "end_date" in kargs:
            end_date = dt.datetime.strptime(kargs["end_date"], "%Y%m%d %H:%M")
            date_subset.append(min([end_date, date_good[1]]))
        else:
            date_subset.append(date_good[1])
    
        time_subset = convert_date_to_time(date_subset)
        
        dat = dat.subset[
                      (time_subset[0] < dat.mpltime) &
                      (dat.mpltime < time_subset[1])]
        
    
        #saves output
        dat.transducer_depth = transducer_depth
        idot = filename.rfind(".")
        filenameOUT = filename[0:idot+1]+"h5"
        dat.to_hdf5(filenameOUT)
        
        #creates instances in the class
        self.filename = filename[0:idot+1]
        self.cabled = cabled
        self.up = up
        self.bottom_depth = bottom_depth
        self.time = dat.mpltime
        self.date = convert_time_to_date(self.time)
        self.u= dat.u
        self.v=dat.v
        self.w=dat.w
        self.eu = dat.vel[3,:,:]
    
        self.transducer_depth = transducer_depth
        self.range = dat.range
        if not self.up:
            r = transducer_depth+self.range
        else:
            r = transducer_depth-self.range

        self.z0 = r  #*np.cos(20*np.pi/180.) #this was incorrect
        self.beam_angle = float(dat.config.beam_angle)
        self.r = self.z0/np.cos(self.beam_angle*np.pi/180.)
        self.corr = dat.signal.corr.astype(float)/255.
        self.echo = dat.signal.echo.astype(float)
        self.prcnt_gd = dat.signal.prcnt_gd.astype(float)
        self.heading = dat.orient.raw.heading
        self.roll = dat.orient.raw.roll
        self.pitch = dat.orient.raw.pitch
        self.temp = dat.env.temperature_C

        
        self.battery = dat.sys.adc[1,:]
        self.xmit_length = dat.config.xmit_pulse #transmit length in m
        self.beam_freq = dat.config.beam_freq_khz
        
    def saves_data(self, file_out):
        out = {}
        vars_to_save = self.__dict__.keys()
        for var0 in vars_to_save:
            exec("out[\""+var0+"\"] = self."+var0)     
        
        jdot = file_out.rfind('.')
        file_ext = file_out[jdot:]
        
        if file_ext == '.mat':
            print "Saving data to mat file"
            out.pop("date")
            for var in out.keys():
                 if type(out[var]) == np.ma.core.MaskedArray:
                      out[var] = out[var].data
            io.savemat(file_out,out)
        else:
            print "Saving data to pickle file"
            with open(file_out,'w+') as f:
                pickle.dump(out,f)
                
        
    def finds_surface_timeseries(self, irt = 100, factor = 1.5, PLOT = False):
        #looks where is the surface in each step, to be sure of the depth where the instrument is
        #in some of the deployements the instrument mooved to a deeper part of the lake and the surface was lost
        #i dont know how to deal with this yet
        print "Finding surfaces"
        d1,d2,d3 = self.echo.shape
        irt = min([irt,d3])
        self.isurf = np.full( d3, 0 ) #index of the surface for each beam
        self.rsurf = np.full( d3, np.nan) #distance from transducer to the surface
        self.z = np.full( (d2,d3), np.nan ) #this variable is the actual depth of each bin at each time step
        self.watercol = np.full ( (d2,d3), True ) #flags the data that is underwater
        for j in range(d3):
            print "Step %d of %d"%(j+1,d3)
            self.isurf[j] = self.finds_surface_1prof( itime = j, irt = irt, factor = factor,PLOT = PLOT)
            self.watercol[self.isurf[j]+1:,j] = False
            self.rsurf[j] = self.range[self.isurf[j]]
            if not self.up:
                 r = (self.bottom_depth - self.rsurf[j])+self.range
            else:
                 r = self.rsurf[j]-self.range
            self.z[:,j] = r #*np.cos(20*np.pi/180.) #this was incorrect
            
            
    def finds_surface_1prof(self, itime, irt = 100, factor = 1.5, PLOT = False):
        #THis function just finds the surface for a given time step and given accoustic beam
        #It probably needs to be improved
        
        #uses the maximum echo
        d1,d2,d3 = self.echo.shape
        istart = itime-irt//2
        iend = itime+irt//2


        if istart<0:
            iend = iend - istart
            istart = 0
        if iend>d3:
            ic = iend-d3
            iend = d3
            istart = istart-ic

            
        itt = np.arange(istart,iend)
        ECHO = np.nanmedian(np.nanmedian(self.echo[:,:,itt], axis = 0), axis = 1)
        #ECHO = np.nanmedian(self.echo[:,:,itime], axis = 0)
        imin = np.argmin(ECHO)
        if imin >= ECHO.size-1:
            imin = 0
        minECHO = np.min(ECHO)
        ECHO = ECHO - minECHO
        maxECHO = np.max(ECHO[imin:])
        imax = np.argmax(ECHO[imin:])+imin
        isurfA = np.where( (ECHO[:imax+1]<maxECHO/factor) )[0]
        if isurfA.size>0:
            isurfA = isurfA[-1]
        else:
            isurfA = imax
            
        #uses the maximum change in echo
        diffECHO = np.diff(ECHO)
        maxdiffECHO = np.max(diffECHO[imin:])
        imaxdiff = np.where(diffECHO==maxdiffECHO)[0][0]
        isurfB = np.where( diffECHO<=maxdiffECHO)[0]
        if imaxdiff ==0:
            imaxdiff = diffECHO.size
        isurfB = isurfB[:imaxdiff][-1]
     
        
        #gets the minimum of the three
        isurf = int(max([isurfA, isurfB+1]))

        if PLOT:
            dz = self.z0[1]-self.z0[0]
            fig,ax = plt.subplots(1,2, sharey = True)
            ax[0].plot(ECHO, self.z0)
            ax[0].plot(ECHO[isurfA],self.z0[isurfA],"o")
            ax[0].plot(ECHO[isurf],self.z0[isurf],"x")
            
            ax[1].plot(diffECHO, self.z0[:-1]+dz/2.)
            ax[1].plot(diffECHO[isurfB],self.z0[isurfB]+dz/2.,"o")
            isurf0 = min([isurf, diffECHO.size-1])
            ax[1].plot(diffECHO[isurf0],self.z0[isurf0]+dz/2.,"x")        
            ax[1].axvline(maxdiffECHO)
  
            plt.show()
        
        return isurf
    
    
    
    def creates_quality_flag(self, mincor = 0.3, maxcor = 0.7, minpcg = 50, errorfactor = 10):#minpcg = 25, errorfactor = 5):
         #Flags the data according different quality criteria (correlation range, percent good and velocity error)
         #It probably needs to be slighly tunned
         print "Creating quality flag"
         
         d1, d2, d3 = self.corr.shape
         self.quality_flag = np.full ( (d2,d3), True )
         if not hasattr(self,"watercol"):
             self.watercol = np.full(self.quality_flag.shape, True)
         
         for j in range(d3):
              for i in range(d2):
                   flg = 0
                   for k in range(d1):
                        cr = self.corr[k,i,j]
                        if (cr<mincor) or (cr>maxcor):
                             flg +=1
                   pcg = self.prcnt_gd[3,i,j]
                   if pcg<minpcg:
                       flg+=1
                   modW = np.sqrt( self.u[i,j]**2+self.v[i,j]**2+self.w[i,j]**2)**0.5
                   if modW<errorfactor*np.abs(self.eu[i,j]):
                       flg+=1
                   if flg>0: 
                       self.quality_flag[i,j] = False
         self.u = np.ma.masked_array(self.u, mask = ~(self.quality_flag & self.watercol))
         self.v = np.ma.masked_array(self.v, mask = ~(self.quality_flag & self.watercol))
         self.w = np.ma.masked_array(self.w, mask = ~(self.quality_flag & self.watercol))
         
                   
    def rotate_velocity(self,alpha):
        #rotates the velocities by an angle alpha (degrees)
        for index in np.ndindex(self.u.shape):
            M = self.__rotation_matrix_2d__(alpha/180*np.pi)
            V = np.array([[self.u[index],self.v[index]]]).T
            V2 = np.dot(M,V)
            self.u[index] = V2[0]
            self.v[index] = V2[1]
            
    def __rotation_matrix_2d__(self,alpha):
     M = np.zeros((2,2))
     M[0,0] = np.cos(alpha)
     M[0,1] = -np.sin(alpha)
     M[1,0] = np.sin(alpha)
     M[1,1] = np.cos(alpha)
     return M

    def calculates_absolute_backscatter(self, kc = 0.45, PLOT = True, msv = -80, Msv = -55, dsv = 0.5):
         print "Absolute backscatter"
         #parameters
         mbat = 32.
         if self.beam_freq == 300:
              C = -140.87
              if not self.cabled:
                   Pdbw = 14.0
              else:
                   Pdbw = 17.5
              alpha = 0.025
         elif self.beam_freq == 600:
              C = -139.09
              if not self.cabled:
                   Pdbw = 9.0
              else:
                   Pdbw = 12.5
              alpha = 0.098
                       
         R = self.r + 0.25*(self.z0[1]-self.z0[0])/np.cos(self.beam_angle*np.pi/180.)
         self.Sv = np.full(self.echo.shape, np.nan)
         shp = self.echo.shape
         for i in range(shp[0]):
              for j in range(shp[1]):
                   for k in range(shp[2]):
                        self.Sv[i,j,k] = C +10*np.log10((self.temp[k]+273.16)*R[j]**2) - 10*np.log10(self.xmit_length) - Pdbw  + 2.*alpha*R[j]+ 10.*np.log10( 10**(kc*(self.echo[i,j,k]-self.Er)/10.)-1 )
                        if not self.cabled:
                             self.Sv[i,j,k] -= 20*np.log10( self.battery[k]/np.nanmean(self.battery)  )
                        #kc*(self.echo[i,j,k]-self.Er)
                        
         if PLOT:
              fig,ax = plt.subplots(figsize = (10,4))
              cc=ax.contourf(self.date,self.z0, np.nanmean(self.Sv, axis = 0), np.arange(msv,Msv,dsv), extend = "both", cmap = CMCV)
              yl = ax.get_ylim()
              ax.set_ylim((np.nanmax(self.z0),np.nanmin(self.z0)))
              ax.set_ylabel("Depth$_0$[m]")
              cb=plt.colorbar(cc)
              cb.set_label("$S_v$ [db]")
              fig.tight_layout()
              plt.savefig(self.filename+"backscatter.png", dpi = 300)
              plt.close(fig)
             
    def plot_velocity(self):
        #makes a plot of the velocity time-series
        print "Plotting velocity"
        su = np.nanstd(self.u)
        sv = np.nanstd(self.v)
        sw = np.nanstd(self.w)
        eu = np.abs(self.eu)
        seu = np.nanstd(eu)
    
        
        fig, ax = plt.subplots(2,2, sharex = True, sharey = True, figsize = (16,8))
        cc=ax[0,0].contourf(self.date,self.range,self.u, np.linspace(-3*su, 3*su,30), vmin = -3*su, vmax = 3*su, extend = "both", cmap = cm.RdBu_r)
        if hasattr(self, "rsurf"):
            ax[0,0].plot(self.date, self.rsurf, "k")
        plt.colorbar(cc,ax = ax[0,0])
        ax[0,0].set_title("u [m/s]")
        ax[0,0].set_ylabel("Depth$_0$ (m)")
        ax[0,0].set_ylim([np.min(self.range), np.max(self.range)])

        
        cc=ax[0,1].contourf(self.date,self.range,self.v, np.linspace(-3*sv,3*sv,30), vmin = -3*sv, vmax = 3*sv, extend = "both", cmap = cm.RdBu_r)
        if hasattr(self, "rsurf"):
            ax[0,1].plot(self.date, self.rsurf, "k")    
        plt.colorbar(cc,ax = ax[0,1])
        ax[0,1].set_title("v [m/s]")
    
        cc=ax[1,0].contourf(self.date,self.range,self.w, np.linspace(-3*sw, 3*sw, 30),  vmin = -3*sw, vmax = 3*sw, extend = "both", cmap = cm.RdBu_r)
        plt.colorbar(cc,ax = ax[1,0])
        if hasattr(self, "rsurf"):
            ax[1,0].plot(self.date, self.rsurf, "k")
        ax[1,0].set_ylabel("Depth$_0$ (m)")
        ax[1,0].set_title("w [m/s]")
        
        cc=ax[1,1].contourf(self.date,self.range,eu, np.linspace(0, 3*seu,30), vmin = 0, vmax = 3*seu, extend = "both",  cmap = cm.hot_r)
        if hasattr(self, "rsurf"):
            ax[1,1].plot(self.date, self.rsurf, "k")
        plt.colorbar(cc,ax = ax[1,1])
        ax[1,1].set_title("eu [m/s]")

        if not self.up:
            yl = ax[0,0].get_ylim()
            ax[0,0].set_ylim((yl[-1],yl[0]))
        
        fig.tight_layout()
        fig.savefig(self.filename+"_velocities.png", dpi = 300)
        plt.close(fig)
        
    def mean_velocities(self, PLOT = True):   
        #calculates the mean velocities (this could be a final product to show to the public)
        print "Computes and plots mean velocities"
        self.mu = np.nanmean(self.u, axis = 0)
        self.mv = np.nanmean(self.v, axis = 0)
        self.mU = (self.mu**2 + self.mv**2)**0.5
        self.mdir = np.arctan2(self.mv,self.mu)*180/np.pi
        self.mdir[self.mdir<0] = 360 + self.mdir[self.mdir<0]

        if not PLOT:
            return

        fig, ax = plt.subplots(3,1, sharex = True, figsize = (8,6))
        ax[0].plot(self.date, self.mu, label = "$East$",lw = 0.5)
        ax[0].plot(self.date, self.mv, label = "$North$", lw = 0.5)
        ax[0].axhline(0, color = "gray",lw = 0.5)
        ax[0].legend(ncol = 2)
        ax[0].grid(True)
        yl = ax[0].get_ylim()
        ylM = max(np.abs(yl))
        ax[0].set_ylim((-ylM, ylM))
        ax[0].set_ylabel("Velocity [m/s]")
        
        ax[1].plot(self.date, self.mU, label = "$U$", color = "k")
        ax[1].set_ylim((0,ylM))
        ax[1].grid(True)
        ax[1].set_ylabel("Speed [m/s]")
        
        ax[2].plot(self.date, self.mdir,lw =0.5)
        ax[2].set_ylim((0,360))
        ax[2].set_ylabel("Direction [$^{\circ}$ over East]")
        ax[2].grid(True)
        fig.tight_layout()
        fig.savefig(self.filename+"_meanvelocities.png", dpi = 300)
        plt.show()
        plt.close(fig)
