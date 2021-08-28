import numpy as np
import scipy.linalg as LA
import scipy.interpolate as intrp
import matplotlib.pyplot as plt

def vertical_modes(d, Nsq, NM = 0, method = "klinck"):
    #calculates normal baroclinic modes
    if method != "kundu" and method != "klinck":
        print "Invalid method, set to klinck"
        method = "klinck"
    iif = np.isfinite(Nsq)
    d = d[iif]
    Nsq = Nsq[iif]
    if d[0]>0:
        z = np.full( d.size+1, np.nan)
        z[-1] = 0.
        z[0:-1] = -d[::-1]
        N2 = np.full( Nsq.size+1, np.nan)
        N2[-1] = Nsq[0]
        N2[0:-1] = Nsq[::-1]
    else:
        z = np.full( d.size, np.nan)
        N2 = np.full( d.size+1, np.nan)
        z = - d[::-1]
        N2 = Nsq[::-1]

    Nd = z.size
    A = np.full( (Nd,Nd), 0.)
    B = np.full( (Nd,Nd), 0.)
    if method == "klinck":
        #calculates eigen vectors for w and then derivates to get v
        for k in range(1,Nd-1):
            dzk =   0.5*(z[k+1] - z[k-1])
            dzkp  =   z[k+1] - z[k] 
            dzkm = z[k] - z[k-1]
            #Klinck
            A[k,k-1] =  1/dzk*1/dzkm
            A[k,k] =  - 1/dzk*( 1/dzkm + 1/dzkp )
            A[k,k+1] =   1/dzk*1/dzkp
        A[0,0] =  1.
        A[-1,-1] =  1.
        for k in range(Nd):
            B[k,k] =   -N2[k]
        lb, w = LA.eig(A, B)
        lb = np.real(lb)
        w = np.real(w)
        #gets eigenvectors for velocities and p
        v = np.full(w.shape, 0.)
        for k in range(Nd):
            v[1:-1,k] = (w[2:,k] - w[0:-2,k])/(z[2:] - z[0:-2])
            v[0,k] = (w[1,k] - w[0,k])/(z[1] - z[0])
            v[-1,k] = (w[-1,k] - w[-2,k])/(z[-1] - z[-2])
        v = -v #to be consistent with kundu method
            
    elif method == "kundu":
        #calculates directly eigenvectors for v
        for k in range(1,Nd-1):
            dzk =   0.5*(z[k+1] - z[k-1])
            dzkp  =   z[k+1] - z[k] 
            dzkm = z[k] - z[k-1]
            N2km = 0.5*N2[k-1] + 0.5*N2[k]
            N2kp = 0.5*N2[k  ] + 0.5*N2[k+1]
            A[k,k-1] = 1/dzk/dzkm/N2km
            A[k,k] =  - 1/dzk*( 1/dzkm/N2km + 1/dzkp/N2kp )
            A[k,k+1] = 1/dzk/dzkp/N2kp
            B[k,k] = -1.
        A[0,1] =  1/(z[1] - z[0])
        A[0,0] =- 1/(z[1] - z[0])
        A[-1,-1] = 1/(z[-1] - z[-2]) + N2[-1]/(4*9.81)
        A[-1,-2] =  - 1/(z[-1] - z[-2]) + N2[-2]/(4*9.81)
        lb, v = LA.eig(A, B)
        lb = np.real(lb)
        v = np.real(v)
        #calculate w modes
        w = np.full( v.shape, 0. )
        for k in range(Nd):
            w[1:,k] = np.cumsum( 0.5*(v[1:,k] + v[:-1,k])*(z[1:] - z[:-1]))
        w = -w #make it equivalent to Klink

    #normalize
    for k in range(Nd):
        intv2 = (np.trapz(v[:,k]**2, x = z))**0.5
        v[:,k] = v[:,k]/intv2
        #uses N2 to normalize potential energy
        intw2 = (np.trapz(N2*w[:,k]**2, x = z))**0.5
        w[:,k] = w[:,k]/intw2
    #gets selected modes    
    ipos = lb>1e-10
    lb = lb[ipos]
    v = v[:,ipos]
    w = w[:,ipos]
    isort = np.argsort(lb)
    lb = lb[isort]
    v = v[:,isort]
    w = w[:,isort]
    #inverts depths
    v = v[::-1,:]
    w = w[::-1,:]
    
    if NM == 0:
        NM = np.sum(ipos)-1
    if method == "klinck":
        lb = lb[0:NM]
        v = v[:,0:NM]
        w = w[:,0:NM]
    elif method == "kundu":
        lb = lb[1:NM+1]
        v = v[:,1:NM+1]
        w = w[:,1:NM+1]

    #velocity
    #ce = (1/lb)**0.5

    return lb, v, w


def __fits_modes_progressive__( zY, Y, zX, X,  min_dat = 0.25, KEfactor = 3):
    #removes the barotropic component
    DZ = zX[-1] - zX[0]
    mY = np.nanmean(Y)#*float(Y.size)/float(np.sum(np.isfinite(Y)))
    Abt = mY*np.sqrt(DZ)
    Y = Y-mY

    #total kinetic energy
    KE = np.nanmean(Y**2)*DZ

    #initialises modes
    Nm = X.shape[1]
    A = np.full(Nm+1, 0.)
    Ymod = np.full((zY.size,Nm+1), np.nan)
    

    #removes nan from modes
    X0 = np.copy(X)
    zX0 = np.copy(zX)
    jj0 = np.isfinite(np.sum(X, axis = 1))
    X = X[jj0,:]
    zX = zX[jj0]
    
    
    #removes nan from velocities and projects in the same grid
    jj = np.isfinite(Y)
    if np.sum(jj)<min_dat*jj.size:
        return A, Ymod.T, np.nan
    Y0 = Y[jj]
    zY0 = zY[jj]
    intY = intrp.interp1d(zY0, Y0, bounds_error = False)
    Y = intY(zX)
    jj = np.isfinite(Y)
    Y = Y[jj]
    X = X[jj,:]

    A[0] = Abt
    A0 = A[1:]
    for i in range(0,Nm):
        XX = X[:,0:i+1]
        invXTX = np.linalg.inv(np.dot(XX.T,XX))
        A1 = np.dot(np.dot(invXTX, XX.T),Y)
        KE1 = 0.5* ( np.sum(A1**2) + Abt**2)
        if KE1>KEfactor*KE:
            break
        else:
            A0[0:i+1] = A1
    A[1:] = A0

    #reconstructs velocities
    Ymod0 = np.full((zX0.size,Nm+1), np.nan)

    Ymod0[jj0,:] = np.c_[np.ones((np.sum(jj0),1))*mY,A[1:]*X0[jj0,:]]
    for i in range(Nm+1):
        ii = np.isfinite(Ymod0[:,i])
        intf = intrp.interp1d(zX0[ii], Ymod0[ii,i], bounds_error = False  )
        Ymod[:,i] = intf( zY)

    Ymod = Ymod.T
    #mod = sm.OLS(Y,X).fit()
    #A = mod.params
    return A, Ymod, KE


def __fits_modes_progressive_fitBT__( zY, Y, zX, X,  min_dat = 0.25, KEfactor = 3):
    #removes the barotropic component
    DZ = zX[-1] - zX[0]
    X = np.c_[np.full((X.shape[0],1), 1./np.sqrt(DZ)),X]
        
    #total kinetic energy
    KE = np.nanmean(Y**2)*DZ

    #initialises modes
    Nm = X.shape[1]
    A = np.full(Nm, 0.)
    Ymod = np.full((zY.size,Nm), np.nan)
    

    #removes nan from modes
    X0 = np.copy(X)
    zX0 = np.copy(zX)
    jj0 = np.isfinite(np.sum(X, axis = 1))
    X = X[jj0,:]
    zX = zX[jj0]
    
    
    #removes nan from velocities and projects in the same grid
    jj = np.isfinite(Y)
    if np.sum(jj)<min_dat*jj.size:
        return A, Ymod.T, np.nan
    Y0 = Y[jj]
    zY0 = zY[jj]
    intY = intrp.interp1d(zY0, Y0, bounds_error = False)
    Y = intY(zX)
    jj = np.isfinite(Y)
    Y = Y[jj]
    X = X[jj,:]

    for i in range(0,Nm):
        XX = X[:,0:i+1]
        invXTX = np.linalg.inv(np.dot(XX.T,XX))
        A1 = np.dot(np.dot(invXTX, XX.T),Y)
        KE1 = 0.5* ( np.sum(A1**2) )
        if i>0 and KE1>KEfactor*KE:
            break
        else:
            A[0:i+1] = A1

    #reconstructs velocities
    Ymod0 = np.full((zX0.size,Nm), np.nan)

    Ymod0[jj0,:] = A*X0[jj0,:]
    for i in range(Nm):
        ii = np.isfinite(Ymod0[:,i])
        intf = intrp.interp1d(zX0[ii], Ymod0[ii,i], bounds_error = False  )
        Ymod[:,i] = intf( zY)

    Ymod = Ymod.T
    #mod = sm.OLS(Y,X).fit()
    #A = mod.params
    return A, Ymod,KE




