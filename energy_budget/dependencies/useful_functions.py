## USEFUL FUNCTIONS
#B. Fernandez Castro 2020.03.03

import numpy as np
import sys
import matplotlib.pyplot as plt

def moving_average(x,n, window = "flat"):
    if n%2 == 0:
        n+=1
    N = x.size
    cx = np.full(x.size, np.nan)
    for i in range(N):
        ii = np.arange(i-n//2, i+n//2+1,1)
        if window == "flat":
            ww = np.ones(ii.size)
        elif window == "gauss":
            xx = ii - i
            
            ww = np.exp(- xx**2/(float(n)/4)**2 )
        elif window == "hanning":
            ww = np.hanning(ii.size)
        ww = ww[ (ii>=0) & (ii<N)]
        ii = ii[ (ii>=0) & (ii<N)]
        
        kk = np.isfinite(x[ii])
        if np.sum(kk)<0.25*ii.size:
            continue
        cx[i] = np.sum(x[ii[kk]]*ww[kk])/np.sum(ww[kk])
    return cx


def linfit_modelII(X,Y):
    # lsqfitma.m                                     by:  Edward T Peltzer, MBARI
    #                                                revised:  2016 Mar 17.
    # 
    # M-file to calculate a "MODEL-2" least squares fit.
    #
    #     The line is fit by MINIMIZING the NORMAL deviates.
    #
    #     The equation of the line is:     y = mx + b.
    #
    #     This line is called the MAJOR AXIS.  All points are given EQUAL
    #       weight.  The units and range for X and Y must be the same.
    #     Equations are from York (1966) Canad. J. Phys. 44: 1079-1086;
    #       re-written from Kermack & Haldane (1950) Biometrika 37: 30-41;
    #       after a derivation by Pearson (1901) Phil. Mag. V2(6): 559-572.
    #
    #     Data are input and output as follows:
    #
    #	    [m,b,r,sm,sb] = lsqfitma(X,Y)
    #
    #             X    =    x data (vector)
    #             Y    =    y data (vector)
    #
    #             m    =    slope
    #             b    =    y-intercept
    #             r    =    correlation coefficient
    #             sm   =    standard deviation of the slope
    #             sb   =    standard deviation of the y-intercept
    #
    #     Note that the equation passes through the centroid:  (x-mean, y-mean)
    
    iif = np.isfinite(X+Y)
    X = X[iif]
    Y = Y[iif]
    n = float(np.sum(iif))

    Sx = np.sum(X)
    Sy = np.sum(Y)
    xbar = Sx/n
    ybar = Sy/n
    U = X - xbar
    V = Y - ybar

    Suv = np.sum(U*V)
    Suu = np.sum(U*U)
    Svv = np.sum(V*V)

    sigx = np.sqrt(Suu/(n-1))
    sigy = np.sqrt(Svv/(n-1))

    m = (Svv - Suu + np.sqrt(((Svv-Suu)**2) + (4*Suv**2)))/(2*Suv)
    b = ybar - m*xbar
    r = Suv/np.sqrt( Suu * Svv )

    sm = (m/r)*np.sqrt((1 - r**2)/n)
    sb1 = (sigy - sigx*m)**2
    sb2 = (2*sigx*sigy) + ((xbar**2*m*(1+r))/r**2)
    sb = np.sqrt( (sb1 + ((1-r)*m*sb2))/n)

    return np.array([m, b]), np.array([sm, sb]), r

def FCD_2d(x, y, axis = 0):
    if x.ndim != 2 or y.ndim !=2:
        sys.exit("Invalid dimensions")
    if axis != 0 and axis != 1:
        sys.exit("Invalid axis")
    if axis == 1:
        x = x.T
        y = y.T
    dy = np.full(y.shape,np.nan)    
    for i in range(x.shape[1]):
        dy[:,i] = first_centered_differences(x[:,i], y[:,i])
        
    if axis == 1:
        dy = dy.T
    return dy

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

def centered_differences(x, y, fill = False):
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
    dy0[1:-1] = (y0[2:] - y0[0:-2])/(x0[2:]- x0[0:-2])
    
    dy[iif] = dy0

    if fill:
        dy[0:iif[0]] = dy[iif[0]]
        dy[iif[-1]+1:] = dy[iif[-1]]
    return dy

def mixed_layer_depth(z0, den0, Dd = 0.05, crit = "diff"):
    #Mixed layer calculation
    if crit != "diff" and crit != "grad":
        crit = "diff"
        print "Incorrect criterion, set to diff"
    c,f = den0.shape
    MLD = np.full(f, np.nan)
    for i in range(f):
        if z0.ndim ==1:
            z = np.copy(z0)
        else:
            z = z0[:,i]
        den = np.sort(den0[:,i])

        iif = np.isfinite(den+z)
        if np.sum(iif)<=1:
            continue
        den = den[iif]
        z = z[iif]

        if crit == "diff":
            sden = den[0]
            denp = den-sden
            imld = np.where( denp>=Dd )[0]
            if imld.size == 0:
                MLD[i] = np.max(z)
            elif imld[0]>0:
                imld = imld[0]
                z2 = z[imld]
                z1 = z[imld-1]
                denp2 = denp[imld]
                denp1 = denp[imld-1]
                MLD[i] = (z2-z1)/(denp2-denp1)*(Dd - denp1) + z1
            else:
                MLD[i] = np.max(z)
                #MLD[i] = z0[0,i]
                
        elif crit == "grad":
            grden = np.abs(first_centered_differences(z, den))
            imld = np.where(grden>=Dd)[0]
            if imld.size == 0:
                MLD[i] = np.max(z)
            elif imld[0]>0:
                imld = imld[0]
                z2 = z[imld]
                z1 = z[imld-1]
                grd2 = grden[imld]
                grd1 = grden[imld-1]
                MLD[i] = (z2-z1)/(grd2-grd1)*(Dd - grd1) + z1
            else:
                MLD[i] = z[0]
                
    return MLD




