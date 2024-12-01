"""VISCOUS/INVISCID INTERACTION CODE
Logan Halstrom
EAE 127
UCD
CREATED:  30 OCT 2014
MODIFIED: 09 NOV 2017
DESCRIPTION: Viscous/Inviscid Interaction (VIvI)
Simulate potential flow over an airfoil and model viscous effects
iteratively by computing boundary layer displacement thickness from
Blausius solution for flat plate using potential flow pressure distribution.
Resulting displacement thickness is added to airfoil geometry and process is
repeated.
"""

import numpy as np
import pandas as pd

import sys
sys.path.append('../../utils') #Add utility library to path
import pyxfoil
import mses

def Cp2V(Cp, Vinf):
    """Return velocity magnitude computed from pressure coefficient using
    Bernoulli's principle
    """
    return np.sqrt(1 - Cp) * Vinf


def MomThickness(x, u, Vinf, mu=1.79E-5, rho=1.225):
    """Compute momentum thickness at a given x-location along chord from
    equation derived by applying Blausius solution for flat plat to
    von Karmen integral relation for boundary layer.
    x --> coordinates along chord
    u --> surface velocity distribution as f(x) (from pot. flow soln)
    """
    #Compute momentum thickness at each X location (via von Karman relation)
    n = len(x) #number of x-locations
    c = max(x) - min(x)
    mom_thk = np.zeros(n) #initialize momentum thickness vector
    #Loop through each x-location
    for i in range(n):
        #integral portion is of surface velocity from LE up to current X
        integral = np.trapz(u[0:i] ** 8.210, x[0:i] * c)
        #momentum thickness from von karman
        mom_thk[i] = np.sqrt( (0.440 * mu) / (rho * u[i] ** 9.210) * integral )
    return mom_thk

def FixTE(x, z, tol=1):
    """The displacement thickness due to the boundary layer at the
    trailing edge is so large that the TE becomes too blunt to solve correctly
    in a potential flow solver.  This function modifies an airfoil geometry
    so that the trailing edge is marginally sharper.
    x,z --> coordinates for ONE side of an airfoil (upper OR lower)
    tol --> surface slope tolerance (currently not used)
    """
    c = max(x) - min(x) #chord length

    #USE DATAFRAME FOR CONVENIENCE
    df = pd.DataFrame({'x' : x, 'z' : z})

    #FIND INDEX OF 90% CHORD LENGTH
    ind = df.loc[df.x >= c*0.9].index[0]

    # #FIND INDEX WHERE SURFACE SLOPE EXCEDES THRESHOLD
    # #Flip surface to look like uppper surface
    # surfactor = -1 if max(abs(df.z)) > max(df.z) else 1
    # df['z'] *= surfactor
    # #Compute surface slope
    # df['dzdx'] = np.gradient(df.z, df.x)
    # #Find index where slope excedes tolerance
    # ind = df.loc[(df.dzdx > -tol) & (df.x > c*0.75)].index[0]
    # print(df.loc[(df.dzdx > -tol) & (df.x > c*0.75)].head())
    # print(df.loc[(df.dzdx > -tol) & (df.x > c*0.75)].tail())
    # #Return surface to proper orientation
    # df['z'] *= surfactor

    #MAKE SURFACE FROM IND TO TE A STRAIGHT LINE (WEDGE)
    #number of points back to calc slope
    back = 5
    #Slope of airfoil surface at ind
    m = (df.loc[ind, 'z'] - df.loc[ind-back, 'z']) / (
         df.loc[ind, 'x'] - df.loc[ind-back, 'x'])
    #Fill rear surface points with straight line
    df.loc[ind:, 'z'] = (df.loc[ind:, 'x'] - df.loc[ind, 'x']) * m + df.loc[ind, 'z']

    return np.array(df['z'])


def ViscousCalcs(df, Vinf, mu, rho):
    """For a SINGLE surface (upper or lower):
    From an inviscid panel method solution,
    compute momentum thickness/displacement thickness from Blausius flat plate/
    von Karman integral relation.
    Add disp. thickness to original geometry to produce "displaced" geometry.
    Compute wall shear stress from momentum thickness.
    (Viscous portion of VIvI)
    df --> dataframe containing parameters for SINGLE surface
    """
    #COMPUTE DISPLACEMENT THICKNESS
    #Compute surface velocity from surface pressure distribution
    df['U'] = Cp2V(df.Cp, Vinf)
    #Compute momentum thickness from Blausius/von Karman
    df['theta'] = MomThickness(df.x, df.U, Vinf, mu, rho)
    #Compute displacement thickness
    H = 2.605 #Shape Parameter for Blausius distribution
    df['delta_star'] = H * df.theta

    #ADD DISP. THICKNESS TO GEOMETRY (POTENTIAL FLOW DIV. STREAMLINE)
    #add or subtract delta* depending on surface side
    surfactor = -1 if max(abs(df.z)) > max(df.z) else 1
    #add disp. thickness to airfoil
    df['zdisp'] = df['z'] + df['delta_star'] * surfactor
    #Sharpen TE for better panel solution
    df['zdisp'] = FixTE(np.array(df['x']), np.array(df['zdisp']))

    #COMPUTE WALL SHEAR STRESS
    print('COMPUTING WALL SHEAR STRESS IN MAIN PROGRAM, NOT FOR RELEASE')
    df['tau'] = 0.220 * mu * df.U / df.theta

    return df


def VIvI(foilfile, alfa, niter=0, Vinf=1, mu=1.79E-5, rho=1.225):
    """SINGLE ITERATION of Viscous/Inviscid Interaction:
    Run invicid panel method simulation with provided geometry,
    compute displacement thickness of BL from surface pressure distribution,
    return "displaced" geometry, which can be used for another VIvI iteration
    foilfile --> path to airfoil geometry file
    alfa --> Single AoA at which to perform VIvI
    niter --> number for current iteration of VIvI
    Vinf, mu, rho --> freestream conditions
    """

    #GET INVISCID PANEL METHOD SURFACE PRESSURE SOLUTION

    #name and settings for specific iterations
    if niter == 0:
        #0th iteration
        num = ''     #no appended number
        pane = False #no panel conditioning
    else:
        #non-0th iteration
        num = '_{}'.format(niter) #append iteration number
        pane = True #smooth panels

    # num = '' if niter == 0 else '_{}'.format(niter)
    #name of current solution depends on VIvI iteration number
    geomfile = 'Data/{}/{}{}.dat'.format(foilfile, foilfile, num)

    #Run inviscid panel method
    invis = pyxfoil.GetPolar(geomfile, False, [alfa], Re=0,
                                SaveCP=True, pane=pane, quiet=True)
    #Read panel method pressure distribution
    pressfile = 'Data/{}{}/{}{}_surfCP_Re0.00e+00a{:1.1f}.dat'.format(
                                            foilfile, num, foilfile, num, alfa)
    xCp = pyxfoil.ReadXfoilSurfPress(pressfile)
    #check of XFOIL solution failed
    if xCp.isnull().values.any():
        sys.exit('XFOIL solution failed for VIvI iteration: {}'.format(niter))

    #SPLIT MSES DATA INTO UPPER/LOWER SURFACES
    #Read current airfoil coordinates
    dfmses = pyxfoil.ReadXfoilAirfoilGeom(geomfile)
    #Compute chord length
    c = max(dfmses.x) - min(dfmses.x)
    #interpolate sufaces to this x
    xnew = np.linspace(0, c, 495)
    #store info for each surface in dictionary
    dfs = {'up' : pd.DataFrame(), 'lo' : pd.DataFrame()}
    #both sides have same x
    dfs['up']['x'], dfs['lo']['x'] = xnew, xnew
    #split surfaces
    dfs['up']['z'], dfs['lo']['z'] = mses.MsesInterp(xnew, dfmses.x, dfmses.z)
    #split surface pressure
    dfs['up']['Cp'], dfs['lo']['Cp'] = mses.MsesInterp(xnew, xCp.x, xCp.Cp)
     #save original geometry for plotting later
    dfog = dfs.copy()

    #VIVI COMPUTATIONS FOR EACH SURFACE
    for side in ['up', 'lo']:
        #compute displacement thickness from pressure distribution, add to geom
        dfs[side] = ViscousCalcs(dfs[side], Vinf, mu, rho)

    #MERGE AND SAVE DISPLACED GEOMETRY
    #Put displaced geometry into MSES format
    dfdisp = pd.DataFrame()
    dfdisp['x'], dfdisp['z'] = mses.MsesMerge(dfs['lo']['x'], dfs['up']['x'],
                                              dfs['lo']['zdisp'], dfs['up']['zdisp'])
    #Save displaced geometry in XFOIL format
    geomfile = 'Data/{}/{}_{}.dat'.format(foilfile, foilfile, niter+1)
    pyxfoil.WriteXfoilFile(geomfile, dfdisp.x, dfdisp.z)

    # #RETURN UPPER/LOWER SURFACE MOMENTUM THICKNESS AND SURFACE VELOCITY DISTS.
    # return dfs['up']['theta'], dfs['up']['U'], dfs['lo']['theta'], dfs['lo']['U']

    #RETURN ALL DATA
    print('RETURNING ALL DATA, NOT FOR RELEASE')
    return dfs