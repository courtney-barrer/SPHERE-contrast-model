#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 17:23:33 2020

@author: benjamin courtney-barrer
"""



import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from astropy.io import fits

coef = pd.read_csv('alphas2.csv',index_col=0)
coef.columns = coef.columns.astype(int)
intercept = pd.read_csv('intercepts2.csv',index_col=0).T
intercept.columns = intercept.columns.astype(int)

CI95 = pd.read_csv('CI95_2.csv',index_col=0)
CI95.columns = CI95.columns.astype(int)

mag2flux_coef = pd.read_csv('flux_model_coefficients.csv',index_col=0) 

offset_dict = {}
offset_dict['wo'] = pd.read_csv('wo_offsets.csv',index_col=0).T
offset_dict['AOfreq'] = pd.read_csv('AOfrq_offsets.csv',index_col=0).T
offset_dict['wfssf'] =pd.read_csv('wfssf_offsets.csv',index_col=0).T

#coef_cov = fits.open('alpha_covariance2.fits')
#coef_std = pd.read_csv('alphas_std.csv',index_col=0)



def select_AO_freq(Gmag):
    #you should get this value from the data header if available.. it won't 
    #always follow these rules:"
    if Gmag <= 10:
        ao_freq = 1380
    elif (Gmag <= 12) & (Gmag > 10):
        ao_freq = 600
    elif (Gmag > 12):
        ao_freq = 300
        
    return(ao_freq)


    
def Gmag2adu(Gmag, AOfreq, airmass=1, LP780=False):
    """
    

    Parameters
    ----------
    Gmag : TYPE
        DESCRIPTION.
    AOfreq : TYPE
        DESCRIPTION.
    airmass : TYPE, optional
        DESCRIPTION. The default is 1.
    LP780 : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    a0,a1,a2 = mag2flux_coef['wfs (G)']
    
    if LP780: #if using the LP780 filter    
        G_flux_adu = 0.3 * 10**((Gmag - a1 * airmass - a2) / (2.5 * a0))  #model
        
    else:
        G_flux_adu = 10**((Gmag - a1 * airmass - a2) / (2.5 * a0))  #model
    
    adu = 0.3 * 1/AOfreq * G_flux_adu
    
    return(adu)


#h band flux ND filter to be scaled by ND filter used in SCIENCE BAND
def Hmag2adu(Hmag=7, ND_filter=0, airmass=1, exptime=32, coadded=64):
    
    a0,a1,a2 = mag2flux_coef['H2']
    
    H_flux_adu = 10**-ND_filter * 10**((Hmag - a1 * airmass - a2) / (2.5 * a0))  #model

    adu = H_flux_adu * exptime * round(coadded/exptime)
    
    return(adu)




def create_contrast_curve(Gmag=6,Hmag = 6, tau0=3,seeing=0.8,airmass=1., ND_filter=0, \
                          exptime=32,coadded=64, sky_class = 'CL',wfssf='SMALL',AOfreq=1380,\
                              LP780=False, contrast_in_magnitude=True):
    
    """
    
    - when we implement this it should not have default values as an error checking process
    - also note model includes wfs flux variability scaling of contrast.. maybe to include in real tim contrast calc for comparison, without this add ~0.15 mag on 95CI
    - should talk with RR about NDfilter
    
    returns
    ======
    predicted 5sigma raw contrast (assuming contrast is calculated
    after high pass filtering and processed in same way that the model
    was trained on) with upper and lower 95CI intervals from out-of-sample 
    tested residuals. 
    
    output format is a tuple of panda series indexed by radial coordinate
    
    parameters
    ======
    Gmag = G band magnitude of target star (mag - float)
    
    Hmag = H band magnitude of target star (mag - float)
    
    tau0 = atmospheric coherence time (ms - float) (hint: milliseconds.. e.g. input 3, not input 0.003)
    
    seeing = atmospheric seeing (arcsec - float)
    
    ND_filter = ND filter applied (unitless - int) (e.g. ND_filter = 2 corresponds to attentuation of 10^-2 in image flux)
    
    airmass = estimated target airmass during observation (unitless - float)
    
    exptime = science exposure time (sec - int)
    
    coadded = total time of stacked images (sec - int)
        
    sky_class = weather officer sky classification (unitless - str) (hint 'PH', 'CL', 'TN', 'TK', 'CY')
    
    wfssf= wavefront sensor spatial filter (unitless - str) (hint check header 'HIERARCH ESO INS4 OPTI22 NAME' .. should be 'SMALL', 'MEDIUM', 'LARGE'
    
    AO_freq= AO frequency (Hz - int), if set as NONE AO freq will be estimated from star magnitude #
    
    LP780 = is the WFS using the LP780 filter? (T/F - boolen) (hint: check header 'HIERARCH ESO INS4 FILT3 NAME') 
    
    contrast_in_magnitude = do you want the output contrast curve in units of magnitude (T/F - boolean)
    
    ***
    NOTE: this model was developed with coadded = 64s, this value should only 
    be changed in the case that exptime > 64s in which case set coadded = exptime)
    (e.g. if exptime = 32s, coadded=64s  then round(64/32) = 2 images will be
    stacked for contrast calculation)
    ***

    -requires the following files.. (i should probably condense these into 1..)
    'alphas2.csv', 'intercepts2.csv','CI95_2.csv','flux_model_coefficients.csv','wo_offsets.csv','AOfrq_offsets.csv','wfssf_offsets.csv'


    """
    
    
    M = dict() #to hold variables, we explicitly name them to avoid confusion
    
    try:
    #set up parameters
        
        if AOfreq not in [1380,600,300] : #if AO freq not defined we estimate it from star magnitude
            print('WARNING: incorrect AO frequency provided. Try use either 1380,600,300 \
                  we will try estimate it from star magnitude for now')       
            AOfreq = select_AO_freq(Gmag)
            
        r0 = 0.98 * 500e-9/(seeing*np.pi/180/3600) 
        tau_ratio = tau0*1e-3 * AOfreq
        SNR_wfs = np.sqrt(Gmag2adu(Gmag, AOfreq, airmass, LP780=False))
        H2_shot = 1/np.sqrt(Hmag2adu(Hmag, ND_filter, airmass, exptime, coadded))
    
    
        #store them in our dict 
        
        M["SNR"] = np.log10( SNR_wfs )
        M["servo"] =  np.log10( tau_ratio )
        M["H2_shot"] = np.log10( H2_shot )
        M["D/r0"] = np.log10(8/r0) 
        M = -2.5 * pd.Series(M)
        
        
    except:
        
        raise TypeError("issue calculating model parameters")
    
    #init dicts for the output
    contrast = dict()
    err = dict()
    
    #check our M dictionary and coef have matching names and order for matrix multiplication 
    if not sum(coef.index != M.keys()): #(if were not going to multipy the wrong things together) 
    
        if contrast_in_magnitude: # if we want contrast in mag
             
            Cm = coef.T @ M + intercept - offset_dict['wo'][sky_class][0] \
                - offset_dict['wfssf'][wfssf][0] - offset_dict['AOfreq'][AOfreq][0] 
            Cm_max = Cm.values[0] - CI95.values[0]
            Cm_min = Cm.values[0] - CI95.values[1]
            
            uncert = pd.DataFrame([Cm_min,Cm_max],index=CI95.index,columns=CI95.columns)
            
            contrast = Cm
            
            
        elif not contrast_in_magnitude: #if we dont want contrast in mag
            Cm = (coef.T @ M + intercept) - offset_dict['wo'][sky_class][0] \
                - offset_dict['wfssf'][wfssf][0] - offset_dict['AOfreq'][AOfreq][0]
            Cm_max = Cm.values[0] - CI95.values[0]
            Cm_min = Cm.values[0] - CI95.values[1]
            
            contrast = 10**( - Cm / 2.5)
        
            uncert1 = 10**( - (Cm_max / 2.5))
            uncert2 = 10**( - (Cm_min / 2.5))
            uncert = pd.DataFrame([uncert2,uncert1],index=CI95.index,columns=CI95.columns)
            
        return(contrast,uncert)
    
    else:
        
        raise TypeError("Cannot perform matrix multiplication. \
                        There is mismatch in variable names and/or order between\
                        the M dictionary and coef dataframe. This\
                        could be caused if the coef file has been changed")

    
#TEST EXAMPLE
c1,er1 = create_contrast_curve(Hmag=5,tau0=8,seeing=0.5,contrast_in_magnitude=False)
c2,er2 = create_contrast_curve(Hmag=5,tau0=3,seeing=1.2, contrast_in_magnitude=True)

fig,ax = plt.subplots(1,2,sharey=False,figsize=(18,9))
ax[0].plot(c1.columns,c1.values[0],color='k',lw=0.8,label='expected contrast')
ax[0].plot(c1.columns, er1.values[0], color='r',linestyle='--',label='bad data limit')
ax[0].fill_between(c1.columns, er1.values[0], er1.values[1],alpha=0.2,label='95CI region')
ax[0].set_yscale('log')
ax[0].legend(loc='upper right',fontsize=18)
ax[0].set_ylabel(r'5$\sigma$ raw contrast',fontsize=18)

ax[1].plot(c2.columns,c2.values[0],color='k',lw=0.8,label='expected contrast')
ax[1].plot(c2.columns,er2.values[0], color='r',linestyle='--',label='bad data limit')
ax[1].fill_between(c2.columns, er2.values[0], er2.values[1],alpha=0.2, label='95CI region')
ax[1].legend(loc='upper right',fontsize=18)
ax[1].set_ylabel(r'5$\sigma$ raw contrast'+'\n(magnitude)',fontsize=18)
ax[1].invert_yaxis()


for axx in ax:
    axx.tick_params(axis='both',labelsize=18)
    axx.set_xlabel('radius (mas)',fontsize=18)
    
    
