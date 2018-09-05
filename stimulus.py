#!/usr/bin/python
# -*- coding: utf-8 -*-a

# Experiment ALEXIS #116 -------------------------------------------------------
# Fuctions for generating stimuli.
#
# MN 2018-09-03
# ------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt  
import soundfile as sf

import mn # User made functions, make sure mn.py is located in the working dir

def create_stimulus(level = 0, left = True, leadclick = True, cdur = 125, sgain = -11, tau = 4.0, 
                    ild = 10.0, itd = 0.325, fs = 48000, dur = 3840, plot = False, rec = False):
    ''' Create stereo vector with click sounds (lead-lag click pair or single (lag-only) click).      
    
    Keyword arguments:
    level -- Lag-lead ratio, LLR [dB], (float). LEad click always = 0 dB. 
             If tau is set to -1, LLR is the relative level of the single lag-click.
    left -- Decides whether spatial info points left or right. True left; False right. 
            NOTE: If both itd and ild are set to 0, then the signal will point straigt ahead no matter if this parameter is 
            set to True or False      
    leadclick -- Decides if there is a lead click preceeding the lag click. 
              If True, a lead-lag ckicl is created, if False only a lag-click    
    cdur -- Click (square pulse) duration in microseconds, rounded to nearest sample (float)
    sgain -- Peak level of standard sound in dB re max, default = -11 (float)
    tau -- Inter-click interval (ms, rounded to nearest sample). If no lead lcick is present
           (leadclick = False), then the lead lcick is replaced with zeros. This 
           ensures that he lag-click will be appear on the same time in lead-lag 
           as in lag-only stimuli, (float)
    ild -- Inter-aural level difference in signal click: max-ear - min-ear (dB , 
           default = 10 (float)
    itd -- Inter-aural time difference (ms, rounded to nearest sample) in signal click: max-ear - min-ear (ms), 
           default = 0.325 (float)
    fs -- Sampling frequency in samples per second, default = 48000 (int)
    dur -- Duration of stimulus in number of samples, default = 3840 (0.08 ms with fs = 48000), (int)
    plot -- If true, will plot the stimulus (default False)
    rec -- If true, will generate wav-file with stimulus, and save it in the current directory  
           (default False).
    
    Returns:
    stimulus -- Vector with click sounds (float, 2xn numpy array)
    '''


    # Create square click surrounded by zeros ----------------------------------
    sqduration = cdur 
    sqsamples = np.int(np.ceil(sqduration  * (fs/1e6)))
    # Read definition of mn.create_squarepulse() for meaning of *zeros arg.
    click = mn.create_squarepulse(sqsamples, leadzeros = 200, lagzeros = 200)
    click = 10**(sgain/20.0) * click  # set gain of standard click
    click = mn.butter_lowpass(click, 20000, fs = fs)
    # --------------------------------------------------------------------------
  
  
    # Create standard (s) component --------------------------------------------
    # ici: inter-click-interval in samples (used to be called tau; but nowadays I prefer ici)
    ici = np.int(np.ceil((fs/1000.0) * tau))  
    gap = np.zeros(ici)
    smask = np.concatenate((click, gap))

    if (not leadclick):  # Just lag-click
        emptyclick = 0.0 * click  # lead click eliminated by setting it to zero
        smask = np.concatenate((emptyclick, gap))
        
    ssignal = np.concatenate((gap, click)) * 10**((level)/20.0)
    s = smask + ssignal
    # --------------------------------------------------------------------------


    # Create variable (v) component---------------------------------------------
    ild = np.abs(ild)  # Side not determined by sign, but by parameter "left"
    itd = np.abs(itd)  
    vmask = smask  # Masker (no ILD no ITD)
    # Add constant ild to signal (could be zero)
    vsignal_maxear = ssignal
    vsignal_minear = ssignal * 10**((-ild)/20.0) 
    # Add constant itd to signal (could be zero)
    zitd = np.zeros(np.int(round(itd * (fs/1000.0)))) # zeros to create itd
    vsignal_minear = np.concatenate((zitd, vsignal_minear)) 
    vsignal_minear = vsignal_minear[0:len(vsignal_maxear)] # remove zeros at end
    vmax = vmask + vsignal_maxear
    vmin = vmask + vsignal_minear  
    
    # Add zeros to make duration (in samples) equal to parameter "dur"
    extrazeros = np.zeros(dur - len(vmax))
    vmax = np.concatenate((extrazeros, vmax)) 
    vmin = np.concatenate((extrazeros, vmin)) 
    # --------------------------------------------------------------------------

    
    # Check for overload -------------------------------------------------------
    if (np.max(s) > 1 or np.max(vmax) > 1 or np.max(vmin) > 1):
        print "Check parameters (sgain, ild), signals are overloaded"
        raise SystemExit
    # --------------------------------------------------------------------------     
    

    # Create stereo signals ----------------------------------------------------     
    if (left): # Signal points left
        stimulus = np.transpose(np.array([vmax,vmin]))  
        side = 'left'   # For naming wav-files     
    else : # Signal points right       
        stimulus = np.transpose(np.array([vmin,vmax])) 
        side = 'right' # For naming wav-files   

    if ((itd == 0) & (ild == 0)):
        side = 'center' # For naming wav-files   
    # --------------------------------------------------------------------------
   
   
    # Plotting -----------------------------------------------------------------
    if (plot):
        print "len(stimulus), number of samples, ms", len(stimulus), len(stimulus)/(0.001*fs) 
        t = (np.arange(0, len(stimulus)) - 2441) / (0.001*fs)   
        plt.plot(t, stimulus[:,0] + 1)  # Left channel
        plt.plot(t, stimulus[:,1])  # Right channel
        plt.show()
    # --------------------------------------------------------------------------
    
    
    # Recording (save as .wav file) --------------------------------------------
    if (rec):
    # Save wavfile
        wavname = 'level_' + str(level) + '_tau_' + str(tau) + '_side_' + side + '_ild_' + str(ild) + '_itd_' + str(itd) + '.wav'
        sf.write(wavname, stimulus, fs) # variable as wav-file
    # --------------------------------------------------------------------------
    
    
    return stimulus
    


