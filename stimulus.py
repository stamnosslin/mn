#!/usr/bin/python
# -*- coding: utf-8 -*-a

# Experiment ALEXIS #116 -------------------------------------------------------
# Fuctions for generating stimuli.
#
# MN 2018-09-05
# ------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt  
import soundfile as sf

import mn # User made functions, make sure mn.py is located in the working dir

def create_stimulus(level = 0, left = True, leadclick = True, cdur = 125, sgain = -11, tau = 4.0, 
                    ild = 10.0, itd = 0.325, fs = 48000, dur = 3840, plot = False, rec = False):
    ''' Create stereo vector with click sounds: lead-lag click pair or single (lag-only) click.      
    
    Keyword arguments:
    level -- Lag-lead ratio, LLR [dB], Relative level of lag-click (re lead-click, i.e. sgain). 
             If leadclick = False, the absolute level of the lag click is level + sgain (float).
    left -- Decides whether spatial info points left or right. True points left; False points right. 
            NOTE: If both itd and ild are set to 0, then the signal will point straigt ahead no matter 
            if this parameter is set to True or False      
    leadclick -- Decides if there is a lead click preceeding the lag click. 
              If True, a lead-lag click is created, if False only a lag-click    
    cdur -- Click (square pulse) duration in microseconds, rounded to nearest sample (float)
    sgain -- Level of lead-click in dB re max (before lowpass filtering), default = -11 (float)
    tau -- Inter-click interval (ms, rounded to nearest sample). If no lead click is present
           (leadclick = False), then the lead click is replaced with zeros. This 
           ensures that the lag-click will be appear on the same time in lead-lag 
           as in lag-only stimuli, (float)
    ild -- Inter-aural level difference in signal click: max-ear - min-ear (dB , 
           default = 10 (float)
    itd -- Inter-aural time difference (ms, rounded to nearest sample) in signal click: max-ear - min-ear (ms), 
           default = 0.325 (float)
    fs -- Sampling frequency in samples per second, default = 48000 (float)
    dur -- Duration of stimulus in number of samples, default = 3840 (0.08 ms with fs = 48000), 
           NOTE: If too short, there will be an error message. Try again with larger integer (int)
    plot -- If true, will plot the stimulus, default False (logical)
    rec -- If true, will generate wav-file with stimulus, and save it in the current directory,  
           default False (logical)
    
    Returns:
    stimulus -- Vector with click sounds (float, 2xn numpy array)
    '''


    # Create square click surrounded by zeros ----------------------------------
    sqduration = cdur 
    sqsamples = np.int(np.ceil(sqduration  * (fs/1e6)))
    # Read definition of mn.create_squarepulse() for meaning of *zeros arg.
    click = mn.create_squarepulse(sqsamples, leadzeros = 100, lagzeros = 250)
    click = 10**(sgain/20.0) * click  # set gain of standard click
    click = mn.butter_lowpass(click, 20000, fs = fs)
    # --------------------------------------------------------------------------
  
  
    # Create lead-click (called mask) -----------------------------------------
    # ici: inter-click-interval in samples (used to be called tau; but nowadays I prefer ici)
    ici = np.int(np.ceil((fs/1000.0) * tau))  
    gap = np.zeros(ici)
    mask = np.concatenate((click, gap))

    if (not leadclick):  # Just lag-click
        emptyclick = 0.0 * click  # lead click eliminated by setting it to zero
        mask = np.concatenate((emptyclick, gap))
    # --------------------------------------------------------------------------


    # Create lag-click (called signal) -----------------------------------------
    signal = np.concatenate((gap, click)) * 10**((level)/20.0)
    ild = np.abs(ild)  # Side not determined by sign, but by parameter "left"
    itd = np.abs(itd)  

    # Add constant ild to signal (could be zero)
    signal_maxear = signal
    signal_minear = signal * 10**((-ild)/20.0) 
    # Add constant itd to signal (could be zero)
    zitd = np.zeros(np.int(round(itd * (fs/1000.0)))) # zeros to create itd
    signal_minear = np.concatenate((zitd, signal_minear)) 
    signal_minear = signal_minear[0:len(signal_maxear)] # remove zeros at end
    stimulusmax = mask + signal_maxear
    stimulusmin = mask + signal_minear  

    # Add zeros to make duration (in samples) equal to parameter "dur"
    extrazeros = np.zeros(dur - len(stimulusmax))
    stimulusmax = np.concatenate((extrazeros, stimulusmax)) 
    stimulusmin = np.concatenate((extrazeros, stimulusmin)) 
    # --------------------------------------------------------------------------

    
    # Check for overload -------------------------------------------------------
    if (np.max(mask) > 1 or np.max(stimulusmax) > 1 or np.max(stimulusmin) > 1):
        print "Check parameters (sgain, ild), signals are overloaded"
        raise SystemExit
    # --------------------------------------------------------------------------     
    

    # Create stereo signals ----------------------------------------------------     
    if (left): # Signal points left
        stimulus = np.transpose(np.array([stimulusmax,stimulusmin]))  
        side = 'left'   # For naming wav-files     
    else : # Signal points right       
        stimulus = np.transpose(np.array([stimulusmin,stimulusmax])) 
        side = 'right' # For naming wav-files   

    if ((itd == 0) & (ild == 0)):
        side = 'center' # For naming wav-files   
    # --------------------------------------------------------------------------
   
   
    # Plotting -----------------------------------------------------------------
    if (plot):
        # Position where stimulus starts, after inital zeros
        start = np.where(stimulus != 0)[0][1]  
        print "len(stimulus), number of samples, ms", len(stimulus), len(stimulus)/(0.001*fs) 
        t = (np.arange(0, len(stimulus)) - start) / (0.001*fs)   
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
