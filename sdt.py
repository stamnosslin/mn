#!/usr/bin/python
# -*- coding: utf-8 -*-a

# Psychophysical concepts by auditory demonstrations ---------------------------
# Basic demonstration of SDT. This just plays a tone in noise or just noise 
# MN 2017-03-03, revised: 2017-10-03
# ------------------------------------------------------------------------------

import psychopy.visual, psychopy.event, psychopy.data, psychopy.gui
import numpy as np
import sounddevice as sd
import soundfile as sf
import time  # function time-sleep()

import mn # User made functions, make sure mn.py is located in the working dir

# Display sound device, output device marked <
sd.default.device = "Analog (1+2) (Babyface Analog (1+2))" # "ASIO4ALL" # 
# print 'Sound device ------------------------------------------------------------'
# print sd.query_devices()
# print '-------------------------------------------------------------------------'


def create_sn(ndur = 0.8, sdur = 0.4, snratio = -40, fs = 48000, rms = -20):      
    
    # Create noise and signal
    noise = mn.create_wn(np.round(ndur * fs))
    signal = mn.create_sinusoid(freq = 1000, phase = 0, dur = sdur, fs = fs)
    silence = np.zeros(np.round(((ndur - sdur) * fs)/2))
    signal = np.concatenate((silence, signal, silence))
    signal = 10**(snratio/20.0) * signal  # Set gain of signal relative to noise
    
    # Combine noise and signal, and apply fade
    combined = mn.fade(noise + signal, fs * 0.05)
    combined = mn.set_gain(combined, rms)
    
    # Make diotic
    s = np.transpose(np.array([combined,combined])) 
    
    return s

def play_sequence(ratiolist, fs = 48000):
    message1 = psychopy.visual.TextStim(win, pos = [0, 0], color = (-1, -1, -1),
               text = "Listen for the tone", height = 0.07)
    message2 = psychopy.visual.TextStim(win, pos = [0, -0.2], color = (-1, -1, -1),
              text = "Hit a key to start", height = 0.05)
    message1.draw()
    message2.draw()
    win.flip()
    psychopy.event.waitKeys()    

    for j in trials:
        message1.text = "Did you hear the tone?"
        message2.text = "Signal level = %.1f: " % (j)
        s = create_sn(snratio = j)
        sd.play(s, fs)
        time.sleep(len(s) / float(fs))
        message1.draw()
        if int(show_sn) == 1:
            message2.draw()
        win.flip()
        key = psychopy.event.waitKeys()
        print 'stimulus = ' + str(j) + '; response = ' + str(key)
        if key == ['escape']:
            print '\n\nUser abort\n\n'
            raise SystemExit  # Abort experiment
        else: pass
        win.flip()
        time.sleep(0.1)      

# Dialog box decide sequence ---------------------------------------------------
id_dict = {'slevel': '-32', 'trials-s+n': '5', 'trials-n': '5', 'show sn': '0'}
dlg = psychopy.gui.DlgFromDict(id_dict, title = 'Enter ID number and Day of testing!')
if dlg.OK:
    level = id_dict['slevel']  # Record id (str)
    sn = id_dict['trials-s+n']  # Record id (str)
    nn = id_dict['trials-n']  # Record id (str)
    show_sn = id_dict['show sn']
else:
    raise SystemExit  # The user hit cancel, so exit  
        
win = psychopy.visual.Window([600, 600], color = (0, 1, 0), pos = (0, 0), 
                   allowGUI = False, monitor = 'testMonitor', units = 'norm')
   
# Trial sequence -----------------------------------------------------------
sn = np.repeat(float(level), float(sn))
nn = np.repeat(-999, float(nn))
trials = np.concatenate((sn, nn))   
np.random.shuffle(trials)
# ------------------------------------------------------------------------------

fs = 48000 

print '\n\n\n-------------------------------------------------------------'

play_sequence(ratiolist = trials)

print '-------------------------------------------------------------\n\n\n'