#!/usr/bin/python
# -*- coding: utf-8 -*-a

# Psychophysical concepts by auditory demonstrations ---------------------------
# Basic demonstration of method of limits. This just plays a tone in noise in
# a descending or ascending (or random) order of signal-to-noise ratios
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
    noise = mn.create_wn(int(np.round(ndur * fs)))
    signal = mn.create_sinusoid(freq = 1000, phase = 0, dur = sdur, fs = fs)
    silence = np.zeros(int(np.round(((ndur - sdur) * fs)/2)))
    signal = np.concatenate((silence, signal, silence))
    signal = 10**(snratio/20.0) * signal  # Set gain of signal relative to noise
    
    # Combine noise and signal, and apply fade
    combined = mn.fade(noise + signal, int(fs * 0.05))
    combined = mn.set_gain(combined, rms)
    
    # Make diotic
    s = np.transpose(np.array([combined,combined])) 
    
    return s

def play_sequence(ratiolist, fs = 48000):
    message1 = psychopy.visual.TextStim(win, pos = [0, 0.2], color = (-1, -1, -1),
               text = "Listen for the tone", height = 0.07)
    message2 = psychopy.visual.TextStim(win, pos = [0, -0.2], color = (-1, -1, -1),
              text = "Hit a key to start", height = 0.05)
    message1.draw()
    message2.draw()
    win.flip()
    psychopy.event.waitKeys()    

    for j in ratiolist:
        message1.text = "Did you hear the tone?\n(integer + return to continue; Esc to abort)"
        message2.text = "Signal level = %.1f: " % (j)
        s = create_sn(snratio = j)
        sd.play(s, fs)
        time.sleep(len(s) / float(fs))
        message1.draw()
        if int(show_sn) == 1:
            message2.draw()
        win.flip()

        # Collect response
        response = None
        all_keys = np.array([])
        while response is None:
            key = psychopy.event.waitKeys(
                  keyList=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
                           'return', 'escape'])
            all_keys = np.append(all_keys, key)

            if key == ['return']:
                r = all_keys[:-1]  # Remove 'return' key
                r = ''.join(r)
                print 'sn-ratio = ' + str(round(j, 1)) + '; response = ' + r
                response = r  # This breaks the while loop
            elif key == ['escape']:
                print '\n\nUser abort\n\n'
                raise SystemExit  # Abort experiment
            else: pass

        win.flip()
        time.sleep(0.1)      

# Dialog box decide sequence ---------------------------------------------------
id_dict = {'high': '0', 'low': '-48', 'trials': '9', 'order (a, d, r)':'d',
            'blanks': '0', 'show sn': '1'}
dlg = psychopy.gui.DlgFromDict(id_dict, title = 'Enter ID number and Day of testing!')
if dlg.OK:
    high = id_dict['high']  # Record id (str)
    low = id_dict['low']  # Record id (str)
    trials = id_dict['trials']  # Record id (str)
    order = id_dict['order (a, d, r)'] # Record id (str)
    blanks = id_dict['blanks']
    show_sn = id_dict['show sn']
else:
    raise SystemExit  # The user hit cancel, so exit  
        
win = psychopy.visual.Window([1366, 780], color = (0, 1, 0), pos = (0, 0), 
                   allowGUI = False, monitor = 'testMonitor', units = 'norm')
   
# Order the sequence -----------------------------------------------------------
snratio = np.linspace(float(low), float(high), num = trials, endpoint = True)
if (order == 'r'):  # r means random order, may include blanks
    lures = np.repeat(-999, float(blanks))
    snratio = np.concatenate((snratio, lures))  
    np.random.shuffle(snratio)
elif (order == 'd'): # r means descending order
    snratio = snratio[::-1]  # Reverses the order
else: pass  # a means ascending order, snratio is in this order to start with
# ------------------------------------------------------------------------------

fs = 48000

print '\n\n\n\n\n\n\n\n\n\n-----------------------------------------------------'

play_sequence(ratiolist = snratio)

print '-----------------------------------------------------\n\n\n\n\n\n\n\n\n\n'

