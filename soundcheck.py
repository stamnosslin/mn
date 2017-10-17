# Test of audio equipment

import numpy as np
import sounddevice as sd
import time  # function time-sleep()

def fade(monosignal,samples):
    '''Apply a raised cosine to the start and end of a mono signal.
    
    Keyword arguments:
    monosignal -- vector (1xn numpy array).
    samples -- number of samples of the fade (integer). Make sure that: 
    2*samples < len(monosignal)
    
    Return:
    out -- faded monosignal (1xn numpy array)
    ''' 
    ramps = 0.5*(1-np.cos(2*np.pi*(np.arange(2*samples))/(2*samples-1)))
    fadein = ramps[0:samples]
    fadeout = ramps[samples:len(ramps)+1]
    plateu = np.ones(len(monosignal)-2*samples)
    weight = np.concatenate((fadein,plateu,fadeout))
    out = weight*monosignal
    return out
    
    
def create_sinusoid (freq = 1000, phase = 0, fs = 48000, dur = 1):
    '''Create a sinusoid of specified length with amplitude -1 to 1. Use
    set_gain() and fade() to set amplitude and fade-in-out.
    
    Keyword arguments:
    frequency -- frequency in Hz (float)
    phase -- phase in radians (float)
    fs -- sampling frequency (int)
    duration -- duration of signal in seconds (float). 
    
    Return:
    sinusoid -- monosignal of sinusoid (1xn numpy array)
    '''    
    t = np.arange(0, dur, 1.0/fs) # Time vector
    sinusoid = np.sin(phase + 2*np.pi* freq * t) # Sinusoid (mono signal)
    return sinusoid


# Set and display sound device, output device marked <
print '\n\n\n  -----------------------------------------------------------------'
print "You may have to set sound device using sd.query_devices()"
print '---------------------------------------------------------------------\n\n'
sd.default.device = "Analog (1+2) (Babyface Analog (1+2))"  
# sd.default.device = "ASIO4ALL"
print 'Sound device ------------------------------------------------------------'
print sd.query_devices()
print '-------------------------------------------------------------------------'

fs = 48000
sgain = 10**(0.3/20)
dur = 3  # duration in seconds
tone = sgain * fade(create_sinusoid(fs = fs, dur = dur), 960)
silence = np.zeros(len(tone))

left  = np.transpose(np.array([tone, silence]))
right = np.transpose(np.array([silence, tone]))


print '\n\n\n'
print 'Start soundcheck -----------------------------------------\n\nSampling frequency: ', fs
print '\n'

time.sleep(1)
print "Now you hear a tone in the LEFT ear: 1 kHz, for 3 seconds"
sd.play(left, fs)
time.sleep(3.5)

print '\n'
print "Now you hear a tone in the RIGHT ear: 1 kHz, for 3 seconds"
sd.play(right, fs)
time.sleep(3.5)
print '\nEnd soundcheck ------------------------------------------\n\n\n'
