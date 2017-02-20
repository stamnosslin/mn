#!/usr/bin/python
# -*- coding: utf-8 -*-

''' In this module, I have collected a number of functions that I tend to use
and reinvent each time a write a new program. Hopefully, collecting functions
in this module will save me some time.
Mats Nilsson (MN)
Gosta Ekman Laboratory, Department of Psychology, Stockholm University

MN: 2016-04-06; Revised: 2017-02-14
'''

import numpy as np
import scipy.signal # for convolution
import os  # to clear console

def cls():
    ''' To clear the python console'''
    os.system('cls' if os.name == 'nt' else 'clear')


def calculate_amplitude(monosignal, type = 'rms', unit = 'level'):
    ''' Calculates the amplitude of a mono signal. See Keyword arguments for 
    implemented definitions of amplitude.
    
    Keyword arguments:
    monosignal -- vector (1xn numpy array).
    amplitude -- String deciding type of amplitude: 'rms' defined as 
                 np.sqrt(np.mean(monosignal**2)); 'peak' defined as  
                 np.max(np.abs(monosignal))
    unit -- String deciding unit amplitude: 'linear' corresponds to pressure
    [-1,1]; 'level' is gain in dB re maximum (0 dB)
        
    Return:
    amplitude -- numeric value representing amplitude (float)
    '''
    if type == 'rms':
        amp = np.sqrt(np.mean(monosignal**2))
    elif type == 'peak':
        amp = np.max(np.abs(monosignal))
    else:
        pass
    
    if unit == 'level':
        amplitude = 20 * np.log10(amp)
    elif unit == 'linear':
        amplitude = amp
    else:
        pass
        
    return amplitude

   
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

    
def create_squarepulse(samples, leadzeros = 10, lagzeros = 10):
    ''' Create square pulse, startig and ending with zeros of specified length
    expressed as multiple of length of the plateau of ones.
    
    Keyword arguments:
    samples -- length of plateau (integer). NB! the total length of the output 
              array = leadzeros*samples + samples + lagzeros*samples 
    leadzeros -- proprtion of zeros before the square. E.g., 10 means 
                that the number of zeros = 10 x samples
    lagzeros -- proportion of zeros after the square. 
    
    Return:
    squarew = numpy array with square wave
    '''
    
    square_list = [0]*(leadzeros*samples) + [1]*samples + [0]*(lagzeros*samples)
    square_array = np.array(square_list)
    return square_array 

 
def cut_middle(mono,cutsamples):
    '''Cuts out samples in the middle of a sound vector.
    
    Keyword arguments:
    mono -- vector (numpy array).
    cutsamples -- number of samples to cut from the middle of the sound 
                 (integer, but float accepted).
    Make sure that cutsamples < len(mono)
    
    Return:
    cut -- monosignal (numpy array).
    '''
    totlength = len(mono)
    middle = totlength // 2 # remember floor division (//) 
    start = middle - (cutsamples // 2)
    end = middle + (cutsamples // 2)
    cut = mono[start:end]
    return cut
    

def create_wn(samples):
    '''Create a white noise of specified length, normalized to [-1,1]. Use
    set_gain() to set amplitude and fade() to fade the signal.
    
    Keyword arguments:
    samples -- number of samples of the noise (integer). 
    
    Return:
    xn -- monosignal of white noise (1xn numpy array)
    '''
    x = np.random.randn(samples)
    maxvalue = np.max(np.abs(x))
    xn = x / maxvalue
    return xn    

    
def delay_fraction(mono, fracdelay, filter_length=151):
    '''Delays a signal (mono) with a fractional sample. Then use delay_integer()
    to combined signals with an integer timeseparation. Example: If you want
    to add two signals s1 and s2 with a timeseparation of 3.5 samples (favoring
    s1), then do the following:
        (1) create s2new = fractional_delay(s2, 0.5)
        (2) create combined_signal = add_timesep(s1, s2new, 3)        
    
    Keyword arguments:
    mono -- sound vector to be delayd (numpy array)
    delay -- fractional part of delay (samples), float in (0, 1)             
    filter_length -- Filter length (in samples), should be an odd integer

    Return:
    mono_delayed -- monosignal delayed with a fraction of sample re input  
    '''
    # Calculate tap weights (= filter coefficients)
    centre_tap = filter_length // 2
    t = np.arange(filter_length)
    x = t-fracdelay
    sinc = np.sin(np.pi*(x-centre_tap))/(np.pi*(x-centre_tap))
    window = 0.54-0.46*np.cos(2.0*np.pi*(x+0.5)/filter_length) # Hamming window
    tap_weight = window*sinc
    
    # Convolute of input signal (hi-pass below uses np.convolve. There is also 
    # scipy.signal.fftconvolve which may be faster. Please check!)
    s = scipy.signal.convolve(mono, tap_weight, mode='full') / sum(tap_weight)
    mono_delayed = s[centre_tap:-centre_tap]
    return mono_delayed

    
def delay_integer(mono1, mono2, timeseparation):
    '''Add two sounds (may be identical) with a specified time-separation 
    
    Keyword arguments:
    mono1 -- sound vector of sound 1 (numpy array).
    mono2 -- sound vector of sound 2 (numpy array). 
    timeseparation -- separation in number of samples (integer, but float is
    accepted and converted to integer: fractional part ignored).
    
    Return:
    combined -- monosignal (numpy array)
    Note that the output will contain a period = timeseparation of mono1 only 
    at the beginning and mono2 only at the end (use cut_middle() to 
    cut out combined part).
    '''
    silence = np.zeros(int(timeseparation))
    direct = np.append(mono1,silence)
    reflect = np.append(silence, mono2)
    combined = direct + reflect
    return combined
 
    
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

 
def hipass(monosignal, fc = 0.05, b = 0.05):
    ''' Hi-pass filters a monosignal.   
    Note: output length > input length. This is fixed using cut_middle()
    Code taken from this web-site:
           tomroelandts.com/articles/how-to-create-a-simple-high-pass-filter
    
    Keyword arguments:
    mono -- signal (numpy array).
    fc -- Cutoff frequency as a fraction of the sample rate (in (0, 0.5)).
          Default = 0.05, corresponding to 2.4 kHz at 48 kHz sampling rate.
    b -- Transition band, as a fraction of the sample rate (in (0, 0.5)).
         Default = 0.05, corresponding to 2.4 kHz at 48 kHz sampling rate.     

    Return:
    s = monosignal (numpy array).    
    '''
    N = int(np.ceil((4 / b)))
    if not N % 2: N += 1  # Make sure that N is odd.
    n = np.arange(N)
    # Compute a low-pass filter.
    h = np.sinc(2 * fc * (n - (N - 1) / 2.))
    w = np.blackman(N)
    h = h * w
    h = h / np.sum(h)
    # Create a high-pass filter from the low-pass filter by spectral inversion.
    h = -h
    h[(N - 1) / 2] += 1
    s = np.convolve(monosignal, h)
    s = cut_middle(s, len(monosignal))  # Cut to same length as input
    return s

    
def set_equal(mono1, mono2, parameter = 'rms'):
    '''Sets rms level of a to be adjusted sound equal to a master sound. 
    
    Keyword arguments:
    mono1 -- master sound, sound vector (numpy array).
    mono2 -- to be adjusted sound (numpy array), set equal in rms as mono1
    parameter -- 'rms' (default): set rms equal; 'peak': set peak equal
    
    Return:
    adjusted -- monosignal type mono2 of same level as mono1 (1xn numpy array)
    '''
    if parameter == 'rms':
        rms1 = np.sqrt(np.mean(mono1**2))
        rms2 = np.sqrt(np.mean(mono2**2))
        adjusted = mono2 * (rms1/rms2)
    elif parameter == 'peak':
        peak1 = np.max(np.abs(mono1))
        peak2 = np.max(np.abs(mono2))
        adjusted = mono2 * (peak1/peak2)
    
    # Print warning if overload, that is, if any abs(sample-value) > 1
    if (np.max(np.abs(adjusted)) > 1):
        message1 = "WARNING: equal_rms() generated overloaded signal!"
        message2 = "max(abs(signal)) = " + str(np.max(np.abs(adjusted))) 
        message3 = ("number of samples >1 = " + 
                    str(np.sum(1 * (np.abs(adjusted) > 1))))
        print message1
        print message2
        print message3
        
    return adjusted

 
def set_gain(mono, gaindb):
    ''' Set gain of mono signal, to get dB(rms) to specified gaindb 
    
    Keyword arguments:
    mono -- vector (numpy array).
    gaindb -- gain of mono in dB re max = 0 dB (float).
    
    Return:
    gained -- monosignal (numpy array)
    '''
    rms = np.sqrt(np.mean(mono**2))
    adjust = gaindb - 20 * np.log10(rms)
    gained = 10**(adjust/20.0) * mono # don't forget to make 20 a float (20.0)
    
    # Print warning if overload, that is, if any abs(sample-value) > 1
    if (np.max(np.abs(gained)) > 1):
        message1 = "WARNING: set_gain() generated overloaded signal!"
        message2 = "max(abs(signal)) = " + str(np.max(np.abs(gained))) 
        message3 = ("number of samples >1 = " + 
                    str(np.sum(1 * (np.abs(gained) > 1))))
        print message1
        print message2
        print message3
 
    return gained
 
 
def butter_bandpass(signal, center, bandwidth, order = 5, fs = 48000.0):
    '''Butterworth bandpass digital filter applied to monosignal (1xn numpy array). 
    
    Keyword arguments:
    signal -- monosignal to be filtered (1xn numpy array)
    center --  center frequency, Hz (float)
    bandwidth -- bandwidth in octaves (float)
    order -- filter order, default = 5 (int)
    fs -- sampling frequency, default = 48000 (float)    
    
    Return:
    out -- bandpassed monosignal (1xn numpy array)
    '''    
    nyq = 0.5 * fs
    q = (2**bandwidth)**(0.5) / (2**bandwidth - 1)
    bw = center / q  # bw = bandwidth in Hz
    f1 = ((center**2) / (2**bandwidth))**(0.5)
    f2 = f1 + bw
    
    low = f1 / nyq
    high = f2 / nyq
    
    b, a = scipy.signal.butter(order, [low, high], btype = 'band')
    out = scipy.signal.lfilter(b, a, signal)
    return out

    
def butter_lowpass(signal, cutoff, order = 5, fs = 48000.0):
    '''Butterworth lowpass digital filter applied to monosignal (1xn numpy array). 
    
    Keyword arguments:
    signal -- monosignal to be filtered (1xn numpy array)
    cutoff --  cut-off frequency, Hz (float)
    order -- filter order, default = 5 (int)
    fs -- sampling frequency, default = 48000 (float)    
    
    Return:
    out -- lowpass filtered monosignal (1xn numpy array)
    '''    

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype = 'low', analog = False)
    out = scipy.signal.lfilter(b, a, signal)
    return out

    
def cutoff_freq(center, octavewidth):
    '''Calculate cut-off frequencies of band defined by center 
    frequency and bandwidth in octaves. 
    
    Keyword arguments:
    center --  center frequency, Hz (float)
    octavewidth -- bandwidth in octaves (float)

    Return:
    out -- low and high cut-off frequency (list, float)
    '''    
    q = (2**octavewidth)**(0.5) / (2**octavewidth - 1)
    bw = center / q  # bw = bandwidth in Hz
    f1 = ((center**2) / (2**octavewidth))**(0.5)
    f2 = f1 + bw
    out = [f1, f2]
    return out

    
def exp_decay(monosignal, decay = 5.0, fs = 48000):
    ''' Amplitude modulation of monosignal by decaying exponential

    Keyword arguments:    
    signal -- monosignal 1xn numpy array (float)
    decay -- time in ms after which the decay reached -40 dB, default is 5 ms (float)
    fs -- sampling frequency in samples per second, default = 48000 (int)
    
    Return:
    out -- modulated monosignal, 1xn numpy array (float)
    '''
    t  = np.arange(len(monosignal)) * (1000.0/fs)  # t in ms
    decay = np.log(0.01) / decay  # decay to 0.01 (-40 dB) after decay ms
    w = np.exp(decay * t)
    out_signal = w * monosignal
    return out_signal