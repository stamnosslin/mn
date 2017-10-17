# Generation of repetition pitch stimuli

# Import libraries -------------------------------------------------------------
import numpy as np
import sounddevice as sd
import time  # Function time-sleep()
import mn  # For generating repetition pitch stimuli
import soundfile as sf  # Just for demo of saving wavfile
# ------------------------------------------------------------------------------


# Set and display sound device, output device marked <
print '\n\n\n  -----------------------------------------------------------------'
print "You may have to set sound device using sd.query_devices()"
print '---------------------------------------------------------------------\n\n'
sd.default.device = "Analog (1+2) (Babyface Analog (1+2))"  
# sd.default.device = "ASIO4ALL"
print 'Sound device ------------------------------------------------------------'
print sd.query_devices()
print '-------------------------------------------------------------------------'


# Set these parameters ---------------------------------------------------------
fs = 48000  # Sampling frequency (samples per second)
timeseparation = fs * 0.008  # Time-separation 2 ms
noisegain = -20.0  # Gain of first noise in dB (re max = 0 dB)
repgain = -20.0  # Gain of repeatead noise (.0 makes it a float)
# ------------------------------------------------------------------------------


# Create stimuli using functions in library mn ---------------------------------
# Create a white noise
noise = mn.create_wn(fs)
noise1 = mn.set_gain(mono = noise, gaindb = noisegain)
noise1 = mn.fade(monosignal = noise1, samples = fs * 0.05)  #Fade in-&-out  

# Add the white noise to itself
noise2 = mn.set_gain(mono = noise, gaindb = repgain)  # Same as noise1, different dB
rep_noise = mn.delay_integer(mono1 = noise1, mono2 = noise2, timeseparation = timeseparation)
rep_noise = mn.fade(monosignal = rep_noise, samples = fs * 0.05) 
rep_nosie = mn.set_equal(noise1, rep_noise, parameter = 'rms')
# ------------------------------------------------------------------------------


# Play white noise -------------------------------------------------------------
print '\n\n\n---------------------- Now you hear a white noise'
sd.play(noise1, fs)  # Play sound
time.sleep(len(noise) / float(fs))  # Wait till sound has been played

# Save wav-file
sf.write('noise_please_delete.wav', noise1, fs)
# ------------------------------------------------------------------------------


# Plat repetition noise --------------------------------------------------------
time.sleep(0.5)  # Short period of silence
print '\n\n\n---------------------- Now you hear a white noise added to itself'
sd.play(rep_noise, fs)
time.sleep(len(rep_noise) / float(fs))  # Wait till sound has been played
print '\n\nTime separation (ms) = ', timeseparation / 48
print 'Repetition level re first sound = ', repgain - noisegain

# Save wav-file
sf.write('repetition_pitch_please_delete.wav', rep_noise, fs)
# ------------------------------------------------------------------------------


print '\n\n---------------------- End of demo \n\n\n'




