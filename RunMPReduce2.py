# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 18:35:40 2016

@author: Yousof Erfani, Unverity of Sherbrooke. Canada
"""

##############################################################################
# te4st wavread-wavwrtie and plot
import scipy.io.wavfile as Wav # wavread and write
import numpy as np
import  matplotlib.pyplot as plt
import time
import mp_map_reduce2
from mp_map_reduce2 import MP_map_reduce_adaptive_segment_size
from mp_map_reduce2 import reconstruct

import scipy.io as sio # for reading mat files
from scipy import linalg # for scipy.linalg.norm

# A test signal
fs,sig = Wav.read ('Original.wav')
print fs
sig_max = np.max(np.abs(sig))
print sig_max
print sig[0:1024] / float(sig_max) # in the denominator should be float to have float results
print np.shape(sig)

Wav.write("writtenfile.wav", 44100, sig)
##############################################################################


plt.close("all")

# Gammatone Filterbank
wraped_vars = sio.loadmat("FBDATA.mat")
filter_bank = wraped_vars['FB']

norm_of_filters = linalg.norm(filter_bank[10, :])

#Initialization
maxIter = 50
Lbase = 4000
knumBases = 25


sig = sig[120000:180000]# in fact of choosing column zero and one for one 


no_segments = np.floor(21*len(sig) / float(44100));

#
#
sig = sig.astype('float')
sig = sig / linalg.norm(sig)
print (linalg.norm(sig))
tic = time.time()
[selected_max_coefficient, selected_time_indx, selected_channel_indx] = MP_map_reduce_adaptive_segment_size(sig, maxIter, filter_bank, no_segments)
#plt.plot(selected_max_coefficient)
toc = time.time()
elapsed_time  = toc - tic
print "elapsed time for decomposition(in sec)=", elapsed_time
plt.plot(sig)
Leng=len(sig)+Lbase

tic=time.time()
out=reconstruct(selected_max_coefficient,selected_channel_indx,selected_time_indx, filter_bank, Leng)
toc=time.time()
elapsed_time = toc - tic

print "elapsed time for reconstruction", elapsed_time
plt.plot(out)
plt.show()

out = out[0: len(sig)]

mSNRR = 10*np.log10( np.sum(sig**2) / np.sum((sig - out.T)**2)  )
print mSNRR

signal_length_Sec = len(sig) / 44100.0
print signal_length_Sec

#print x,y

#plt.plot(x)
#plt.show()

