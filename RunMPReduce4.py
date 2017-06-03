# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 17:24:00 2016

@author: Yousof Erfani, University of Sherbrooke, Canada
"""

##############################################################################
# te4st wavread-wavwrtie and plot
import scipy.io.wavfile as Wav # wavread and write
import numpy as np
import  matplotlib.pyplot as plt
import time
fs,sig= Wav.read ('/home/usef/Work/erfy1801/trunk/Sparse-Watermarking/WSpace/Original.wav')
print fs
sig_max=np.max(np.abs(sig))
print sig_max
print sig[0:1024]/float(sig_max) # in the denominator should be float to have float results
print np.shape(sig)
#plt.plot(sig[100:800000]/float(sig_max))
#plt.ylabel('some numbers')
#plt.show()
Wav.write("writtenfile.wav", 44100, sig)
##############################################################################
import mp_map_reduce4
from mp_map_reduce4 import MP_map_reduce_adaptive_segment_size
from mp_map_reduce4 import reconstruct

import scipy.io as sio # for reading mat files
from scipy import linalg # for scipy.linalg.norm



#clc
#clear all
plt.close("all")
#load fc
#wraped_vars=sio.loadmat("/home/usef/Work/erfy1801/trunk/Sparse-Watermarking/WspaceRamin/W38.mat")

wraped_vars=sio.loadmat("/home/usef/Work/erfy1801/trunk/Sparse-Watermarking/WspaceRamin/FBDATA.mat")
filter_bank=wraped_vars['FB']
#plt.plot(filter_bank[10,:])
#plt.show()
norm_of_filters=linalg.norm(filter_bank[10,:])
#
#% for no_segments=5:5:40
maxIter=100
#tic

#LC=10;
#fc=fc24;
Lbase=4000;
knumBases=25;
#% sig=sig(200000:360000,1);
sig=sig[120000:180000]# in fact of choosing column zero and one for one 
# dimensional ve3ctor we choose [:,]
#
no_segments=np.floor(21*len(sig)/float(44100));
#
#[selected_max_coefficient,selected_time_indx,selected_channel_indx]  = MP_MR3_addaptive_segment_size(sig, maxIter,FB,no_segments);
#plot(selected_max_coefficient)
#Leng=length(sig);
#[out]=reconstruct(selected_max_coefficient,selected_channel_indx,selected_time_indx,FB,Leng+Lbase);
#toc
#out=out(1:length(sig));
#out=out';
#% wavplay(out,fs)
#mSNRR=10*log10( sum(sig.^2)/sum((sig-out).^2)  )
#% end
#plot(sig);hold on;plot(out,'r')
#signal_length_Sec=length(sig)/44100
#% figure
#
#
sig = sig.astype('float')
sig=sig/linalg.norm(sig)
print (linalg.norm(sig))
tic=time.time()
[selected_max_coefficient,selected_time_indx,selected_channel_indx]=MP_map_reduce_adaptive_segment_size(sig, maxIter,filter_bank,no_segments)
#plt.plot(selected_max_coefficient)
toc=time.time()
elapsed_time=toc-tic
print "elapsed time for decomposition(in sec)=", elapsed_time
plt.plot(sig)
Leng=len(sig)+Lbase
# When function has only one output we dont put it in a bracket 
tic=time.time()
out=reconstruct(selected_max_coefficient,selected_channel_indx,selected_time_indx, filter_bank, Leng)
toc=time.time()
elapsed_time=toc-tic
print "elapsed time for reconstruction", elapsed_time
plt.plot(out)
plt.show()

out=out[0:len(sig)]

mSNRR=10*np.log10( np.sum(sig**2)/np.sum((sig-out.T)**2)  )
print mSNRR

signal_length_Sec=len(sig)/44100.0
print signal_length_Sec

#print x,y

#plt.plot(x)
#plt.show()

