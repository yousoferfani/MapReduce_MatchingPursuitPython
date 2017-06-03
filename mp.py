# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 18:39:34 2016

@author: usef erfani, Universite de Sherbrooke
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def MP(input_signal, maxIter, gammatone_filterbank):
    projection1 = np.zeros( (25, len(input_signal) + gammatone_filterbank.shape[1] - 1 ))
    selected_max_coefficient=[]
    selected_time_indx=[]
    selected_channel_indx=[]
    cnt=0


    length_signal= len(input_signal)
    kernel_length= gammatone_filterbank.shape[1]
    
    num_bases= gammatone_filterbank.shape[0]
    residual= input_signal.transpose()

    zz =  np.zeros(kernel_length+1)
    residual =  np.concatenate( (residual, zz), axis=0 )
    filter_bank_reverse=gammatone_filterbank[:,::-1] # for reversing the time axis along the row [::-1] is used for reversing  column indices

    for iter in range (maxIter):
        print iter

        for k  in range (num_bases):
            projection1[k,:] = signal.fftconvolve( residual [0:len (input_signal) ],filter_bank_reverse [k,:] , mode="full" )

        projection_matrix1 =  projection1 [:,  kernel_length-1:]
        selected_max_coefficient.append(0)
 
        projection_max1 = np.amax( abs (projection_matrix1),axis=0)
        indices_max1 = np.argmax( abs (projection_matrix1),axis=0 )
        selected_time_indx.append(np.argmax(projection_max1));
        selected_channel_indx.append (indices_max1 [selected_time_indx[cnt] ])
        selected_max_coefficient[cnt]= projection_matrix1[selected_channel_indx[cnt],selected_time_indx[cnt]]
        residual[ selected_time_indx[cnt]: selected_time_indx[cnt]+kernel_length] = residual[selected_time_indx[cnt]: selected_time_indx[cnt]+kernel_length]-selected_max_coefficient[cnt]*gammatone_filterbank[selected_channel_indx[cnt],:]
        cnt=cnt+1;
        
    return [selected_max_coefficient,selected_time_indx,selected_channel_indx]
   
        
        
        
        
def reconstruct(VAL,CH,T,FB,Leng):
    out=np.zeros(Leng)
    for i in range (len(VAL)):
        out[ T[i]:T[i]+4000] = out[ T[i]:T[i]+4000]+VAL[i]*FB[CH[i],:]
    return out 



   
