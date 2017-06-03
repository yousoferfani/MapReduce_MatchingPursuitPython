# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 11:08:52 2016

@author: Yousof Erfani
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def MP_map_reduce_adaptive_segment_size(input_signal, maxIter,filter_bank,no_segments):
    
    length_signal = len(input_signal) # numpy len
    kernel_length = filter_bank.shape[1]
    num_bases = filter_bank.shape[0]
    residual = input_signal
    cnt = 0;
    zz = np.zeros(kernel_length+1)
    print zz
   # residual.append(zz)
    #residual=np.concatenate([residual zz]);
    print len (residual)
    residual = np.concatenate( (residual, zz), axis=0 )
    print "\n"
    print len (residual)
    nsMax = no_segments;
    nsPerSec = []
    selected_time_indx = []
    selected_channel_indx = []
    selected_max_coefficient = []
    #ind=range(len(indices_max1))
    ind=range(len(input_signal))
  
    sparse_coefficients=[]
    time_samples=[]
    channel_numbers=[]
    filter_bank_reverse=filter_bank[:,::-1] # for reversing the time axis along the row [::-1] is used for reversing  column indices
    #print np.shape (filter_bank_reverse)
    filter_bank_reverse_col=filter_bank_reverse.T
    filter_tensor=filter_bank_reverse_col[np.newaxis,:,:]
    #
    for iter in range (maxIter):
        
        #Adaptive segment size
        nsPerSec = int ( 30* (maxIter-iter) / maxIter ) + 1
        no_segments = int ( nsPerSec* len (input_signal) / 44100.0 )+1
        processing_window =  int( length_signal / float(no_segments) )
        indx_matrix=index_generator(length_signal, processing_window, kernel_length)

        print iter
      
       
       
       ##############################################################################
        residual_matrix=residual[indx_matrix]
        projection_tensor1=signal.fftconvolve(residual_matrix[:,:,np.newaxis], filter_tensor, mode='full')
       # proj = np.amax( abs (XX[:8,:]), axis = 0 )
        projection_tensor=projection_tensor1[:,kernel_length-1:-kernel_length,:]
        
        projections_max_matrix = np.amax(abs( projection_tensor), axis = 1 )
        projections_max_indices = np.argmax( abs(projection_tensor), axis = 1 )
        
        
        
        proj_max1=  np.amax( projections_max_matrix[:,:8],axis=1)
        channel_indx1 =  np.argmax( projections_max_matrix[:,:8],axis=1)
        time_indx1=projections_max_indices[np.arange(len(channel_indx1)),channel_indx1]
        
        selected_coefficients_vector1=projection_tensor[np.arange(no_segments), time_indx1, channel_indx1]
        
        
        
        
        
        proj_max2=  np.amax( projections_max_matrix[:,8:16],axis=1)
        channel_indx2 =  np.argmax( projections_max_matrix[:,8:16],axis=1)+8
        time_indx2=projections_max_indices[np.arange(len(channel_indx2)),channel_indx2]
        
        selected_coefficients_vector2=projection_tensor[np.arange(no_segments), time_indx2, channel_indx2]

            
            
        proj_max3=  np.amax( projections_max_matrix[:,16:24],axis=1)
        channel_indx3 =  np.argmax( projections_max_matrix[:,16:24],axis=1)+16
        time_indx3=projections_max_indices[np.arange(len(channel_indx3)),channel_indx3]
        
        selected_coefficients_vector3=projection_tensor[np.arange(no_segments), time_indx3, channel_indx3]
        for k in range(len(time_indx1)):
            time_indx1[k]=time_indx1[k]+int(k*processing_window)
            time_indx2[k]=time_indx2[k]+int(k*processing_window)
            time_indx3[k]=time_indx3[k]+int(k*processing_window)
            
            
        sparse_coefficients_4this_iteration=np.concatenate( (selected_coefficients_vector1,selected_coefficients_vector2,selected_coefficients_vector3))
        channel_numbers_4this_iteration=np.concatenate((channel_indx1,channel_indx2,channel_indx3))
       
        time_indices_4this_iteration=np.concatenate((time_indx1,time_indx2,time_indx3 ))
        
        reduce_sig=reconstruct(sparse_coefficients_4this_iteration, channel_numbers_4this_iteration, time_indices_4this_iteration,filter_bank, length_signal+kernel_length+1)
            
        residual=residual-reduce_sig
       
        
        sparse_coefficients.extend(sparse_coefficients_4this_iteration)
        #sparse_coefficients.extend(selected_coefficients_vector2)
        #sparse_coefficients.extend(selected_coefficients_vector3)
        time_samples.extend(time_indices_4this_iteration)
       # time_samples.extend(time_indx2)
        #time_samples.extend(time_indx3)
        channel_numbers.extend(channel_numbers_4this_iteration)
       # channel_numbers.extend(channel_indx2)
        #channel_numbers.extend(channel_indx3)

        



            
       
      
    print cnt
    return [sparse_coefficients,time_samples,channel_numbers]


def reconstruct(VAL,CH,T,FB,Leng):
    out=np.zeros(Leng)
    for i in range (len(VAL)):
        out[ T[i]:T[i]+4000] = out[ T[i]:T[i]+4000]+VAL[i]*FB[CH[i],:]
    return out 
def index_generator(signal_length,frame_length,filter_length):
    ind=np.zeros((signal_length/(frame_length),frame_length+filter_length), dtype=np.int)
    x=np.array([1,2,3,4,5,6,7,8,9,10,11])
    no_segments=(signal_length)/frame_length
    for i in range(int(no_segments)):
        ind[i,:]= np.arange(i*frame_length, (i+1)*frame_length+filter_length)
        
#ind=[[1,1],[0,2]]
    return ind
    

