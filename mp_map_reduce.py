# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 16:13:40 2016

@author: usef
"""
import numpy as np
import matplotlib.pyplot as plt
import signal
def MP_map_reduce_adaptive_segment_size(input_signal, maxIter,filter_bank,no_segments):
    
    #, maxIter, gammatone_filterbank, no_segments    
    #% Localized matching pursuit- Frame based=> much faster than MP in 50
    #% iterations you reas 20 dB SNR  while with ordinary MP it is not
    #% possibnle- sparsity of the coefficients still to be checked: for this
    #% goal we hould have an iteratibe method is odd iteration odd segments and
    #% in even itwertion even segments
    #% Using the adptive segnent size appch- We can get the same sparseity
    #% quality as MP but in the 3 percent timing required by MP this idea can be
    #% used for audio fingerprining based on STRF 
    #% 

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
    #
    for iter in range (maxIter):
        
        #Adaptive segment size
#        nsPerSec = np.floor ( 20* (maxIter-iter) / maxIter ) + 1
#        no_segments = np.floor ( nsPerSec* len (input_signal) / 44100.0 )
        print iter
        filter_bank_reverse=filter_bank[:,::-1] # for reversing the time axis along the row [::-1] is used for reversing  column indices
        #print np.shape (filter_bank_reverse)
        projection1 = np.zeros( (25, len(input_signal) + filter_bank.shape[1] - 1 ))
        projection2 = np.zeros( (25, len(input_signal) + filter_bank.shape[1] - 1 ))
        projection3 = np.zeros( (25, len(input_signal) + filter_bank.shape[1] - 1 ))

#        
#        for k in range (7):
#            projection1[k,:] = np.convolve( residual [0:len (input_signal) ],filter_bank_reverse [k,:] , mode="full" )
#            #print np.shape (projection1)
#    #    iter
#        for k in range(8,15):
#            projection2[k,:] = np.convolve( residual [0:len (input_signal) ],filter_bank_reverse [k,:] , mode="full" )
#
#   #    
#        for k in  range(16,24):
#            projection3[k,:] = np.convolve( residual [0:len (input_signal) ],filter_bank_reverse [k,:] , mode="full" )
#       
        for k in range (7):
            projection1[k,:] = np.correlate( residual [0:len (input_signal) ],filter_bank [k,:] , mode="full" )
            #print np.shape (projection1)
    #    iter
        for k in range(7,15):
            projection2[k,:] = np.correlate( residual [0:len (input_signal) ],filter_bank [k,:] , mode="full" )

   #    
        for k in  range(16,24):
            projection3[k,:] = np.correlate( residual [0:len (input_signal) ],filter_bank [k,:] , mode="full" )
    #  
    #    
        projection_matrix1 =  projection1 [:7,  kernel_length:]
        projection_matrix2 =  projection2 [7:15,  kernel_length:]
        projection_matrix3 =  projection3 [16:,  kernel_length:]
#        
#        projection_matrix1 =  projection1 [:,  kernel_length-1:length_signal]
#        projection_matrix2 =  projection2 [:,  kernel_length-1:length_signal]
#        projection_matrix3 =  projection3 [:,  kernel_length-1:length_signal]
    #
    #    
    #    
        projection_max1 = np.amax( abs (projection_matrix1), axis = 0 )
        indices_max1 = np.argmax( abs (projection_matrix1), axis = 0 )
        #print indices_max1

        projection_max2 =  np.amax( abs (projection_matrix2), axis = 0 )
        indices_max2 = np.argmax( abs (projection_matrix2), axis = 0 )
        
        projection_max3 =  np.amax( abs (projection_matrix3), axis = 0 )
        indices_max3 = np.argmax( abs (projection_matrix3), axis = 0 )
    #    [projection_max2,indices_max2]=max(abs(projection_matrix2));
    #    [projection_max3,indices_max3]=max(abs(projection_matrix3));
    #
    #
    #    
        processing_window =  int( len( projection_max1 ) / float(no_segments) )
       # print processing_window
        #plt.plot(projection_max1)
        #plt.show()
    #    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for iter2 in range(int(no_segments-2)):
            selected_max_coefficient.append( np.amax( projection_max1[(iter2)*processing_window: (iter2+1)* processing_window] ) )
#            selected_time_indx(cnt) = 
            selected_time_indx.append(np.argmax(projection_max1[(iter2)*processing_window: (iter2+1)*processing_window]));
            
            selected_time_indx[cnt] = selected_time_indx[cnt] + (iter2)*processing_window
           # print selected_time_indx[cnt]
            #print selected_time_indx[cnt]
            selected_channel_indx.append (indices_max1 [selected_time_indx[cnt] ])
            #print selected_channel_indx[cnt]
            selected_max_coefficient[cnt]= projection_matrix1[selected_channel_indx[cnt],selected_time_indx[cnt]]
            residual[ selected_time_indx[cnt]: selected_time_indx[cnt]+kernel_length] = residual[selected_time_indx[cnt]: selected_time_indx[cnt]+kernel_length]-selected_max_coefficient[cnt]*filter_bank[selected_channel_indx[cnt],:]
            cnt=cnt+1;
            
        for iter2 in range(int(no_segments-2)):
            selected_max_coefficient.append( np.amax( projection_max2[(iter2)*processing_window: (iter2+1)* processing_window] ) )
#            selected_time_indx(cnt) = 
            selected_time_indx.append(np.argmax(projection_max2[(iter2)*processing_window: (iter2+1)*processing_window]));
            
            selected_time_indx[cnt] = selected_time_indx[cnt] + (iter2)*processing_window
           # print selected_time_indx[cnt]
            #print selected_time_indx[cnt]
            selected_channel_indx.append (indices_max2 [selected_time_indx[cnt] ])
            #print selected_channel_indx[cnt]
            selected_max_coefficient[cnt]= projection_matrix2[selected_channel_indx[cnt],selected_time_indx[cnt]]
            residual[ selected_time_indx[cnt]: selected_time_indx[cnt]+kernel_length] = residual[selected_time_indx[cnt]: selected_time_indx[cnt]+kernel_length]-selected_max_coefficient[cnt]*filter_bank[selected_channel_indx[cnt],:]
            cnt=cnt+1;
            
        for iter2 in range(int(no_segments-2)):
            selected_max_coefficient.append( np.amax( projection_max3[(iter2)*processing_window: (iter2+1)* processing_window] ) )
#            selected_time_indx(cnt) = 
            selected_time_indx.append(np.argmax(projection_max3[(iter2)*processing_window: (iter2+1)*processing_window]));
            
            selected_time_indx[cnt] = selected_time_indx[cnt] + (iter2)*processing_window
           # print selected_time_indx[cnt]
            #print selected_time_indx[cnt]
            selected_channel_indx.append (indices_max3 [selected_time_indx[cnt] ])
            #print selected_channel_indx[cnt]
            selected_max_coefficient[cnt]= projection_matrix3[selected_channel_indx[cnt],selected_time_indx[cnt]]
            residual[ selected_time_indx[cnt]: selected_time_indx[cnt]+kernel_length] = residual[selected_time_indx[cnt]: selected_time_indx[cnt]+kernel_length]-selected_max_coefficient[cnt]*filter_bank[selected_channel_indx[cnt],:]
            cnt=cnt+1;  
    

    #    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #    for iter2=1:1:no_segments 
    #        [selected_max_coefficient(cnt), selected_time_indx(cnt)]=max(projection_max2(1+(iter2-1)*processing_window:iter2*processing_window));
    #        selected_time_indx(cnt)=selected_time_indx(cnt)+(iter2-1)*processing_window;
    #
    #        selected_channel_indx(cnt)=indices_max2(selected_time_indx(cnt));
    #        selected_max_coefficient(cnt)= projection_matrix2(selected_channel_indx(cnt),selected_time_indx(cnt));
    #        residual(selected_time_indx(cnt): selected_time_indx(cnt)+kernel_length-1)=residual(selected_time_indx(cnt): selected_time_indx(cnt)+kernel_length-1)-selected_max_coefficient(cnt)*gammatone_filterbank(selected_channel_indx(cnt),:);
    #        cnt=cnt+1;
    #    end
    #     
    #    
    #    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%          
    #    for iter2=1:1:no_segments 
    #        [selected_max_coefficient(cnt), selected_time_indx(cnt)]=max(projection_max3(1+(iter2-1)*processing_window:iter2*processing_window));
    #        selected_time_indx(cnt)=selected_time_indx(cnt)+(iter2-1)*processing_window;
    #
    #        selected_channel_indx(cnt)=indices_max3(selected_time_indx(cnt));
    #        selected_max_coefficient(cnt)= projection_matrix3(selected_channel_indx(cnt),selected_time_indx(cnt));
    #        residual(selected_time_indx(cnt): selected_time_indx(cnt)+kernel_length-1)=residual(selected_time_indx(cnt): selected_time_indx(cnt)+kernel_length-1)-selected_max_coefficient(cnt)*gammatone_filterbank(selected_channel_indx(cnt),:);
    #        cnt=cnt+1;
    #    end
    #
    
    #cnt
     #return [selected_max_coefficient,selected_time_indx,selected_channel_indx] 
#    plt.plot(projection1[20,:])
#    plt.show()
    print cnt
    return [selected_max_coefficient,selected_time_indx,selected_channel_indx]


def reconstruct(VAL,CH,T,FB,Leng):
    out=np.zeros(Leng)
    for i in range (len(VAL)):
        out[ T[i]:T[i]+4000] = out[ T[i]:T[i]+4000]+VAL[i]*FB[CH[i],:]
    return out 


