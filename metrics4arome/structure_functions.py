#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 11:11:45 2022

@author: brochetc

Radial structure correlation functions

"""
import numpy as np
from math import ceil
import random
from glob import glob

def load_batch(path,number,CI,Shape=(3,128,128), option='fake'):
    
    if option=='fake':
        
        list_files=glob(path+'_Fsample_*.npy')

        Mat=np.zeros((number, Shape[0], Shape[1], Shape[2]), dtype=np.float32)
        
        list_inds=random.sample(list_files, number)
        for i in range(number):
            Mat[i]=np.load(list_inds[i])[:,:Shape[1],:Shape[2]].astype(np.float32)
            
    elif option=='real':
        
        list_files=glob(path+'_sample*')
        Shape=np.load(list_files[0])[1:4,CI[0]:CI[1], CI[2]:CI[3]].shape
        Mat=np.zeros((number, Shape[0], Shape[1], Shape[2]), dtype=np.float32)
        
        list_inds=random.sample(list_files, number)
        for i in range(number):
            Mat[i]=np.load(list_inds[i])[1:4,CI[0]:CI[1], CI[2]:CI[3]].astype(np.float32)
            
            
        Means=np.load(path+'mean_with_orog.npy')[1:4].reshape(1,3,1,1).astype(np.float32)
        Maxs=np.load(path+'max_with_orog.npy')[1:4].reshape(1,3,1,1).astype(np.float32)
        Mat=(0.95)*(Mat-Means)/Maxs
        

    return Mat


def radial_bin(data, center):
    """
    provide simple radial binning of the data with respect to the center
    
    Inputs:
        data : numpy ndarray, data to be binned
        center : 2 points iterable (list, tuple, array) the center to compute binning from
        
    Returns :
        radial_profile : numpy array, of size R=#(distinct_radial_values_in array)
    """
    
    y, x= np.indices(data.shape)
    r=np.sqrt((x-center[0])**2+(y-center[1])**2)
    r=r.astype(int)
    
    tbin=np.bincount(r.ravel(), data.ravel())
    nr=np.bincount(r.ravel())
    
    radial_profile=tbin/nr
    
    return radial_profile


def increments(data, max_length):
    """
    compute the increments array using max_length as the maximum pixel distance
    selects regularly spaced points to be bases for increments, 
    at the expense of letting boundaries of the sample out of calculation
    
    Inputs:
    
        input_data : np array of shape B x C x H x W
        
        max_length : number of pixels to compute increments on
                     costs decrease quadratically with max_length growth
    
    Returns :
        
        output : np array of shape 
            C x R x H//(2*max_length+1) x C//(2*max_length+1)
            (averaged over batch samples)^
            
        rad_max : int, the maximum radial distance given max_length 
                    (ie the max bin edge in the radial bin)
    
    """
    
    ###### data dimensions
    
    batch=data.shape[0]
    channels=data.shape[1]
    width=data.shape[2]
    
    stride=2*max_length+1
    
    rad_max=ceil(np.sqrt(2*max_length**2))
    
    output=np.zeros((batch, channels,rad_max,width//stride,width//stride))
    
    ######## selecting regularly spaced points to compute increments
    
    for i in range(width//stride):
        for j in range(width//stride):
            
            increm=data[:,:,i*stride:(i+1)*stride, j*stride : (j+1)*stride]

            ref=increm[:,:,max_length, max_length].reshape(batch, channels,1,1)
            increm=np.abs(increm-ref).reshape(batch*channels, stride, stride)
            
            
            
            radial_increm=np.zeros((batch*channels,rad_max))
            
            ##### radial binning for each sample and channels
            
            for k in range(increm.shape[0]):
                radial_increm[k]=radial_bin(increm[k],center=(max_length,max_length))
            
            radial_increm=radial_increm.reshape(batch, channels, rad_max)
            
            output[:,:,:,i,j]=radial_increm
            
    return output, rad_max

def structure_function(increm, p_list):
    """
    compute the structure function of order p from the increment array
    
    Inputs :
        increm: numpy array as returned from the increments function
            shape is B x C x R x h x w
            
        p_list : iterable containing orders of the structure function
        
    Returns :
        struct : numpy array of shape C x R x P  -> input to the power p, averaged
        over spatial dimensions and samples
        
    """
    struct=np.zeros((increm.shape[1],increm.shape[2], len(p_list)))
    for i,p in enumerate(p_list):
        struct[:,:,i]=np.mean(increm**p, axis=(0,3,4))
    return struct

def structure_dist(real_data, fake_data):
    
    real_st=structure_function(increments(real_data,32),[1])
    fake_st=structure_function(increments(fake_data,32),[1])
    
    return np.sqrt(np.mean(real_st-fake_st)**2)
