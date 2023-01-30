#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:52:42 2022

@author: brochetc
"""

import artistic as art
import numpy as np
import random
import matplotlib.pyplot as plt
from glob import glob

data_dir_fake='/home/brochetc/Bureau/Thèse/présentations_thèse/images_des_entrainements/echantillons/Norm/'
data_dir='/home/brochetc/Bureau/Thèse/présentations_thèse/images_des_entrainements/echantillons/'
data_dir_vent='/home/brochetc/Bureau/Thèse/présentations_thèse/images_des_entrainements/echantillons_vent/'
path_plot='/home/brochetc/Bureau/Thèse/présentations_thèse/images_des_entrainements/'
CI=(78,207,55,184)

Maxs=np.load(data_dir+'max_with_orog.npy')[1:4].reshape(3,1,1)
Means=np.load(data_dir+'mean_with_orog.npy')[1:4].reshape(3,1,1)


if __name__=="__main__":
    
    var_names=[('u', 'm/s'), ('v', 'm/s'), ('t2m', 'K')]
    
    
        
    litrue=glob(data_dir+"_sample*.npy")
    lifalse=glob(data_dir_fake+"_Fsample_49000_0*.npy")
    
    data_r_name=random.sample(litrue,1)[0][len(data_dir):]
    data_f_name=random.sample(lifalse,3)
    
    data=np.zeros((4,3,129,129))
    
    
    data0=np.expand_dims(np.load(data_dir+data_r_name),axis=0)
    data[0]=art.standardize_samples(data0, chan_ind=[1,2,3], crop_inds=CI)
    
    
    can=art.canvasHolder("SE_for_GAN",129,129)
    
    for j in range(3):
        dataname=data_f_name[j][len(data_dir_fake):]
        data_j=np.expand_dims(np.load(data_dir_fake+dataname),axis=0)
        data[j+1]=art.standardize_samples(data_j,
            normalize=[0],norm_vectors=(Means,Maxs))
        
    #data_name="_sample1911.npy"
    
    can.plot_data_normal(data,var_names,path_plot, 'new_artistic0.png')
    can.plot_data_wind(data[:,0:2,:,:], path_plot,'new_artistic_wind.png',withQuiver=False)
    can.plot_data_wind(data[:,0:2,:,:], path_plot,'new_artistic_wind_quiver.png',withQuiver=True)