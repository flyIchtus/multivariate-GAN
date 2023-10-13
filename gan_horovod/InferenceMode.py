#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 12:00:34 2022

@author: brochetc

Model in inference mode test

"""

import os
import glob
import torch
import torch.nn as nn
import resnets_antialiasing as RN
import numpy as np
import argparse
import network_analysis_tools as nat
import pickle


parser=argparse.ArgumentParser()
parser.add_argument('--expe_dir', type=str, \
                    default='./')
args=parser.parse_args()

model_dir=args.expe_dir+'models/'
output_dir=args.expe_dir+'samples/'

os.chdir(output_dir)
if not os.path.exists('Best_model_dataset'):
    os.mkdir('Best_model_dataset')
if not os.path.exists('Best_model_dataset/'):
    os.mkdir('Best_model_dataset/')
output_dir=output_dir+'Best_model_dataset/'
os.chdir(model_dir)




def load_models(path,step, cuda=False):
    """
    load models according to step and
    
    """
    
    nameD='bestdisc_'+str(step)
    nameG='bestgen_'+str(step)
    
    modelG=RN.ResNet_G(64,129,3)
    modelG.apply(RN.Add_Spectral_Norm)
    modelG.apply(RN.Orthogonal_Init)
    
    modelD=RN.ResNet_D(129,3)
    modelD.apply(RN.Add_Spectral_Norm)
    modelD.apply(RN.Orthogonal_Init)
    
    modelG.load_state_dict(torch.load(path+nameG))
    modelG.eval()
    modelG.train=False
    
    modelD.load_state_dict(torch.load(path+nameD))
    modelD.eval()
    modelD.train=False

    if cuda:
        modelG.cuda()
        modelD.cuda()
    latent_dim=modelG.nz
    #channels=modelG.output_channels
    return modelD, modelG, latent_dim
  

def save_network_data(dic,data_type, model_type,savedir,code=''):
    assert data_type in ['gradients', 'features']
    pickle.dump(dic,savedir+model_type+'_'+data_type+'sample_'+code+'.p')


def save_samples(Mat, savedir,code=''):
    assert len(Mat.shape)==4
    Mat=Mat.detach().cpu().numpy()
    for j in range(Mat.shape[0]):
        sample=Mat[j,:,:,:]
        np.save(savedir+'_Fsample_'+code+'_'+str(j)+'.npy',sample)
        
def extract(steps,layers_dic,path,data_type, N_samples,savedir,batch_size,cuda=False):
    
    """
    run models in inference mode for a given set of training steps and produces
    samples, either only real samples (data_type='outputs'), or activations features
    (data_type='features')
    
    
     gradients : to be added (maybe) later
    """
    
    assert data_type in ['outputs', 'gradients', 'features']
    
    for step in steps:
        modelD, modelG, latent_dim=load_models(path, step, cuda)
        if data_type=='features':
            moduleG=nat.FeatureExtractor(modelG, layers_dic['generator'])
            moduleD=nat.FeatureExtractor(modelD, layers_dic['discriminator'])
            
            for i in range(N_samples//batch_size):
                print('Step {}'.format(i))
                if i<N_samples//batch_size-1:
                    Z=torch.empty((batch_size, latent_dim)).normal_()
                else:
                    Z=torch.empty((N_samples-i*batch_size, latent_dim)).normal_()
                
                if cuda :
                    Z=Z.cuda()
                    
                with torch.no_grad():
                    
                    actiG, X_fake=moduleG(Z)
                    actiD, _=moduleD(X_fake)
                    save_network_data(actiG,data_type,'gen',\
                              savedir,code=str(step)+'_'+str(i))
                    
                    save_network_data(actiD,data_type,'disc',\
                              savedir,code=str(step)+'_'+str(i))
                    
                    save_samples(X_fake, savedir, code=str(step)+'_'+str(i))
                    
        if data_type=='outputs':
            
           for i in range(N_samples//batch_size):
                print('Step {}'.format(i))
                if i<N_samples//batch_size-1:
                    Z=torch.empty((batch_size, latent_dim)).normal_()
                else:
                    Z=torch.empty((N_samples-i*batch_size, latent_dim)).normal_()
                
                if cuda :
                    Z=Z.cuda()
                    
                with torch.no_grad():
                   X_fake=modelG(Z)
                   _=modelD(X_fake)
                    
                   save_samples(X_fake, savedir, code=str(step)+'_'+str(i))
        


################################### RUNTIME ###################################

N_samples=66048
batch_size=1024

list_models=glob.glob('bestdisc_*')

list_steps=[48000]# for ch in list_models]
print(list_steps)
extract(list_steps,{}, model_dir, 'outputs',N_samples, 
        output_dir, batch_size, cuda=True) 






