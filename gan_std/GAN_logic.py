#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 17:59:36 2022

@author: brochetc

GAN logical steps for different algorithms
"""
import torch
import torch.autograd as autograd
from torch.nn.functional import relu



###############################################################################
##################### Adversarial components training steps ###################
###############################################################################



###############################################################################

def Generator_Step_Wasserstein(real, modelD, modelG, device):
    """
    perform wasserstein generator step
    """
    z=torch.empty(real.size(0), modelG.nz).normal_().to(device) 
    fake=modelG(z)
    out_fake=modelD(fake)
    loss=-(out_fake).mean()

    loss.backward()
            
    return loss
    
    
    return loss

def Discrim_Step_Hinge(real, modelD, modelG, device):
    """
    perform hinge loss (Wasserstein) discriminator step
    """
    
    
    out_real=modelD(real)
    z=torch.empty(real.size(0), modelG.nz).normal_().to(device)
    
    for param in modelD.parameters():
        param.grad=None
    for param in modelG.parameters():
        param.grad=None
    
    fake=modelG(z)
    out_fake=modelD(fake)
    loss=relu(1.0-out_real).mean()+relu(1.0+out_fake).mean()        

    loss.backward()
        
    return loss

###############################################################################

