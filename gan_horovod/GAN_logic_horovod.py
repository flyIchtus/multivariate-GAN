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
import horovod.torch as hvd


###############################################################################
##################### Adversarial components training steps ###################
###############################################################################


def Generator_Step_Wasserstein(real, modelD, modelG, optim_D,optim_G,
                               use_amp=False): # scaler=None):
    """
    perform wasserstein generator step
    """
    z=torch.empty(real.size(0), modelG.nz).normal_().cuda()
    for param in modelG.parameters():
        param.grad=None

    if use_amp :
        with torch.cuda.amp.autocast():
            fake=modelG(z)
            out_fake=modelD(fake)
            loss=-(out_fake).mean()
        """for param in modelG.parameters():
           p= param.grad[0].norm() if param.grad is not None else -1
           print('Gen grad gen step ',p)
        for param in modelD.parameters():
           p= param.grad[0].norm() if param.grad is not None else -1
           print('Disc grad gen step ',p)"""
    
    else:
        fake=modelG(z)
        out_fake=modelD(fake)
        loss=-(out_fake).mean()

    loss.backward()
            
    return loss

def Discrim_Step_Hinge(real, modelD, modelG, optim_D,optim_G,\
                       use_amp=False): #scaler=None):
    """
    perform hinge loss (Wasserstein) discriminator step
    """

    z=torch.empty(real.size(0), modelG.nz).normal_().cuda()
    for param in modelD.parameters():
        param.grad=None
    for param in modelG.parameters():
        param.grad=None
    if use_amp:

        with torch.cuda.amp.autocast():
            out_real=modelD(real)
            with torch.no_grad():
                fake=modelG(z)
            out_fake=modelD(fake)
            loss=relu(1.0-out_real).mean()+relu(1.0+out_fake).mean() 
    else:
        out_real=modelD(real)
        fake=modelG(z)
        out_fake=modelD(fake)
        loss=relu(1.0-out_real).mean()+relu(1.0+out_fake).mean()

    loss.backward()
                
    return loss

