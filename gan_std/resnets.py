#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 17:29:56 2022

@author: brochetc

resnets

"""



import torch
import torch.nn as nn
from torch.nn.init import orthogonal_
from torch.nn.utils import spectral_norm
from math import log2, sqrt
import network_analysis_tools as nat

###############################################################################    
    ### BASE BLOCKS ###
###############################################################################

class ResBlockUp(nn.Module):
    def __init__(self, input_channels, hidden_channels,output_channels,ksize=3):
        super().__init__()
        
        self.BN1=nn.BatchNorm2d(input_channels)
        self.relu1=nn.LeakyReLU(0.1)
        self.Conv1=nn.Conv2d(input_channels, hidden_channels, (ksize,ksize),\
                             padding=(ksize-1)//2,\
                             stride=(1,1), bias=False)
        
        
        self.BN2=nn.BatchNorm2d(hidden_channels)
        self.relu2=nn.LeakyReLU(0.1)
        self.Conv2=nn.Conv2d(hidden_channels, output_channels, (ksize,ksize),\
                             padding=(ksize-1)//2,\
                             stride=(1,1), bias=False)
        
        
        self.Upscale=nn.Upsample(scale_factor=2, mode='nearest')
        
        
        self.ConvShort=nn.Conv2d(input_channels, output_channels,(1,1),\
                                padding=0, stride=(1,1), bias=False)
    def forward(self,x):
        
        ### block_path ##
        x1=self.BN1(x)
        x1=self.relu1(self.Upscale(x1))
        x1=self.Conv1(x1)
        x1=self.relu2(self.BN2(x1))
        x1=self.Conv2(x1)
        #shortcut upsample
        
        x_up=self.Upscale(x)
        x_up=self.ConvShort(x_up)
        
        
        return x_up+x1

    
class ResBlockDown(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, ksize=3):
        super().__init__()        
        
        self.Conv1=nn.Conv2d(input_channels, hidden_channels,(ksize,ksize),\
                             padding=(ksize-1)//2,\
                             stride=(1,1), bias=False)
        self.relu1=nn.LeakyReLU(0.1)
        
        
        self.Conv2=nn.Conv2d(hidden_channels, output_channels,\
                             (ksize,ksize), padding=(ksize-1)//2,\
                             stride=(1,1), bias=False)

        self.relu2=nn.LeakyReLU(0.1)
        
        
        self.ConvShort=nn.Conv2d(input_channels, output_channels,(3,3),padding=1,\
                                 stride=(2,2), bias=False)
        
        self.Downscale=nn.AvgPool2d((3,3), stride=(2,2), padding=1)
        
    def forward(self,x):
        
        ### block path ##
        x1=self.relu1(x)
        x1=self.Conv1(x1)
        x1=self.relu2(x1)
        x1=self.Conv2(x1)
        x1=self.Downscale(x1)
        ##shortcut - downsample
        x_down=self.ConvShort(x)
        #x_down=self.Downscale(x_down)
        
        return x_down+x1
    

class SimpleBlock_D(nn.Module):
    def __init__(self, input_channels, hidden_channels, ksize=3):
        super().__init__()        
        
        self.Conv1=nn.Conv2d(input_channels, hidden_channels,(ksize,ksize),\
                             padding=(ksize-1)//2,
                             stride=(1,1),bias=False)
        self.relu1=nn.LeakyReLU(0.1)
        
        
        self.Conv2=nn.Conv2d(hidden_channels, input_channels,(ksize,ksize),
                             padding=(ksize-1)//2,\
                             stride=(1,1),bias=False)
        self.relu2=nn.LeakyReLU(0.1)
        
    def forward(self,x):
        
        ## block path
        x1=self.Conv1(self.relu1(x))
        x1=self.Conv2(self.relu2(x1))
        
        
        return x+x1
    
class SimpleBlock_G(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super().__init__()       
        
        self.BN1=nn.BatchNorm2d(input_channels)
        self.Conv1=nn.Conv2d(input_channels, hidden_channels,(3,3), padding=1,\
                             stride=(1,1),bias=False)
        self.relu1=nn.LeakyReLU(0.1)
        
        
        self.BN2=nn.BatchNorm2d(hidden_channels)
        self.Conv2=nn.Conv2d(hidden_channels, input_channels,(3,3), padding=1,\
                             stride=(1,1),bias=False)
        self.relu2=nn.LeakyReLU(0.1)
        
        
    def forward(self,x):
        
        ## block path
        x1=self.BN1(x)
        x1=self.Conv1(self.relu1(x1))
        
        x1=self.BN2(x1)
        x1=self.Conv2(self.relu2(x1))
        
        return x+x1
    
###############################################################################    
    ### FULL NETWORKS ###
###############################################################################
    
#################### Discriminators ###########################################

class ResNet_D128(nn.Module):
    """
    buit upon the architecture provided in Miyato et al., 2018
    initial image size is assumed to be 128*128
    """
    def __init__(self,input_channels):
        super().__init__()
        self.in_ch=input_channels
        
        
        self.Model=nn.ModuleList([
                ResBlockDown(self.in_ch,self.in_ch, 64,3),
                ResBlockDown(64,64,128,3),
                ResBlockDown(128,128,256,3),
                ResBlockDown(256,256,512,3),
                ResBlockDown(512,512,1024,3),
                SimpleBlock_D(1024,1024)])
    
    
        self.Dense=nn.Linear(1024,1)
        
    def forward(self,x):
        for m in self.Model:
            x=m(x)
            
        ### global average pooling
        x=nn.AvgPool2d(kernel_size=x.size()[-2:],)(x)
        x=x.view((-1,1,x.size()[1]))
        ## final classification layer
        x=self.Dense(x)
        return x.view((-1,1,))

class ResNet_D(nn.Module):
    """
    modular version of the above
    fixes adequate number of layers to match input_dimensions
        input_dim : int -> nmber W of witdh location features (W=H is assumed)
        input_dim must be equal to a power of 2
        input_channels : int -> number of variables considered
    """
    
    def __init__(self, input_dim,input_channels):
        super().__init__()
        self.HW=input_dim
        self.in_ch=input_channels
        
        loc_feats=[self.HW//(2**i) for i in range(0,int(log2(self.HW))-2)]
        layNum=len(loc_feats) #number of downsampling residual blocks
        channels=[self.in_ch]+[64*(2**i) for i in range(layNum)]
        k_sizes=[3 for i in range(layNum)]
        
        self.Model=nn.ModuleList(
                [ResBlockDown(channels[i],channels[i], channels[i+1], k_sizes[i])\
                        for i in range(layNum)]\
                 
                 +[SimpleBlock_D(channels[-1], channels[-1])]
                )
        self.Avg=nn.AvgPool2d(kernel_size=(loc_feats[-1]//2,loc_feats[-1]//2),)
        
        self.Dense=nn.Linear(channels[-1],1, bias=False)
    
    def forward(self,x):
        for m in self.Model:
            x=m(x)
        
        ### global average pooling
        x=self.Avg(x)
        #x=nn.AvgPool2d(kernel_size=x.size()[-2:],)(x)
        x=x.view((-1,1,x.size()[1]))
        ## final classification layer
        x=self.Dense(x)
        return x.view((-1,1,))

##################### Generators ##############################################
        

class ResNet_G128(nn.Module):
    """
    buit upon the architecture provided in Miyato et al., 2018
    initial image size is assumed to be 128*128
    """
    def __init__(self, latent_dim, output_channels):
        super().__init__()
        self.nz=latent_dim
        
        #first dense layer
        self.Dense0=nn.Linear(latent_dim,128*128, bias=False)
        
        
        self.Model=nn.ModuleList([
                ResBlockUp(1024,1024,1024),
                ResBlockUp(1024,512, 512),
                ResBlockUp(512,256,256),
                ResBlockUp(256,128,128),
                ResBlockUp(128,64,64)])
    
        #final transforms
        self.BN0=nn.BatchNorm2d(64)
        self.relu0=nn.LeakyReLU(0.1)
        self.ConvFinal=nn.Conv2d(64, output_channels, (3,3),padding=1,\
                                 stride=(1,1), bias=False)
        
        self.th=nn.Tanh()
        
        
    def forward(self, z):
        
        x1=self.Dense0(z)
        x1=x1.view((-1,1024,4,4))
        
        for m in self.Model:
            x1=m(x1)
            
        x1=self.ConvFinal(self.relu0(self.BN0(x1)))
        
        x1=self.th(x1)
        
        return x1
    
class ResNet_G128_rr(nn.Module):
    """
    same as the above but treating first output channel differently 
    to handle precipitation
    """
    def __init__(self, latent_dim, output_channels):
        super().__init__()
        self.nz=latent_dim
        
        #first dense layer
        self.Dense0=nn.Linear(latent_dim,128*128, bias=False)
        
        
        self.Model=nn.ModuleList([
                ResBlockUp(1024,1024,1024),
                ResBlockUp(1024,512, 512),
                ResBlockUp(512,256,256),
                ResBlockUp(256,128,128),
                ResBlockUp(128,64,64)])
    
        #final transforms
        self.BN0=nn.BatchNorm2d(64)
        self.relu0=nn.LeakyReLU(0.1)
        self.ConvFinal=nn.Conv2d(64, output_channels, (3,3),padding=1,\
                                 stride=(1,1), bias=False)
        
        self.th=nn.Tanh()
        self.RELU=nn.LeakyReLU(0.1)
        
    def forward(self, z):
        
        x1=self.Dense0(z)
        x1=x1.view((-1,1024,4,4))
        
        for m in self.Model:
            x1=m(x1)
            
        x1=self.ConvFinal(self.relu0(self.BN0(x1)))
        
        x_rr=x1[:,:1,:,:]
        x_else=x1[:,1:,:,:]
        
        outputs=(self.RELU(x_rr), self.th(x_else))
        x1=torch.cat(outputs, dim=1)
        return x1

class ResNet_G(nn.Module):
    """
    modular version of the above
    fixes adequate number of layers to match output_dimensions
        output_dim : int -> nmber W of witdh location features (W=H is assumed)
            output_dim must be equal to a power of 2
        output_channels : int -> nmber of variables considered
        latent_dim : int -> dimension for latent space; preferably power of 2
    """
    def __init__(self, latent_dim=128, output_dim=128, output_channels=4):
        super().__init__()
        self.nz=latent_dim
        self.HWo=output_dim
        #first dense layer
        self.Dense0=nn.Linear(latent_dim,128*latent_dim, bias=False)
        self.FirstHiddenChannels=8*latent_dim
        
        layNum=int(log2(self.HWo//4)) #number of upsampling residual blocks
        
        channels=[self.FirstHiddenChannels//(2**i) for i in range(layNum)]
        
        self.Model=nn.ModuleList(
                [ResBlockUp(channels[0], channels[0], channels[0])]\
                
            + [ResBlockUp(channels[i],channels[i+1], channels[i+1])\
                for i,_ in  enumerate(channels[:-1])])

    
        #final transforms
        self.BN0=nn.BatchNorm2d(channels[-1])
        self.relu0=nn.LeakyReLU(0.1)
        self.ConvFinal=nn.Conv2d(channels[-1], output_channels, (3,3),padding=1,\
                                 stride=(1,1), bias=False)
       
        self.th=nn.Tanh()
        
        
    def forward(self, z):
        x1=self.Dense0(z)
        
        x1=x1.view((-1,self.FirstHiddenChannels,4,4))
        
        for m in self.Model:
            x1=m(x1)
        
        x1=self.ConvFinal(self.relu0(self.BN0(x1)))
        
        x1=self.th(x1)
        
        return x1
    

def Add_Spectral_Norm(model):
    classname=model.__class__.__name__
    if classname.find('Conv')!=-1:
        model=spectral_norm(model)
    elif classname.find('Linear')!=-1:
        model=spectral_norm(model)

def Orthogonal_Init(model, alsoLinear=False):
    classname=model.__class__.__name__
    if classname.find('Conv')!=-1:
        orthogonal_(model.weight.data, gain=sqrt(2)) #gain to account for subsequ. ReLU
    elif classname.find('Linear')!=-1 and alsoLinear :
        orthogonal_(model.weight.data)

def Name_and_Grad(model):
    li=[]
    classname=model.__class__.__name__
    if classname not in ['ReLU', 'Tanh','Upsample']:
        gradMean=torch.mean(torch.tensor([p.grad[0].norm() for p in model.parameters()]))
        print(classname, gradMean)
        li.append((classname, gradMean))
    return li
