#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 10:09:34 2022

@author: brochetc


Wavelet like metrics 

Using package kymatio

"""

from kymatio import Scattering2D
import numpy as np
from glob import glob
import torch
import random


def load_batch(path,number,CI,Shape=(3,128,128), option='fake'):
    
    if option=='fake':
        
        list_files=glob(path+'_Fsample_*.npy')

        Mat=np.zeros((number, Shape[0], Shape[1], Shape[2]))
        
        list_inds=random.sample(list_files, number)
        for i in range(number):
            Mat[i]=np.load(list_inds[i])[:,:Shape[1],:Shape[2]]
            
    elif option=='real':
        
        list_files=glob(path+'_sample*')
        Shape=np.load(list_files[0])[1:4,CI[0]:CI[1], CI[2]:CI[3]].shape
        Mat=np.zeros((number, Shape[0], Shape[1], Shape[2]))
        
        list_inds=random.sample(list_files, number)
        for i in range(number):
            Mat[i]=np.load(list_inds[i])[1:4,CI[0]:CI[1], CI[2]:CI[3]]
            
            
        Means=np.load(path+'mean_with_orog.npy')[1:4].reshape(1,3,1,1)
        Maxs=np.load(path+'max_with_orog.npy')[1:4].reshape(1,3,1,1)
        Mat=(0.95)*(Mat-Means)/Maxs
        

    return Mat      

class scatteringHandler():
    """
    Manipulation interface for Numpy Scattering2D functions
    This class hold objects that can be dynamically computed only once
    and then are stored to avoid computational overhead in case of multiple calls
    
    It enables the computation and disentangling of scattering coefficients of 
    1st and 2d order, making them ready to be plotted by ad-hoc functions
    
    """
    
    def __init__(self,J,shape,L=8, \
                 max_order=2,pre_pad=False, \
                 frontend='numpy',\
                 backend='numpy',out_type='array', cuda=False):
        
        self.scattering=Scattering2D(J,shape,L,
             max_order=max_order,pre_pad=pre_pad,
             frontend=frontend,backend=backend,
             out_type=out_type)  #kymatio scattering object
        self.cuda=cuda
        if cuda and torch.cuda.is_available():
            self.scattering.cuda()
        self.J=J
        self.L=L
        self.shape=shape
        self.coeff=None # where the result of the call we be stored
        self.S1_j1=None
        self.S2_j1j2=None
        self.order2_ind=None
        self.s21_j1j2=None 
        self.s22_j1j2=None

    
    def __call__(self, x):
        """
        Calculate scattering coefficients with Kymatio's conventions
        (often the main computational effort)
        if cuda is available, only this part is done on GPU 
        (approximate 100x speed-up for rgb 128x128 images)
        
        Inputs:
            x : numpy array or torch tensor; shape is batch x Var x Height x Width
            
            batch is preferably a power of 2
        
        Returns :
            
            coeff : numpy.array
        """
     
        if self.coeff is None:
            
            Nbatch, Nch, Size=x.shape[0],x.shape[1], x.shape[-1]
            Sizemax=self.shape[-1]
            
            
            if Size>Sizemax:
                x=x[:,:,:Sizemax,:Sizemax]
            elif Size<Sizemax:
                raise ValueError('Wrong data dimension')
            
            Ncoeffs=1 + self.J * self.L + (self.L**2) * (self.J * (self.J-1) //2)
            
            
            if self.cuda :
                
                
                if Nbatch>1024: # if the data shape is too large for GPU memory, 
                                    # slicing into chunks is performed
                    steps=Nbatch//1024
                    Slice=Nbatch//steps
                    res=torch.zeros((Nbatch,Nch,Ncoeffs, Sizemax//2**self.J,Sizemax//(2**self.J)))
                    
                    for i in range(steps) :       
                        
                        y=torch.tensor(x[Slice*i:Slice*(i+1)],dtype=torch.float32).cuda()
        
                        res0=self.scattering(y)
                        res[Slice*i: Slice*(i+1)]=res0.cpu()
                        del res0
                        
                else :
                    res=self.scattering(x.cuda())
                
                self.coeff=res.numpy()
                
                
            else :
                self.coeff=self.scattering(x)

            
        return self.coeff
    
    def reset(self):
        """
        put all relevant fields to None so that computing can be done with new data
        """
        
        self.coeff=None # where the result of the call we be stored
        self.S1_j1=None
        self.S2_j1j2=None
        self.order2_ind=None
        self.s21_j1j2=None 
        self.s22_j1j2=None
        
    
    def order_extractor(self,order):
        """
        Extract relevant coefficients from the results array
        corresponding to a given order :
            0 : mean coefficient
            1 : pixel-averaged first-order coefficients (scale, orientation dependent)
            2 : pixel-averaged second-order coefficients (scale 1, scale 2, orientation dependent)
            distance_perVar=np.zeros(est_real.shape[1])
        
        for i in range(est_real.shape[1]):
            distance_perVar[i]=sw.sliced_wasserstein(est_real[:,i], est_fake[:,i],\
                                                dirs_repeat=128, dirs_per_repeat=128)
        Returns :
            numpy array or torch tensor corresponding to the extracted indexes
        """
        
        if order==0:
            return np.mean(self.coeff[:,:,0,:,:],axis=(2,3))
        if order==1:
            return np.mean(self.coeff[:,:,1:self.J*self.L+1,:,:], axis=(3,4))
        
        if order==2:
            return np.mean(self.coeff[:,:,self.J*self.L+1:,:,:], axis=(3,4))
    
    def isotropicEstimator(self, order,scat=None):
        
        assert order>0
        
        if scat is None:
            scat = self.order_extractor(order)
        
        if order==1:
            self.S1_j1 = np.reshape(scat, 
                                    (scat.shape[0], scat.shape[1],
                                     self.J, self.L))
            
            return np.mean(self.S1_j1, axis=3)
            
        else: #assuming order==2 here
            if self.order2_ind is None:
                #indexing as provided by the kymatio documentation
                
                self.order2_ind=[[[[self.L**2*(j1*(self.J-1)-j1*(j1-1)//2)\
                               +self.L*(j2-j1-1)\
                               +self.L*(self.J-j1-1)*l1\
                               +l2 \
                                for l2 in range(self.L)]\
                                for l1 in range(self.L)]\
                                for j2 in range(j1+1,self.J)] \
                                for j1 in range(self.J-1)] 
    
            if self.S2_j1j2 is not None:
                return self.S2_j1j2
                    
            S2_j1j2=[]
            for j1 in range(self.J-1):
                for j2 in range(self.J-j1-1):
                    S2_j1j2.append(np.mean(
                            scat[:,:,self.order2_ind[j1][j2]],\
                                           axis=(-1,-2))
                            )
            self.S2_j1j2=np.array(S2_j1j2).transpose(1,2,0)
            return self.S2_j1j2
    
    def sparsityEstimator(self, scat=None):
        """
        for order 2 computation of the s21 scattering sparsity estimator
        use of order 1 S1_j1
        """
        
        if self.s21_j1j2 is not None:
            return self.s21_j1j2
        if scat is None:
            scat=self.order_extractor(2)
        if self.order2_ind is None:
            #indexing as provided by the kymatio documentation
            
            self.order2_ind=[[[[self.L**2*(j1*(self.J-1)-j1*(j1-1)//2)\
                           +self.L*(j2-j1-1)\
                           +self.L*(self.J-j1-1)*l1\
                           +l2 \
                            for l2 in range(self.L)]\
                            for l1 in range(self.L)]\
                            for j2 in range(j1+1,self.J)] \
                            for j1 in range(self.J-1)] 
        if self.S1_j1 is None:
            _=self.isotropicEstimator(1, scat=None)
            
        # averaging on 2d order orientations
        S2_j1j2l1=[]
        for j1 in range(self.J-1):
            for j2 in range(self.J-j1-1):
                for l1 in range(self.L):
                    
                    S2_j1j2l1.append(
                            np.mean(
                            scat[:,:,self.order2_ind[j1][j2][l1]],\
                            axis=-1)
                            )
                            
        S2_j1j2l1=np.array(S2_j1j2l1).transpose(1,2,0)
        
        s21_j1j2=[]
        for j1 in range(self.J-1):
            for j2 in range(self.J-j1-1):
                s21_j1j2l1=[]
                for l1 in range(self.L):
                    
                    s21_j1j2l1.append(
                            S2_j1j2l1[:,:,j1*(self.J-1)+(self.J-j1-1)*j2+l1]
                            /self.S1_j1[:,:,j1,l1]
                            )
                    
                    
                s21_j1j2.append(np.array(s21_j1j2l1).mean(axis=0))
                
        self.s21_j1j2=np.array(s21_j1j2).transpose(1,2,0)
        
        return self.s21_j1j2
    
    def shapeEstimator(self, scat=None):
        
        """
        for order 2 computation of the s22 scattering sparsity estimator
        use of order 1 S1_j1
        """
        
        if self.s22_j1j2 is not None:
            return self.s22_j1j2
        
        if scat is None:
            scat=self.order_extractor(2)
            
        if self.order2_ind is None:
            #indexing as provided by the kymatio documentation
            
            self.order2_ind=[[[[self.L**2*(j1*(self.J-1)-j1*(j1-1)//2)\
                                   +self.L*(j2-j1-1)\
                                   +self.L*(self.J-j1-1)*l1\
                                   +l2 \
                                    for l2 in range(self.L)]\
                                    for l1 in range(self.L)]\
                                    for j2 in range(j1+1,self.J)] \
                                    for j1 in range(self.J-1)]  
        
        s22_j1j2=[]

        for j1 in range(self.J-1):
            for j2 in range(self.J-j1-1):
                s22_j1j2l1_eq=[]
                s22_j1j2l1_ortho=[]
                for l1 in range(self.L):
                    s22_j1j2l1_eq.append(
                            scat[:,:,self.order2_ind[j1][j2][l1][l1]]
                            )
                    
                    s22_j1j2l1_ortho.append(
                            scat[:,:,self.order2_ind[j1][j2][l1][(l1+self.L//2)%self.L]]
                            )
                    
                s22_j1j2l1_eq=np.array(s22_j1j2l1_eq)
                s22_j1j2l1_ortho=np.array(s22_j1j2l1_ortho)
                
                s22_j1j2.append(
                        np.mean(
                                s22_j1j2l1_eq/s22_j1j2l1_ortho,
                                axis=0))

        self.s22_j1j2=np.array(s22_j1j2).transpose(1,2,0)
        
        return self.s22_j1j2