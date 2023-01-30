#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 11:13:58 2022

@author: brochetc

scattering metrics

"""

import metrics4arome.scattering_funcs as scf
import numpy as np
import metrics4arome.wasserstein_distances as wd
import metrics4arome.sliced_wasserstein as sw
from time import perf_counter

class scattering_metric():
    def __init__(self, J,L, shape, estimators, backend='numpy', frontend='numpy', cuda=False):
        """
        Inputs:
        
        scat_real, scat_fake  : scatteringHandlers obj, one for each data array
            
        estimators : list of str, choice between s22 (shape), s21 (sparsity)
                     the idea is to compute scattering coefficients only once
                     and to derive all estimators from them
        
        """
        
        self.cuda=cuda
        self.J=J
        self.L=L
        self.shape=shape
        
        self.backend=backend
        self.frontend=frontend
        
        self.count=0
        
        self.estNames=[]
        
        for est in estimators:  # set of scattering estimators to be computed
            if est=='s21':
                self.estNames.append('sparsityEstimator')
            elif est=='s22':
                self.estNames.append('shapeEstimator')
            elif est=='S1':
                self.estNames.append('isotropicEstimator')
            else:
                raise ValueError('Unknown estimator')
            
    
    def scattering_rmse(self, real_data, fake_data):
        
        if self.count==0:
            self.scat_real=scf.scatteringHandler(self.J,self.shape,L=self.L, 
                                             frontend=self.frontend, 
                                             backend=self.backend, cuda=self.cuda)
            self.scat_fake=scf.scatteringHandler(self.J, self.shape, L=self.L,
                                             frontend=self.frontend,
                                             backend=self.backend, cuda=self.cuda)
            self.count+=1

        else:
            self.scat_real.reset()
            self.scat_fake.reset()

        _=self.scat_real(real_data)
        _=self.scat_fake(fake_data)
        
        RMSEs=[]
        
        for estName in self.estNames:
        
            est_real=getattr(self.scat_real, estName)()
            est_real = est_real.reshape(est_real.shape[0],est_real.shape[1],-1)
            est_fake=getattr(self.scat_fake, estName)()
            est_fake = est_fake.reshape(est_fake.shape[0], est_fake.shape[1],-1)
        
        
            # computing rmse distance (each component of estimator being 1 d.o.f)
            # on average scattering spectra
            rmse=np.sqrt(((est_fake.mean(axis=0)-est_real.mean(axis=0))**2).mean(axis=-1)) 
        
            RMSEs.append(rmse)
        
        return np.concatenate([np.expand_dims(r, axis=0) for r in RMSEs], axis=0)

    def scattering_distance_crude(self,real_data,fake_data):
        """
        return average of pixel-wise Wasserstein distances between sets of scattering estimators
        for real and generated (fake) data
        
        Same as scattering_distance_crude, but selects only small scales to amplify difference
        
        Inputs :
            real_data, fake_data : numpy arrays of shape B x C x H x W
            
        
        Returns :
            
            distance : numpy array, shape (n_estimators,), result of Mean(Wasserstein(estim_real, estim_fake))
        """
        if self.count==0:
            self.scat_real=scf.scatteringHandler(self.J,self.shape,L=self.L, 
                                             frontend=self.frontend, 
                                             backend=self.backend, cuda=self.cuda)
            self.scat_fake=scf.scatteringHandler(self.J, self.shape, L=self.L,
                                             frontend=self.frontend,
                                             backend=self.backend, cuda=self.cuda)
            self.count+=1

        else:
            self.scat_real.reset()
            self.scat_fake.reset()

        _=self.scat_real(real_data)
        _=self.scat_fake(fake_data)
        
        distances=[]
        for estName in self.estNames:
        
            distances.append(
                    wd.pointwise_W1(getattr(self.scat_real, estName)(),\
                    getattr(self.scat_fake, estName)()).mean(axis=-1)
                )
                
        return np.array(distances)
    
    def scattering_distance_refined(self,real_data,fake_data):
        
        """
        return average of Wasserstein distances between sets of scattering estimators
        for real and generated (fake) data
        
        Same as scattering_distance_crude, but selects only small scales to amplify difference
        
        Inputs :
            real_data, fake_data : numpy arrays
            
        
        Returns :
            
            distance : numpy array, shape (n_estimators,) result of Mean(Wasserstein(estim_real, estim_fake))
            
        
        """

        if self.count==0:
            self.scat_real=scf.scatteringHandler(self.J,self.shape,L=self.L, 
                                             frontend=self.frontend, 
                                             backend=self.backend, cuda=self.cuda)
            self.scat_fake=scf.scatteringHandler(self.J, self.shape, L=self.L,
                                             frontend=self.frontend,
                                             backend=self.backend, cuda=self.cuda)
            self.count+=1

        else:
            self.scat_real.reset()
            self.scat_fake.reset()

        _=self.scat_real(real_data)
        _=self.scat_fake(fake_data)

        distances=[]
        for estName in self.estNames:
        
            distances.append(
                    wd.pointwise_W1(getattr(self.scat_real, estName)()[:,:,:self.J-1],\
                    getattr(self.scat_fake, estName)()[:,:,:self.J-1]).mean(axis=-1)
                )

        
        return np.array(distances)

    
    
    
    ## Normalizing or not normalizing here ?
    ##
    ## In their implementation of SWD, Karras et al. normalize their samples
    ## both in space and in neighbourhoods. Don't we lose something here ?
    ## Mean bias and variance are important in comparing distributions !
    ## I think this is why SWD as implemented by K.etal cannot detect mode collapse to 0
    ## Moreover, normalisation is an important operation which cannot be done 
    ## "under the hood", especially when comparing distributions.
    ## Yet, in the spirit of batch normalization, it is indeed interesting to compare
    ## distributions with same 1st and 2e moments 
    ## (directly probing non gaussianity).
    ##
    ## hence : -> use of scattering_rmse to differentiate the mean and std scattering spectra
    ##         -> use of normalized data to compute SWD on them
    ##         -> use of unnormalized, per-channel SWD  
    ## pros : -> fully interpretable metrics
    ##        -> hopefully smooth training dynamics
    ## cons : -> more computations
    ##        -> more metrics !!!

    
    def scattering_sliced(self, real_data, fake_data, dir_repeats=4, dirs_per_repeat=128):
        """
        
        return Sliced Wasserstein Distance between sets of scattering estimators
        for real and generated (fake) data
        
        Slices are taken using channels as effective dimensions.
        
        Inputs :
            real_data, fake_data : numpy arrays of shape B x C x H x W
            
        
        Returns :
            
            distance : numpy array, shape n_estimators x C+C+1+C, result of 
                              -> Scattering RMSE of mean (per-channel) [0:C]
                              -> Scattering RMSE of std  (per-channel) [C+1:2*C]
                              -> Scattering SWD(estim_real, estim_fake)) [2*C+1:2*C+2]
                              -> Scattering, per channel SWD [2*C+2: 3*C+1]
        """
        
        
        if self.count==0:
            self.scat_real=scf.scatteringHandler(self.J,self.shape,L=self.L, 
                                             frontend=self.frontend, 
                                             backend=self.backend, cuda=self.cuda)
            self.scat_fake=scf.scatteringHandler(self.J, self.shape, L=self.L,
                                             frontend=self.frontend,
                                             backend=self.backend, cuda=self.cuda)
            self.count+=1

        else:
            self.scat_real.reset()
            self.scat_fake.reset()
        
        
        t0=perf_counter()
        _=self.scat_real(real_data)
        _=self.scat_fake(fake_data)
        print('scattering time',perf_counter()-t0)
        
        distances=[]
        
        for estName in self.estNames : # applying the same logic for different estimators
            
            
            t1=perf_counter()
    
            est_real=getattr(self.scat_real, estName)()
            est_real = est_real.reshape(est_real.shape[0],est_real.shape[1],-1)
            est_fake=getattr(self.scat_fake, estName)()
            est_fake = est_fake.reshape(est_fake.shape[0], est_fake.shape[1],-1)
            print('estimator time',perf_counter()-t1)
            
            ### 1st and 2d moment distances on every d.o.f
            
            rmse_mean=np.sqrt(((est_fake.mean(axis=0)-est_real.mean(axis=0))**2).mean(axis=-1))
            rmse_std=np.sqrt(((est_fake.std(axis=0)-est_real.std(axis=0))**2).mean(axis=-1))
            
            
            ### per variable swd
            distance_perVar=np.zeros(est_real.shape[1])
            t2=perf_counter()
            for i in range(est_real.shape[1]):
                distance_perVar[i]=sw.sliced_wasserstein(est_real[:,i], est_fake[:,i],\
                                                    dir_repeats=dir_repeats, dirs_per_repeat=dirs_per_repeat)
            print('SWD perVar time', perf_counter()-t2)
            
            ### per channel normalisation (equiv to batch + instance norm)
            
            est_real -= est_real.mean(axis=(0,-1), keepdims=True)
            est_real /= est_real.std(axis=(0,-1), keepdims=True)
            
            est_fake -= est_fake.mean(axis=(0,-1), keepdims=True)
            est_fake /= est_fake.std(axis=(0,-1), keepdims=True)
            
            
            ### swd on normalized samples
            
            est_real = est_real.reshape(est_real.shape[0],-1)
            est_fake = est_fake.reshape(est_fake.shape[0], -1)
            
            t3=perf_counter()
            swd=sw.sliced_wasserstein(est_real, est_fake, \
                                           dir_repeats=dir_repeats, \
                                           dirs_per_repeat=dirs_per_repeat)
            print('SWD time', perf_counter()-t3)
            
            distance=np.concatenate((rmse_mean, rmse_std, np.array([swd]), distance_perVar), axis=0)
            print(estName+'_distance',distance)
            distances.append(distance)
            
        return np.concatenate([np.expand_dims(d, axis=0) for d in distances], axis=0)
    
    
    
    def scattering_renorm(self, real_data, fake_data, dir_repeats=4, dirs_per_repeat=128):
        """
        
        return Sliced Wasserstein Distance between sets of scattering estimators
        for real and generated (fake) data
        
        Slices are taken using channels as effective dimensions.
        
        Inputs :
            real_data, fake_data : numpy arrays of shape B x C x H x W
            
        
        Returns :
            
            distance : numpy array, shape n_estimators x {C+C+1+C}, result of 
                              -> Scattering RMSE of mean (per-channel) [0:C]
                              -> Scattering RMSE of std  (per-channel) [C+1:2*C]
                              -> Scattering SWD(estim_real, estim_fake)) [2*C+1:2*C+2]
                              -> Scattering, per channel SWD [2*C+2: 3*C+1]
        """
        
        
        if self.count==0:
            self.scat_real=scf.scatteringHandler(self.J,self.shape,L=self.L, 
                                             frontend=self.frontend, 
                                             backend=self.backend, cuda=self.cuda)
            self.scat_fake=scf.scatteringHandler(self.J, self.shape, L=self.L,
                                             frontend=self.frontend,
                                             backend=self.backend, cuda=self.cuda)
            self.count+=1

        else:
            self.scat_real.reset()
            self.scat_fake.reset()
            
        ### per channel normalisation (equiv to batch + instance norm)
        
        real_data -= real_data.mean(axis=(0,2,3), keepdims=True)
        fake_data -= fake_data.mean(axis=(0,2,3), keepdims=True)
        
        real_data /= real_data.std(axis=(0,2,3), keepdims=True)
        fake_data /= fake_data.std(axis=(0,2,3), keepdims=True)
        
        t0=perf_counter()
        
        _=self.scat_real(real_data)
        _=self.scat_fake(fake_data)
        print('scattering time',perf_counter()-t0)
        
        distances=[]
        for estName in self.estNames : # applying the same logic for different estimators
            
            t1=perf_counter()
            
            est_real=getattr(self.scat_real, estName)()
            est_real = est_real.reshape(est_real.shape[0],est_real.shape[1],-1)
            est_fake=getattr(self.scat_fake, estName)()
            est_fake = est_fake.reshape(est_fake.shape[0], est_fake.shape[1],-1)
            
            print('estimator time',perf_counter()-t1)
    
            
            ### per variable swd
            
            distance_perVar=np.zeros(est_real.shape[1])
            t2=perf_counter()
            for i in range(est_real.shape[1]):
                distance_perVar[i]=sw.sliced_wasserstein(est_real[:,i], est_fake[:,i],\
                                                    dir_repeats=dir_repeats, 
                                                    dirs_per_repeat=dirs_per_repeat)
            print('SWD perVar time', perf_counter()-t2)
            
            
            ### swd on full samples
            
            est_real = est_real.reshape(est_real.shape[0],-1)
            est_fake = est_fake.reshape(est_fake.shape[0], -1)
            
            t3=perf_counter()
            swd=sw.sliced_wasserstein(est_real, est_fake, \
                                           dir_repeats=dir_repeats, \
                                           dirs_per_repeat=dirs_per_repeat)
            print('SWD time', perf_counter()-t3)
            
            distance=np.concatenate((np.array([swd]), distance_perVar), axis=0)
            
            print(estName+' distance',distance)
            
            distances.append(distance)
            
        return np.concatenate([np.expand_dims(d, axis=0) for d in distances], axis=0)
    
    def scattering_sliced_perVar(self, real_data, fake_data, dir_repeats=4, 
                                 dirs_per_repeat=128):
        """
        return Sliced Wasserstein distance between sets of scattering estimators 
        for real and generated (fake) data
        
        Slices are taken on each channel separately to have one distance per channel
        
        Inputs:
            
            real_data, fake_data : numpy arrays of shape B x C x H x W
            
        
        Returns :
            
            distance : numpy array of shape (n_estmators, C), result of SWD(estim_real, estim_fake)) for each channel
        
        """
        self.scat_real.reset()
        self.scat_fake.reset()

        _=self.scat_real(real_data)
        _=self.scat_fake(fake_data)
        
        distances=[]
        for estName in self.estNames : # applying the same logic for different estimators
        
            est_real=getattr(self.scat_real, estName)()
            est_real = est_real.reshape(est_real.shape[0],est_real.shape[1],-1)
            est_fake=getattr(self.scat_fake, estName)()
            est_fake = est_fake.reshape(est_fake.shape[0], est_fake.shape[1],-1)
            
            distance_perVar=np.zeros(est_real.shape[1])
            
            for i in range(est_real.shape[1]):
                distance_perVar[i]=sw.sliced_wasserstein(est_real[:,i], est_fake[:,i],\
                                                    dir_repeats=dir_repeats, 
                                                    dirs_per_repeat=dirs_per_repeat)
            distances.append(distance_perVar)
        return np.concatenate([np.expand_dims(d, axis=0) for d in distances], axis=0)

