#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 13:54:10 2022

@author: brochetc

DataSet/DataLoader classes from Importance_Sampled images
DataSet:DataLoader classes for test samples
"""

import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import ToTensor, Normalize, Compose
from filelock import FileLock


################ reference dictionary to know what variables to sample where
################ do not modify unless you know what you are doing 

var_dict={'rr' : 0, 'u' : 1, 'v' : 2, 't2m' :3 , 'orog' : 4}

################
class ISDataset(Dataset):
    
    def __init__(self, data_dir, ID_file, var_indexes, crop_indexes,\
                 transform=None,add_coords=False):
        
        self.data_dir=data_dir
        self.transform=transform
        self.labels=pd.read_csv(data_dir+ID_file)
        
        ## portion of data to crop from (assumed fixed)
        
        self.CI=crop_indexes
        self.VI=var_indexes
        
  
        
        ## adding 'positional encoding'
        self.add_coords=add_coords
        
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        
        sample_path=os.path.join(self.data_dir, self.labels.iloc[idx,0])
        
        sample=np.float32(np.load(sample_path+'.npy'))\
        [self.VI,self.CI[0]:self.CI[1],self.CI[2]:self.CI[3]] 
        
        importance=self.labels.iloc[idx,1]
        position=self.labels.iloc[idx,2]
        
        ## transpose to get off with transform.Normalize builtin transposition
        sample=sample.transpose((1,2,0))
        
        
        if self.transform:
            sample = self.transform(sample)
            
            
        ## adding coordinates as channels
        
        if self.add_coords:
            
            
            Size=sample.shape[1]
            CoordsX=np.array([[(i/Size) for i in range(Size)] for j in range(Size)],
                               dtype=np.float32)
            CoordsX=0.9*(CoordsX-0.5)/0.5 #centering around 0
            
            CoordsX=CoordsX.reshape(1,Size,Size)
            
            CoordsY=np.array([[(j/Size) for i in range(Size)] for j in range(Size)], 
                               dtype=np.float32)
            CoordsY=0.9*(CoordsY-0.5)/0.5 #centering around 0
            
            CoordsY=CoordsY.reshape(1,Size,Size)
            
            sample=np.concatenate((sample, CoordsX, CoordsY), axis=0)
            
            
        return sample, importance, position


class ISData_Loader():
    
    def __init__(self, path, batch_size, var_indexes, crop_indexes,\
                 shuf=False, add_coords=False):
        
        self.path = path
        self.batch = batch_size
        
        self.shuf = shuf #shuffle performed once per epoch
        
        
        self.VI=var_indexes
        self.CI=crop_indexes
        
        Means=np.load(path+'mean_with_orog.npy')[self.VI]
        Maxs=np.load(path+'max_with_orog.npy')[self.VI]
        
        self.means=list(tuple(Means))
        self.stds=list(tuple((1.0/0.95)*(Maxs)))
        self.add_coords=add_coords
        
    def transform(self, totensor, normalize):
        options = []
        if totensor:
            options.append(ToTensor())

        if normalize:
            options.append(Normalize(self.means, self.stds))
        
        transform = Compose(options)
        return transform
    
    def loader(self, hvd_size=None, hvd_rank=None, kwargs=None):
        
        if kwargs is not None :
            with FileLock(os.path.expanduser("~/.horovod_lock")):    
                dataset=ISDataset(self.path, 'IS_method_labels.csv',\
                                  self.VI,self.CI,self.transform(True,True),\
                                  add_coords=self.add_coords) # coordinates system

        self.sampler=DistributedSampler(
                    dataset, num_replicas=hvd_size,rank=hvd_rank
                    )
        if kwargs is not None:
            loader = DataLoader(dataset=dataset,
                            batch_size=self.batch,
                            shuffle=self.shuf,
                            sampler=self.sampler,
                            **kwargs
                            )
        else:
            loader = DataLoader(dataset=dataset,
                            batch_size=self.batch,
                            shuffle=self.shuf,
                            sampler=self.sampler,
                            )
        return loader
