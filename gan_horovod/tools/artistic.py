#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 15:13:47 2022

@author: brochetc and mouniera

artistic stuff with matplotlib & cartopy
To plot nice visualizations of GAN _generated or AI generated fields
"""

import h5py,os
import numpy as np
import random
import datetime
import pickle
import matplotlib
#matplotlib.use("Agg")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import AxesGrid
from cartopy.mpl.geoaxes import GeoAxes
from glob import glob


data_dir_fake='./'
data_dir='./'
data_dir_vent='./'
path_plot='./'
CI=(78,207,55,184)

Maxs=np.load(data_dir+'max_with_orog.npy')[1:4].reshape(3,1,1)
Means=np.load(data_dir+'mean_with_orog.npy')[1:4].reshape(3,1,1)

def extract_lonlat() :
    #Extraction des longitudes,latitudes des pdg. A modifier si grille modifiée !
    with open('/home/brochetc/gan4arome/gan_horovod/tools/latlon.file','rb') as f:
               lonlat=pickle.load(f,encoding="latin1")
    return lonlat


def grid_to_lat_lon(X,Y):
    """
    renormalize a pixel-wise grid to Latitude/longitude Coordinates
    
    Constants are fixed to center the domain on AROME domain 
    
    """
    Lat_min=37.5
    Lat_max=55.4
    Lon_min=-12.0
    Lon_max=16.0
    n_lat=717
    n_lon=1121
    Lat=Lat_min+Y*(Lat_max-Lat_min)/n_lat
    Lon=Lon_min+X*(Lon_max-Lon_min)/n_lon
    return Lat,Lon

def get_boundaries(Zone) :
    """
    retrieves left-most, bottom-most domain boundaries for distinct climatologic regions 
    over AROME-FRANCE
    
    Inputs : 
        Zone : str the region to be selected
    
    Returns : 
        X_min, Y_min : int; the indexes on AROME grid of the bottom left corner of the region
    
    """
    Zone_l=["NO","SO","SE","NE","C","SE_for_GAN"]
    X_min=[230,250,540,460,360,540+55]
    Y_min=[300,150,120,300,220,120+78]
    index=Zone_l.index(Zone)
    return X_min[index],Y_min[index]
 
def cartesian2polar(u,v):
    """
    
    Transform cartesian notation (u,v) of a planar vector
    into its module+direction notation
    
    -------------
    Inputs :
    
        u, v, respectively the x- and y- coords of the vector
    
    Returns :
        module, direction :tuple
        
        The module and normalized coordinates (respectively cos(theta), sin(theta) in the polar framework)
        
    -------------
    """
    
    
    module=np.sqrt(u**2+v**2)
    direction=(u/module, v/module)
    return module, direction

        
def standardize_samples(data,
                     normalize=False, norm_vectors=(0.0,1.0),
                     chan_ind=None,
                     crop_inds=None):
    """
    --------------------------------
    Inputs :
    
        normalize : list (optional) -> list of samples to be normalized
            
        norm_vectors : tuple(array) (optional) -> target mean and absolute maximum
                    to be used for normalization
                    array should be of size len(var_names)
                    
        crop : bool (optional) -> check if data should be cropped
            
        crop_inds : list(int) -> crop indexes
                                (xmin, xmax, ymin, ymax)"""
    
    if normalize:
            Means=norm_vectors[0]
            Maxs=norm_vectors[1]
    if chan_ind==None and crop_inds==None:
        data_clean=data
    elif chan_ind is not None and crop_inds==None:    
        data_clean=data[data.shape[0], chan_ind, 
                             data.shape[2], data.shape[3]]
    else:
        data_clean=data[:, chan_ind, 
                             crop_inds[0]:crop_inds[1],
                             crop_inds[2]:crop_inds[3]]

    if normalize:            
            
        data_clean=(1/0.95)*data_clean*Maxs+Means

    return data_clean
   
class canvasHolder():
    def __init__(self, Zone, nb_lon, nb_lat):
        
        
        self.X_min, self.Y_min=get_boundaries(Zone)
        self.nb_lon, self.nb_lat=nb_lon,nb_lat
        
        
        self.lonlat=extract_lonlat()
        
        self.Coords=[
                self.lonlat[0][self.Y_min:(self.Y_min+nb_lat),
                     self.X_min:(self.X_min+nb_lon)],\
                self.lonlat[1][self.Y_min:(self.Y_min+nb_lat),
                      self.X_min:(self.X_min+nb_lon)]
                ]
        
        self.proj0=ccrs.Stereographic(central_latitude=46.7,central_longitude=2)
        
        self.proj_plot=ccrs.PlateCarree()
        self.axes_class= (GeoAxes,dict(map_projection=self.proj0))
        
        
    def project(self,padX=(10,45), padY=(10,10), ax=None) :
        """
        
        create a plt.axes object using  a Stereographic projection
        Using grid indexes coordinates and width
        possibly reuses an existing ax object
        
        Inputs :
            
            X_min, Y_min : int, bottom left corner coordinates
            nb_lat, nb_lon : int, pixel width in latitude and longitude
            ax : None / plt.axes object, to be either manipulated or created
            
        Returns :
            
            ax : a plt.axes object ready to be filled with data, 
            incorporating Borders and Coastline
        
        """
        
        if ax==None:
            ax = plt.axes(projection=self.proj0)
       
        Lat_min,Lon_min= grid_to_lat_lon(self.X_min-padX[0],self.Y_min-padY[0])
        Lat_max,Lon_max= grid_to_lat_lon(self.X_min+self.nb_lon+padX[0],
                                         self.Y_min+self.nb_lat+padY[0])
        lon_bor=[Lon_min,Lon_max]
        lat_bor=[Lat_min,Lat_max]
       
        #Projecting boundaries onto stereographic grid
        lon_lat_1=self.proj0.transform_point(lon_bor[0],lat_bor[0],
                                             ccrs.PlateCarree())
        lon_lat_2=self.proj0.transform_point(lon_bor[1],lat_bor[1],
                                             ccrs.PlateCarree())
        
        #Properly redefining boundaries
        lon_bor=[lon_lat_1[0],lon_lat_2[0]]
        lat_bor=[lon_lat_1[1],lon_lat_2[1]]
        borders = lon_bor + lat_bor
        
        ax.set_extent(borders,self.proj0) # map boundaries under the right projection
        ax.add_feature(cfeature.COASTLINE.with_scale('10m')) # adding coastline
        ax.add_feature(cfeature.BORDERS.with_scale('10m')) # adding borders
        
        return ax

    def plot_data_normal(self,data, var_names,
                  plot_dir, pic_name,
                  contrast=False, cvalues=(-1.0,1.0)):
        
        """
        
        use self-defined axes structures and projections to plot numerical data
        and save the figure in dedicated directory
        
        Inputs :
            
            data: np.array -> data to be plotted shape Samples x Channels x Lat x Lon
                          with  Channels being the number of variables to be plotted
                
            plot_dir : str -> the directory to save the figure in
            
            pic_name : str -> the name of the picture to be saved
                
            
            contrast : bool (optional) -> check if boundary values for plot 
                                         shoud be imposed (same value for all variables)
            
            cvalues : tuple (optional) -> bottom and top of colorbar plot [one
             for each variable]
            
            withQuiver : bool (optional) -> adding wind direction arrows on top of wind magnitude
            
    
        Returns :
            
            
        Note :
            
            last docstring review by C .Brochet 15/04/2022
            
        """
            
            
        fig=plt.figure(figsize=(20,24))
        axes={}
        ims={}
        
        grid=AxesGrid(fig, 111, axes_class=self.axes_class,
                  nrows_ncols=(len(var_names),data.shape[0]),
                  axes_pad=0.4, cbar_mode= 'None',
                  cbar_pad=0.0,label_mode='')
        
        for ind in range(data.shape[0]):
            #plotting each sample 
            
            
            data_plot=data[ind,:,:,:]
                
                
            for i, var in enumerate(var_names):
                
                Var=var[0]
                unit=var[1]
                
                cmap='coolwarm' if Var=='t2m' else 'viridis'
                axes[Var+str(ind)]=self.project(ax=grid[i*data.shape[0]+ind])
                if not contrast:
                    ims[Var+str(ind)]=axes[Var+str(ind)].pcolormesh(
                        self.Coords[0],\
                        self.Coords[1],\
                        data_plot[i,:,:],\
                        cmap=cmap,alpha=1,transform=self.proj_plot)
                else :
                    ims[Var+str(ind)]=axes[Var+str(ind)].pcolormesh(
                        self.Coords[0],\
                        self.Coords[1],\
                        data_plot[i,:,:],\
                        cmap=cmap,alpha=1,vmin=cvalues[0], vmax=cvalues[1],
                        transform=self.proj_plot)
                
                varTitle=Var+' ('+unit+')'
                if ind==0:
                    #Title+=' GAN'
                    axes[Var+str(ind)].title.set_text('PEARO, '+varTitle)
                elif i==0:
                    axes[Var+str(ind)].title.set_text('GAN')
        Title='Full state'
        st=fig.suptitle(Title, fontsize='25')
        st.set_y(0.98)
        fig.canvas.draw()
        fig.tight_layout()
        plt.savefig(plot_dir+pic_name)
        plt.close()
        
        """cbar=grid.cbar_axes[i+3*ind].colorbar(ims[var+str(ind)])
        yvalues=np.linspace(data[i,:,:].min(), data[i,:,:].max(),15)
        cbar.ax.set_yticks(yvalues)
    
        ylabels=['{:.1f}'.format(np.float32(xa)) for xa in yvalues]
        cbar.ax.set_yticklabels(ylabels, va='center',fontsize=8)"""
        
        
    def plot_data_wind(self, data,
                       plot_dir, pic_name,
                       contrast=False, cvalues=(-1.0,1.0),
                       withQuiver=True):
        """
        
        use self-defined axes structures and projections to plot wind magnitude and
        direction
        
        Inputs :
            
            data: np.array -> data to be plotted shape Samples x 2 x Lat x Lon
                          with 0 dimension being 'meridian wind' and
                          1 dimension being 'zonal wind'
                
            plot_dir : str -> the directory to save the figure in
            
            pic_name : str -> the name of the picture to be saved
                
            
            contrast : bool (optional) -> check if boundary values for plot 
                                         shoud be imposed (same value for all variables)
            
            cvalues : tuple (optional) -> bottom and top of colorbar plot [one
             for each variable]
            
            withQuiver : bool (optional) -> adding wind direction arrows on top of wind magnitude
            
    
        Returns :
            
            
        Note :
            
            last docstring review by C .Brochet 15/04/2022
            
        """
            
        fig=plt.figure(figsize=(20,9))

        axes={}
        ims={}
        
        grid=AxesGrid(fig, 111, axes_class=self.axes_class,
                  nrows_ncols=(1,data.shape[0]),
                  axes_pad=0.4, cbar_mode= 'None',
                  cbar_pad=0.0,label_mode='')

        u,v=data[:,0,:,:], data[:,1,:,:]
        quiSub=u.shape[1]//32 #subsampling ratio for arrow drawings
        
        module, direction=cartesian2polar(u,v)
        
        axes={}
        ims={}
        
        for ind in range(data.shape[0]):
            Title='Wind magnitude (m/s)'
            axes['wind_mag'+str(ind)]=self.project(ax=grid[ind])
            if not contrast :
                ims['wind_mag'+str(ind)]=axes['wind_mag'+str(ind)].pcolormesh(
                    self.Coords[0],\
                   self.Coords[1],\
                   module[ind,:,:],\
                   cmap='plasma',alpha=1,transform=self.proj_plot)
            else :
                ims['wind_mag'+str(ind)]=axes['wind_mag'+str(ind)].pcolormesh(
                    self.Coords[0],\
                   self.Coords[1],\
                   module,\
                   cmap='plasma',alpha=1,
                   vmin=cvalues[0], vmax=cvalues[1],
                   transform=self.proj_plot)
                

            if withQuiver:   
                axes['wind_dir'+str(ind)]=self.project(ax=grid[ind])
                ims['wind_dir'+str(ind)]=axes['wind_dir'+str(ind)].quiver(
                    x=np.array(self.Coords[0])[::quiSub,::quiSub],\
                    y=np.array(self.Coords[1])[::quiSub,::quiSub],
                    u=direction[0][ind,::quiSub,::quiSub],
                    v=direction[1][ind,::quiSub,::quiSub],
                    scale=32.0, color='white',
                    transform=self.proj_plot)
                Title='Wind magnitude (m/s) and direction'
            if ind==0:
               #Title+=' GAN'
               axes['wind_mag'+str(ind)].title.set_text('PEARO')
            else:
               axes['wind_mag'+str(ind)].title.set_text('GAN')

        st=fig.suptitle(Title, fontsize='25')
        st.set_y(0.96)
        fig.canvas.draw()
        fig.tight_layout()
        plt.savefig(plot_dir+pic_name)
        plt.close()
