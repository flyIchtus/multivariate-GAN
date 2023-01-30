#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 10:03:46 2022

@author: brochetc

Plotting Functions for 2D experiments

"""

import matplotlib.pyplot as plt
from torch import empty
from numpy import log10, histogram
import pandas as pd

def online_distrib_plot(epoch,n_samples, train_samples, modelG, device):
    z=empty((n_samples,modelG.nz)).normal_().to(device)
    modelG.train=False
    out_fake=modelG(z).cpu().detach().numpy()
    modelG.train=True
    data_list=[train_samples.cpu().detach(), out_fake]
    legend_list=["Data", "Generated"]
    var_names=["rr" , "u", "v", "t2m"]
    title="Model Performance after epoch "+str(epoch+1)
    option="climato"
    plot_distrib_simple(data_list, legend_list, var_names, title, option)
    
    
def plot_distrib_simple(data_list,legend_list, var_names,title, option):
    """
    plot the distribution of data -- one distribution for each value of the last axis
    """
    fig=plt.figure(figsize=(6,8))
    st=fig.suptitle(title+" "+option, fontsize="x-large")
    data=data_list[0]
    N_var=data.shape[-1]
    columns=1
    for i in range(N_var):
        ax=plt.subplot(N_var, columns, i+1)
        for j,data in enumerate(data_list):
            if var_names[i].find('rr')!=-1:
                o1,o2=histogram(data[:,i], bins=200)
                o2_=o2[:-1]
                o1log=log10(o1)
                ax.plot(o2_, o1log, 'o', label=legend_list[j])
            else :
                ax.hist(data[:,i], bins=200, density=True, label=legend_list[j])
        ax.set_ylabel(var_names[i])
        ax.legend()
    fig.tight_layout()
    st.set_y(0.95)
    fig.subplots_adjust(top=0.9)
    plt.savefig(title+"_"+option+".png")
    plt.show()
    plt.close()

    return 0

    
def plot_GAN_metrics(metrics_list):
    fig=plt.figure(figsize=(6,8))
    st=fig.suptitle("Metrics", fontsize="x-large")
    rows=len(metrics_list)
    columns=1
    for i,metric in enumerate(metrics_list) :
        ax=plt.subplot(rows, columns, i+1)
        ax.plot(range(1,len(metric.data)+1,1),metric.data)
        
        ylabel=metric.name
        ax.set_ylabel(ylabel)
    plt.xlabel("Number of epochs")
    fig.tight_layout()
    st.set_y(0.95)
    fig.subplots_adjust(top=0.9)
    plt.savefig("GAN_metrics_graph.png")
    plt.close()
    
def online_sample_plot(batch, n_samples, Step, var_names, path,coords=False):
    
    batch_to_print=batch[:n_samples]
    IMG_SIZE=batch.shape[2]
    for i, var in enumerate(var_names):
        if var=='t2m':
            varname='2m temperature'
            cmap='coolwarm'
        elif var=='rr':
            varname='Rain rate'
            cmap='Blues'
        elif var=='orog':
            varname='Orography'
            cmap='terrain'
        else :
            varname='Wind '+var
            cmap='viridis'
        fig=plt.figure(figsize=(20,20))
        rows=4
        columns=4
        st=fig.suptitle(varname, fontsize='30')
        st.set_y(0.96)
        for j in range(batch_to_print.shape[0]) :
            b=batch_to_print[j][i].view(IMG_SIZE, IMG_SIZE)
            ax=fig.add_subplot(rows, columns, j+1)
            im=ax.imshow(b.cpu().detach().numpy()[::-1,:], cmap=cmap, vmin=-0.5, vmax=0.5)
        fig.subplots_adjust(bottom=0.05,top=0.9, left=0.05, right=0.9)
        cbax=fig.add_axes([0.92,0.05,0.02,0.85])
        cb=fig.colorbar(im, cax=cbax)
        cb.ax.tick_params(labelsize=20)
        plt.savefig(path+"/Samples_at_Step_"+str(Step)+'_'+var+".png")
        plt.close()
    if coords:
        for i in [-1,-2]:
            fig=plt.figure(figsize=(20,20))
            rows=4
            columns=4
            st=fig.suptitle('Coords', fontsize='30')
            st.set_y(0.96)
            for j in range(batch_to_print.shape[0]) :
                b=batch_to_print[j][i].view(IMG_SIZE, IMG_SIZE)
                ax=fig.add_subplot(rows, columns, j+1)
                im=ax.imshow(b.cpu().detach().numpy()[::-1,:], cmap=cmap, vmin=-1, vmax=1)
            fig.subplots_adjust(bottom=0.05,top=0.9, left=0.05, right=0.9)
            cbax=fig.add_axes([0.92,0.05,0.02,0.85])
            cb=fig.colorbar(im, cax=cbax)
            cb.ax.tick_params(labelsize=20)
            if i==-1 :
                name='CoordsY'
            else:
                name='CoordsX'
            plt.savefig(path+"/Samples_at_Step_"+str(Step)+'_'+name+'.png')
            plt.close()
    return 0

def plot_metrics_from_csv(log_path,filename, metrics_list=[]):
    """
    file structure should be 'Step,metric1,metric2,etc...'
    """
    df=pd.read_csv(log_path+filename)
    if len(metrics_list)==0:
        metrics_list=df.columns[1:]
    N_metrics=len(df.columns[1:])

    figure=plt.figure(figsize=(6,N_metrics*4))
    st=figure.suptitle("Metrics", fontsize="x-large")
    for i,metric in enumerate(metrics_list):
        ax=plt.subplot(N_metrics,1,i+1)
        ax.plot(df['Step'], df[metric])
        ax.set_ylabel(metric)
    plt.xlabel('Iteration step')
    figure.tight_layout()
    st.set_y(0.95)
    figure.subplots_adjust(top=0.9)
    plt.savefig(log_path+"GAN_metrics_graph.png")
    plt.close()
    
def plot_metrics_on_same_plot(log_path,filename, metrics_list=[], targets=[]):
    """
    file structure should be 'Step,metric1,metric2,etc...'
    """
    df=pd.read_csv(log_path+filename)
    if len(metrics_list)==0:
        metrics_list=df.columns[1:]
    
    colors=['b','r','k', 'g', 'orange']

    figure=plt.figure(figsize=(10,10))
    st=figure.suptitle("Metrics", fontsize="x-large")
    for i,metric in enumerate(metrics_list):
        if metric=='criterion':
            label='W1_crop'
        else: 
            label=metric
        plt.plot(df['Step'], df[metric], label=label, color=colors[i])
    plt.hlines(targets,xmin=0,xmax=49000,colors=colors[:len(metrics_list)], linestyles='--')
    plt.grid()
    plt.legend()
    #plt.set_ylabel('Distance')
    plt.xlabel('Iteration step')
    plt.yscale('log')
    figure.tight_layout()
    st.set_y(0.95)
    figure.subplots_adjust(top=0.9)
    plt.savefig(log_path+"GAN_metrics_graph.png")
    plt.close()