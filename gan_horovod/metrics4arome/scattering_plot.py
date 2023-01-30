#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 13:51:11 2022

@author: brochetc


Scattering coefficients plotting functions


"""
import matplotlib.pyplot as plt

def plot_2o_Estimators(estim_fake, estim_real, name, J,L, output_dir):
    """
    Plot and save figures for second order_like scattering estimators 
    of fake (generated) and real (extracted from dataset) samples.
    
    Inputs :
        
        estim_fake : numpy array representing 2d order scattering estimator
        of the GAN generated samples
        shape is Batch x Variables x J(J-1)//2 where J is the nmber of scales used
        
        estim_true : numpy array representing 2d order scattering estimator
        of the dataset samples
        shape is Batch x Variables x J(J-1)//2 where J is the nmber of scales used
    
        name : str, name of the estimator 
        (either "shape" or "sparsity" as far as we are interested)
        
        J : int, number of scales involved in the computation
        
        L : int, number of directions involved in the computation
        
        output_dir : str, the directory to save figures in
    
    """
    

    for i, var in enumerate(['$u$', '$v$', '$t_{2m}$']):

        print(var)
        fig, axs = plt.subplots(1, J-1, figsize=(15, 6),
                            sharex=False, sharey=True)
        for j_up in range(1,J):
            
            si=sum([k for k in range(j_up+1, J)]) #starting index
            ei=si+j_up #ending index
            
            
            axs[J-1-j_up].plot([1+j2+(J-j_up) for j2 in range(j_up)], 
               estim_fake[:,i,si:ei].mean(axis=0), 'b-', label='Mean GAN')
            
            axs[J-1-j_up].plot([1+j2+(J-j_up) for j2 in range(j_up)], 
               estim_fake[:,i,si:ei].mean(axis=0)
                                    +estim_fake[:,i,si:ei].std(axis=0), 
                                    'b--', label=r'Mean $\pm$ Std GAN')
            axs[J-1-j_up].plot([1+j2+(J-j_up) for j2 in range(j_up)],
               estim_fake[:,i,si:ei].mean(axis=0)
                                    -estim_fake[:,i,si:ei].std(axis=0), 'b--')
            
            axs[J-1-j_up].plot([1+j2+(J-j_up) for j2 in range(j_up)], 
               estim_fake[:,i,si:ei].max(axis=0), 'b+', label='Max/Min GAN')
            axs[J-1-j_up].plot([1+j2+(J-j_up) for j2 in range(j_up)], 
               estim_fake[:,i,si:ei].min(axis=0), 'b+')
        

            axs[J-1-j_up].plot([1+j2+(J-j_up) for j2 in range(j_up)], 
               estim_real[:,i,si:ei].mean(axis=0), 'r-', label='Mean PEARO')
            
            axs[J-1-j_up].plot([1+j2+(J-j_up) for j2 in range(j_up)], 
               estim_real[:,i,si:ei].mean(axis=0)
                 +estim_real[:,i,si:ei].std(axis=0), 
                 'r--', label='Mean $\pm$ Std PEARO')
            axs[J-1-j_up].plot([1+j2+(J-j_up) for j2 in range(j_up)], 
               estim_real[:,i,si:ei].mean(axis=0)
                 -estim_real[:,i,si:ei].std(axis=0),'r--')
            
            axs[J-1-j_up].plot([1+j2+(J-j_up) for j2 in range(j_up)], 
               estim_real[:,i,si:ei].max(axis=0), 'r+',label='Max/Min PEARO')
            axs[J-1-j_up].plot([1+j2+(J-j_up) for j2 in range(j_up)], 
               estim_real[:,i,si:ei].min(axis=0), 'r+')
            
            axs[J-1-j_up].set_xticks(list(range(J-j_up+1, J+1)))
            axs[J-1-j_up].set_xlabel(r"Interaction scale $j_2 > j_1$")
            axs[J-1-j_up].title.set_text(r"Scale $j_1$={}".format(J-j_up))
            
        axs[0].set_ylabel("{} estimator".format(name))
        axs[J-2].legend(bbox_to_anchor=(1.0,0.5),loc='center left')
        
        fig.subplots_adjust(bottom=0.05,top=0.9, left=0.05, right=0.95)
        st=fig.suptitle('Scattering {} estimators for {}, $J$={}, $L$={}'.format(name,var, J,L), fontsize='14')
        st.set_y(1.0)
        
        plt.savefig(output_dir+'scattering_{}_{}_{}_{}.png'.format(var[1:-1],J,L, name), bbox_inches='tight')
        
        plt.show()
        plt.close()
        
        
def plot_1o_Estimators(estim_fake, estim_real, name, J,L, output_dir):
    """
    Plot and save figures for first order_like scattering estimators (S1)
    of fake (generated) and real (extracted from dataset) samples.
    
    Inputs :
        
        estim_fake : numpy array representing 2d order scattering estimator
        of the GAN generated samples
        shape is either Batch x Variables x J x L -> one plot per L value
                 or     Batch x Variable x J -> one plot only
        
        estim_true : numpy array representing 2d order scattering estimator
        of the dataset samples
        shape is either Batch x Variables x J x L -> one plot per L value
                 or     Batch x Variable x J -> one plot only
    
        name : str, name of the estimator 
        (First order coefficient)
        
        J : int, number of scales involved in the computation
        
        L : int, number of directions involved in the computation
        
        output_dir : str, the directory to save figures in
    
    """
    if len(estim_fake.shape)==4:
        for i, var in enumerate(['$u$', '$v$', '$t_{2m}$']):

            print(var)
            fig, axs = plt.subplots(1, L, figsize=(15, 6),
                                sharex=False, sharey=True)
            for ell in range(L):
                
                axs[ell].plot([1+j2 for j2 in range(J)], 
                   estim_fake[:,i,:,ell].mean(axis=0), 'b-', label='Mean GAN')
                
                axs[ell].plot([1+j2 for j2 in range(J)], 
                   estim_fake[:,i,:,ell].mean(axis=0)
                                        +estim_fake[:,i,:,ell].std(axis=0), 
                                        'b--', label=r'Mean $\pm$ Std GAN')
                axs[ell].plot([1+j2 for j2 in range(J)],
                   estim_fake[:,i,:,ell].mean(axis=0)
                                        -estim_fake[:,i,:,ell].std(axis=0), 'b--')
                
                axs[ell].plot([1+j2 for j2 in range(J)], 
               estim_fake[:,i,:,ell].max(axis=0), 'b+', label='Max/Min GAN')
                
                axs[ell].plot([1+j2 for j2 in range(J)], 
               estim_fake[:,i,:,ell].min(axis=0), 'b+')
                
                
                axs[ell].plot([1+j2 for j2 in range(J)], 
                   estim_real[:,i,:,ell].mean(axis=0), 'r-', label='Mean PEARO')
                
                axs[ell].plot([1+j2 for j2 in range(J)], 
                   estim_real[:,i,:,ell].mean(axis=0)
                                        +estim_real[:,i,:,ell].std(axis=0), 
                                        'r--', label=r'Mean $\pm$ Std PEARO')
                axs[ell].plot([1+j2 for j2 in range(J)],
                   estim_real[:,i,:,ell].mean(axis=0)
                                        -estim_real[:,i,:,ell].std(axis=0), 'r--')
                
                axs[ell].plot([1+j2 for j2 in range(J)], 
               estim_real[:,i,:,ell].max(axis=0), 'r+', label='Max/Min PEARO')
                
                axs[ell].plot([1+j2 for j2 in range(J)], 
               estim_real[:,i,:,ell].min(axis=0), 'r+')
                
                axs[ell].set_xticks(list(range(1,J+1)))
                axs[ell].set_xlabel(r"Scale $j_1$")
                axs[ell].title.set_text(r"Orientation $ \ell=$"+str(ell)+r"$\frac{\pi}{L}$")
            
                
            axs[0].set_ylabel("{} estimator".format(name))
            axs[L-1].legend(bbox_to_anchor=(1.0,0.5),loc='center left')
        
            fig.subplots_adjust(bottom=0.05,top=0.9, left=0.05, right=0.95)
            st=fig.suptitle('{} estimators for {}, $J$={}, $L$={}'.format(name,var, J,L), fontsize='14')
            st.set_y(1.0)
            
            plt.savefig(output_dir+'{}_{}_{}_{}.png'.format(var[1:-1],J,L, name), bbox_inches='tight')
            
            plt.show()
            plt.close()
            
    elif len(estim_fake.shape)==3:
         for i, var in enumerate(['$u$', '$v$', '$t_{2m}$']):
             print(var)
             fig=plt.figure(figsize=(8,8))
                
             plt.plot([1+j2 for j2 in range(J)], 
               estim_fake[:,i].mean(axis=0), 'b-', label='Mean GAN')
            
             plt.plot([1+j2 for j2 in range(J)], 
               estim_fake[:,i].mean(axis=0)
                                    +estim_fake[:,i].std(axis=0), 
                                    'b--', label=r'Mean $\pm$ Std GAN')
             plt.plot([1+j2 for j2 in range(J)],
               estim_fake[:,i].mean(axis=0)
                                    -estim_fake[:,i].std(axis=0), 'b--')
            
             plt.plot([1+j2 for j2 in range(J)], 
           estim_fake[:,i].max(axis=0), 'b+', label='Max/Min GAN')
            
             plt.plot([1+j2 for j2 in range(J)], 
           estim_fake[:,i].min(axis=0), 'b+')
            
            
             plt.plot([1+j2 for j2 in range(J)], 
               estim_real[:,i].mean(axis=0), 'r-', label='Mean PEARO')
            
             plt.plot([1+j2 for j2 in range(J)], 
               estim_real[:,i].mean(axis=0)
                                    +estim_real[:,i].std(axis=0), 
                                    'r--', label=r'Mean $\pm$ Std PEARO')
             plt.plot([1+j2 for j2 in range(J)],
               estim_real[:,i].mean(axis=0)
                                    -estim_real[:,i].std(axis=0), 'r--')
            
             plt.plot([1+j2 for j2 in range(J)], 
           estim_real[:,i].max(axis=0), 'r+', label='Max/Min PEARO')
            
             plt.plot([1+j2 for j2 in range(J)], 
           estim_real[:,i].min(axis=0), 'r+')
            
             #plt.set_xticks(list(range(J)))
             plt.xlabel(r"Scale $j_1$")
            
               
             plt.ylabel("{} estimator".format(name))
             plt.legend(bbox_to_anchor=(1.0,0.5),loc='center left')
        
             fig.subplots_adjust(bottom=0.05,top=0.9, left=0.05, right=0.95)
             st=fig.suptitle('{} estimators for {}, $J$={}, $L$={}'.format(name,var, J,L), fontsize='14')
             st.set_y(0.95)
            
             plt.savefig('{}_{}_{}_{}.png'.format(var[1:-1],J,L, name), bbox_inches='tight')
            
             plt.show()
             plt.close()
