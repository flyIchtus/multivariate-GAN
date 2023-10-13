#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 09:33:21 2022

@author: brochetc
"""

import torch

import horovod.torch as hvd
import trainer_horovod as trainer

import plotting_functions as plf
from expe_init import get_expe_parameters

hvd.init()

torch.cuda.set_device(hvd.local_rank())

import resnets as RN
###############################################################################
############################# INITIALIZING EXPERIMENT #########################
###############################################################################

config=get_expe_parameters().parse_args()


###############################################################################
############################ BUILDING MODELS ##################################
###############################################################################

load_optim=False

modelG=RN.ResNet_G(config.latent_dim, config.g_output_dim, config.g_channels)

if config.train_type=='wgan-hinge' and config.sn_on_g: 
    modelG.apply(RN.Add_Spectral_Norm)

if config.ortho_init : modelG.apply(RN.Orthogonal_Init)


modelD=RN.ResNet_D(config.d_input_dim, config.d_channels)

if config.train_type=='wgan-hinge': 
    modelD.apply(RN.Add_Spectral_Norm)

if config.ortho_init : modelD.apply(RN.Orthogonal_Init)

if config.pretrained_model>0:

    modelG.load_state_dict(torch.load(config.output_dir+'/models/bestgen_{}'.format(config.pretrained_model)))
    modelD.load_state_dict(torch.load(config.output_dir+'/models/bestdisc_{}'.format(config.pretrained_model)))
    load_optim=True
 
   
###############################################################################    
######################### LOADING models and Data #############################
###############################################################################

TRAINER=trainer.Trainer(config,criterion="W1_Center",\
                        test_metrics=["W1_random","SWD_metric_torch"])
                        
# names used in test_metrics should belong to the metrics namespace

TRAINER.instantiate(modelG, modelD,load_optim)

###############################################################################
################################## TRAINING ###################################
##########################   (and online testing)  ############################
###############################################################################

TRAINER.fit_(modelG, modelD)

###############################################################################
############################## POST-PROCESSING ################################
############################ (of training data) ###############################

plf.plot_metrics_from_csv(config.output_dir+'/log/', 'metrics.csv')
