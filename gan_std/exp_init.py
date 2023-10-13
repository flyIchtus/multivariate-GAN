#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 11:56:21 2022

@author: brochetc


experiment initializer
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 09:33:21 2022

@author: brochetc
"""

import os
from glob import glob
import argparse

def str2bool(v):
    return v.lower() in ('true')

def get_parameters():

    parser = argparse.ArgumentParser()

    # Model architecture hyper-parameters
    parser.add_argument('--model', type=str, default='resnet', \
                        choices=['resnet', 'coco-gan'])
    parser.add_argument('--train_type', type=str, default='wgan-hinge',\
                        choices=['vanilla','wgan-gp', 'wgan-hinge'])
    parser.add_argument('--version', type=str, default='resnet_128')
    parser.add_argument('--patch_size', type=int, default=128)
    
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--g_channels', type=int, default=3)
    parser.add_argument('--d_channels', type=int, default=3)
    parser.add_argument('--g_output_dim', type=int, default=128)
    parser.add_argument('--d_input_dim', type=int, default=128)
    parser.add_argument('--lamda_gp', type=float, default=10.0)
    parser.add_argument('--ortho_init',type=str2bool, default=False)
    
    parser.add_argument('--sn_on_g', type=str2bool, default=False,\
                        help='Apply spectral normalisation on Generator')

    # Training setting
    parser.add_argument('--epochs_num', type=int, default=100,\
                        help='how many times to go through dataset')
    parser.add_argument('--total_step', type=int, default=10,\
                        help='how many times to update the generator')
    
    parser.add_argument('--n_dis', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--accum_steps', type=int, default=1,\
                        help="Accumulation factor for batch_size")
    
    parser.add_argument('--lr_G', type=float, default=0.0001)
    parser.add_argument('--lr_D', type=float, default=0.0002)
    
    parser.add_argument('--beta1_D', type=float, default=0.0)
    parser.add_argument('--beta2_D', type=float, default=0.9)
    
    parser.add_argument('--beta1_G', type=float, default=0.0)
    parser.add_argument('--beta2_G', type=float, default=0.9)
    
    
    #Training setting -schedulers
    parser.add_argument('--lrD_sched', type=str, default=None, \
                        choices=[None,'exp', 'linear'])
    parser.add_argument('--lrG_sched', type=str, default=None, \
                        choices=[None,'exp', 'linear'])
    parser.add_argument('--lrD_gamma', type=float, default=0.95)
    parser.add_argument('--lrG_gamma', type=float, default=0.95)
    
    
    # Testing and plotting settings
    parser.add_argument('--test_samples',type=int, default=128)
    parser.add_argument('--plot_samples', type=int, default=16)
    
    # using pretrained
    parser.add_argument('--pretrained_model', type=int, default=None,\
                        help='step at which pretrained model have been saved')

    # Misc
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--use_amp', type=str2bool, default=True)
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)
    parser.add_argument('--num_cpu_workers', type=int, default=2)
    parser.add_argument('--num_gpu_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')

    # Path
    parser.add_argument('--data_dir', type=str, \
                        default='./')
    parser.add_argument('--output_dir', type=str, \
                        default='/scratch/mrmn/brochetc/GAN_2D_11')

    # Step size
    parser.add_argument('--log_step', type=int, default=-1) #-> default is at the end of each epoch
    parser.add_argument('--sample_step', type=int, default=100) # set to 0 if not needed
    parser.add_argument('--plot_step', type=int, default=100) #set to 0 if not needed
    parser.add_argument('--save_step', type=int, default=0) # set to 0 if not needed
    parser.add_argument('--test_step', type=int, default=100) #set to 0 if not needed
    
    # Channel data description
    parser.add_argument('--var_names', type=list, default=['u', 'v', 't2m'])#, 'orog'])

    config=parser.parse_args()
    assert config.g_channels==len(config.var_names) and config.d_channels==len(config.var_names)
    assert config.log_step%config.test_step==0
    
    return parser.parse_args()

###############################################################################
############################# INITIALIZING EXPERIMENT #########################
###############################################################################

if __name__=='__main__':
    config=get_parameters()
    
    
    NAME=config.version+'_'+config.train_type+'_'+str(config.latent_dim)+'_'+str(config.batch_size)+'_'+str(config.n_dis)+'_'+str(config.lr_D)
    
    #### DO NOT ADD any print command as stdo is read from another bash script
    
    os.chdir(config.output_dir)
    if not os.path.exists(NAME):
        os.mkdir(NAME)
    os.chdir(NAME)
    
    previous_inst=len(glob('*Instance*'))
    INSTANCE_NUM=previous_inst+1
    os.mkdir('Instance_'+str(INSTANCE_NUM))
    os.chdir('Instance_'+str(INSTANCE_NUM))
    
    
    with open('ReadMe_'+str(INSTANCE_NUM)+'.txt', 'a') as f:
        f.write('-----------------------------------------\n')
        for arg in config.__dict__.keys():
            f.write(arg+'\t:\t'+str(config.__dict__[arg])+'\n')
        f.close
    if not os.path.exists('log'):
        os.mkdir('log')
        
    if not os.path.exists('models'):
        os.mkdir('models')
        
    if not os.path.exists('samples'):
        os.mkdir('samples')
    print(os.getcwd())
