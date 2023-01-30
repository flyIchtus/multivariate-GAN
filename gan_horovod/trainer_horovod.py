#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 17:36:23 2022

@author: brochetc

"""

########## NN libs ############################################################
import torch
import torch.optim as optim
from numpy import save

########## MP libs ############################################################
import horovod.torch as hvd
#import torch.multiprocessing as mp

########## data handling and plotting libs ####################################

import metrics4arome as METR
import plotting_functions as plotFunc
import DataSet_Handler_horovod as DSH
import LookAheadMiniMax as LA

import GAN_logic_horovod as GAN


from horovod.torch.mpi_ops import allreduce
from time import perf_counter



###################### Scheduler choice function ##############################

def AllocScheduler(sched, optimizer, gamma):
    if sched=="exp":
        if hvd.rank()==0 : print("exponential scheduler")
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif sched=="linear" :
        if hvd.rank()==0 : print("linear scheduler")
        lambda0= lambda epoch: 1.0/(1.0+gamma*epoch)
        return optim.lr_scheduler.LambdaLR(optimizer, lambda0)
    else:
        if hvd.rank()==0 : print("Scheduler set to None")
        return None

############## A Horovod auxliliary function with Adasum ######################

def passive_step(self):
    for p, (handle, ctx) in self._handles.items():
        self._allreduce_delay[p]=self.backward_passes_per_step
    self._handles.clear()

###############################################################################
######################### Trainer class #######################################
###############################################################################



class Trainer():
    """
    main training class
    
    inputs :
        
        config -> model, training, optimizer, dirs, file names parameters 
                (see constructors and main.py file)
        optimizer -> the type of optimizer (if working with LA-Minimax)
        
        test_metrics -> list of names from metrics namespace
        criterion -> specific metric name (in METR) used as save/stop criterion
        
        metric_verbose -> check if metric long name should be printed @ test time
                default to True
    
    outputs :
        
        models saved along with optimizers and schedulers at regular steps
        plots and scores logged regularly
        return of metrics and losses lists
        
        
    bound methods :
        
        "get-ready" methods:
            instantiate_optimizers : prepare distributed optimizers settings
            prepare_data_pipeline : create Dataset and Dataloader instances
            choose_algorithm : select GAN logic to implement
            instantiate_metric_log : prepare output files
            instantiate : builder from the above methods + load and broadcast 
                          models
        
        logging methods :
            log : save scores in file for later data analysis
            plot : plot samples from generated and real distributions
            message : print message in stdo
            save_models : save models, optimizers, schedulers
            save_samples : save a large chunk of samples in a single array
        training methods :
            Discrim_Update
            Generator_Update
            test_ : provide models evaluation over variate metrics
            fit_ : train the models according to selected logic
                   pilot the testing
                   callbacks management
    """
    
    def __init__(self, config, criterion,test_metrics={},\
                metric_verbose=True ):
        
        self.config=config
        self.instance_flag=False
        
        self.LA_optimizer=self.config.LA_optimizer
        self.test_metrics=[getattr(METR, name) for name in test_metrics]
        self.criterion=getattr(METR,criterion)
        
        self.metric_verbose=metric_verbose
        self.batch_size=self.config.batch_size
        self.n_dis=self.config.n_dis
        self.test_step=self.config.test_step
        self.save_step=self.config.save_step
        self.plot_step=self.config.plot_step
        
        self.var_indexes=[DSH.var_dict[var] for var in self.config.var_names]
        self.crop_indexes=self.config.crop_indexes
        
        
    ########################## GET-READY FUNCTIONS ############################
    
    def instantiate_optimizers(self, modelG, modelD, load_optim, load_sched):
        """
        instantiate and prepare optimizers and lr schedulers
        """
        if load_optim:
            self.optim_G=torch.load(self.config.output_dir+'/models/optimGen_{}'.format(self.config.pretrained_model))
            self.optim_D=torch.load(self.config.output_dir+'/models/optimDisc_{}'.format(self.config.pretrained_model))
        else:        
            self.optim_G=optim.Adam(modelG.parameters(), lr=self.config.lr_G, 
                           betas=(self.config.beta1_G, self.config.beta2_G))
            self.optim_D=optim.Adam(modelD.parameters(), lr=self.config.lr_D,
                           betas=(self.config.beta1_D, self.config.beta2_D))
                
        if load_sched:
            if hvd.rank()==0:
                self.scheduler_G=torch.load(self.config.output_dir+\
                                            '/models/SchedGen_{}'.format(self.config.pretrained_model))
                hvd.broadcast_object(self.scheduler_G.state_dict(),0,name='sched_G')
                
                self.scheduler_D=torch.load(self.config.output_dir+\
                                            '/models/SchedDisc_{}'.format(self.config.pretrained_model))
                hvd.broadcast_object(self.scheduler_G.state_dict(),0,name='sched_D')
        else:
            self.scheduler_G=AllocScheduler(self.config.lrD_sched,\
                                            self.optim_G, self.config.lrD_gamma)
            self.scheduler_D=AllocScheduler(self.config.lrG_sched,\
                                            self.optim_D, self.config.lrG_gamma)
        
        if self.LA_optimizer :
                _alpha=self.config.LA_alpha
                _k=self.config.LA_k
                self.optim_G=LA.LookAheadOpt(self.optim_G, alpha=_alpha,k=_k)
                self.optim_D=LA.LookAheadOpt(self.optim_D, alpha=_alpha,k=_k)
        
        ###### horovodizing stuff
        
        self.optim_G=hvd.DistributedOptimizer(self.optim_G,\
               named_parameters=modelG.named_parameters(prefix='generator'),
               backward_passes_per_step=self.config.accum_steps)                                              
        self.optim_D=hvd.DistributedOptimizer(self.optim_D,\
               named_parameters=modelD.named_parameters(prefix='discriminator'),
               backward_passes_per_step=self.config.accum_steps)
        
        
        # If working with Adasum
        
        #self.optim_G.passive_step=passive_step.__get__(self.optim_G)
        #self.optim_D.passive_step=passive_step.__get__(self.optim_D)
        
        hvd.broadcast_optimizer_state(self.optim_D, root_rank=0)
        hvd.broadcast_optimizer_state(self.optim_D, root_rank=0)
    
    def prepare_data_pipeline(self):
        """
        
        instantiate datasets and dataloaders to be used in training
        set GPU-CPU communication parameters
        see ISData_Loader classes for details
        
        """
        torch.set_num_threads(1)

        if hvd.rank()==0 : print("Loading data")
        self.Dl_train=DSH.ISData_Loader(self.config.data_dir,self.batch_size,
                                        self.var_indexes, self.crop_indexes,\
                                        add_coords=self.config.coords)
        self.Dl_test=DSH.ISData_Loader(self.config.data_dir, self.config.test_samples,
                                       self.var_indexes, self.crop_indexes,
                                       add_coords=self.config.coords)
        
        kwargs={'pin_memory': True}
        
        self.train_dataloader=self.Dl_train.loader(hvd.size(),hvd.rank(), kwargs)
        self.test_dataloader=self.Dl_test.loader(hvd.size(), hvd.rank(), kwargs)
        
    def choose_algorithm(self):

        if self.config.train_type=="wgan-gp":
            self.D_backward=GAN.Discrim_Wasserstein(self.config.lamda_gp).Discrim_Step
            self.G_backward=GAN.Generator_Step_Wasserstein
        elif self.config.train_type=="vanilla":
            self.D_backward=GAN.Discrim_Step
            self.G_backward=GAN.Generator_Step
        elif self.config.train_type=="wgan-hinge":
            self.D_backward=GAN.Discrim_Step_Hinge
            self.G_backward=GAN.Generator_Step_Wasserstein
    
    def instantiate_metrics_log(self):
        
        """create metrics to be logged
        prepare log file
        # metrics list is absolutely agnostic, so a rough list of metric functions
        # as basis for test_metrics seems good practice here"""
        
        self.lossD_list=[]
        self.lossG_list=[]
        self.Metrics={name:[] for metr in self.test_metrics for name in metr.names}
        self.crit_List=[]
        
        with open(self.config.output_dir+'/log/metrics.csv','w') as file:
            file.write('Step,loss_D,loss_G,criterion')
            for mname in self.Metrics.keys():
                file.write(','+mname)
            file.write('\n')
            file.close()
    
    def instantiate(self, modelG, modelD,load_optim=False, load_sched=False):
    
        torch.manual_seed(self.config.seed)
        modelD.cuda()
        modelG.cuda()
        
        hvd.broadcast_parameters(modelG.state_dict(), root_rank=0)
        hvd.broadcast_parameters(modelD.state_dict(),root_rank=0)
        
        ### horovod : scaling learning rates by number of gpus allowed
    
        self.config.lr_G=self.config.lr_G*(hvd.size())
        self.config.lr_D=self.config.lr_D*(hvd.size())
        
        if hvd.rank()==0 : print('WARNING : Updated learning rates by {} factor'.format(hvd.size()))
        
        ###
        
        self.instantiate_optimizers(modelG, modelD, load_optim, load_sched)
        
        self.instantiate_metrics_log()
        
        self.choose_algorithm()
        
        self.prepare_data_pipeline()
        
        #######################################################################
        
    
        if hvd.rank()==0 : print("Data loaded, Trainer instantiated")
        self.instance_flag=True
    
    ################################ LOGGING FUNCTIONS #######################
    
    def log(self, Step):

        data=[Step,self.lossD_list[-1], self.lossG_list[-1],\
                          self.crit_List[-1]]
        if len(self.Metrics)!=0:
            data+=[self.Metrics[name][-1]\
                                        for name in self.Metrics.keys()]
        data_to_write='%d,%.6f,%.6f,%.6f'
        
        for mname in self.Metrics.keys():
            data_to_write=data_to_write+',%.6f'
        data_to_write=data_to_write %tuple(data)
        
        with open(self.config.output_dir+'/log/metrics.csv','a') as file:
            file.write(data_to_write)
            file.write('\n')
            file.close()
    
    def plot(self, Step, modelG, samples):
         print('Plotting')
                     
         modelG.train=False
         z=torch.empty(3*(self.config.plot_samples//4),modelG.nz).normal_().cuda()
         with torch.no_grad():
             batch=modelG(z)
             
         modelG.train=True
         
         batch=torch.cat((batch, samples[:self.config.plot_samples//4,:,:,:]), dim=0)
         
         plotFunc.online_sample_plot(batch,self.config.plot_samples,\
                                Step,self.config.var_names, \
                                self.config.output_dir+'/samples',
                                coords=self.config.coords)
    
    def message(self,loss, network, step, N_batch):
        """
        this loss is computed on batch_size*hvd_size() samples
        not on batch_size*hvd.size()*accum_steps samples
        """
        if network=='D':
            name='discrim_loss'
            msg="Discrim loss : "
            li=self.lossD_list
            if hvd.rank()==0: print("Dataset percentage done :", str(100*step/N_batch)[:4])
        elif network=='G':
            name='gen_loss'
            msg="Gen loss : "
            li=self.lossG_list
        else :
            raise ValueError("Network must be 'D' or 'G', got {} ".format(network))
            
        loss=allreduce(loss, name=name)
        
        if hvd.rank()==0:
            li+=[loss.item()]
            print(msg+"%.4f"%(loss.item()))
    
    def save_models(self,Step,modelG,modelD):
        print("Saving")
        
        #models
        torch.save(modelD.state_dict(), \
                   self.config.output_dir+\
                   "/models/bestdisc_{}".format(Step))
        torch.save(modelG.state_dict(), \
                    self.config.output_dir+\
                   "/models/bestgen_{}".format(Step))
        
        #optimizers
        torch.save(self.optim_D.state_dict(), \
                   self.config.output_dir+\
                   "/models/optimDisc_{}".format(Step))
        torch.save(self.optim_G.state_dict(), \
                    self.config.output_dir+\
                   "/models/optimGen_{}".format(Step))
        
        #schedulers
        if self.scheduler_D is not None :
            torch.save(self.scheduler_D.state_dict(), \
                    self.config.output_dir+\
                   "/models/SchedDisc_{}".format(Step))
        if self.scheduler_G is not None :
            torch.save(self.scheduler_G.state_dict(), \
                    self.config.output_dir+\
                   "/models/SchedGen_{}".format(Step))
            
    def save_samples(self,number,Step, modelG):
        print("Saving samples")
        
        for i in range(16):
            z=torch.empty(number,modelG.nz).normal_().cuda()
            with torch.no_grad():
                out=modelG(z).cpu().numpy()

            save(self.config.output_dir+'/samples/_Fsample_{}_{}.npy'.format(Step,i), out)
        
        return 0
     ########################### TRAINING FUNCTIONS ############################
            
    def Discrim_Update(self,modelD, modelG, train_iter):
        for i in range(self.n_dis):

            for acc in range(self.config.accum_steps):
                samples, _,_=next(train_iter)
                samples=samples.cuda()            
                
                loss=self.D_backward(samples, modelD, modelG, self.optim_D,\
                                          self.optim_G,\
                                          self.config.use_amp)

                self.optim_D.synchronize()

                #torch.nn.utils.clip_grad_norm_(modelD.parameters(),3.0)

            with self.optim_D.skip_synchronize():
                self.optim_D.step()

            self.optim_G.synchronize()

            
        return loss, samples
    
    def Generator_Update(self, modelD, modelG,samples):
        for acc in range(self.config.accum_steps):
            loss=self.G_backward(samples, modelD, modelG,\
                    self.optim_D,self.optim_G, self.config.use_amp)
        
            self.optim_G.synchronize()
            #torch.nn.utils.clip_grad_norm_(modelG.parameters(),3.0)
    
        with self.optim_G.skip_synchronize():
            self.optim_G.step()
        self.optim_D.synchronize()
        return loss
    
    def warmup(self, modelD, modelG, epoch,length=1, option=('n_dis',2), test=False):
        """
        perform warmup steps to avoid early explosion
        
        Inputs :
            epoch -> int to set the Dataloader epoch seed
            modelD, modelG
            length : number of steps to perform warm-up
    
            option : tuple giving a warm-up method id and a parameter
            
                case 'n_dis' :
                    modelD is updated n_dis number of times as modelG is trained once
                
                case 'lr' :
                    learning rate is fixed to (lower) lr during warm-up
            
            test : bool, tests and save the scores at the end of warm-up
        """
        self.train_dataloader.sampler.set_epoch(epoch)
        warmUp_iter=iter(self.train_dataloader)
        
        if hvd.rank()==0 :
            print('------------------- WARM-UP STEPS ------------------------')
            
        assert option[0] in ['lr', 'n_dis']
        
        if option[0]=='n_dis':
            self.n_dis=option[1]
            if hvd.rank()==0 : print(self.n_dis)
            assert length<len(self.train_dataloader)//self.n_dis-1
            
        else :
            self.optim_D.lr=option[1]
            self.optim_G.lr=option[1]
            
        wu_step=0
        
        while wu_step<length:
            print('wu_step',wu_step)
            loss,samples=self.Discrim_Update(modelD, modelG, warmUp_iter)
            if wu_step==length-1 : self.message(loss, 'D',wu_step,length)
            
            loss=self.Generator_Update(modelD, modelG, samples)
            if wu_step==length-1 : self.message(loss, 'G',wu_step,length)

            wu_step+=1
            
        
        if option[0]=='n_dis' : self.n_dis=self.config.n_dis
        if hvd.rank()==0 : print(self.n_dis)
        
        if option[1]=='lr' : 
            self.optim_D.lr=self.config.lr_D
            self.optim_G.lr=self.config.lr_G
            
        if test:
            self.test_(modelG, modelD,length, warmUp_iter)
            if hvd.rank()==0 :
                self.log(length)
                self.plot(length, modelG, samples)
        if hvd.rank()==0:
            print('------------------ WARM-UP END ---------------------------')
        
    def test_(self, modelG, modelD, Step, DataIter):
        """
        test samples in parallel for each metric
        iterates through test dataset with DataIter
        
        """
        
        modelD.train=False
        modelG.train=False
        real_samples,_,_=next(DataIter)
        real_samples=real_samples.cuda()
        sample_num=min(real_samples.shape[0],self.config.test_samples)
        
        # using sample_num samples PER GPU to compute scores
        
        z=torch.empty((sample_num,modelG.nz)).normal_().cuda()
        
        with torch.no_grad():
                fake_samples=modelG(z)
                
        for metr in self.test_metrics:                
            res=metr(real_samples,fake_samples)
            assert torch.isfinite(res).all()

            RES=hvd.allreduce(res, name=metr.names[0])
                
            
            if hvd.rank()==0:
                
                for i,name in enumerate(metr.names):
                    self.Metrics[name].append(RES[i].item())  
                    
                if self.metric_verbose :
                    print(metr.long_name+"%.4f"%(RES.mean().item()))
                    
        crit=self.criterion(real_samples,fake_samples)
        assert torch.isfinite(crit).all()

        CRIT=hvd.allreduce(crit, name=self.criterion.long_name)
        
        if hvd.rank()==0:
            self.crit_List.append(CRIT.item())
            
            if CRIT.item()==min(self.crit_List): 
                print("Best criterion score until now : %.4f "%(CRIT.item()))
                
        #saving models, optimizers, schedulers
        self.save_models(Step, modelG, modelD)
                
    
        modelG.train=True
        modelD.train=True
            
    def fit_(self, modelG, modelD):
        
        assert self.instance_flag #security flag avoiding meddling with uncomplete init
        
        modelG.train()
        modelD.train()
        Step=0
        N_batch=len(self.train_dataloader)//(self.config.accum_steps*self.n_dis)
        
        if self.config.warmup :
            self.warmup(modelD, modelG,self.config.epochs_num+1,length=5, option=('lr',1e-5),test=False)
        
        for e in range(self.config.epochs_num):
            if Step>=self.config.total_steps : break
        
            self.train_dataloader.sampler.set_epoch(e)
            self.test_dataloader.sampler.set_epoch(e)
            train_Dl_iter=iter(self.train_dataloader)    
            
            if hvd.rank()==0 : print("----------------------------------------------------------")
            if hvd.rank()==0 : print("Epoch num ", e+1, "/",self.config.epochs_num)
            if hvd.rank()==0 : t00=perf_counter()
            
            step=0
            while step<N_batch:

                if hvd.rank()==0 and step%100==0 : print(step)
                Step=e*N_batch+step
                if Step>=self.config.total_steps : break
            
                if hvd.rank()==0 : t0=perf_counter()        
                
                ############################### Discriminator Updates #########
                loss,samples=self.Discrim_Update(modelD, modelG, train_Dl_iter)
                
                ############################# Store and print value ###########
                
                if self.config.log_step>0 and Step%self.config.log_step==0:
                    self.message(loss, 'D', step, N_batch)
                        
                ############################ Generator Update #################
                
                loss=self.Generator_Update(modelD, modelG,samples)
                
                ############################ Look_Ahead Step if need be #######
                
                if self.LA_optimizer and Step>0 and Step%self.optim_G.k==0:
                    
                    self.optim_G.look_ahead_step()
                    self.optim_D.look_ahead_step()

                ########################### Store and print value #############
                
                if Step%self.config.log_step==0:
                    self.message(loss, 'G', step, N_batch)

                if hvd.rank()==0 : 
                    t1=perf_counter()-t0
                    #print('gan step time',t1)
                    
                ################ advancing schedulers for learning rate #######
                if step==N_batch-1:
                    if self.scheduler_G!=None :
                        self.scheduler_G.step()
                    if self.scheduler_D!=None :
                        self.scheduler_D.step()
                
                if hvd.rank()==0:
                    t11=perf_counter()-t00
                    #print('total step time',t11)
                    #print('step time net from GAN', t11-t1)
                    t00=t11+t00
                    
                ############### saving models and samples #################
                
                #if self.save_step>0 and (Step%self.save_step==0 or Step==self.config.total_steps-1) and hvd.rank()==0:
                #    self.save_models(Step, modelG, modelD)
                
                if self.save_step>0 and (Step%self.save_step==0 or Step==self.config.total_steps-1) and hvd.rank()==0:
                    self.save_samples(self.config.sample_num,Step, modelG)
                
                ############### testing samples at test step ##################
                
                if self.test_step>0 and (Step%self.test_step==0 or Step==self.config.total_steps-1):
                    test_Dl_iter=iter(self.test_dataloader)
                    self.test_(modelG, modelD, Step,test_Dl_iter)
                    
                ############# logging experiment data #############
                
                if self.config.log_step>0 and (Step%self.config.log_step==0 or Step==self.config.total_steps-1) and hvd.rank()==0:
                    self.log(Step)
                
                ############### plotting distribution at plot step ############
                if self.plot_step>0 and (Step%self.plot_step==0 or Step==self.config.total_steps-1) and hvd.rank()==0:
                     self.plot(Step, modelG, samples)
                     
                    
                step+=1
                ############ END OF STEP ######################################
                
            ################ END OF EPOCH #####################################
            
        ##########################
        return self.lossD_list, self.lossG_list, self.crit_List, self.Metrics
###############################################################################
