# Copyright (c) HuaWei, Inc. and its affiliates.
# liu.haiyang@huawei.com
# Train script for audio2pose

import os
import signal
import time
import csv
import sys
import warnings
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import numpy as np
import time
import pprint
from loguru import logger
import wandb

from utils import config, logger_tools, other_tools
from dataloaders import data_tools
from dataloaders.build_vocab import Vocab
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func

import faulthandler
faulthandler.enable()

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

class BaseTrainer(object):
    def __init__(self, args):
        self.args = args
        self.notes = args.notes
        self.ddp = args.ddp
        self.rank = 0
        self.checkpoint_path = args.root_path+args.out_root_path + "custom/" + args.name + args.notes + "/" #wandb.run.dir #args.root_path+args.out_root_path+"/"+args.name
        self.batch_size = args.batch_size
        self.gpus = len(args.gpus)
        
        self.best_epochs = {
            "rec_val": [np.inf, 0],
        }
        self.loss_meters = {
            "rec_val": other_tools.AverageMeter("rec_val"),
            "pae": other_tools.AverageMeter("pae"),
            "all": other_tools.AverageMeter("all"),
            "rec": other_tools.AverageMeter("rec"),
            "gen": other_tools.AverageMeter("gen"),
            "dis": other_tools.AverageMeter("dis"),
            "cel": other_tools.AverageMeter("cel"),
        }
        self.pose_version = args.pose_version
        # data and path
        self.mean_pose = np.load(args.root_path+args.mean_pose_path+f"{args.pose_rep}/bvh_mean.npy")
        self.std_pose = np.load(args.root_path+args.mean_pose_path+f"{args.pose_rep}/bvh_std.npy")
        
        # pose
        self.pose_rep = args.pose_rep 
        self.pose_fps = args.pose_fps
        self.pose_dims = args.pose_dims
        # audio
        self.audio_rep = args.audio_rep
        self.audio_fps = args.audio_fps
        #self.audio_dims = args.audio_dims
        # facial
        self.facial_rep = args.facial_rep
        self.facial_fps = args.facial_fps
        self.facial_dims = args.facial_dims
        
        self.pose_pae = args.pose_pae
        
        # model para    
        self.pre_frames = args.pre_frames
        self.rec_loss = get_loss_func("huber_loss")
        self.adv_loss = get_loss_func("bce_loss")
        self.vel_loss = get_loss_func("l2_loss")
        self.acc_loss = get_loss_func("l2_loss")
        # TODO: 
        # self.pos_loss        
        self.rec_weight = args.rec_weight
        self.adv_weight = args.adv_weight
        self.vel_weight = args.vel_weight
        self.acc_weight = args.acc_weight
        self.grad_norm = args.grad_norm 
      
        self.no_adv_epochs = args.no_adv_epochs
        self.log_period = args.log_period
        self.test_demo = args.root_path + args.test_data_path + f"{args.pose_rep}_vis/"
        
        self.train_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).CustomDataset(args, "train")
        self.train_loader = torch.utils.data.DataLoader(
            self.train_data, 
            batch_size=args.batch_size,  
            shuffle=False if self.ddp else True,  
            num_workers=args.loader_workers,
            drop_last=True,
            sampler=torch.utils.data.distributed.DistributedSampler(self.train_data) if self.ddp else None, 
        )
        self.train_length = len(self.train_loader)
        logger.info(f"Init train dataloader success")
       
        self.val_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).CustomDataset(args, "val")  
        self.val_loader = torch.utils.data.DataLoader(
            self.val_data, 
            batch_size=args.batch_size,  
            shuffle=False,  
            num_workers=args.loader_workers,
            drop_last=True,
            sampler=torch.utils.data.distributed.DistributedSampler(self.val_data) if self.ddp else None, 
        )
        logger.info(f"Init val dataloader success")
        self.test_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).CustomDataset(args, "test")
        self.test_loader = torch.utils.data.DataLoader(
            self.test_data, 
            batch_size=1,  
            shuffle=True,  
            num_workers=args.loader_workers,
            drop_last=True,
        )
        logger.info(f"Init test dataloader success")
        
        model_module = __import__(f"models.{args.model}", fromlist=["something"])
        
        if self.ddp:
            self.model = getattr(model_module, args.g_name)(args).to(self.rank)
            process_group = torch.distributed.new_group()
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model, process_group)   
            self.model = DDP(self.model, device_ids=[self.rank], output_device=self.rank,
                             broadcast_buffers=False, find_unused_parameters=False)
        else: 
            self.model = torch.nn.DataParallel(getattr(model_module, args.g_name)(args), args.gpus).cuda()
        if self.rank == 0:
            # logger.info(self.model)
            # wandb.watch(self.model)
            logger.info(f"init {args.g_name} success")
        
        if args.d_name is not None:
            if self.ddp:
                self.d_model = getattr(model_module, args.d_name)(args).to(self.rank)
                self.d_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.d_model, process_group)   
                self.d_model = DDP(self.d_model, device_ids=[self.rank], output_device=self.rank, 
                                   broadcast_buffers=False, find_unused_parameters=False)
            else:    
                self.d_model = torch.nn.DataParallel(getattr(model_module, args.d_name)(args), args.gpus).cuda()
            if self.rank == 0:
                logger.info(self.d_model)
                wandb.watch(self.d_model)
                logger.info(f"init {args.d_name} success")
            self.opt_d = create_optimizer(args, self.d_model, lr_weight=args.d_lr_weight)
            self.opt_d_s = create_scheduler(args, self.opt_d)
            
        self.opt = create_optimizer(args, self.model)
        self.opt_s = create_scheduler(args, self.opt)

        model = self.model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module  # 取出包裹的原始模型

        audio_encoder = getattr(model, "audio_encoder", None)
        if audio_encoder is not None:
            for encoder_name in ("audio_encoder_cont", "audio_encoder_emo"):
                encoder = getattr(audio_encoder, encoder_name, None)
                if encoder is not None:
                    for param in encoder.parameters():
                        param.requires_grad = False

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"trainable parameters: {trainable_params:,}")
       
    def recording(self, epoch, its, its_len, loss_meters, lr_g, lr_d, t_data, t_train, mem_cost):
        if self.rank == 0:
            pstr = "[%03d][%03d/%03d]  "%(epoch, its, its_len)
            for name, loss_meter in self.loss_meters.items():
                if "val" not in name:
                    if loss_meter.count > 0:
                        pstr += "{}: {:.3f}\t".format(loss_meter.name, loss_meter.avg)
                        wandb.log({loss_meter.name: loss_meter.avg}, step=epoch*self.train_length+its)
                        loss_meter.reset()
            pstr += "glr: {:.1e}\t".format(lr_g)
            pstr += "dlr: {:.1e}\t".format(lr_d)
            wandb.log({'glr': lr_g, 'dlr': lr_d}, step=epoch*self.train_length+its)
            pstr += "dtime: %04d\t"%(t_data*1000)        
            pstr += "ntime: %04d\t"%(t_train*1000)
            pstr += "mem: {:.2f} ".format(mem_cost*self.gpus)
            logger.info(pstr)
     
    def val_recording(self, epoch, meters):
        if self.rank == 0: 
            pstr_curr = "Curr info >>>>  "
            pstr_best = "Best info >>>>  "

            for name, meter in meters.items():
                if "val" in name:
                    if meter.count > 0:
                        pstr_curr += "{}: {:.3f}     \t".format(meter.name, meter.avg)
                        wandb.log({meter.name: meter.avg}, step=epoch*self.train_length)
                        if meter.avg < self.best_epochs[meter.name][0]:
                            self.best_epochs[meter.name][0] = meter.avg
                            self.best_epochs[meter.name][1] = epoch
                            other_tools.save_checkpoints(os.path.join(self.checkpoint_path, f"{meter.name}.bin"), self.model, opt=None, epoch=None, lrs=None)        
                        meter.reset()
            for k, v in self.best_epochs.items():
                pstr_best += "{}: {:.3f}({:03d})\t".format(k, v[0], v[1])
            logger.info(pstr_curr)
            logger.info(pstr_best)  

@logger.catch
def main_worker(rank, world_size, args):
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    # dist.init_process_group("nccl", rank=rank, world_size=world_size)
        
    logger_tools.set_args_and_logger(args, rank)
    other_tools.set_random_seed(args)
    other_tools.print_exp_info(args)
      
    # return one intance of trainer
    trainer = __import__(f"{args.trainer}_trainer", fromlist=["something"]).CustomTrainer(args) if args.trainer != "base" else BaseTrainer(args) 
    
    logger.info("Training from starch ...")          
    start_time = time.time()
    for epoch in range(args.epochs):

        if trainer.ddp: trainer.val_loader.sampler.set_epoch(epoch)
        trainer.val(epoch)
        if trainer.ddp: trainer.train_loader.sampler.set_epoch(epoch) 
        trainer.train(epoch)
        epoch_time = time.time()-start_time
        if trainer.rank == 0: logger.info("Time info >>>>  elapsed: %.2f mins\t"%(epoch_time/60)+"remain: %.2f mins"%((args.epochs/(epoch+1e-7)-1)*epoch_time/60))

        if (epoch+1) % args.test_period == 0:
            if rank == 0:
                # trainer.test(epoch)
                other_tools.save_checkpoints(os.path.join(trainer.checkpoint_path, f"last_{epoch}.bin"), trainer.model, opt=None, epoch=None, lrs=None)
                
        # trainer.save_parameters()
    for k, v in trainer.best_epochs.items():
        wandb.log({f"{k}_best": v[0], f"{k}_epoch": v[1]})
    
    if rank == 0:
        wandb.finish()
    
            
if __name__ == "__main__":

    os.environ["MASTER_ADDR"]='localhost'
    os.environ["MASTER_PORT"]='2222'
    args = config.parse_args()
    
    if args.ddp:
        mp.set_start_method("spawn", force=True)
        mp.spawn(
            main_worker,
            args=(len(args.gpus), args,),
            nprocs=len(args.gpus),
                )
    else:
        main_worker(0, 1, args)
