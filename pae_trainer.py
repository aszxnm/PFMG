# Copyright (c) HuaWei, Inc. and its affiliates.
# liu.haiyang@huawei.com
# Train script for audio2pose

import os
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
import numpy as np
import time
import pprint
from loguru import logger

from utils import config, logger_tools, other_tools
from dataloaders import data_tools
from dataloaders.build_vocab import Vocab
from train import BaseTrainer
from models.utils import Utility as utility

import Plotting as plot
import matplotlib.pyplot as plt

#Initialize drawing
plt.ion()
_, ax1 = plt.subplots(6,1)
_, ax2 = plt.subplots(20,5)
_, ax3 = plt.subplots(1,2)
_, ax4 = plt.subplots(2,1)
dist_amps = []
dist_freqs = []
plotting_interval = 500
loss_history = utility.PlottingWindow("Loss History", ax=ax4, min=0, drawInterval=plotting_interval)  
           
def Item(value):
    return value.detach().cpu()     

class CustomTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.input_type = "velocity"
        self.g_name = args.g_name
        self.pose_length = args.pose_length
        self.loss_meters = {
            'rec_val': other_tools.AverageMeter('rec_val'),
            'vel_val': other_tools.AverageMeter('vel_val'),
            'kl_val': other_tools.AverageMeter('kl_val'),
            'all': other_tools.AverageMeter('all'),
            'rec_l1': other_tools.AverageMeter('rec_l1'), 
            'vel_l1': other_tools.AverageMeter('vel_l1'),
            'kl_loss': other_tools.AverageMeter('kl_loss'),
            #'acceleration_loss': other_tools.AverageMeter('acceleration_loss'),
        }
        self.best_epochs = {
            'rec_val': [np.inf, 0],
            'vel_val': [np.inf, 0],
            'kl_val': [np.inf, 0],
                           }
        self.rec_loss = torch.nn.L1Loss(reduction='none')
        self.vel_loss = torch.nn.MSELoss(reduction='none')
        self.variational_encoding = args.variational_encoding
        self.rec_weight = args.rec_weight
        self.vel_weight = args.vel_weight

    def train(self, epoch):
        self.model.train()
        its_len = len(self.train_loader)
        t_start = time.time()

        for its, dict_data in enumerate(self.train_loader):
            tar_pose = dict_data[self.input_type]
            tar_pose = tar_pose[:, :, 46:]
            # tar_pose = torch.cat((tar_pose[:, :, 70:154], tar_pose[:, :, 166:250]), axis=2)
            tar_pose = tar_pose.cuda()
            t_data = time.time() - t_start 
            self.opt.zero_grad()
            recon_data, latent, signal, params = \
                self.model(tar_pose)
            recon_loss = self.vel_loss(recon_data, tar_pose) # 128*34*123
            recon_loss = torch.mean(recon_loss, dim=(1, 2)) # 128
            self.loss_meters['rec_l1'].update(torch.mean(recon_loss).item()*self.rec_weight)
            recon_loss = torch.sum(recon_loss*self.rec_weight)
            # rec vel loss
            if self.vel_weight > 0:  # use pose diff
                target_diff = tar_pose[:, 1:] - tar_pose[:, :-1]
                recon_diff = recon_data[:, 1:] - recon_data[:, :-1]
                vel_rec_loss = torch.mean(self.vel_loss(recon_diff, target_diff), dim=(1, 2))
                self.loss_meters['vel_l1'].update(torch.sum(vel_rec_loss).item()*self.vel_weight)
                recon_loss += (torch.sum(vel_rec_loss)*self.vel_weight)
            # KLD
            
            loss = recon_loss
            self.loss_meters['all'].update(loss.item())
            if self.grad_norm != 0 and "LSTM" in self.g_name: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
#             logger.warning(total_norm)
            loss.backward()
            self.opt.step()
            t_train = time.time() - t_start - t_data
            t_start = time.time()
            mem_cost = torch.cuda.memory_reserved() / 1E9
            lr_g = self.opt.param_groups[0]['lr']
            # --------------------------- recording ---------------------------------- #
            if its % self.log_period == 0:
                self.recording(epoch, its, its_len, self.loss_meters, lr_g, 0, t_data, t_train, mem_cost)   

            #End Visualization Section
        self.opt_s.step(epoch)
        
                    
    def val(self, epoch):

        args = self.args
        self.model.eval()
        with torch.no_grad():
            pca_indices = []
            pca_batches = []
            pivot = 0
            pca_sequence_count = 100
            r = 0
            its_len = len(self.val_loader)

            for its, dict_data in enumerate(self.val_loader):
                if its >= its_len:
                    raise StopIteration
                tar_pose = dict_data[self.input_type]
                # print(tar_pose.shape)
                tar_pose = tar_pose[:, :, 46:]
                # tar_pose = torch.cat((tar_pose[:, :, 70:154], tar_pose[:, :, 166:250]), axis=2)
                tar_pose = tar_pose.cuda()
                # tar_pose = torch.cat((tar_pose[:, :, :18], tar_pose[:, :, 75:84]), axis=2)
                # tar_pose = tar_pose[:, 1:, :] - tar_pose[:, :-1, :]
                # tmp = torch.zeros((tar_pose.shape[0], tar_pose.shape[1], 1)).cuda()

                recon_data, latent, signal, params = \
                self.model(tar_pose)
                if self.vel_weight > 0:  # use pose diff
                    target_diff = tar_pose[:, 1:] - tar_pose[:, :-1]
                    recon_diff = recon_data[:, 1:] - recon_data[:, :-1]
                    vel_rec_loss = torch.mean(self.vel_loss(recon_diff, target_diff), dim=(0, 1, 2))
                    self.loss_meters['vel_val'].update(vel_rec_loss.item())
                #print(recon_data.shape, tar_pose.shape)    
                recon_loss = self.vel_loss(recon_data, tar_pose)
                recon_loss = torch.mean(recon_loss, dim=(0, 1, 2))
                self.loss_meters['rec_val'].update(recon_loss.item())
                
                if epoch >= 1 and (epoch+1) % 2 == 0:
                    #Start Visualization Section
                    _a_ = Item(params[2]).squeeze().numpy()
                    for i in range(_a_.shape[0]):
                        dist_amps.append(_a_[i,:])
                    while len(dist_amps) > 10000:
                        dist_amps.pop(0)

                    _f_ = Item(params[1]).squeeze().numpy()
                    for i in range(_f_.shape[0]):
                        dist_freqs.append(_f_[i,:])
                    while len(dist_freqs) > 10000:
                        dist_freqs.pop(0)

                    loss_history.Add(
                        (Item(recon_loss).item(), "Reconstruction Loss")
                    )
                
                    plot.Functions(ax1[0], Item(tar_pose[0]).reshape(args.pose_dims,args.pose_length), -1.0, 1.0, -5.0, 5.0, title="Motion Curves" + " " + str(args.pose_dims) + "x" + str(args.pose_length), showAxes=False)
                    plot.Functions(ax1[1], Item(latent[0]), -1.0, 1.0, -2.0, 2.0, title="Latent Convolutional Embedding" + " " + str(args.embedding_channels) + "x" + str(args.pose_length), showAxes=False)
                    plot.Circles(ax1[2], Item(params[0][0]).squeeze(), Item(params[2][0]).squeeze(), title="Learned Phase Timing"  + " " + str(args.embedding_channels) + "x" + str(2), showAxes=False)
                    plot.Functions(ax1[3], Item(signal[0]), -1.0, 1.0, -2.0, 2.0, title="Latent Parametrized Signal" + " " + str(args.embedding_channels) + "x" + str(args.pose_length), showAxes=False)
                    plot.Functions(ax1[4], Item(recon_data[0]).reshape(args.pose_dims,args.pose_length), -1.0, 1.0, -5.0, 5.0, title="Curve Reconstruction" + " " + str(args.pose_dims) + "x" + str(args.pose_length), showAxes=False)
                    plot.Function(ax1[5], [Item(tar_pose[0]), Item(recon_data[0])], -1.0, 1.0, -5.0, 5.0, colors=[(0, 0, 0), (0, 1, 1)], title="Curve Reconstruction (Flattened)" + " " + str(1) + "x" + str(args.pose_dims*args.pose_length), showAxes=False)
                    plot.Distribution(ax3[0], dist_amps, title="Amplitude Distribution")
                    plot.Distribution(ax3[1], dist_freqs, title="Frequency Distribution")

                    for i in range(int(args.embedding_channels)):
                        phase = params[0][:,i]
                        freq = params[1][:,i]
                        amps = params[2][:,i]
                        offs = params[3][:,i]
                        plot.Phase1D(ax2[i,0], Item(phase), Item(amps), color=(0, 0, 0), title=("1D Phase Values" if i==0 else None), showAxes=False)
                        plot.Phase2D(ax2[i,1], Item(phase), Item(amps), title=("2D Phase Vectors" if i==0 else None), showAxes=False)
                        plot.Functions(ax2[i,2], Item(freq).transpose(0,1), -1.0, 1.0, 0.0, 4.0, title=("Frequencies" if i==0 else None), showAxes=False)
                        plot.Functions(ax2[i,3], Item(amps).transpose(0,1), -1.0, 1.0, 0.0, 1.0, title=("Amplitudes" if i==0 else None), showAxes=False)
                        plot.Functions(ax2[i,4], Item(offs).transpose(0,1), -1.0, 1.0, -1.0, 1.0, title=("Offsets" if i==0 else None), showAxes=False)
                    
                    #Visualization
                    if r < pca_sequence_count:
                        a = Item(params[2]).squeeze()
                        p = Item(params[0]).squeeze()
                        b = Item(params[3]).squeeze()
                        m_x = a * np.sin(2.0 * np.pi * p) + b
                        m_y = a * np.cos(2.0 * np.pi * p) + b
                        manifold = torch.hstack((m_x, m_y))
                        pca_indices.append(pivot + np.arange(self.pose_length))
                        pca_batches.append(manifold)
                        pivot += self.pose_length
                        r += 1
            if epoch >= 1 and (epoch+1) % 2 == 0:
                plot.PCA2D(ax4[0], pca_indices, pca_batches, "Phase Manifold (" + str(pca_sequence_count) + " Random Sequences)")
                np.save('./pae_data/pca_indeices_' + str(epoch) + '.npy', pca_indices)
                np.save('./pae_data/pca_batches_' + str(epoch) + '.npy', pca_batches)
                plt.gcf().canvas.draw_idle()
                plt.gcf().canvas.start_event_loop(1e-5)
                plt.savefig('performance.jpg')
            self.val_recording(epoch, self.loss_meters)
            
    def test(self, epoch):
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        start_time = time.time()
        total_length = 0
        test_seq_list = os.listdir(self.test_demo)
        test_seq_list.sort()
        self.model.eval()
        # self.std_pose = np.concatenate(self.std_pose[:18], self.std_pose[75:84])
        # self.mean_pose =  np.concatenate(self.std_pose[:18], self.std_pose[75:84])
        with torch.no_grad():
            if not os.path.exists(results_save_path):
                os.makedirs(results_save_path)
            for its, dict_data in enumerate(self.test_loader):
                tar_pose = dict_data[self.input_type]
                tar_pose = tar_pose[:, :, 46:]
                # tar_pose = torch.cat((tar_pose[:, :, 70:154], tar_pose[:, :, 166:250]), axis=2)
                tar_pose = tar_pose.cuda() # no mean
                # tar_pose = torch.cat((tar_pose[:, :, :18], tar_pose[:, :, 75:84]), axis=2)
                # tar_pose = tar_pose[:, 1:, :] - tar_pose[:, :-1, :]
                # tmp = torch.zeros((tar_pose.shape[0], tar_pose.shape[1], 1)).cuda()
                if "LSTM" in self.g_name or "multi_length" in self.notes:
                    recon_data, latent, signal, params = \
                    self.model(poses=tar_pose)
                    out_final = recon_data.cpu().numpy().reshape(-1, self.pose_dims)                
                else:
                    for i in range(tar_pose.shape[1]//(self.pose_length)):
                        tar_pose_new = tar_pose[:,i*(self.pose_length):i*(self.pose_length)+self.pose_length,:]
                        recon_data, latent, signal, params = \
                        self.model(tar_pose_new)
                        out_sub = recon_data.cpu().numpy().reshape(-1, self.pose_dims)
                        if i != 0:
                            out_final = np.concatenate((out_final,out_sub), 0)
                        else:
                            out_final = out_sub
                
                total_length += out_final.shape[0]
                with open(f"{results_save_path}result_raw_{test_seq_list[its]}", 'w+') as f_real:
                    for line_id in range(out_final.shape[0]): #,args.pre_frames, args.pose_length
                        line_data = np.array2string(out_final[line_id], max_line_width=np.inf, precision=6, suppress_small=False, separator=' ')
                        f_real.write(line_data[1:-2]+'\n')
            data_tools.result2target_vis(self.pose_version, results_save_path, results_save_path, self.test_demo, False)
            end_time = time.time() - start_time
            logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.pose_fps)} s motion")
              
    def save_parameters(self,):
        self.model.eval()
        with open(self.checkpoint_path+'/Parameters'+'.txt', 'w') as file:
            for i, dict_data in enumerate(self.train_loader):
                tar_pose = dict_data["pose"]
                tar_pose = tar_pose.cuda()
                self.opt.zero_grad()
                _, _, _, params = \
                    self.model(tar_pose)
                p = utility.ToNumpy(params[0]).squeeze()
                f = utility.ToNumpy(params[1]).squeeze()
                a = utility.ToNumpy(params[2]).squeeze()
                b = utility.ToNumpy(params[3]).squeeze()
                for j in range(p.shape[0]):
                    params = np.concatenate((p[j,:],f[j,:],a[j,:],b[j,]))
                    line = ' '.join(map(str, params))
                    if (i+j) == (len(self.train_loader)*self.batch_size-1):
                        file.write(line)
                    else:
                        file.write(line + '\n')