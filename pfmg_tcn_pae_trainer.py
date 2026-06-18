import train
import os
import time
import torch
import numpy as np
from loguru import logger

from utils import other_tools
from dataloaders import data_tools


class CustomTrainer(train.BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.word_rep = args.word_rep
        self.emo_rep = args.emo_rep
        self.sem_rep = args.sem_rep
        self.speaker_id = args.speaker_id
        self.best_epochs = {"rec_val": [np.inf, 0]}
        self.loss_meters = {
            "rec_val": other_tools.AverageMeter("rec_val"),
            "all": other_tools.AverageMeter("all"),
            "rec": other_tools.AverageMeter("rec"),
            "gen": other_tools.AverageMeter("gen"),
            "pae": other_tools.AverageMeter("pae"),
            "dis": other_tools.AverageMeter("dis"),
        }

    def train(self, epoch):
        use_adv = bool(epoch >= self.no_adv_epochs)
        self.model.train()
        self.d_model.train()
        its_len = len(self.train_loader)
        t_start = time.time()
        for its, batch_data in enumerate(self.train_loader):
            t_data = time.time() - t_start
            tar_pose = batch_data["pose"].cuda()
            in_audio = batch_data["audio"].cuda() if self.audio_rep is not None else None
            in_facial = batch_data["facial"].cuda() if self.facial_rep is not None else None
            in_pae = batch_data["pae"].cuda() if self.pose_pae is not None else None
            in_id = batch_data["id"].cuda() if self.speaker_id else None
            in_word = batch_data["word"].cuda() if self.word_rep is not None else None
            in_emo = batch_data["emo"].cuda() if self.emo_rep is not None else None
            in_sem = batch_data["sem"].cuda() if self.sem_rep is not None else None

            in_pre_pose = tar_pose.new_zeros((tar_pose.shape[0], tar_pose.shape[1], tar_pose.shape[2] + 1)).cuda()
            in_pre_pose[:, 0:self.pre_frames, :-1] = tar_pose[:, 0:self.pre_frames]
            in_pre_pose[:, 0:self.pre_frames, -1] = 1
            t_data = time.time() - t_start

            d_loss_final = 0
            if use_adv:
                self.opt_d.zero_grad()
                out_pose, *_ = self.model(in_pre_pose, in_audio=in_audio, in_facial=in_facial, in_pae=in_pae, in_text=in_word, in_id=in_id, in_emo=in_emo)
                out_d_fake = self.d_model(out_pose)
                out_d_real = self.d_model(tar_pose)
                d_loss_adv = torch.sum(-torch.mean(torch.log(out_d_real + 1e-8) + torch.log(1 - out_d_fake + 1e-8)))
                d_loss_final += d_loss_adv
                self.loss_meters["dis"].update(d_loss_final.item())
                d_loss_final.backward()
                self.opt_d.step()
            self.opt.zero_grad()

            g_loss_final = 0
            out_pose, pre_pae, *_ = self.model(in_pre_pose, in_audio=in_audio, in_facial=in_facial, in_pae=in_pae, in_text=in_word, in_id=in_id, in_emo=in_emo)
            if self.sem_rep is not None:
                huber_value = self.rec_loss(tar_pose * (in_sem.unsqueeze(2) + 1), out_pose * (in_sem.unsqueeze(2) + 1))
            else:
                huber_value = self.rec_loss(tar_pose, out_pose)
            pae_loss = self.rec_loss(pre_pae, in_pae)
            self.loss_meters["pae"].update(pae_loss.item())

            huber_value *= self.rec_weight
            self.loss_meters["rec"].update(huber_value.item())
            g_loss_final += huber_value + pae_loss
            if use_adv:
                dis_out = self.d_model(out_pose)
                d_fake_value = -torch.mean(torch.log(dis_out + 1e-8))
                d_fake_value *= self.adv_weight * d_fake_value
                self.loss_meters["gen"].update(d_fake_value.item())

            self.loss_meters["all"].update(g_loss_final.item())
            g_loss_final.requires_grad_(True)
            g_loss_final.backward()
            if self.grad_norm != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
            self.opt.step()

            t_train = time.time() - t_start - t_data
            t_start = time.time()
            mem_cost = torch.cuda.memory_reserved() / 1E9
            lr_g = self.opt.param_groups[0]["lr"]
            lr_d = self.opt_d.param_groups[0]["lr"]
            if its % self.log_period == 0:
                self.recording(epoch, its, its_len, self.loss_meters, lr_g, lr_d, t_data, t_train, mem_cost)
            if self.args.max_train_batches and its + 1 >= self.args.max_train_batches:
                break
        self.opt_s.step(epoch)
        self.opt_d_s.step(epoch)

    def val(self, epoch):
        self.model.eval()
        with torch.no_grad():
            for its, batch_data in enumerate(self.val_loader):
                tar_pose = batch_data["pose"].cuda()
                in_audio = batch_data["audio"].cuda() if self.audio_rep is not None else None
                in_facial = batch_data["facial"].cuda() if self.facial_rep is not None else None
                in_pae = batch_data["pae"].cuda() if self.pose_pae is not None else None
                in_id = batch_data["id"].cuda() if self.speaker_id else None
                in_word = batch_data["word"].cuda() if self.word_rep is not None else None
                in_emo = batch_data["emo"].cuda() if self.emo_rep is not None else None

                in_pre_pose = tar_pose.new_zeros((tar_pose.shape[0], tar_pose.shape[1], tar_pose.shape[2] + 1)).cuda()
                in_pre_pose[:, 0:self.pre_frames, :-1] = tar_pose[:, 0:self.pre_frames]
                in_pre_pose[:, 0:self.pre_frames, -1] = 1

                out_pose, _, *_ = self.model(in_pre_pose, in_audio=in_audio, in_facial=in_facial, in_pae=in_pae, in_text=in_word, in_id=in_id, in_emo=in_emo)
                huber_value = self.rec_loss(tar_pose, out_pose)
                huber_value *= self.rec_weight
                self.loss_meters["rec_val"].update(huber_value.item())
                if self.args.max_val_batches and its + 1 >= self.args.max_val_batches:
                    break
            self.val_recording(epoch, self.loss_meters)

    def test(self, epoch):
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        start_time = time.time()
        total_length = 0
        test_seq_list = os.listdir(self.test_demo)
        test_seq_list.sort()
        self.model.eval()
        with torch.no_grad():
            if not os.path.exists(results_save_path):
                os.makedirs(results_save_path)
            for its, batch_data in enumerate(self.test_loader):
                tar_pose = batch_data["pose"].cuda()
                in_audio = batch_data["audio"].cuda() if self.audio_rep is not None else None
                in_facial = batch_data["facial"].cuda() if self.facial_rep is not None else None
                in_pae = batch_data["pae"].cuda() if self.pose_pae is not None else None
                in_id = batch_data["id"].cuda() if self.speaker_id else None
                in_word = batch_data["word"].cuda() if self.word_rep is not None else None
                in_emo = batch_data["emo"].cuda() if self.emo_rep is not None else None

                pre_pose = tar_pose.new_zeros((tar_pose.shape[0], tar_pose.shape[1], tar_pose.shape[2] + 1)).cuda()
                pre_pose[:, 0:self.pre_frames, :-1] = tar_pose[:, 0:self.pre_frames]
                pre_pose[:, 0:self.pre_frames, -1] = 1

                in_audio = in_audio.reshape(1, -1)
                out_dir_vec, *_ = self.model(pre_seq=pre_pose, in_audio=in_audio, in_text=in_word, in_facial=in_facial, in_pae=in_pae, in_id=in_id, in_emo=in_emo)
                out_final = (out_dir_vec.cpu().numpy().reshape(-1, self.pose_dims) * self.std_pose) + self.mean_pose
                total_length += out_final.shape[0]

                with open(f"{results_save_path}result_raw_{test_seq_list[its]}", "w+") as f_real:
                    for line_id in range(out_final.shape[0]):
                        line_data = np.array2string(out_final[line_id], max_line_width=np.inf, precision=6, suppress_small=False, separator=" ")
                        f_real.write(line_data[1:-2] + chr(10))
                if self.args.max_test_batches and its + 1 >= self.args.max_test_batches:
                    break
        data_tools.result2target_vis(self.pose_version, results_save_path, results_save_path, self.test_demo, False)
        end_time = time.time() - start_time
        logger.info(f"generated {int(total_length / self.pose_fps)} s motion in {int(end_time)} s")
