import os
import time

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

import train
from utils import other_tools


class CustomTrainer(train.BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.word_rep = args.word_rep
        self.emo_rep = args.emo_rep
        self.speaker_id = args.speaker_id
        self.best_epochs = {"rec_val": [np.inf, 0]}
        self.loss_meters = {
            "rec_val": other_tools.AverageMeter("rec_val"),
            "all": other_tools.AverageMeter("all"),
            "eye": other_tools.AverageMeter("eye"),
            "exp": other_tools.AverageMeter("exp"),
            "mouth": other_tools.AverageMeter("mouth"),
            "gen": other_tools.AverageMeter("gen"),
            "dis": other_tools.AverageMeter("dis"),
        }
        self.face_loss = nn.MSELoss()

    def _collect_inputs(self, batch_data):
        in_audio = batch_data["audio"].cuda() if self.audio_rep is not None else None
        in_facial = batch_data["facial"].cuda() if self.facial_rep is not None else None
        in_id = batch_data["id"].cuda() if self.speaker_id else None
        in_word = batch_data["word"].cuda() if self.word_rep is not None else None
        in_emo = batch_data["emo"].cuda() if self.emo_rep is not None else None
        return in_audio, in_facial, in_word, in_id, in_emo

    def _face_reconstruction_loss(self, out_face, target_face):
        target_delta = target_face[:, 1:, :] - target_face[:, :-1, :]
        output_delta = out_face[:, 1:, :] - out_face[:, :-1, :]

        eye = self.face_loss(out_face[:, :, :8], target_face[:, :, :8])
        eye_vel = 0.5 * self.face_loss(output_delta[:, :, :8], target_delta[:, :, :8])
        exp = self.face_loss(out_face[:, :, 8:26], target_face[:, :, 8:26])
        exp_vel = 0.5 * self.face_loss(output_delta[:, :, 8:26], target_delta[:, :, 8:26])
        mouth = self.face_loss(out_face[:, :, 26:49], target_face[:, :, 26:49])
        mouth_vel = 0.5 * self.face_loss(output_delta[:, :, 26:49], target_delta[:, :, 26:49])

        return {
            "eye": eye + eye_vel,
            "exp": exp + exp_vel,
            "mouth": mouth + mouth_vel,
        }

    def train(self, epoch):
        use_adv = bool(epoch >= self.no_adv_epochs)
        self.model.train()
        self.d_model.train()
        its_len = len(self.train_loader)
        t_start = time.time()
        for its, batch_data in enumerate(self.train_loader):
            t_data = time.time() - t_start
            in_audio, in_facial, in_word, in_id, in_emo = self._collect_inputs(batch_data)

            if use_adv:
                self.opt_d.zero_grad()
                out_face, *_ = self.model(in_audio=in_audio, in_text=in_word, in_id=in_id, in_emo=in_emo)
                out_d_fake = self.d_model(out_face.detach())
                out_d_real = self.d_model(in_facial)
                d_loss = torch.sum(-torch.mean(torch.log(out_d_real + 1e-8) + torch.log(1 - out_d_fake + 1e-8)))
                self.loss_meters["dis"].update(d_loss.item())
                d_loss.backward()
                self.opt_d.step()

            self.opt.zero_grad()
            out_face, *_ = self.model(in_audio=in_audio, in_text=in_word, in_id=in_id, in_emo=in_emo)
            losses = self._face_reconstruction_loss(out_face, in_facial)
            rec_loss = losses["eye"] + losses["exp"] + losses["mouth"]
            g_loss = rec_loss * self.rec_weight

            if use_adv:
                dis_out = self.d_model(out_face)
                adv_loss = -torch.mean(torch.log(dis_out + 1e-8))
                adv_loss = self.adv_weight * adv_loss
                self.loss_meters["gen"].update(adv_loss.item())
                g_loss = g_loss + adv_loss

            for name, value in losses.items():
                self.loss_meters[name].update(value.item())
            self.loss_meters["all"].update(g_loss.item())
            g_loss.backward()
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
                in_audio, in_facial, in_word, in_id, in_emo = self._collect_inputs(batch_data)
                out_face, *_ = self.model(in_audio=in_audio, in_text=in_word, in_id=in_id, in_emo=in_emo)
                losses = self._face_reconstruction_loss(out_face, in_facial)
                rec_loss = losses["eye"] + losses["exp"] + losses["mouth"]
                self.loss_meters["rec_val"].update((rec_loss * self.rec_weight).item())
                if self.args.max_val_batches and its + 1 >= self.args.max_val_batches:
                    break
            self.val_recording(epoch, self.loss_meters)

    def test(self, epoch):
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        start_time = time.time()
        total_length = 0
        test_seq_list = sorted(os.listdir(self.test_demo))
        self.model.eval()
        with torch.no_grad():
            os.makedirs(results_save_path, exist_ok=True)
            for its, batch_data in enumerate(self.test_loader):
                in_audio, _, in_word, in_id, in_emo = self._collect_inputs(batch_data)
                if in_audio is not None:
                    in_audio = in_audio.reshape(1, -1)
                out_face, *_ = self.model(in_audio=in_audio, in_text=in_word, in_id=in_id, in_emo=in_emo)
                total_length += out_face.shape[1]
                save_name = test_seq_list[its].replace(".bvh", ".npy")
                np.save(f"{results_save_path}result_raw_{save_name}", out_face.cpu().numpy())
                if self.args.max_test_batches and its + 1 >= self.args.max_test_batches:
                    break
        end_time = time.time() - start_time
        logger.info(f"generated facial coefficients for {int(total_length / self.pose_fps)} s in {int(end_time)} s")
