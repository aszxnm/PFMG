# Copyright (c) HuaWei, Inc. and its affiliates.
# liu.haiyang@huawei.com
# Inference script for PFMG.

import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.distributed as dist
from loguru import logger
import wandb

from utils import config, logger_tools, other_tools
from dataloaders import data_tools
from dataloaders.build_vocab import Vocab


class BaseTrainer(object):
    def __init__(self, args):
        self.args = args
        self.notes = args.notes
        self.ddp = args.ddp
        self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.checkpoint_path = args.root_path + args.out_root_path + "custom/" + args.name + args.notes + "/"
        self.trainer_name = args.trainer

        self.pose_version = args.pose_version
        self.mean_pose = np.load(args.root_path + args.mean_pose_path + f"{args.pose_rep}/bvh_mean.npy")
        self.std_pose = np.load(args.root_path + args.mean_pose_path + f"{args.pose_rep}/bvh_std.npy")

        self.pose_rep = args.pose_rep
        self.pose_fps = args.pose_fps
        self.pose_dims = args.pose_dims
        self.pose_pae = args.pose_pae
        self.audio_rep = args.audio_rep
        self.facial_rep = args.facial_rep
        self.word_rep = args.word_rep
        self.emo_rep = args.emo_rep
        self.speaker_id = args.speaker_id
        self.pre_frames = args.pre_frames
        self.test_demo = args.root_path + args.test_data_path + f"{args.pose_rep}_vis/"

        self.test_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).CustomDataset(args, "test")
        self.test_loader = torch.utils.data.DataLoader(
            self.test_data,
            batch_size=1,
            shuffle=False,
            num_workers=args.loader_workers,
            drop_last=False,
        )
        logger.info("Init test dataloader success")

        model_module = __import__(f"models.{args.model}", fromlist=["something"])
        self.model = getattr(model_module, args.g_name)(args)
        other_tools.load_checkpoints(self.model, args.root_path + args.test_ckpt, args.g_name)
        self.model = torch.nn.DataParallel(self.model, args.gpus).cuda()
        if self.rank == 0:
            logger.info(self.model)
            wandb.watch(self.model)
            logger.info(f"init {args.g_name} success")

    def _run_model(self, pre_pose, in_audio, in_facial, in_pae, in_word, in_id, in_emo):
        if self.trainer_name == "pfmg_tcn_pae":
            out_dir_vec, *_ = self.model(
                pre_seq=pre_pose,
                in_audio=in_audio,
                in_text=in_word,
                in_facial=in_facial,
                in_pae=in_pae,
                in_id=in_id,
                in_emo=in_emo,
            )
            return out_dir_vec, None
        if self.trainer_name == "pfmg":
            out_dir_vec = self.model(
                pre_seq=pre_pose,
                in_audio=in_audio,
                in_text=in_word,
                in_facial=in_facial,
                in_pae=in_pae,
                in_id=in_id,
                in_emo=in_emo,
            )
            return out_dir_vec, None
        if self.trainer_name == "audio2face":
            out_face, *_ = self.model(in_audio=in_audio, in_text=in_word, in_id=in_id, in_emo=in_emo)
            return out_face, None
        if self.trainer_name == "habibie":
            out_dir_vec, out_face = self.model(
                pre_seq=pre_pose,
                in_audio=in_audio,
                in_text=in_word,
                in_facial=in_facial,
                in_id=in_id,
                in_emo=in_emo,
            )
            return out_dir_vec, out_face
        if self.trainer_name == "a2g":
            out_dir_vec = self.model(
                pre_seq=pre_pose,
                in_audio=in_audio,
                in_pose=None,
                in_text=in_word,
                in_facial=in_facial,
                in_id=in_id,
                in_emo=in_emo,
            )
            return out_dir_vec, None
        if self.trainer_name == "multi":
            out_dir_vec, *_ = self.model(pre_seq=pre_pose, in_audio=in_audio, in_word=in_word, in_id=in_id)
            return out_dir_vec, None
        if self.trainer_name == "com":
            raise NotImplementedError("External comparison results are not part of the public PFMG release.")
        out_dir_vec = self.model(
            pre_seq=pre_pose,
            in_audio=in_audio,
            in_text=in_word,
            in_facial=in_facial,
            in_id=in_id,
            in_emo=in_emo,
        )
        return out_dir_vec, None

    def test(self, epoch):
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        start_time = time.time()
        total_length = 0
        test_seq_list = sorted(os.listdir(self.test_demo))
        self.model.eval()

        with torch.no_grad():
            os.makedirs(results_save_path, exist_ok=True)
            for its, batch_data in enumerate(self.test_loader):
                seq_name = test_seq_list[its]
                logger.info(f"Generating {results_save_path}result_raw_{seq_name}")
                if "1_1_1" in seq_name:
                    continue

                tar_pose = batch_data["pose"].cuda()
                in_audio = batch_data["audio"].cuda() if self.audio_rep is not None else None
                in_facial = batch_data["facial"].cuda() if self.facial_rep is not None else None
                in_pae = batch_data["pae"].cuda() if self.pose_pae is not None else None
                in_id = batch_data["id"].cuda() if self.speaker_id else None
                in_word = batch_data["word"].cuda() if self.word_rep is not None else None
                in_emo = batch_data["emo"].cuda() if self.emo_rep is not None else None

                if in_id is not None:
                    in_id = in_id + 2

                pre_pose = tar_pose.new_zeros((tar_pose.shape[0], tar_pose.shape[1], tar_pose.shape[2] + 1)).cuda()
                pre_pose[:, 0:self.pre_frames, :-1] = tar_pose[:, 0:self.pre_frames]
                pre_pose[:, 0:self.pre_frames, -1] = 1

                if in_audio is not None:
                    in_audio = in_audio.reshape(1, -1)

                out_dir_vec, out_face = self._run_model(pre_pose, in_audio, in_facial, in_pae, in_word, in_id, in_emo)
                if self.trainer_name == "audio2face":
                    np.save(f"{results_save_path}result_raw_{seq_name}", out_dir_vec.cpu().numpy())
                    continue

                out_dir_vec = out_dir_vec[:, :tar_pose.shape[1]]
                out_final = (out_dir_vec.cpu().numpy().reshape(-1, self.pose_dims) * self.std_pose) + self.mean_pose
                total_length += out_final.shape[0]

                with open(f"{results_save_path}result_raw_{seq_name}", "w+") as f_real:
                    for line_id in range(out_final.shape[0]):
                        line_data = np.array2string(out_final[line_id], max_line_width=np.inf, precision=6, suppress_small=False, separator=" ")
                        f_real.write(line_data[1:-2] + chr(10))

                if out_face is not None:
                    save_path = seq_name.replace(".bvh", ".npy")
                    np.save(f"{results_save_path}result_raw_{save_path}", out_face.cpu())

                if self.args.max_test_batches and its + 1 >= self.args.max_test_batches:
                    break

        data_tools.result2target_vis(self.pose_version, results_save_path, results_save_path, self.test_demo, False)
        end_time = time.time() - start_time
        logger.info(f"generated {int(total_length / self.pose_fps)} s motion in {int(end_time)} s")


@logger.catch
def main_worker(rank, world_size, args):
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    if args.ddp:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    logger_tools.set_args_and_logger(args, rank)
    other_tools.set_random_seed(args)
    other_tools.print_exp_info(args)

    trainer = BaseTrainer(args)
    logger.info("Generating motion from checkpoint ...")
    trainer.test(9999)


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "2222"
    args = config.parse_args()
    main_worker(0, 1, args)
