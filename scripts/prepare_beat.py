#!/usr/bin/env python3
"""Prepare a BEAT-style cache for PFMG.

This script assumes the raw BEAT data has already been converted into the
feature folders consumed by `dataloaders/beat.py`, for example:

  beat_4english_15_141/
    train/{bvh_rot,wave16k,facial52,pae,text,emo,sem}/
    val/{bvh_rot,wave16k,facial52,pae,text,emo,sem}/
    test/{bvh_rot,wave16k,facial52,pae,text,emo,sem}/

It then computes pose statistics, optionally builds the vocabulary, and
materializes LMDB caches (`*_cache`) for each split.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloaders.beat import CustomDataset
from dataloaders.build_vocab import build_vocab


def _with_slash(path: Path) -> str:
    return str(path.resolve()) + "/"


def compute_pose_stats(train_pose_dir: Path) -> None:
    pose_files = sorted(train_pose_dir.glob("*.bvh"))
    if not pose_files:
        raise FileNotFoundError(f"No BVH files found in {train_pose_dir}")

    chunks = []
    for pose_file in pose_files:
        frames = []
        with pose_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                values = np.fromstring(line, dtype=float, sep=" ")
                if values.size:
                    frames.append(values)
        if frames:
            chunks.append(np.asarray(frames, dtype=np.float32))

    if not chunks:
        raise ValueError(f"No numeric pose frames found in {train_pose_dir}")

    poses = np.concatenate(chunks, axis=0)
    mean = poses.mean(axis=0)
    std = poses.std(axis=0)
    std[std < 1e-8] = 1.0
    np.save(train_pose_dir / "bvh_mean.npy", mean)
    np.save(train_pose_dir / "bvh_std.npy", std)
    print(f"Wrote pose stats to {train_pose_dir}")


def make_dataset_args(args: argparse.Namespace, split: str) -> SimpleNamespace:
    cache_root = args.cache_root.resolve()
    train_path = _with_slash(cache_root / "train")
    return SimpleNamespace(
        root_path="",
        train_data_path=train_path,
        val_data_path=_with_slash(cache_root / "val"),
        test_data_path=_with_slash(cache_root / "test"),
        mean_pose_path=train_path,
        std_pose_path=train_path,
        new_cache=args.overwrite,
        pose_length=args.pose_length,
        stride=args.stride,
        pose_fps=args.pose_fps,
        pose_dims=args.pose_dims,
        speaker_dims=args.speaker_dims,
        audio_rep=args.audio_rep,
        pose_rep=args.pose_rep,
        facial_rep=args.facial_rep,
        pose_pae=args.pose_pae,
        pose_world=args.pose_world,
        pose_vel=args.pose_vel,
        word_rep=args.word_rep,
        emo_rep=args.emo_rep,
        sem_rep=args.sem_rep,
        audio_fps=args.audio_fps,
        speaker_id=args.speaker_id,
        disable_filtering=args.disable_filtering,
        clean_first_seconds=args.clean_first_seconds,
        clean_final_seconds=args.clean_final_seconds,
        multi_length_training=args.multi_length_training if split != "test" else [1.0],
        audio_norm=False,
        facial_norm=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True, help="Prepared BEAT feature root.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--overwrite", action="store_true", help="Rebuild existing LMDB caches.")
    parser.add_argument("--skip-lmdb", action="store_true", help="Only compute stats/vocab; do not build LMDB caches.")
    parser.add_argument("--skip-vocab", action="store_true", help="Do not build vocab.pkl.")
    parser.add_argument("--pose-rep", default="bvh_rot")
    parser.add_argument("--audio-rep", default="wave16k")
    parser.add_argument("--facial-rep", default="facial52")
    parser.add_argument("--pose-pae", default="pae")
    parser.add_argument("--pose-world", default=None)
    parser.add_argument("--pose-vel", default=None)
    parser.add_argument("--word-rep", default="text")
    parser.add_argument("--emo-rep", default="emo")
    parser.add_argument("--sem-rep", default="sem")
    parser.add_argument("--speaker-id", default="id")
    parser.add_argument("--pose-length", type=int, default=34)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--pose-fps", type=int, default=15)
    parser.add_argument("--audio-fps", type=int, default=16000)
    parser.add_argument("--pose-dims", type=int, default=141)
    parser.add_argument("--speaker-dims", type=int, default=30)
    parser.add_argument("--multi-length-training", type=float, nargs="+", default=[1.0])
    parser.add_argument("--disable-filtering", action="store_true")
    parser.add_argument("--clean-first-seconds", type=int, default=0)
    parser.add_argument("--clean-final-seconds", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.cache_root = args.cache_root.resolve()

    train_pose_dir = args.cache_root / "train" / args.pose_rep
    compute_pose_stats(train_pose_dir)

    if not args.skip_vocab and args.word_rep:
        build_vocab(
            "beat_english_15_141",
            _with_slash(args.cache_root),
            str(args.cache_root / "vocab.pkl"),
            None,
            300,
        )
        print(f"Wrote vocabulary to {args.cache_root / 'vocab.pkl'}")

    if args.skip_lmdb:
        return

    for split in args.splits:
        print(f"Building LMDB cache for {split}")
        dataset_args = make_dataset_args(args, split)
        dataset = CustomDataset(dataset_args, split, build_cache=True)
        print(f"{split}: {len(dataset)} samples")


if __name__ == "__main__":
    main()
