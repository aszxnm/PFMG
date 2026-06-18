# Release Checklist

This file is for maintainers. User-facing instructions are in `README.md`, `docs/DATA.md`, and `docs/MODEL_ZOO.md`.

## Finished

- Removed local datasets, LMDB caches, checkpoints, generated outputs, notebooks, and nested git histories from the release tree.
- Added `.gitignore` rules for data, outputs, checkpoints, caches, and binary artifacts.
- Changed configs to work with `--root_path .` and local `data/`.
- Added `--wandb_mode` and `--wandb_entity` to avoid hard-coded W&B settings.
- Added `--max_train_batches`, `--max_val_batches`, and `--max_test_batches` for smoke tests.
- Added `scripts/prepare_beat.py` for BEAT-style feature cache preparation.
- Rewrote README and docs around the PFMG / PFMG-PAE release path.
- Removed score computation from the public PFMG-PAE generation path.

## Local validation on 2026-06-18

The following checks passed in the private workspace with local data and weights after the PFMG-PAE generation-only cleanup:

```shell
python -m py_compile train.py test.py audio2face_trainer.py pfmg_tcn_pae_trainer.py models/audio2face.py scripts/prepare_beat.py
python scripts/prepare_beat.py --help
```

Audio2Face training smoke test passed with one validation batch and one training batch. Use batch size 2 or larger because the facial heads use BatchNorm:

```shell
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled \
PFMG_WAV2VEC2_MODEL=/path/to/wav2vec2-large-xlsr-53-english \
PFMG_WAV2VEC2_EMOTION_MODEL=/path/to/wav2vec-english-speech-emotion-recognition \
python train.py \
  -c configs/audio2face_4english_15_141.yaml \
  --root_path . \
  --wandb_mode disabled \
  --batch_size 2 \
  --epochs 1 \
  --max_train_batches 1 \
  --max_val_batches 1 \
  --test_period 9999 \
  --loader_workers 0 \
  --gpus 0 \
  --no_adv_epochs 9999
```

PFMG-PAE training smoke test passed with one validation batch and one training batch:

```shell
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled \
PFMG_WAV2VEC2_MODEL=/path/to/wav2vec2-large-xlsr-53-english \
PFMG_WAV2VEC2_EMOTION_MODEL=/path/to/wav2vec-english-speech-emotion-recognition \
python train.py \
  -c configs/pfmg_tcn_pae_4english_15_141.yaml \
  --root_path . \
  --wandb_mode disabled \
  --batch_size 1 \
  --epochs 1 \
  --max_train_batches 1 \
  --max_val_batches 1 \
  --test_period 9999 \
  --loader_workers 0 \
  --gpus 0 \
  --no_adv_epochs 9999
```

Pretrained generation smoke passed with `pfmg_tcn_pae.bin` and `--max_test_batches 1`; the run generated `result_raw_*.bvh` and converted it to `res_*.bvh`.

## Before public release

- Fill project page, paper link, and final BibTeX in `README.md`.
- Upload checkpoints and fill URLs/SHA256 values in `docs/MODEL_ZOO.md`.
- Confirm BEAT data instructions and redistribution boundaries.
- Confirm the selected license and all third-party notices with coauthors and institutional stakeholders.
- Confirm the redistributable license for the included pymo snapshot.
- Run a clean clone smoke test without access to the original private workspace.

## Clean clone smoke target

```shell
python -m py_compile train.py test.py audio2face_trainer.py pfmg_tcn_pae_trainer.py models/audio2face.py scripts/prepare_beat.py
python scripts/prepare_beat.py --help
python train.py -c configs/audio2face_4english_15_141.yaml --root_path . --wandb_mode disabled --batch_size 2 --epochs 1 --max_train_batches 1 --max_val_batches 1
python train.py -c configs/pfmg_tcn_pae_4english_15_141.yaml --root_path . --wandb_mode disabled --epochs 1 --max_train_batches 1 --max_val_batches 1
python test.py -c configs/pfmg_tcn_pae_4english_15_141.yaml --root_path . --wandb_mode disabled --max_test_batches 1
```
