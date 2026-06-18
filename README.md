# PFMG

This repository contains the PyTorch implementation of PFMG, an audio-driven co-speech gesture generation method built on BEAT-style body, expression, audio, text, emotion, speaker, and PAE motion features.

The public release focuses on the PFMG-PAE generation path. Given speech-related conditions and PAE features, the model generates upper-body gesture motion and writes BVH-compatible result files for visualization.

[Project Page: TBD](#) | [Paper: TBD](#) | [Pretrained Models: TBD](docs/MODEL_ZOO.md) | [Data Preparation](docs/DATA.md)

## News

- **[2026/06]** Release training and inference code for PFMG.
- **[2026/06]** Add a clean BEAT-style cache preparation script in `scripts/prepare_beat.py`.
- **[TBD]** Public pretrained checkpoints will be released after final hosting and license review.

## Release Plans

- [x] PFMG-PAE training code
- [x] PFMG-PAE inference code
- [x] BEAT-style dataloader and cache helper
- [x] Audio2Face training code for producing `face.bin`
- [x] PFMG-PAE release config and auxiliary checkpoint layout
- [ ] Public pretrained checkpoint links
- [ ] Final project page and paper links

## Contents

- `train.py`, `test.py`: common training and motion-generation entry points.
- `audio2face_trainer.py`: trains the Audio2Face auxiliary model used to produce `face.bin`.
- `models/`: PFMG-PAE, PFMG-TCN, audio-to-face, and PAE-related model code.
- `dataloaders/`: BEAT-style LMDB dataloader, vocabulary builder, and motion utilities.
- `configs/`: release configs for Audio2Face, PFMG-PAE, and PAE feature preparation.
- `scripts/prepare_beat.py`: builds statistics, vocabulary, and LMDB caches from prepared BEAT feature folders.

## Installation

We recommend Python `==3.8`. The release smoke tests were run with Python 3.8, CUDA, PyTorch 1.12, and `pyarrow==4.0.0`.

```shell
conda create -n pfmg python=3.8
conda activate pfmg
pip install -r requirements-legacy.txt
```

If you use local HuggingFace Wav2Vec2 directories, set:

```shell
export PFMG_WAV2VEC2_MODEL=/path/to/wav2vec2-large-xlsr-53-english
export PFMG_WAV2VEC2_EMOTION_MODEL=/path/to/wav2vec-english-speech-emotion-recognition
```

## Download weights

Pretrained weights are not included in git. Please place downloaded checkpoints under:

```text
./data/beat_cache/beat_4english_15_141/weights/
```

The main PFMG-PAE path expects:

```text
face.bin
pfmg_tcn.bin
pfmg_tcn_pae.bin
```

`face.bin` and `pfmg_tcn.bin` are auxiliary checkpoints loaded by the final PFMG-PAE generator.

See `docs/MODEL_ZOO.md` for checkpoint names and download placeholders.

## Data Preparation

This release does not redistribute BEAT data or generated caches. After obtaining BEAT data and converting it into the feature folders described in `docs/DATA.md`, run:

```shell
python scripts/prepare_beat.py \
  --cache-root ./data/beat_cache/beat_4english_15_141
```

The expected feature folders are:

```text
train/val/test
|-- bvh_rot
|-- wave16k
|-- facial52
|-- pae
|-- text
|-- emo
`-- sem
```

## Training and Motion Generation

Commands below assume they are run from this repository root.

### Train Audio2Face

PFMG-TCN loads `face.bin` to generate facial conditions before body-motion generation. To train this auxiliary model from the prepared BEAT cache, run:

```shell
python train.py \
  -c configs/audio2face_4english_15_141.yaml \
  --root_path . \
  --wandb_mode disabled
```

The best validation checkpoint is saved as:

```text
outputs/audio2pose/custom/<audio2face_exp_name>/rec_val.bin
```

Rename or copy this checkpoint to:

```text
data/beat_cache/beat_4english_15_141/weights/face.bin
```

### Train PFMG-PAE

PFMG-PAE expects the auxiliary `face.bin` and `pfmg_tcn.bin` checkpoints under `data/beat_cache/beat_4english_15_141/weights/`. Then run:

```shell
python train.py \
  -c configs/pfmg_tcn_pae_4english_15_141.yaml \
  --root_path . \
  --wandb_mode disabled
```

### Generate motion with pretrained PFMG-PAE

```shell
python test.py \
  -c configs/pfmg_tcn_pae_4english_15_141.yaml \
  --root_path . \
  --wandb_mode disabled \
  --gpus 0
```

The script writes raw generated motion to `result_raw_*.bvh` files and then converts them to `res_*.bvh` files under:

```text
outputs/audio2pose/custom/<exp_name>/9999/
```

These generated BVH files can be loaded into Blender with the same visualization flow used by BEAT.

## Quick smoke test

For checking the environment and data path quickly:

```shell
python -m py_compile train.py test.py audio2face_trainer.py pfmg_tcn_pae_trainer.py models/audio2face.py scripts/prepare_beat.py
python scripts/prepare_beat.py --help
python train.py -c configs/audio2face_4english_15_141.yaml --root_path . --wandb_mode disabled --batch_size 2 --epochs 1 --max_train_batches 1 --max_val_batches 1 --test_period 9999 --loader_workers 0 --gpus 0
python train.py -c configs/pfmg_tcn_pae_4english_15_141.yaml --root_path . --wandb_mode disabled --batch_size 1 --epochs 1 --max_train_batches 1 --max_val_batches 1 --test_period 9999 --loader_workers 0 --gpus 0
python test.py -c configs/pfmg_tcn_pae_4english_15_141.yaml --root_path . --wandb_mode disabled --max_test_batches 1 --loader_workers 0 --gpus 0
```

## Acknowledgements

This codebase builds on BEAT / PantoMatrix audio-to-gesture code and the BEAT 2022 CaMN baseline. We thank the authors of BEAT, CaMN, HuggingFace Transformers, timm, pymo, and related open-source projects. See `NOTICE.md` for details.

## Citation

If you find this code useful for your research, please cite the PFMG paper. The final BibTeX will be added before public release.

```bibtex
@article{pfmg2026,
  title   = {PFMG},
  author  = {TBD},
  journal = {TBD},
  year    = {2026}
}
```

## License

This release is for non-commercial research use. See `LICENSE.md` and `NOTICE.md` for license and third-party notices.
