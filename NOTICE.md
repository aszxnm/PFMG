# Third-Party Notices

This repository is a cleaned release candidate for the PFMG experiments. It contains project code plus several upstream components needed for reproducibility. Do not remove upstream copyright headers from source files.

## Primary upstream code

- BEAT / PantoMatrix audio-to-pose code. The original local source remote was `https://github.com/PantoMatrix/BEAT`. Several files retain Huawei copyright headers, including `train.py`, `test.py`, trainer files, dataloaders, optimizers, and utility code. The upstream non-commercial license snapshot is preserved as `LICENSE.upstream-cc-by-nc.md`.
- BEAT 2022 baseline code. `baselines/beat/` is a cleaned snapshot adapted from the local `codes/beat` workspace whose original remote was `https://github.com/beat2022dataset/beat`. Its upstream README is kept at `baselines/beat/README.upstream.md`.

## Optimizer and scheduler implementations

- `optimizers/timm/` and `baselines/beat/optimizers/timm/` contain optimizer and scheduler implementations derived from timm / pytorch-image-models by Ross Wightman and related upstream sources. These files include their upstream attribution comments.
- `optimizers/timm/adafactor.py` is derived from fairseq and carries Facebook copyright and MIT license comments.
- `optimizers/timm/adamp.py` and `optimizers/timm/sgdp.py` are derived from NAVER CLOVA AdamP/SGDP implementations and carry MIT license comments.
- `optimizers/timm/adahessian.py` is derived from David Samuel AdaHessian implementation and carries MIT license comments.
- `optimizers/timm/rmsprop_tf.py` is derived from PyTorch RMSprop and notes PyTorch BSD-style license.

## Motion processing utilities

- `dataloaders/pymo/` and `baselines/beat/dataloaders/pymo/` include pymo-style BVH parsing and preprocessing utilities by Omid Alemi. Some parser logic is based on `https://gist.github.com/johnfredcee/2007503`. The exact redistributable license for this snapshot should be confirmed before public release.

## External runtime dependencies

- HuggingFace Transformers is used for Wav2Vec2 audio encoders in `models/audio2face.py`. Transformers is distributed separately under its own license.
- The default Wav2Vec2 model identifiers or local directories are not redistributed in this repository. Users must obtain those models from their official sources and follow the model-card terms. Local paths can be set with `PFMG_WAV2VEC2_MODEL` and `PFMG_WAV2VEC2_EMOTION_MODEL`.
- fastText embeddings, if used, are not included and must be obtained separately under their own terms.

## Dataset and project checkpoints

- BEAT raw data, derived feature folders, LMDB caches, videos, audio files, BVH files, and transcripts are not included. Users must obtain BEAT from the official source and comply with the BEAT dataset license.
- Project pretrained weights are not committed to git. Public download URLs, checksums, and terms should be filled in `docs/MODEL_ZOO.md` before release.

## Items needing final human review

- Confirm that every redistributed upstream file is compatible with the chosen non-commercial release terms.
- Confirm the exact license of the included pymo snapshot.
- Confirm that pretrained checkpoints may be redistributed, and publish them with explicit license/usage terms.
