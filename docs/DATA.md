# Data Preparation

This repository does not include BEAT raw data, videos, audio files, BVH files, facial features, PAE features, transcripts, LMDB caches, or generated intermediate files.

Please obtain BEAT from the official source and follow the BEAT license. Put local files under `data/`, which is ignored by git.

## Expected layout

The released PFMG-PAE config assumes:

```text
data/
  beat_cache/
    beat_4english_15_141/
      train/
        bvh_rot/*.bvh
        wave16k/*.npy
        facial52/*.json
        pae/*.txt
        text/*.TextGrid
        emo/*.csv
        sem/*.txt
        bvh_mean.npy
        bvh_std.npy
        bvh_rot_cache/
      val/
        bvh_rot/
        wave16k/
        facial52/
        pae/
        text/
        emo/
        sem/
        bvh_rot_cache/
      test/
        bvh_rot/
        wave16k/
        facial52/
        pae/
        text/
        emo/
        sem/
        bvh_rot_vis/
        bvh_rot_cache/
      vocab.pkl
      weights/
        face.bin
        pfmg_tcn.bin
        pfmg_tcn_pae.bin
```

The YAML files use paths such as `/data/beat_cache/...`. The code prefixes these paths with `--root_path`, so `--root_path .` resolves them to `./data/beat_cache/...`.

## Build cache

After raw BEAT sequences are converted to the feature folders above, run:

```shell
python scripts/prepare_beat.py \
  --cache-root ./data/beat_cache/beat_4english_15_141
```

Useful options:

- `--overwrite`: rebuild existing LMDB caches.
- `--skip-lmdb`: only compute pose statistics and vocabulary.
- `--skip-vocab`: skip `vocab.pkl` generation.
- `--splits train val test`: choose which split caches to build.

## Build vocabulary only

```shell
python dataloaders/build_vocab.py \
  --data_path ./data/beat_cache/beat_4english_15_141/ \
  --cache_path ./data/beat_cache/beat_4english_15_141/vocab.pkl
```

Optional fastText embeddings can be passed with `--word_vec_path`.

## Notes

- `scripts/prepare_beat.py` starts from prepared feature folders. It is not a full raw-BEAT converter.
- Do not commit local data, feature folders, LMDB caches, weights, or generated outputs.
- For another dataset, implement a dataloader with the same fields as `dataloaders/beat.py`, then set `dataset` in the YAML config.
