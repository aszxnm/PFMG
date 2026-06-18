# Model Zoo

Pretrained weights are not included in git. Download links and SHA256 checksums will be filled after the public checkpoint host is finalized.

Put downloaded weights under:

```text
data/beat_cache/beat_4english_15_141/weights/
```

## PFMG-PAE checkpoints

| Model | Config | Checkpoint | SHA256 | Notes |
| --- | --- | --- | --- | --- |
| Audio2Face | `configs/audio2face_4english_15_141.yaml` | `face.bin` | TBD | Auxiliary checkpoint loaded by PFMG-TCN |
| PFMG-TCN | TBD | `pfmg_tcn.bin` | TBD | Aperiodic generator loaded by PFMG-PAE |
| PFMG-PAE | `configs/pfmg_tcn_pae_4english_15_141.yaml` | `pfmg_tcn_pae.bin` | TBD | Main released motion-generation model |

Download URL placeholders:

```text
face.bin: TBD
pfmg_tcn.bin: TBD
pfmg_tcn_pae.bin: TBD
```

## Checksum

After uploading weights, run:

```shell
sha256sum data/beat_cache/beat_4english_15_141/weights/*.bin
```
