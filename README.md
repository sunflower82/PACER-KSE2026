# PACER

Official implementation and reproducibility materials for:

**PACER: A Perturbation-Aware Contrastive Framework for Long-Tail
Multimodal Recommendation**

Accepted at KSE 2026.

## Repository status

This repository is public and released under the MIT License (`LICENSE`).

## Structure

- `paper/`: camera-ready manuscript source and bibliography (PDF withheld until IEEE Xplore)
- `src/`: model implementation (DVR/NRDMC-lite, MACP, Interest-Tree, trainer)
- `scripts/`: training, preprocessing, evaluation, and HPO drivers
- `configs/`: Clothing and Sports experiment configurations (YAML)
- `data/README.md`: Hugging Face download links, expected layout, and checksums
- `notebooks/`: supporting notebooks (to be added)
- `results/`: aggregated experimental results (to be added)

## Copyright notice (accepted manuscript)

The accepted manuscript of this paper (KSE 2026) is withheld from the
public repository until it appears in IEEE Xplore. Copyright will be
transferred to IEEE upon publication. Personal use of this material is
permitted. Permission from IEEE must be obtained for all other uses,
including reprinting/republishing this material for advertising or
promotional purposes, creating new collective works, for resale or
redistribution to servers or lists, or reuse of any copyrighted component
of this work in other works. The IEEE Xplore DOI will be added after
publication.

## Paper artefacts

- `paper/PACER_KSE2026_camera_ready.tex`
- `paper/references_v6.bib`
- `paper/README.md`

Compile the TeX from `paper/` with the IEEE conference template.
The manuscript PDF is withheld until IEEE Xplore publication.

## Setup

```bash
pip install -r requirements.txt
```

Place Amazon Clothing / Sports 5-core files under `data/<Dataset>/`
following `data/README.md`. The loader expects
`5-core/{train,val,test}.json` plus frozen `image_feat.npy` and
`text_feat.npy`. Those payloads are not shipped here.

## Training

From the repository root:

```bash
PYTHONPATH=src python src/main_tercile.py --data_path ./data \
    --dataset Clothing --embed_size 320 --UI_layers 3 \
    --enable_nrdmc_lite 1 --nrdmc_lite_layers 2 \
    --enable_logq 0 --enable_tamer 1
```

Locked five-seed Clothing protocol:

```bash
python scripts/run_kse_final_5seed.py --grid honest
```

Paper hyperparameters live in `configs/clothing.yaml` and
`configs/sports.yaml`. DVR is `src/damps/nrdmc_lite.py`.

## Citation

See `CITATION.cff`. After IEEE Xplore assigns a DOI, that record will
be updated. Until then, cite the KSE 2026 accepted version.

## Contact

Maintainer: sunflower82
