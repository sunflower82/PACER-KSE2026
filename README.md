# PACER

Official implementation and reproducibility materials for:

**PACER: A Perturbation-Aware Contrastive Framework for Long-Tail
Multimodal Recommendation**

Accepted at KSE 2026.

## Repository status

This repository is public. Camera-ready materials and a sanitized
training snapshot from
[DAMPS_upgrade_for_MMHCL_randoms_Amazon_Clothing](https://github.com/sunflower82/DAMPS_upgrade_for_MMHCL_randoms_Amazon_Clothing)
(`wave2/branchA-prime-nrdmc-lite`) are included.

Notebooks, Excel files, extra PDFs, JSON splits, and aggregated
result dumps are intentionally omitted until the paper is published.

No license is attached yet. Default copyright remains with the authors.

## Structure

- `paper/`: camera-ready manuscript source, bibliography, and PDF
- `src/`: model implementation (DVR/NRDMC-lite, MACP, Interest-Tree, trainer)
- `scripts/`: training, preprocessing, evaluation, and HPO drivers
- `configs/`: Clothing and Sports experiment configurations (YAML)
- `notebooks/`: supporting notebooks (to be added)
- `results/`: aggregated experimental results (to be added)

## Paper artefacts

- `paper/PACER_KSE2026_camera_ready.tex`
- `paper/PACER_KSE2026_camera_ready.pdf`
- `paper/references_v6.bib`

Compile the TeX from `paper/` with the IEEE conference template.

## Setup

```bash
pip install -r requirements.txt
```

Place Amazon Clothing / Sports 5-core files under `data/<Dataset>/`.
The loader expects `train.json`, `val.json`, `test.json`, and frozen
modality features. Those payloads are not shipped here.

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

Citation information will be added after publication metadata
becomes available.

## Contact

Maintainer: sunflower82
