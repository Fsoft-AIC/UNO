# [WACV 2026] UNO: Unifying One-stage Video Scene Graph Generation via Object-Centric Visual Representation Learning

[![Conference](https://img.shields.io/badge/WACV-2026-FGD94D.svg)](https://wacv.thecvf.com/)
[![Paper](https://img.shields.io/badge/Paper-arxiv.2506.03589-FF6B6B.svg)](https://arxiv.org/pdf/2509.06165)

</div>

The official implementation of WACV 2026 paper [UNO: Unifying One-stage Video Scene Graph Generation via Object-Centric Visual Representation Learning](https://arxiv.org/pdf/2509.06165)

## 📌 Citation

If you find this paper useful, please consider staring 🌟 this repo and citing 📑 our paper:

```
@inproceedings{le2026uno,
  title={UNO: Unifying One-stage Video Scene Graph Generation via Object-Centric Visual Representation Learning},
  author={Le, Huy and Chung, Nhat and Kieu, Tung and Yang, Jingkang and Le, Ngan},
  booktitle={WACV},
  year={2026},
}
```

## 📕 Overview

Video Scene Graph Generation (VidSGG) aims to represent dynamic visual content by detecting objects and modeling their temporal interactions as structured graphs. Prior studies typically target either coarse-grained box-level or fine-grained panoptic pixel-level VidSGG, often requiring task-specific architectures and multi-stage training pipelines. In this paper, we present UNO (UNified Object-centric VidSGG), a single-stage, unified framework that jointly addresses both tasks within an end-to-end architecture. UNO is designed to minimize task-specific modifications and maximize parameter sharing, enabling generalization across different levels of visual granularity. The core of UNO is an extended slot attention mechanism that decomposes visual features into object and relation slots. To ensure robust temporal modeling, we introduce object temporal consistency learning, which enforces consistent object representations across frames without relying on explicit tracking modules. Additionally, a dynamic triplet prediction module links relation slots to corresponding object pairs, capturing evolving interactions over time. We evaluate UNO on standard box-level and pixel-level VidSGG benchmarks. Results demonstrate that UNO not only achieves competitive performance across both tasks but also offers improved efficiency through a unified, object-centric design.

### Setup code environment

```shell
conda create -n uno python=3.9
conda activate uno
pip install -r requirements.txt
```

## Dataset

### Data preperation

We use two datasets Action Genome and PVSG to train/evaluate our method.

- For Action Genome dataset please process the downloaded dataset with the [Toolkit](https://github.com/JingweiJ/ActionGenome) and put the processed annotation [files](https://drive.google.com/drive/folders/1tdfAyYm8GGXtO2okAoH1WgVHVOTl1QYe?usp=share_link) with COCO style into `annotations` folder.
  The directories of the dataset should look like:

```
|-- action-genome
    |-- annotations   # gt annotations
        |-- ag_train_coco_style.json
        |-- ag_test_coco_style.json
        |-- ...
    |-- frames        # sampled frames
    |-- videos        # original videos
```

- For PVSG dataset please follow this repo to download and pre-process the dataset [PVSG](https://github.com/LilyDaytoy/OpenPVSG).
  The directories of the dataset should look like:

```
|-- pvsg
    |-- pvsg.json   # gt annotations
    |-- ego4d/epic_kitchen/vidor   # video sources
        |-- masks        # sampled masks
        |-- frames        # sampled frames
        |-- videos        # original videos
```

## DSGG

### Training

You can follow the scripts below to train UNO:

in SGDET, SGCLS and PredCLS tasks.

Notably, manually tuning LR drop may be needed to obtain the best performance.

- For SGDET task

```
bash scripts/train_ocdsg_sgdet.sh
```

- For SGCLS task

```
bash scripts/train_ocdsg_sgcls.sh
```

- For PredCLS task

```
bash scripts/train_ocdsg_predcls.sh
```

### Evaluation

Please download the [checkpoints]() used in the paper and put it into `exps/dsgg` folder.
You can use the scripts below to evaluate the performance of OED.

- For SGDET task

```
bash scripts/eval_ocdsg_sgdet.sh
```

- For SGCLS task

```
bash scripts/eval_ocdsg_sgcls.sh
```

- For PredCLS task

```
bash scripts/eval_ocdsg_predcls.sh
```

## Acknowledgement

We thanks all of the authors from the following code for the excellent code they have released. Our framework is built upon these following repos:

- [OED](https://github.com/guanw-pku/OED/tree/main)
- [PVSG](https://github.com/LilyDaytoy/OpenPVSG)
- [SMTC](https://github.com/shvdiwnkozbw/SMTC/tree/main)
- [Learnable_Regions](https://github.com/yuanze-lin/Learnable_Regions)
- [DINOv2](https://github.com/facebookresearch/dinov2/tree/main)
- [xFormers](https://github.com/facebookresearch/xformers)
