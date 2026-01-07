# Modified by Lu He
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path
import torch
from datasets.AG.pipelines.torchvision_datasets import VidCocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.AG.pipelines.transforms_single as T
from datasets.AG.utils import ConvertCocoPolysToMask
from datasets.AG.pipelines.transforms_single import make_coco_transforms
from datasets.AG.pipelines.coco_video_parser import CocoVID

class AGSingleVideoDataset(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, cache_mode=False, local_rank=0, local_size=1, is_train=False):
        super(AGSingleVideoDataset, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(transforms)
        self.is_train = is_train
        self.ann_file = ann_file
        self.cocovid = CocoVID(self.ann_file)

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        imgs = []
        targets = []
        coco = self.coco
        video_id = self.video_ids[idx]
        img_ids = self.cocovid.get_img_ids_from_vid(video_id)    
        # print(f'[{video_id}]\n{img_ids}\n')
        
        for img_id in img_ids:
            ann_ids = coco.getAnnIds(imgIds=img_id)
            target = coco.loadAnns(ann_ids)
            img_info = coco.loadImgs(img_id)[0]
            path = img_info['file_name']
            video_id = img_info['video_id']
            img = self.get_image(path)

            target = {'video_id': video_id, 'image_id': img_id, 'annotations': target}

            img, target = self.prepare(img, target)
            if not self.is_train:
                target['img_path'] = path
            imgs.append(img.unsqueeze(0))        
            targets.append(target)
        return torch.cat(imgs, dim=0), targets

def build(image_set, args):
    root = Path('/your_dir')
    assert root.exists(), f'provided Action Genome path {root} does not exist'
    PATHS = {
        "train": (root, root / "annotations" / 'ag_train_coco_style.json'),
        "val": (root, root / "annotations" / 'ag_test_coco_style.json'),
    }
    img_folder, anno_file = PATHS[image_set]
    dataset = AGSingleVideoDataset(img_folder, anno_file, transforms=make_coco_transforms(image_set, 448))

    return dataset
