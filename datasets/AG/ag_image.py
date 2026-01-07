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

from datasets.AG.pipelines.torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.AG.pipelines.transforms_single as T
from datasets.AG.utils import ConvertCocoPolysToMask
from datasets.AG.pipelines.transforms_single import make_coco_transforms

class AGImageDataset(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, cache_mode=False, local_rank=0, local_size=1, is_train=False):
        super(AGImageDataset, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(transforms)
        self.is_train = is_train

    def __getitem__(self, idx):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        # idx若为675834，则img_id为675835(img_id=idx+1)
        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # print(ann_ids)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = self.get_image(path)
        # print(img.shape)
        image_id = img_id
        target = {'image_id': image_id, 'annotations': target}

        img, target = self.prepare(img, target)
        if not self.is_train:
            target['img_path'] = path
            
        return img, target



def build(image_set, args):
    root = Path(args.data_path)
    assert root.exists(), f'provided Action Genome path {root} does not exist'
    PATHS = {
        "train": (root, root / "annotations" / 'ag_train_coco_style.json'),
        "val": (root, root / "annotations" / 'ag_test_coco_style.json'),
    }
    img_folder, anno_file = PATHS[image_set]
    dataset = AGImageDataset(img_folder, anno_file, transforms=make_coco_transforms(image_set, args.image_size), cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size(), is_train=(not args.eval))

    return dataset
