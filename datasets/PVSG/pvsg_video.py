import os, glob, json
import random
from pathlib import Path
from typing import List
from typing_extensions import Literal
import pdb

import copy
from torch.utils.data import Dataset

import numpy as np
from PIL import Image, ImageDraw

from utils import SeqObj, vpq_eval, PVSGAnnotation

class PVSGVideoDataset(Dataset):
    def __init__(self,
                 data_root='/your_dir',
                 annotation_file='pvsg.json',
                 split='train',
                 transforms=None):
        
        self.divisor = 10000
        self.is_instance_only = False
        self.with_ps_id = False
        self._transforms = transforms
        
        
        assert data_root is not None
        data_root = Path(data_root)
        anno_file = data_root / annotation_file

        with open(anno_file, 'r') as f:
            anno = json.load(f)

        # collect class names
        self.THING_CLASSES = anno['objects']['thing']  # 115
        self.STUFF_CLASSES = anno['objects']['stuff']  # 11
        self.BACKGROUND_CLASSES = ['background']
        self.CLASSES = self.THING_CLASSES + self.STUFF_CLASSES
        self.num_thing_classes = len(self.THING_CLASSES)
        self.num_stuff_classes = len(self.STUFF_CLASSES)
        self.num_classes = len(self.CLASSES)  # 126

        # collect video id within the split
        video_ids, img_names = [], []
        for data_source in ['vidor', 'epic_kitchen', 'ego4d']:
            for video_id in anno['split'][data_source][split]:
                video_ids.append(video_id)
                img_names += glob.glob(
                    os.path.join(data_root, data_source, 'frames', video_id,
                                 '*.png'))

        assert anno_file.exists()
        assert data_root.exists()

        # get annotation file
        anno = PVSGAnnotation(anno_file, video_ids)

        # find all images
        images = []
        vid2seq_id = {}
        seq_count = 0
        for itm in img_names:
            tokens = itm.split(sep='/')
            vid, img_id = tokens[-2], tokens[-1].split(sep='.')[0]
            vid_anno = anno[vid]  # annotation_dict of this video

            # map vid to seq_id (seq_id starts from 0)
            if vid in vid2seq_id:
                seq_id = vid2seq_id[vid]
            else:
                seq_id = seq_count
                vid2seq_id[vid] = seq_count
                seq_count += 1

            images.append(
                SeqObj({
                    'seq_id': seq_id,
                    'img_id': int(img_id),
                    'img_path': itm,
                    'ann_path': itm.replace('frames', 'masks'),
                    'objects': vid_anno['objects'],
                }))

            assert os.path.exists(images[-1]['img_path'])
            assert os.path.exists(images[-1]['ann_path'])

        self.vid2seq_id = vid2seq_id

        self.sequences = images
        
    def cates2id(self, category):
        class2ids = dict(
            zip(self.CLASSES + self.BACKGROUND_CLASSES,
                range(len(self.CLASSES + self.BACKGROUND_CLASSES))))
        return class2ids[category]

    def bitmasks2bboxes(self, bitmasks):
        bitmasks_array = bitmasks
        boxes = np.zeros((bitmasks_array.shape[0], 4), dtype=np.float32)
        x_any = np.any(bitmasks_array, axis=1)
        y_any = np.any(bitmasks_array, axis=2)
        for idx in range(bitmasks_array.shape[0]):
            x = np.where(x_any[idx, :])[0]
            y = np.where(y_any[idx, :])[0]
            if len(x) > 0 and len(y) > 0:
                boxes[idx, :] = np.array((x[0], y[0], x[-1], y[-1]),
                                        dtype=np.float32)
        return boxes

    def _load_annotations(self, results):
        img = Image.open(results['img_path']).convert('RGB')
        pan_mask = np.array(Image.open(results['ann_path']).convert('P')).astype(
            np.int64)  # palette format saved one-channel image
        # default:int16, need to change to int64 to avoid data overflow
        objects_info = results['objects']

        gt_semantic_seg = -1 * np.ones_like(pan_mask)
        classes = []
        masks = []
        instance_ids = []
        for instance_id in np.unique(pan_mask):  # 0,1...n object id
            # filter background (void) class
            if instance_id == 0:  # no segmentation area
                category = 'background'
                gt_semantic_seg[pan_mask == instance_id] = self.cates2id(
                    category)  # 61
            else:  # gt_label & gt_masks do not include "void"
                category = objects_info[instance_id - 1]['category']
                semantic_id = self.cates2id(category)
                gt_semantic_seg[pan_mask == instance_id] = semantic_id
                classes.append(semantic_id)
                instance_ids.append(instance_id)
                masks.append((pan_mask == instance_id).astype(np.int64))

        # check semantic mask
        gt_semantic_seg = gt_semantic_seg.astype(np.int64)
        assert -1 not in np.unique(gt_semantic_seg).astype(np.int64)
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'] = ['gt_semantic_seg']

        # add panoptic_seg in "vps encoded format" for evaluation use --------------------------------------
        ps_id = gt_semantic_seg * self.divisor + pan_mask
        results['gt_panoptic_seg'] = ps_id
        # --------------------------------------------------------------------------------------------------

        if len(classes) == 0:  # this image is annotated as "all background", no classes, no masks... (very few images)
            print('{} is annotated as all background!'.format(
                results['filename']))
            gt_labels = np.array(classes).astype(np.int64)  # empty array
            gt_instance_ids = np.array(instance_ids).astype(np.int64)
            _height, _width = pan_mask.shape
            gt_masks = np.empty((0, _height, _width), dtype=np.uint8)

        else:
            gt_labels = np.stack(classes).astype(np.int64)
            gt_instance_ids = np.stack(instance_ids).astype(np.int64)
            _height, _width = pan_mask.shape
            gt_masks = np.stack(masks).reshape(-1, _height, _width)

            # check the sanity of gt_masks
            verify = np.sum(gt_masks.astype(np.int64), axis=0)
            assert (verify == (pan_mask != 0).astype(
                verify.dtype)).all()  # none-background area exactly same

            # for instance only -- might not use
            if self.is_instance_only:
                gt_masks = np.delete(gt_masks,
                                           gt_labels >= results['thing_upper'],
                                           axis=0)
                gt_instance_ids = np.delete(
                    gt_instance_ids,
                    gt_labels >= results['thing_upper'],
                )
                gt_labels = np.delete(
                    gt_labels,
                    gt_labels >= results['thing_upper'],
                )
                
        results['img'] = img
        results['gt_labels'] = gt_labels
        results['gt_masks'] = gt_masks
        results['gt_instance_ids'] = gt_instance_ids  # ??
        results['mask_fields'] = ['gt_masks']

        # generate boxes
        boxes = self.bitmasks2bboxes(gt_masks)
        results['gt_bboxes'] = boxes
        results['bbox_fields'] = ['gt_bboxes']
        return results

    def _concat_video_list(self, results):
        assert (isinstance(results, list)), 'results must be list'
        outs = results[:1]
        for i, result in enumerate(results[1:], 1):
            if 'img' in result:
                img = result['img']
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                if i == 1:
                    result['img'] = np.expand_dims(img, -1)
                else:
                    outs[1]['img'] = np.concatenate(
                        (outs[1]['img'], np.expand_dims(img, -1)), axis=-1)
            for key in ['img_metas', 'gt_masks']:
                if key in result:
                    if i == 1:
                        result[key] = [result[key]]
                    else:
                        outs[1][key].append(result[key])
            for key in [
                    'proposals',
                    'gt_bboxes',
                    'gt_bboxes_ignore',
                    'gt_labels',
                    'gt_instance_ids',
            ]:
                if key not in result:
                    continue
                value = result[key]
                if value.ndim == 1:
                    value = value[:, None]
                N = value.shape[0]
                value = np.concatenate((np.full(
                    (N, 1), i - 1, dtype=np.float32), value),
                                       axis=1)
                if i == 1:
                    result[key] = value
                else:
                    outs[1][key] = np.concatenate((outs[1][key], value),
                                                  axis=0)
            if 'gt_semantic_seg' in result:
                if i == 1:
                    result['gt_semantic_seg'] = result['gt_semantic_seg'][...,
                                                                          None,
                                                                          None]
                else:
                    outs[1]['gt_semantic_seg'] = np.concatenate(
                        (outs[1]['gt_semantic_seg'],
                         result['gt_semantic_seg'][..., None, None]),
                        axis=-1)

            if 'gt_depth' in result:
                if i == 1:
                    result['gt_depth'] = result['gt_depth'][..., None, None]
                else:
                    outs[1]['gt_depth'] = np.concatenate(
                        (outs[1]['gt_depth'], result['gt_depth'][..., None,
                                                                 None]),
                        axis=-1)
            if i == 1:
                outs.append(result)
        return outs

    def __getitem__(self, idx):
        outputs = []
        print(len(self.sequences[idx]))
        for output in self.sequences[idx]:
            print(output)
            output = self._load_annotations(output)
            # print(len(output["img"]))
            outputs.append(output)
        res = self._concat_video_list(outputs)
            # outputs = self._transforms(outputs)
            # print(len(outputs))
        return outputs
    
    def __len__(self):
        return len(self.sequences)


def build(image_set, args):
    root = Path('/your_dir')
    assert root.exists(), f'provided PVSG path {root} does not exist'
   
    img_folder = root
    anno_file = 'pvsg.json'
    
    dataset = PVSGVideoDataset(img_folder, anno_file, image_set, transforms=make_coco_transforms(image_set, args.image_size))

    return dataset
