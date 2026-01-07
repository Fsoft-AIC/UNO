from typing_extensions import Literal
from unicodedata import category
import torch
import numpy as np
from PIL import Image

import os
from util.pvsg.mask_ops import bitmasks2bboxes


class LoadAnnotationsDirect:
    """New version of VPS dataloader PVSG dataset."""
    def __init__(
        self,
        cates2id,
        num_obj_classes,   
        num_rel_classes,   
        divisor: int = 10000,
        instance_only: bool = False,
        with_ps_id: bool = False,
    ):
        self.cates2id = cates2id
        self.divisor = divisor
        self.is_instance_only = instance_only
        self.with_ps_id = with_ps_id  # do we need this?
        self.num_rel_classes = num_rel_classes

    def __call__(self, data_root, video_results, frame_results):
        targets = {}
        img = Image.open(os.path.join(data_root, frame_results['img_path'])).convert('RGB')
        pan_mask = np.array(Image.open(os.path.join(data_root, frame_results['ann_path'])).convert('P')).astype(np.int64)  # palette format saved one-channel image
        # default:int16, need to change to int64 to avoid data overflow
        objects_info = video_results['objects']

        gt_semantic_seg = -1 * np.ones_like(pan_mask)
        classes = []
        masks = []
        instance_ids = []
        
        for instance_id in np.unique(pan_mask):  # 0,1...n object id
            # filter background (void) class
            if instance_id == 0:  # no segmentation area
                category = 'background'
                gt_semantic_seg[pan_mask == instance_id] = self.cates2id(category)  # 126
            else:  # gt_label & gt_masks do not include "void"
                if (instance_id-1) >= len(objects_info):
                    print(f"video id: {frame_results['video_id']} bad anno")
                else:
                    category = objects_info[instance_id - 1]['category']
                semantic_id = self.cates2id(category)
                gt_semantic_seg[pan_mask == instance_id] = semantic_id
                classes.append(semantic_id)
                instance_ids.append(instance_id)
                mask = (pan_mask == instance_id).astype(np.int64)
                masks.append(mask)

        # check semantic mask
        gt_semantic_seg = gt_semantic_seg.astype(np.int64)
        assert -1 not in np.unique(gt_semantic_seg).astype(np.int64)

        # add panoptic_seg in "vps encoded format" for evaluation use --------------------------------------
        ps_id = gt_semantic_seg * self.divisor + pan_mask
        # --------------------------------------------------------------------------------------------------

        if len(classes) == 0:  # this image is annotated as "all background", no classes, no masks... (very few images)
            # print(f"video{frame_results['video_id']} - frame: {frame_results['frame_id']} is annotated as all background!")
            gt_labels = np.array(classes).astype(np.int64)  # empty array
            gt_instance_ids = np.array(instance_ids).astype(np.int64)
            _height, _width = pan_mask.shape
            ps_height, ps_width = ps_id.shape
            gt_masks = np.empty((0, _height, _width), dtype=np.uint8)

        else:
            gt_labels = np.stack(classes).astype(np.int64)
            gt_instance_ids = np.stack(instance_ids).astype(np.int64)
            _height, _width = pan_mask.shape
            ps_height, ps_width = ps_id.shape

            gt_masks = np.stack(masks).reshape(-1, _height, _width)

            # check the sanity of gt_masks
            verify = np.sum(gt_masks.astype(np.int64), axis=0)
            assert (verify == (pan_mask != 0).astype(
                verify.dtype)).all()  # none-background area exactly same
        
        id2idx = dict()
        for idx, id in enumerate(gt_instance_ids):
            id2idx[id] = idx
        # generate boxes
        boxes = bitmasks2bboxes(gt_masks)
    #   {
    #       "sub_name": "adult",
    #       "sub": 0,
    #       "sub_id": 22,
    #       "obj_name": "child",
    #       "obj": 33,
    #       "obj_id": 17,
    #       "rel": "holding",
    #       "rel_id": 20
    #     },
        gt_rel_labels, gt_sub_labels, gt_sub_ids, gt_obj_labels, gt_obj_ids  = [], [], [], [], []
        gt_sub_masks, gt_sub_boxes, gt_obj_masks, gt_obj_boxes  = [], [], [], []
        for relation in frame_results['relations']:
            if relation['sub_id'] not in id2idx or relation['obj_id'] not in id2idx:
                # print(f"{relation['sub_id']} - {relation['obj_id']} - {relation['rel']}")
                continue
            gt_rel_label = np.zeros((self.num_rel_classes)).astype(np.float32)
            gt_rel_label[relation['rel_id']] = 1
            gt_rel_labels.append(gt_rel_label)

            gt_sub_labels.append(self.cates2id(relation['sub_name']))
            gt_sub_ids.append(relation['sub_id'])
            gt_sub_masks.append(gt_masks[id2idx[relation['sub_id']]])
            gt_sub_boxes.append(boxes[id2idx[relation['sub_id']]])
            
            gt_obj_labels.append(self.cates2id(relation['obj_name']))            
            gt_obj_ids.append(relation['obj_id'])
            gt_obj_masks.append(gt_masks[id2idx[relation['obj_id']]])
            gt_obj_boxes.append(boxes[id2idx[relation['obj_id']]])

        if len(gt_rel_labels) == 0:  # this image is annotated as "all background", no classes, no masks... (very few images)
            # print(f"video{frame_results['video_id']} - frame: {frame_results['frame_id']} is annotated as no relationship!")

            _height, _width = pan_mask.shape

            gt_rel_labels = np.zeros((0, self.num_rel_classes)).astype(np.float32)

            gt_sub_labels = np.zeros((0,)).astype(np.int64)
            gt_sub_ids = np.zeros((0,)).astype(np.int64)
            gt_sub_masks = np.empty((0, _height, _width), dtype=np.uint8)
            gt_sub_boxes = np.zeros((0, 4)).astype(np.float32)
            
            gt_obj_labels = np.zeros((0,)).astype(np.int64)     
            gt_obj_ids = np.zeros((0,)).astype(np.int64)
            gt_obj_masks = np.empty((0, _height, _width), dtype=np.uint8)
            gt_obj_boxes = np.zeros((0, 4)).astype(np.float32)

        else:        
            gt_rel_labels = np.stack(gt_rel_labels).astype(np.float32)

            gt_sub_labels = np.stack(gt_sub_labels).astype(np.int64)
            gt_sub_ids = np.stack(gt_sub_ids).astype(np.int64)
            gt_sub_masks = np.stack(gt_sub_masks).astype(np.int64)
            gt_sub_boxes = np.stack(gt_sub_boxes).astype(np.float32)
            
            gt_obj_labels = np.stack(gt_obj_labels).astype(np.int64)       
            gt_obj_ids = np.stack(gt_obj_ids).astype(np.int64)
            gt_obj_masks = np.stack(gt_obj_masks).astype(np.int64)
            gt_obj_boxes = np.stack(gt_obj_boxes).astype(np.float32)

        targets['orig_size'] = torch.as_tensor([int(ps_height), int(ps_width)]) # h, w!
        targets['orig_img'] = torch.as_tensor(np.array(img))

        targets['semantic_seg'] = torch.as_tensor(gt_semantic_seg, dtype=torch.int64)
        targets['panoptic_seg'] = torch.as_tensor(ps_id, dtype=torch.int64)

        targets['labels'] = torch.as_tensor(gt_labels, dtype=torch.int64)
        targets['masks'] = torch.as_tensor(gt_masks,  dtype=torch.int64)
        targets['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)

        targets['instance_ids'] = torch.as_tensor(gt_instance_ids, dtype=torch.int64)  # ??

        targets['rel_labels'] = torch.as_tensor(gt_rel_labels, dtype=torch.float32)
        # print(targets['rel_labels'].shape)
        targets['sub_labels'] = torch.as_tensor(gt_sub_labels,  dtype=torch.int64)
        targets['sub_ids'] = torch.as_tensor(gt_sub_ids,  dtype=torch.int64)
        targets['sub_masks'] = torch.as_tensor(gt_sub_masks,  dtype=torch.int64)
        targets['sub_boxes'] = torch.as_tensor(gt_sub_boxes, dtype=torch.float32)

        targets['obj_labels'] = torch.as_tensor(gt_obj_labels, dtype=torch.int64)  # ??
        targets['obj_ids'] = torch.as_tensor(gt_obj_ids,  dtype=torch.int64)
        targets['obj_masks'] = torch.as_tensor(gt_obj_masks,  dtype=torch.int64)
        targets['obj_boxes'] = torch.as_tensor(gt_obj_boxes, dtype=torch.float32)
        
        return img, targets


class LoadMultiAnnotationsDirect(LoadAnnotationsDirect):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, frame_results):
        outs = []
        for _frame_results in frame_results:
            _frame_results = super().__call__(_frame_results)
            if _frame_results is None:
                return None
            outs.append(_frame_results)
        return outs
