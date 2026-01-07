
import torch
import torch.utils.data
from pycocotools import mask as coco_mask

class ConvertCocoPolysToMask(object):
    
    def __init__(self, transforms):
        self._transforms = transforms

    def __call__(self, image, target):
        w, h = image.size
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        # print(anno)
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0 or obj['iscrowd'] == -1]
        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
        attn_labels, spatial_labels, contacting_labels = [], [], []
        for oid in range(1, len(anno)):
            obj = anno[oid]
            attn_obj_label = torch.zeros(3)
            spatial_obj_label = torch.zeros(6)
            contacting_obj_label = torch.zeros(17)
            attn_obj_label[obj['attention_rel']] = 1
            spatial_obj_label[torch.tensor(obj['spatial_rel']) - 3] = 1
            contacting_obj_label[torch.tensor(obj['contact_rel']) - 9] = 1

            # attn_labels.append(obj['attention_rel'])
            # spatial_labels.append(obj['spatial_rel'])
            # contacting_labels.append(obj['contact_rel'])

            attn_labels.append(attn_obj_label)
            spatial_labels.append(spatial_obj_label)
            contacting_labels.append(contacting_obj_label)
        attn_labels = torch.stack(attn_labels, dim=0)
        spatial_labels = torch.stack(spatial_labels, dim=0)
        contacting_labels = torch.stack(contacting_labels, dim=0)
        
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])[keep]
        area = torch.tensor([obj["area"] for obj in anno])[keep]
        attn_labels = attn_labels[keep[1:]]
        spatial_labels = spatial_labels[keep[1:]]
        contacting_labels = contacting_labels[keep[1:]]
        
        # attn_labels = [attn_labels[kpidx-1] for kpidx in torch.where(keep)[0][1:]]
        # spatial_labels = [spatial_labels[kpidx-1] for kpidx in torch.where(keep)[0][1:]]
        # contacting_labels = [contacting_labels[kpidx-1] for kpidx in torch.where(keep)[0][1:]]

        num_objs = len(boxes) - 1
        target = {}
        target["orig_size"] = torch.as_tensor([int(h), int(w)]) # h, w!
        target["size"] = torch.as_tensor([int(h), int(w)])
        target['boxes'] = boxes
        target['labels'] = classes
        target["iscrowd"] = iscrowd
        target["area"] = area
        image, target = self._transforms(image, target)

        if num_objs == 0 or (classes == 1).sum() == 0:
            target['sub_labels'] = torch.zeros((0,), dtype=torch.int64)
            target['obj_labels'] = torch.zeros((0,), dtype=torch.int64)
            target['sub_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['obj_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['attn_labels'] = torch.zeros((0, 3), dtype=torch.float32)
            target['spatial_labels'] = torch.zeros((0, 6), dtype=torch.float32)
            target['contacting_labels'] = torch.zeros((0, 17), dtype=torch.float32)
            target['matching_labels'] = torch.zeros((0,), dtype=torch.int64)
        else:
            target['obj_labels'] = target['labels'][1:]
            target['sub_boxes'] = target['boxes'][0].repeat((num_objs, 1))
            target['sub_labels'] = torch.ones(target['sub_boxes'].shape[0], dtype=torch.int64)
            target['obj_boxes'] = target['boxes'][1:]
            target['attn_labels'] = attn_labels
            target['spatial_labels'] = spatial_labels
            target['contacting_labels'] = contacting_labels
            # print(target['boxes'].shape)
            # print(target['labels'].shape)
            # print(target['obj_labels'].shape)
            # print('\n')
            target['matching_labels'] = torch.ones_like(target['obj_labels'])

        return image, target


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks
