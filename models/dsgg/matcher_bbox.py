import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import pdb

from util.dsgg.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

class HungarianMatcherbboxDSGG(nn.Module):
    def __init__(self, cost_obj_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_obj_class = cost_obj_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_obj_class != 0 or cost_bbox != 0 or cost_giou != 0, 'all costs cant be 0'

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs['pred_logits'].shape[:2]
        indices = []

        for b in range(bs):
            out_obj_prob = outputs['pred_logits'][b].softmax(-1)
            out_bbox = outputs['pred_boxes'][b]

            tgt_obj_labels = targets[b]['labels'] 
            tgt_boxes = targets[b]['boxes'] 
            
            cost_obj_class = -out_obj_prob[:, tgt_obj_labels]
            cost_bbox = torch.cdist(out_bbox, tgt_boxes, p=1)
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_boxes))

            C = self.cost_obj_class * cost_obj_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou

            C = C.view(num_queries, -1).cpu()
            indices.append(linear_sum_assignment(C))

        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]
        
def build_bbox_matcher(args):
    return HungarianMatcherbboxDSGG(cost_obj_class=args.set_cost_obj_class,
                               cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)

