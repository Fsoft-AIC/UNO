import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import pdb

from util.dsgg.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

class HungarianMatcherDSGG(nn.Module):
    def __init__(self, cost_obj_class: float = 1, cost_rel_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_obj_class = cost_obj_class
        self.cost_rel_class = cost_rel_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_obj_class != 0 or cost_rel_class != 0 or cost_bbox != 0 or cost_giou != 0, 'all costs cant be 0'

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs['pred_obj_logits'].shape[:2]
        indices = []

        for b in range(bs):
            out_obj_prob = outputs['pred_obj_logits'][b].softmax(-1)
            out_attn_prob = outputs['pred_attn_logits'][b].softmax(-1)
            out_spatial_prob = outputs['pred_spatial_logits'][b].sigmoid()
            out_contacting_prob = outputs['pred_contacting_logits'][b].sigmoid()
            out_sub_bbox = outputs['pred_sub_boxes'][b]
            out_obj_bbox = outputs['pred_obj_boxes'][b]

            tgt_obj_labels = targets[b]['obj_labels'] 
            tgt_sub_boxes = targets[b]['sub_boxes'] 
            tgt_obj_boxes = targets[b]['obj_boxes'] 
            tgt_attn_labels = targets[b]['attn_labels'] 
            tgt_spatial_labels = targets[b]['spatial_labels'] 
            tgt_contacting_labels = targets[b]['contacting_labels']
            tgt_attn_labels_permute = tgt_attn_labels.permute(1, 0)
            tgt_spatial_labels_permute = tgt_spatial_labels.permute(1, 0)
            tgt_contacting_labels_permute = tgt_contacting_labels.permute(1, 0)

            cost_obj_class = -out_obj_prob[:, tgt_obj_labels]
            cost_sub_bbox = torch.cdist(out_sub_bbox, tgt_sub_boxes, p=1)
            cost_obj_bbox = torch.cdist(out_obj_bbox, tgt_obj_boxes, p=1) * (tgt_obj_boxes != 0).any(dim=1).unsqueeze(0)
            if cost_sub_bbox.shape[1] == 0:
                cost_bbox = cost_sub_bbox
            else:
                cost_bbox = torch.stack((cost_sub_bbox, cost_obj_bbox)).max(dim=0)[0]
            cost_sub_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_sub_bbox), box_cxcywh_to_xyxy(tgt_sub_boxes))
            cost_obj_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_obj_bbox), box_cxcywh_to_xyxy(tgt_obj_boxes)) + \
                            cost_sub_giou * (tgt_obj_boxes == 0).all(dim=1).unsqueeze(0)
            if cost_sub_giou.shape[1] == 0:
                cost_giou = cost_sub_giou
            else:
                cost_giou = torch.stack((cost_sub_giou, cost_obj_giou)).max(dim=0)[0]

            out_attn_prob = out_attn_prob[:, :tgt_attn_labels_permute.shape[0]]
            cost_attn_class = -(out_attn_prob.matmul(tgt_attn_labels_permute) / \
                                (tgt_attn_labels_permute.sum(dim=0, keepdim=True) + 1e-4) + \
                                (1 - out_attn_prob).matmul(1 - tgt_attn_labels_permute) / \
                                ((1 - tgt_attn_labels_permute).sum(dim=0, keepdim=True) + 1e-4)) / 2
            
            cost_spatial_class = -(out_spatial_prob.matmul(tgt_spatial_labels_permute) / \
                                (tgt_spatial_labels_permute.sum(dim=0, keepdim=True) + 1e-4) + \
                                (1 - out_spatial_prob).matmul(1 - tgt_spatial_labels_permute) / \
                                ((1 - tgt_spatial_labels_permute).sum(dim=0, keepdim=True) + 1e-4)) / 2
            cost_contacting_class = -(out_contacting_prob.matmul(tgt_contacting_labels_permute) / \
                                (tgt_contacting_labels_permute.sum(dim=0, keepdim=True) + 1e-4) + \
                                (1 - out_contacting_prob).matmul(1 - tgt_contacting_labels_permute) / \
                                ((1 - tgt_contacting_labels_permute).sum(dim=0, keepdim=True) + 1e-4)) / 2
            C = self.cost_obj_class * cost_obj_class + self.cost_rel_class * (cost_attn_class + cost_spatial_class + cost_contacting_class) + \
                self.cost_bbox * cost_bbox + self.cost_giou * cost_giou

            if self.use_matching:
                tgt_matching_labels = torch.cat([v['matching_labels'] for v in targets])
                out_matching_prob = outputs['pred_matching_logits'].flatten(0, 1).softmax(-1)
                cost_matching = -out_matching_prob[:, tgt_matching_labels]
                C += self.cost_matching * cost_matching


            C = C.view(num_queries, -1).cpu()
            indices.append(linear_sum_assignment(C))
            
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]
        
def build_matcher(args):
    return HungarianMatcherDSGG(cost_obj_class=args.set_cost_obj_class, cost_rel_class=args.set_cost_rel_class,
                               cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)

