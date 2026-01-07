import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import pdb

class HungarianMatcherPointerDSGG(nn.Module):
    def __init__(self, cost_rel_class: float = 1, cost_pointer: float = 1):
        super().__init__()
        self.cost_rel_class = cost_rel_class
        self.cost_pointer = cost_pointer
        assert cost_rel_class != 0 or cost_pointer != 0, 'all costs cant be 0'

    def reduce_redundant_gt_box(self, tgt_bbox, indices):
        """Filters redundant Ground-Truth Bounding Boxes
        Due to random crop augmentation, there exists cases where there exists
        multiple redundant labels for the exact same bounding box and object class.
        This function deals with the redundant labels for smoother HOTR training.
        """
        tgt_bbox_unique, map_idx, idx_cnt = torch.unique(tgt_bbox, dim=0, return_inverse=True, return_counts=True)

        k_idx, bbox_idx = indices
        triggered = False
        if (len(tgt_bbox) != len(tgt_bbox_unique)):
            map_dict = {k: v for k, v in enumerate(map_idx)}
            map_bbox2kidx = {int(bbox_id): k_id for bbox_id, k_id in zip(bbox_idx, k_idx)}

            bbox_lst, k_lst = [], []
            for bbox_id in bbox_idx:
                if map_dict[int(bbox_id)] not in bbox_lst:
                    bbox_lst.append(map_dict[int(bbox_id)])
                    k_lst.append(map_bbox2kidx[int(bbox_id)])
            bbox_idx = torch.tensor(bbox_lst)
            k_idx = torch.tensor(k_lst)
            tgt_bbox_res = tgt_bbox_unique
        else:
            tgt_bbox_res = tgt_bbox
        bbox_idx = bbox_idx.to(tgt_bbox.device)
        return tgt_bbox_res, k_idx, bbox_idx

    @torch.no_grad()
    def forward(self, outputs, targets, indices):
        bs, num_queries = outputs['pred_attn_logits'].shape[:2]
        return_list = []
        for b in range(bs):
            out_attn_prob = outputs['pred_attn_logits'][b].softmax(-1)
            out_spatial_prob = outputs['pred_spatial_logits'][b].sigmoid()
            out_contacting_prob = outputs['pred_contacting_logits'][b].sigmoid()

            tgt_attn_labels = targets[b]['attn_labels'] 
            tgt_spatial_labels = targets[b]['spatial_labels'] 
            tgt_contacting_labels = targets[b]['contacting_labels']
            tgt_attn_labels_permute = tgt_attn_labels.permute(1, 0)
            tgt_spatial_labels_permute = tgt_spatial_labels.permute(1, 0)
            tgt_contacting_labels_permute = tgt_contacting_labels.permute(1, 0)
            
            tgt_labels = targets[b]['labels'] 
            tgt_sub_labels = targets[b]['sub_labels'] 
            tgt_obj_labels = targets[b]['obj_labels'] 

            tgt_boxes = targets[b]["boxes"] # (num_boxes, 4)
            tgt_sub_boxes = targets[b]["sub_boxes"] # (num_pair_boxes, 4)
            tgt_obj_boxes = targets[b]["obj_boxes"] # (num_pair_boxes, 4)

            # find which gt boxes match the h, o boxes in the pair
            k_idx, bbox_idx = indices[b]
            k_idx, bbox_idx = k_idx.cuda(), bbox_idx.cuda()

            sbox_with_cls = torch.cat([tgt_sub_boxes, tgt_sub_labels.unsqueeze(-1)], dim=1)
            obox_with_cls = torch.cat([tgt_obj_boxes, tgt_obj_labels.unsqueeze(-1)], dim=1)
            bbox_with_cls = torch.cat([tgt_boxes, tgt_labels.unsqueeze(-1)], dim=1)

            cost_sbox = torch.cdist(sbox_with_cls, bbox_with_cls, p=1)
            cost_obox = torch.cdist(obox_with_cls, bbox_with_cls, p=1)

            # find which gt boxes matches which prediction in K
            s_match_indices = torch.nonzero(cost_sbox == 0, as_tuple=False) # (num_hbox, num_boxes)
            o_match_indices = torch.nonzero(cost_obox == 0, as_tuple=False) # (num_obox, num_boxes)
            tgt_sids, tgt_oids = [], []

            # obtain ground truth indices for h
            if len(s_match_indices) != len(o_match_indices):
                pdb.set_trace()

            for s_match_idx, o_match_idx in zip(s_match_indices, o_match_indices):
                sbox_idx, S_bbox_idx = s_match_idx
                obox_idx, O_bbox_idx = o_match_idx
          
                GT_idx_for_S = (bbox_idx == S_bbox_idx).nonzero(as_tuple=False).squeeze(-1)
                query_idx_for_S = k_idx[GT_idx_for_S]
                tgt_sids.append(query_idx_for_S)

                GT_idx_for_O = (bbox_idx == O_bbox_idx).nonzero(as_tuple=False).squeeze(-1)
                query_idx_for_O = k_idx[GT_idx_for_O]
                tgt_oids.append(query_idx_for_O)

            # check if empty
            if len(tgt_sids) == 0: tgt_sids.append(torch.as_tensor([-1])) # we later ignore the label -1
            if len(tgt_oids) == 0: tgt_oids.append(torch.as_tensor([-1])) # we later ignore the label -1
            
            
            tgt_sids = torch.cat(tgt_sids)
            tgt_oids = torch.cat(tgt_oids)

            out_sprob = outputs["pred_sidx"][b].softmax(-1)
            out_oprob = outputs["pred_oidx"][b].softmax(-1)

            cost_sclass = -out_sprob[:, tgt_sids] # [batch_size * num_queries, detr.num_queries+1]
            cost_oclass = -out_oprob[:, tgt_oids] # [batch_size * num_queries, detr.num_queries+1]
            

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
            
            C = self.cost_rel_class * (cost_attn_class + cost_spatial_class + cost_contacting_class) + \
                self.cost_pointer * (cost_sclass +  cost_oclass)


            C = C.view(num_queries, -1).cpu()
            return_list.append(linear_sum_assignment(C))
            
            targets[b]["s_labels"] = tgt_sids.to(tgt_sub_labels.device)
            targets[b]["o_labels"] = tgt_oids.to(tgt_obj_labels.device)

        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in return_list
            ], targets
        
def build_pointer_matcher(args):
    return HungarianMatcherPointerDSGG(cost_rel_class=args.set_cost_rel_class, cost_pointer=args.set_cost_pointer)

