from scipy.optimize import linear_sum_assignment

import torch
from torch import nn
import torch.nn.functional as F

from util.dsgg.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
import numpy as np
import copy
import einops

from .matcher_bbox import build_bbox_matcher
from .matcher_pointer_dsgg import build_pointer_matcher

from models.backbone import build_backbone, MLPProjector
from models.slot_attention import SlotAttention
from functools import partial
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union

# === Utility Functions for Monkey-Patching ===
def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper

class OCDSG_pointer_SGDET(nn.Module):

    def __init__(self, device, backbone, num_obj_classes, aux_loss=False, args=None):
        super().__init__()
        # task determines whether object pair detection results are given
        self.dsgg_task = args.dsgg_task

        #Visual Backbone
        self.backbone = backbone
        self.backbone.eval()
        self.patch_size = self.backbone.body.patch_size

        print(f"Use DINO layer: {args.dino_layer}")
        self.backbone.body.forward = unpack_tuple(
            partial(self.backbone.body.get_intermediate_layers, n={args.dino_layer - 1})
        )
        self.encoder_dim = self.backbone.body.embed_dim
        self.out_dim = self.encoder_dim
        self.out_dim = 512
        self.visual_proj = MLPProjector(self.encoder_dim, self.out_dim)
       
        #Object Decoder - Slot Attention
        self.object_decoder = SlotAttention(device=device, num_slots=args.num_object_slots, encoder_dims=self.out_dim, iters=args.slot_iters, hidden_dim=args.slot_hidden_dim, pos_embed=args.use_pos_slot)
        self.relation_decoder = SlotAttention(device=device, num_slots=args.num_relation_slots, encoder_dims=self.out_dim, iters=args.slot_iters, hidden_dim=args.slot_hidden_dim, pos_embed=False)

        #Object-Subject Pointer
        self.sub_pointer_embed = nn.Sequential(
                                nn.Linear(self.out_dim, self.out_dim, bias=True),
                                nn.ReLU(),
                                nn.Linear(self.out_dim, self.out_dim, bias=True),
                                )
        
        self.obj_pointer_embed = nn.Sequential(
                                nn.Linear(self.out_dim, self.out_dim, bias=True),
                                nn.ReLU(),
                                nn.Linear(self.out_dim, self.out_dim, bias=True),
                                )

        self.bbox_embed = MLP(self.out_dim, self.out_dim, 4, 3)
        self.obj_class_embed = nn.Linear(self.out_dim, num_obj_classes + 1)

        self.attn_class_embed = nn.Linear(self.out_dim, args.num_attn_classes + 1) # add 1 for background
        self.spatial_class_embed = nn.Linear(self.out_dim, args.num_spatial_classes)
        self.contacting_class_embed = nn.Linear(self.out_dim, args.num_contacting_classes)
        
        # hyperparameters
        self.aux_loss = aux_loss
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / args.pointer_temperature))


    # def test(self, samples, targets=None):
    #     if not isinstance(samples, NestedTensor):
    #         samples = nested_tensor_from_tensor_list(samples)
    #     images = samples.tensors
    #     b,f,c,h,w = images.shape

    #     out = self.backbone(samples)

    #     features, mask = out.decompose()
    #     features = self.visual_proj(features)
    #     # assert mask is not None

    #     object_slots, obj_attn = self.object_decoder(features, True) 
    #     relation_slots, rel_attn = self.relation_decoder(features, True)

    #     object_slots = object_slots.unsqueeze(0)       
    #     relation_slots = relation_slots.unsqueeze(0)
    #     obj_attn = einops.rearrange(obj_attn, 'b s (h w) -> b s h w', h=h//self.patch_size)
    #     rel_attn = einops.rearrange(rel_attn, 'b s (h w) -> b s h w', h=h//self.patch_size)

    #     obj_attn = obj_attn.unsqueeze(0)
    #     rel_attn = rel_attn.unsqueeze(0)

    #     inst_embeds = F.normalize(object_slots, p=2, dim=-1) # instance representations
    #     sub_pointer_embeds = F.normalize(self.sub_pointer_embed(relation_slots), p=2, dim=-1)
    #     obj_pointer_embeds = F.normalize(self.obj_pointer_embed(relation_slots), p=2, dim=-1)
    #     logit_scale = self.logit_scale.exp()

    #     outputs_sidx = torch.einsum("t b r c, t b s c -> t b r s", sub_pointer_embeds,  inst_embeds) * logit_scale
    #     outputs_oidx = torch.einsum("t b r c, t b s c -> t b r s", obj_pointer_embeds,  inst_embeds) * logit_scale
        
    #     outputs_coord = self.bbox_embed(object_slots).sigmoid()
    #     outputs_class = self.obj_class_embed(object_slots)

    #     outputs_attn_class = self.attn_class_embed(relation_slots)
    #     outputs_spatial_class = self.spatial_class_embed(relation_slots)
    #     outputs_contacting_class = self.contacting_class_embed(relation_slots)

    #     out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],
    #            'pred_sidx': outputs_sidx[-1], 'pred_oidx': outputs_oidx[-1],
    #            'pred_attn_logits': outputs_attn_class[-1], 'pred_spatial_logits': outputs_spatial_class[-1], 
    #            'pred_contacting_logits': outputs_contacting_class[-1],
    #            'ins_attn_weight': obj_attn[-1], 'rel_attn_weight': rel_attn[-1],
    #            'raw_imgs': images,
    #         }
    #     return out


    def forward(self, samples, targets=None):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        out = self.backbone(samples)

        features, mask = out.decompose()
        features = self.visual_proj(features)
        # assert mask is not None

        object_slots = self.object_decoder(features) 
        relation_slots= self.relation_decoder(features)

        object_slots = object_slots.unsqueeze(0)       
        relation_slots = relation_slots.unsqueeze(0)

        inst_embeds = F.normalize(object_slots, p=2, dim=-1) # instance representations
        sub_pointer_embeds = F.normalize(self.sub_pointer_embed(relation_slots), p=2, dim=-1)
        obj_pointer_embeds = F.normalize(self.obj_pointer_embed(relation_slots), p=2, dim=-1)
        logit_scale = self.logit_scale.exp()

        outputs_sidx = torch.einsum("t b r c, t b s c -> t b r s", sub_pointer_embeds,  inst_embeds) * logit_scale
        outputs_oidx = torch.einsum("t b r c, t b s c -> t b r s", obj_pointer_embeds,  inst_embeds) * logit_scale
        
        outputs_coord = self.bbox_embed(object_slots).sigmoid()
        outputs_class = self.obj_class_embed(object_slots)

        outputs_attn_class = self.attn_class_embed(relation_slots)
        outputs_spatial_class = self.spatial_class_embed(relation_slots)
        outputs_contacting_class = self.contacting_class_embed(relation_slots)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],
               'pred_sidx': outputs_sidx[-1], 'pred_oidx': outputs_oidx[-1],
               'pred_attn_logits': outputs_attn_class[-1], 'pred_spatial_logits': outputs_spatial_class[-1], 
               'pred_contacting_logits': outputs_contacting_class[-1]}
        if self.aux_loss:                      
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord,
                                                    outputs_sidx, outputs_oidx,
                                                    outputs_attn_class, outputs_spatial_class,
                                                    outputs_contacting_class)

        return out
        
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, output_sidx, output_oidx, \
            outputs_attn_class, outputs_spatial_class, outputs_contacting_class):
        return [{'pred_logits': a, 'pred_boxes': b, 'pred_sidx': c, 'pred_oidx': d, 'pred_attn_logits': e, \
                    'pred_spatial_logits': f, 'pred_contacting_logits': g}
                for a, b, c, d, e, f, g, h in zip(outputs_class[:-1], outputs_coord[:-1], \
                                                output_sidx[:-1], output_oidx[:-1], \
                                                outputs_attn_class[:-1], \
                                                outputs_spatial_class[:-1], outputs_contacting_class[:-1])]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SetCriterionPointerDSGG(nn.Module):

    def __init__(self, num_obj_classes, bbox_matcher, pointer_matcher, weight_dict, eos_coef, losses, pointer_losses, args):
        super().__init__()

        self.num_obj_classes = num_obj_classes
        self.num_attn_classes = args.num_attn_classes
        self.num_spatial_classes = args.num_spatial_classes
        self.num_contacting_classes = args.num_contacting_classes
        self.bbox_matcher = bbox_matcher
        self.pointer_matcher = pointer_matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.pointer_losses = pointer_losses
        empty_weight = torch.ones(self.num_obj_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.alpha = args.alpha


    def cal_weights(self, label_nums, p=0.5):
        num_fgs = len(label_nums[:-1])
        weight = [0] * (num_fgs + 1)
        num_all = sum(label_nums[:-1])

        for index in range(num_fgs):
            if label_nums[index] == 0: continue
            weight[index] = np.power(num_all/label_nums[index], p)

        weight = np.array(weight)
        weight = weight / np.mean(weight[weight>0])

        weight[-1] = np.power(num_all/label_nums[-1], p) if label_nums[-1] != 0 else 0

        weight = torch.FloatTensor(weight).cuda()
        return weight

    def loss_obj_labels(self, outputs, targets, indices, num_interactions, log=True):
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_obj_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        obj_weights = self.empty_weight
        loss_obj_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, obj_weights)
        # loss_obj_ce = F.cross_entropy(src_logits[idx], target_classes_o, obj_weights)
        losses = {'loss_obj_ce': loss_obj_ce}

        if log:
            losses['obj_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_obj_cardinality(self, outputs, targets, indices, num_interactions):
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['labels']) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'obj_cardinality_error': card_err}
        return losses


    def loss_relation_labels(self, outputs, targets, indices, num_interactions):
        # loss = focal_loss(inputs, targets, gamma=self.args.action_focal_loss_gamma, alpha=self.args.action_focal_loss_alpha, prior_verb_label_mask=prior_verb_label_mask)
        num_attn_rel, num_spatial_rel, num_contacting_rel = 3, 6, 17
        attn_logits = outputs['pred_attn_logits'].reshape(-1, num_attn_rel + 1)
        spatial_logits = outputs['pred_spatial_logits'].reshape(-1, num_spatial_rel)
        contacting_logits = outputs['pred_contacting_logits'].reshape(-1, num_contacting_rel)
        attn_probs = attn_logits.softmax(dim=-1)
        spatial_probs = spatial_logits.sigmoid()
        contacting_probs = contacting_logits.sigmoid()

        idx = self._get_src_permutation_idx(indices)
        idx = (idx[0].to(attn_logits.device), idx[1].to(attn_logits.device))
        target_attn_classes_o = torch.cat([t['attn_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_spatial_classes_o = torch.cat([t['spatial_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_contacting_classes_o = torch.cat([t['contacting_labels'][J] for t, (_, J) in zip(targets, indices)])
  
        ## only select matched query to calculate loss
        sel_idx = idx[0] * outputs['pred_attn_logits'].shape[1] + idx[1]
        attn_logits = attn_logits[sel_idx]
        spatial_probs = spatial_probs[sel_idx]
        contacting_probs = contacting_probs[sel_idx]
        target_attn_classes = target_attn_classes_o
        target_spatial_classes = target_spatial_classes_o
        target_contacting_classes = target_contacting_classes_o
        ## -----------------------------------

        target_attn_labels = torch.where(target_attn_classes)[1]
        loss_attn_ce = F.cross_entropy(attn_logits, target_attn_labels)
        loss_spatial_ce = self._neg_loss(spatial_probs, target_spatial_classes, alpha=self.alpha)
        loss_contacting_ce = self._neg_loss(contacting_probs, target_contacting_classes, alpha=self.alpha)
        
        losses = {'loss_attn_ce': loss_attn_ce, 'loss_spatial_ce': loss_spatial_ce, 'loss_contacting_ce': loss_contacting_ce}

        return losses

    def loss_boxes(self, outputs, targets, indices, num_interactions):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}
        if src_boxes.shape[0] == 0:
            losses['loss_bbox'] = src_boxes.sum()
            losses['loss_giou'] = src_boxes.sum()
        else:
            loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
            losses['loss_bbox'] = loss_bbox.sum() / num_interactions
            loss_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes),
                                                               box_cxcywh_to_xyxy(target_boxes)))
        
            losses['loss_giou'] = loss_giou.sum() / num_interactions
        return losses

    def _neg_loss(self, pred, gt, weights=None, alpha=0.25):
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        loss = 0

        pos_loss = alpha * torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        if weights is not None:
            pos_loss = pos_loss * weights[:-1]

        neg_loss = (1 - alpha) * torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    #######################################################################################################################
    # * Pointer Losses
    #######################################################################################################################
    # >>> pointer Losses 1 : SO Pointer
    def loss_pointer_labels(self, outputs, targets, pointer_indices, num_interactions, log=True):
        assert ('pred_sidx' in outputs and 'pred_oidx' in outputs)
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        src_sidx = outputs['pred_sidx']
        src_oidx = outputs['pred_oidx']

        idx = self._get_src_permutation_idx(pointer_indices)

        target_sidx_classes = torch.full(src_sidx.shape[:2], -1, dtype=torch.int64, device=src_sidx.device)
        target_oidx_classes = torch.full(src_oidx.shape[:2], -1, dtype=torch.int64, device=src_oidx.device)

        # H Pointer loss        
        target_classes_s = torch.cat([t["s_labels"][J] for t, (_, J) in zip(targets, pointer_indices)])
        target_sidx_classes[idx] = target_classes_s

        # O Pointer loss
        target_classes_o = torch.cat([t["o_labels"][J] for t, (_, J) in zip(targets, pointer_indices)])
        target_oidx_classes[idx] = target_classes_o

        # print(src_oidx.shape)
        # print(target_oidx_classes.shape)
        
        loss_s = F.cross_entropy(src_sidx.transpose(1, 2), target_sidx_classes, ignore_index=-1)
        loss_o = F.cross_entropy(src_oidx.transpose(1, 2), target_oidx_classes, ignore_index=-1)
        losses = {'loss_sidx': loss_s, 'loss_oidx': loss_o}

        if log:
            losses['sub_pointer_error'] = 100 - accuracy(src_sidx[idx], target_sidx_classes[idx])[0]
            losses['obj_pointer_error'] = 100 - accuracy(src_oidx[idx], target_oidx_classes[idx])[0]

        return losses
    
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num, **kwargs):
        loss_map = {
            'obj_labels': self.loss_obj_labels,
            'obj_cardinality': self.loss_obj_cardinality,
            'obj_boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num, **kwargs)

    def get_pointer_loss(self, loss, outputs, targets, indices, num, **kwargs):
        loss_map = {
            'pointer_labels': self.loss_pointer_labels,
            'relation_labels': self.loss_relation_labels,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        indices = self.bbox_matcher(outputs_without_aux, targets)
        # print(f'ind {len(indices)}')

        input_targets = [copy.deepcopy(target) for target in targets]
        pointer_indices, pointer_targets = self.pointer_matcher(outputs_without_aux, input_targets, indices)

        num_interactions = sum(len(t['labels']) for t in targets)
        num_interactions = torch.as_tensor([num_interactions], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_interactions)
        num_interactions = torch.clamp(num_interactions / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_interactions))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.bbox_matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'obj_labels':
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_interactions, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # pointer detection losses
        if self.pointer_losses is not None:
            for loss in self.pointer_losses:
                losses.update(self.get_pointer_loss(loss, outputs, pointer_targets, pointer_indices, num_interactions))

            if 'aux_outputs' in outputs:
                for i, aux_outputs in enumerate(outputs['aux_outputs']):
                    input_targets = [copy.deepcopy(target) for target in targets]
                    pointer_indices, targets_for_aux = self.pointer_matcher(aux_outputs, input_targets, indices)
                    for loss in self.pointer_losses:
                        kwargs = {}
                        if loss == 'pair_targets': kwargs = {'log': False} # Logging is enabled only for the last layer
                        l_dict = self.get_pointer_loss(loss, aux_outputs, pointer_targets, pointer_indices, num_interactions, **kwargs)
                        l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)

        return losses


class PostProcessPointerDSGG(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.threshold = 0

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_cls_logits = outputs['pred_logits']
        out_boxes = outputs['pred_boxes']
        out_attn_logits = outputs['pred_attn_logits']
        out_spatial_logits = outputs['pred_spatial_logits']
        out_contacting_logits = outputs['pred_contacting_logits']

        assert len(out_cls_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        attn_probs = out_attn_logits[..., :-1].softmax(-1)
        spatial_probs = out_spatial_logits.sigmoid()
        contacting_probs = out_contacting_logits.sigmoid()

        s_prob = F.softmax(outputs['pred_sidx'], -1)
        s_idx_score, s_indices = s_prob.max(-1)

        o_prob = F.softmax(outputs['pred_oidx'], -1)
        o_idx_score, o_indices = o_prob.max(-1)
        
        assert len(s_indices) == len(o_indices)

        sub_boxes, obj_boxes, obj_logits = [], [], []
        for batch_id, (cls_logits, box, s_idx, o_idx) in enumerate(zip(out_cls_logits, out_boxes, s_indices, o_indices)):
            sub_boxes.append(box[s_idx, :])
            obj_boxes.append(box[o_idx, :])
            obj_logits.append(cls_logits[o_idx, :])
        sub_boxes = torch.stack(sub_boxes, dim=0)
        obj_boxes = torch.stack(obj_boxes, dim=0)
        obj_logits = torch.stack(obj_logits, dim=0)
        final_out_boxes = torch.cat([sub_boxes, obj_boxes], dim=1)

        obj_prob = F.softmax(obj_logits, -1)
        obj_scores, obj_labels = obj_prob[..., :-1].max(-1)

        results = []
        for index in range(len(target_sizes)):
            frame_pred = {}
            frame_pred['pred_scores'] = torch.cat([torch.ones(sub_boxes.shape[1]), obj_scores[index].cpu()]).numpy()
            frame_pred['pred_labels'] = torch.cat([torch.ones(sub_boxes.shape[1]), obj_labels[index].cpu()]).numpy()
            frame_pred['pred_boxes'] = final_out_boxes[index].cpu().numpy()
            frame_pred['pair_idx'] = torch.cat([torch.arange(sub_boxes.shape[1])[:, None], \
                                                torch.arange(sub_boxes.shape[1], 2 * sub_boxes.shape[1])[:, None]], dim=1).cpu().numpy()
            frame_pred['attention_distribution'] = attn_probs[index].cpu().numpy()
            frame_pred['spatial_distribution'] = spatial_probs[index].cpu().numpy()
            frame_pred['contacting_distribution'] = contacting_probs[index].cpu().numpy()

            results.append(frame_pred)

        return results



def build(args):

    num_classes = 36 + 1

    device = torch.device(args.device)

    visual_backbone = build_backbone(args)


    model = OCDSG_pointer_SGDET(
        device=device,
        backbone=visual_backbone,
        num_obj_classes=num_classes,
        aux_loss=args.aux_loss,
        args=args
    )

    bbox_matcher = build_bbox_matcher(args)
    pointer_matcher = build_pointer_matcher(args)

    weight_dict = {}
    weight_dict['loss_attn_ce'] = args.rel_loss_coef
    weight_dict['loss_spatial_ce'] = args.rel_loss_coef
    weight_dict['loss_contacting_ce'] = args.rel_loss_coef
    weight_dict['loss_obj_ce'] = args.obj_loss_coef
    weight_dict['loss_bbox'] = args.bbox_loss_coef
    weight_dict['loss_giou'] = args.giou_loss_coef
    weight_dict['loss_sidx'] = args.pointer_loss_coef
    weight_dict['loss_oidx'] = args.pointer_loss_coef

    losses = ['obj_labels','obj_boxes', 'obj_cardinality']
    pointer_losses = ['pointer_labels', 'relation_labels']

    criterion = SetCriterionPointerDSGG(num_classes, bbox_matcher=bbox_matcher, pointer_matcher=pointer_matcher, \
                                        weight_dict=weight_dict, eos_coef=args.eos_coef, \
                                        losses=losses, pointer_losses=pointer_losses, args=args)

    criterion.to(device)
    postprocessors = {'dsgg': PostProcessPointerDSGG(args)}

    return model, criterion, postprocessors
