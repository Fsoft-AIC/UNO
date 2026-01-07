import math
import os
import sys
from typing import Iterable
import numpy as np
import copy
import itertools

import torch
import pdb
import pickle

import util.misc as utils
from models.dsgg.evaluate_dsgg import BasicSceneGraphEvaluator
import torchvision.transforms as T

import cv2
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from thop import profile
from tqdm import tqdm
import time
import torch.nn.functional as F



def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler,
                    device: torch.device, epoch: int, max_norm: float = 0, 
                    args=None, writer = None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if hasattr(criterion, 'loss_labels'):
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    elif 'obj_labels' in criterion.losses:
        metric_logger.add_meter('obj_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    maxIter = len(data_loader)
    cur_iter = epoch * maxIter
    total_loss_list = []
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        if type(targets[0]) == dict:
            targets = [{k: v.to(device) for k, v in t.items() if k != 'filename'} for t in targets]
        else:
            targets = [{k: v.to(device) for k, v in t.items() if k != 'filename'} for target in targets for t in target]
        in_targets = targets if (args.dsgg_task != 'sgdet' or args.use_matched_query) else None
        if 'cur_idx' in targets[0].keys():
            cur_idx = targets[0]['cur_idx'].item()
            outputs = model(samples, targets=in_targets, cur_idx=cur_idx)
        else:
            outputs = model(samples, targets=in_targets)
        if len(targets) > 1 and args.dsgg_task == 'sgdet' and args.dataset_file == 'multi':
            # pdb.set_trace()
            targets = [targets[0]]
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        # # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        if hasattr(model, 'module'):
            torch.clamp_(model.module.logit_scale.data, max=np.log(100))
        else:
            torch.clamp_(model.logit_scale.data, max=np.log(100))


        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if hasattr(criterion, 'loss_labels'):
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        elif 'obj_class_error' in loss_dict_reduced.keys():
            metric_logger.update(obj_class_error=loss_dict_reduced['obj_class_error'])
        if 'sub_pointer_error' in loss_dict_reduced.keys():
            metric_logger.update(sub_pointer_error=loss_dict_reduced['sub_pointer_error'])
        if 'obj_pointer_error' in loss_dict_reduced.keys():
            metric_logger.update(obj_pointer_error=loss_dict_reduced['obj_pointer_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer is not None:
            writer.add_scalar('loss/total_loss', loss_value, cur_iter)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar('loss/{}'.format(k), v, cur_iter)
        cur_iter += 1
        total_loss_list.append(loss_value)
    if args.lr_drop != 0:
        scheduler.step()
    else:   
        scheduler.step(np.mean(total_loss_list))


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_dsgg(dataset_file, model, epoch, postprocessors, data_loader, device, args):
    model.eval()
    prefix = ""
    if args.eval:
        prefix = "_testing"
    evaluator1 = BasicSceneGraphEvaluator(args=args, mode=args.dsgg_task, iou_threshold=0.5, model_version=args.pretrained, save_file=f'{args.output_dir}/ep{epoch}{prefix}_{args.dsgg_task}_eval.txt', constraint='with', nms_filter=args.use_nms_filter)
    evaluator2 = BasicSceneGraphEvaluator(args=args, mode=args.dsgg_task, iou_threshold=0.5, model_version=args.pretrained, save_file=f'{args.output_dir}/ep{epoch}{prefix}_{args.dsgg_task}_eval.txt', constraint='no', nms_filter=args.use_nms_filter)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    save_prediction = False
    preds_dict = {}
    to_test = 1000
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        # samples.tensors = samples.tensors.half() # TODO
        samples = samples.to(device)
        if type(targets[0]) == list:
            targets = [{k: v.to(device) if type(v) == torch.tensor else v for k, v in t.items() if k != 'filename'} for target in targets for t in target]
        in_targets = targets if (args.dsgg_task != 'sgdet' or args.use_matched_query) else None
        if 'cur_idx' in targets[0].keys():
            cur_idx = targets[0]['cur_idx'].item()
            outputs = model(samples, targets=in_targets, cur_idx=cur_idx)
        else:
            outputs = model(samples, targets=in_targets)
        if len(targets) > 1 and args.dsgg_task == 'sgdet' and args.dataset_file == 'multi':
            targets = [targets[0]]

        # if args.dsgg_task == 'sgdet':
        #     if args.dataset_file == 'multi':
        #         targets = [{k: v.to(device) if type(v) == torch.tensor else v for k, v in t.items() } for target in targets for t in target]
        #         if 'cur_idx' in targets[0].keys():
        #             cur_idx = targets[0]['cur_idx'].item()
        #     in_targets = targets
        # if 'cur_idx' in targets[0].keys():
        #     cur_idx = targets[0]['cur_idx'].item()
        #     outputs = model(samples, targets=in_targets, cur_idx=cur_idx)
        # else:
        #     outputs = model(samples, targets=in_targets)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # scores = outputs['pred_obj_logits'].max(-1)[0] # * outputs['pred_attn_logits'].max(-1)[0] * outputs['pred_spatial_logits'].max(-1)[0] * outputs['pred_contacting_logits'].max(-1)[0]
        # sorted_idx_topk = scores[0].sort()[1].cpu().numpy()[::-1][:5]
        # for idx in sorted_idx_topk: # plot 3 queries
        #     ins_mask = outputs['ins_attn_weight'][0, idx].cpu().numpy()
        #     rel_mask = outputs['rel_attn_weight'][0, idx].cpu().numpy()
        #     plot_attn_weight(ins_mask, targets[0]['img_path'], 'query_{}_ins'.format(idx))
        #     plot_attn_weight(rel_mask, targets[0]['img_path'], 'query_{}_rel'.format(idx))

        # to_test -= 1
        # if to_test == 0:
        #     break
        # # continue

        if args.dsgg_task == 'sgdet':
            results = postprocessors['dsgg'](outputs, orig_target_sizes)
        else:
            cur_idx = 0
            if args.dataset_file == 'multi':
                cur_idx = targets[0]['cur_idx']
                targets = [targets[cur_idx]]
            if args.seq_sort:
                results = postprocessors['dsgg'](outputs, targets, cur_idx)
            else:
                results = postprocessors['dsgg'](outputs, targets)
        if save_prediction:
            preds_dict[targets[0]['img_path']] = results

        evaluator1.evaluate_scene_graph(targets, results)
        evaluator2.evaluate_scene_graph(targets, results)
        # to_test -= 1
        # if to_test == 0:
        #     break
    metric_logger.synchronize_between_processes()

    # if save_prediction:
    #     with open('sgdet_single_preds_dict.pkl', 'wb') as f:
    #         pickle.dump(preds_dict, f)

    stats = {}
    print('-------------------------with constraint-------------------------------')
    stats['with'] = evaluator1.print_stats()
    print('-------------------------no constraint-------------------------------')
    stats['no'] = evaluator2.print_stats()

    return stats
