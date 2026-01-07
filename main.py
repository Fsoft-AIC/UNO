import argparse
import time
import datetime
import random
from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
from datasets import build_dataset
from engine import train_one_epoch, evaluate_dsgg, viz_slot_dsgg
from models import build_model
import os
import pdb
from tqdm import tqdm

import util.misc as utils

from torch.utils.tensorboard import SummaryWriter


from util.arg_parser import get_dsgg_args_parser


def main(args):    
    if not args.dataset_file == 'single' or args.dataset_file == 'multi':
        AssertionError('No support for this dataset')

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)

    assert args.dsgg_task in ['sgdet', 'sgcls', 'predcls']

    device = torch.device(args.device)

    # TODO random
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_n_parameters = sum(p.numel() for p in model.parameters())
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if p.requires_grad]},
        
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    if args.lr_drop != 0:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    else:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=1, factor=0.1, threshold=1e-1, threshold_mode="abs", min_lr=1e-4)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers, pin_memory=False)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers, pin_memory=False)
    current_epoch = 0
    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            args.resume = os.path.join(args.output_path, args.resume)
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
         
            args.start_epoch = checkpoint['epoch'] + 1
            # if args.start_epoch >= args.lr_drop:
            # optimizer.param_groups[0]['lr'] *= 0.1 # TODO: mannual learning rate drop
            # optimizer.param_groups[1]['lr'] *= 0.1
        if (checkpoint['epoch'] % args.eval_epoch == 0 and checkpoint['epoch'] != 0) or checkpoint['epoch'] == 22:
            test_stats = evaluate_dsgg(args.dataset_file, model, current_epoch, postprocessors, data_loader_val, device, args)

    elif args.pretrained:
        args.pretrained = os.path.join(args.output_path, args.pretrained)
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        if args.eval:
            current_epoch = checkpoint['epoch']
            model_without_ddp.load_state_dict(checkpoint['model'], strict=False)

    if args.eval:
        if args.eval_mode == 'acc':
            test_stats = evaluate_dsgg(args.dataset_file, model, current_epoch, postprocessors, data_loader_val, device, args)
        else:
            viz_slot_dsgg(args.dataset_file, model, current_epoch, postprocessors, data_loader_val, device, args)
        return

    print("Start training")
    start_time = time.time()
    best_performance = 0
    writer = None
    if 'pdb' not in args.output_dir:
        writer = SummaryWriter(args.output_dir + 'curve')
    for epoch in range(args.start_epoch, args.epochs):
        current_epoch = epoch
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, lr_scheduler, device, epoch,
            args.clip_max_norm, args, writer)
        
        if epoch == args.max_epochs - 1:
            checkpoint_path = os.path.join(output_dir, 'checkpoint_last.pth')
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)

        # new added save script
        if epoch <= args.max_epochs:
            checkpoint_path = os.path.join(output_dir, 'checkpoint_{}.pth'.format(epoch))
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)

        # evaluation one epoch every some epochs and save the best model accoding to the evaluation metic
        if (epoch < 30) and ((epoch % args.eval_epoch == 0 and epoch != 0) or (epoch == 22)):
            test_stats = evaluate_dsgg(args.dataset_file, model, current_epoch, postprocessors, data_loader_val, device, args)
            # test_with_stats = test_stats['with']
            # test_semi_stats = test_stats['semi']
            # test_no_stats = test_stats['no']
        
            # if performance > best_performance:
            #     checkpoint_path = os.path.join(output_dir, 'checkpoint_best.pth')
            #     utils.save_on_master({
            #         'model': model_without_ddp.state_dict(),
            #         'optimizer': optimizer.state_dict(),
            #         'lr_scheduler': lr_scheduler.state_dict(),
            #         'epoch': epoch,
            #         'args': args,
            #     }, checkpoint_path)
            
            #     best_performance = performance

        log_stats = {'epoch': epoch,
                    'n_parameters': n_parameters,
                    'total_n_parameters': total_n_parameters,
                    **{f'train_{k}': v for k, v in train_stats.items()},
                     # **{f'test_with_{k}': v for k, v in test_with_stats.items()},
                     # **{f'test_semi_{k}': v for k, v in test_semi_stats.items()},
                     # **{f'test_no_{k}': v for k, v in test_no_stats.items()},
                    }

        if args.output_dir and utils.is_main_process() and epoch <= args.max_epochs:
            with (output_dir / f"{args.dsgg_task}_training_log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    writer.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('OCDSG training and evaluation script', parents=[get_dsgg_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        args.output_dir = os.path.join(args.output_path, args.output_dir)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
