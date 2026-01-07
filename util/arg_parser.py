import argparse

def get_pvsg_args_parser():
    parser = argparse.ArgumentParser('PVSG Hyperparameters', add_help=False)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_drop', default=0, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--max_epochs', default=30, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--image_size', default=448, type=int)
    parser.add_argument('--image_size_test', default=448, type=int)

    #loss Hyperparameters
    parser.add_argument('--use_new_seg', action='store_true', default=False)
    parser.add_argument('--use_matched_query', action='store_true', default=False)
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    
    #Matcher Hyperparameters
    parser.add_argument('--set_cost_mask', default=2.5, type=float,
                        help="L1 mask coefficient in the matching cost")
    parser.add_argument('--set_cost_dice', default=2.5, type=float,
                        help="L1 mask coefficient in the matching cost")
    parser.add_argument('--set_cost_obj_class', default=2, type=float,
                        help="Object class coefficient in the matching cost")
    parser.add_argument('--set_cost_rel_class', default=1, type=float)
    parser.add_argument('--set_cost_pointer', default=20, type=float)

    #Loss coefficients Hyperparameters
    parser.add_argument('--mask_loss_coef', default=2.5, type=float)
    parser.add_argument('--dice_loss_coef', default=2.5, type=float)
    parser.add_argument('--obj_loss_coef', default=2, type=float)
    parser.add_argument('--rel_loss_coef', default=10, type=float)
    parser.add_argument('--pointer_loss_coef', default=1, type=float)    
    
    parser.add_argument('--pointer_temperature', default=0.05, type=float)

    parser.add_argument('--alpha', default=0.5, type=float, help='focal loss alpha')
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    
    #Mask training Hyperparameters
    parser.add_argument('--oversample_ratio', default=3.0, type=float)
    parser.add_argument('--importance_sample_ratio', default=0.75, type=float)
    parser.add_argument('--train_num_points', default=112*112, type=int)
    
    #DINO Hyperparameters
    parser.add_argument('--dino_version', default='v2', type=str)
    parser.add_argument('--dino_type', default='vit_large', type=str)
    
    #Seg Head
    parser.add_argument('--dino_layer_list', default=[22], type=list) # +1 to find real layers since layer start from 0
    parser.add_argument('--dino_layer_dim', default=[1024, 1024, 1024], type=list) # +1 to find real layers since layer start from 0
    parser.add_argument('--seg_hidden_dim', default=256, type=int)
    parser.add_argument('--final_output_dim', default=512, type=int)
    parser.add_argument('--max_per_image', default=100, type=int)
    parser.add_argument('--object_mask_thr', default=0.5, type=float)
    parser.add_argument('--object_cls_thr', default=0.5, type=float)
    parser.add_argument('--iou_thr', default=0.8, type=float)
    parser.add_argument('--filter_low_score', default=True, type=bool)
    parser.add_argument('--INSTANCE_OFFSET', default=1000, type=int) #pan_id = ins_id * INSTANCE_OFFSET + cat_id
    parser.add_argument('--train_seg_only', action='store_true', default=False)

    #Slot Attention Hyperparameters
    parser.add_argument('--use_pos_slot', action='store_true',
                        help="use pointer matching, default not use")
    parser.add_argument('--slot_iters', default=3, type=int)
    parser.add_argument('--num_object_slots', default=80, type=int)
    parser.add_argument('--num_relation_slots', default=64, type=int)
    parser.add_argument('--slot_hidden_dim', default=256, type=int)

    # Data Hyperparameters
    parser.add_argument('--pretrained', type=str, default='',
                        help='Pretrained model path')
    parser.add_argument('--dataset_file', default='single')
    parser.add_argument('--code_path', type=str, default='')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--output_path', type=str, default='')
    parser.add_argument('--output_dir', default='test',
                        help='path where to save, empty for no saving')
    
    # Training Hyperparameters
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval_epoch', default=5, type=int,
                        help='eval epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_mode', default='vpq', type=str)

    # Distributed Training Hyperparameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    # Eval Hyperparameters
    parser.add_argument('--use_nms_filter', action='store_true', help='Use pair nms filter, default not use')
    parser.add_argument('--thres_nms', default=0.7, type=float)
    parser.add_argument('--nms_alpha', default=1.0, type=float)
    parser.add_argument('--nms_beta', default=0.5, type=float)
    parser.add_argument('--json_file', default='results.json', type=str)
    parser.add_argument('--cache_mode', action='store_true', default=False)


    # PVSG Hyperparameters
    parser.add_argument('--num_thing_classes', type=int, default=115, # TODO
                        help="Number of thing classes")
    parser.add_argument('--num_stuff_classes', type=int, default=11, # TODO
                        help="Number of stuff classes") 
    parser.add_argument('--num_obj_classes', type=int, default=126, # TODO
                        help="Number of object classes")
    parser.add_argument('--num_rel_classes', default=57, type=int)
    parser.add_argument('--dsgg_task', default='pvsg', type=str)
    parser.add_argument('--seq_sort', action='store_true', default=False)

    parser.add_argument('--obj_reweight', action='store_true', default=False)
    parser.add_argument('--rel_reweight', action='store_true', default=False)
    parser.add_argument('--use_static_weights', action='store_true', default=False)
    parser.add_argument('--queue_size', default=4704*1.0, type=float,
                        help='Maxsize of queue for obj and verb reweighting, default 1 epoch')
    parser.add_argument('--p_obj', default=0.7, type=float,
                        help='Reweighting parameter for obj')
    parser.add_argument('--p_rel', default=0.7, type=float,
                        help='Reweighting parameter for verb')
    parser.add_argument('--subject_category_id', default=1, type=int)
    parser.add_argument('--verb_loss_type', type=str, default='focal',
                        help='Loss type for the verb classification')

    return parser


def get_dsgg_args_parser():
    parser = argparse.ArgumentParser('OCDSG Hyperparameters', add_help=False)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_drop', default=0, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument('--max_epochs', default=30, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--image_size', default=448, type=int)
    parser.add_argument('--image_size_test', default=448, type=int)

    #loss Hyperparameters
    parser.add_argument('--use_pos_slot', action='store_true',
                        help="use pointer matching, default not use")
    parser.add_argument('--use_pointer_matching', action='store_true',
                        help="use pointer matching, default not use")
    parser.add_argument('--use_matching', action='store_true',
                        help="Use obj/sub matching 2class loss in first decoder, default not use")
    parser.add_argument('--use_matched_query', action='store_true', default=False)
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    
    #Matcher Hyperparameters
    parser.add_argument('--set_cost_bbox', default=2.5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_cost_obj_class', default=1, type=float,
                        help="Object class coefficient in the matching cost")
    parser.add_argument('--set_cost_matching', default=1, type=float,
                        help="Sub and obj box matching coefficient in the matching cost")
    parser.add_argument('--set_cost_rel_class', default=1, type=float)
    parser.add_argument('--set_cost_pointer', default=1, type=float)

    #Loss coefficients Hyperparameters
    parser.add_argument('--bbox_loss_coef', default=2.5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--obj_loss_coef', default=1, type=float)
    parser.add_argument('--rel_loss_coef', default=1, type=float)
    parser.add_argument('--pointer_loss_coef', default=1, type=float)
    parser.add_argument('--alpha', default=0.5, type=float, help='focal loss alpha')
    parser.add_argument('--matching_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    
    parser.add_argument('--pointer_temperature', default=0.05, type=float)
    #DINO Hyperparameters
    parser.add_argument('--dino_version', default='v2', type=str)
    parser.add_argument('--dino_type', default='vit_large', type=str)
    parser.add_argument('--dino_layer', default=22, type=int) # +1 to find real layers since layer start from 0

    #Slot Attention Hyperparameters
    parser.add_argument('--slot_iters', default=3, type=int)
    parser.add_argument('--num_object_slots', default=24, type=int)
    parser.add_argument('--num_relation_slots', default=24, type=int)
    parser.add_argument('--slot_hidden_dim', default=256, type=int)
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    # Data Hyperparameters
    parser.add_argument('--pretrained', type=str, default='',
                        help='Pretrained model path')
    parser.add_argument('--dataset_file', default='single')
    parser.add_argument('--code_path', type=str, default='')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--output_path', type=str, default='')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    
    # Training Hyperparameters
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--hold', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval_epoch', default=5, type=int,
                        help='eval epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_mode', default='acc', type=str)
    parser.add_argument('--type_slot_viz', default='obj', type=str)


    # Distributed Training Hyperparameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    # Eval Hyperparameters
    parser.add_argument('--use_nms_filter', action='store_true', help='Use pair nms filter, default not use')
    parser.add_argument('--thres_nms', default=0.7, type=float)
    parser.add_argument('--nms_alpha', default=1.0, type=float)
    parser.add_argument('--nms_beta', default=0.5, type=float)
    parser.add_argument('--json_file', default='results.json', type=str)

    parser.add_argument('--cache_mode', action='store_true', default=False)

    # DSGG Hyperparameters
    parser.add_argument('--num_obj_classes', type=int, default=36, # TODO
                        help="Number of object classes")
    parser.add_argument('--num_attn_classes', default=3, type=int)
    parser.add_argument('--num_spatial_classes', default=6, type=int)
    parser.add_argument('--num_contacting_classes', default=17, type=int)
    parser.add_argument('--dsgg_task', default='sgdet', type=str)
    parser.add_argument('--seq_sort', action='store_true', default=False)

    parser.add_argument('--obj_reweight', action='store_true', default=False)
    parser.add_argument('--rel_reweight', action='store_true', default=False)
    parser.add_argument('--use_static_weights', action='store_true', default=False)
    parser.add_argument('--queue_size', default=4704*1.0, type=float,
                        help='Maxsize of queue for obj and verb reweighting, default 1 epoch')
    parser.add_argument('--p_obj', default=0.7, type=float,
                        help='Reweighting parameter for obj')
    parser.add_argument('--p_rel', default=0.7, type=float,
                        help='Reweighting parameter for verb')
    parser.add_argument('--subject_category_id', default=1, type=int)
    parser.add_argument('--verb_loss_type', type=str, default='focal',
                        help='Loss type for the verb classification')


    return parser