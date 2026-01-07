from .AG.ag_image import build as build_ag_image
from .AG.ag_video import build as build_ag_video
from .AG.ag_single_video import build as build_ag_video_single
from .PVSG.pvsg_image import build as build_pvsg_single
from .PVSG.pvsg_image_single import build as build_pvsg_single_test

def build_dataset(image_set, args):
    if args.dsgg_task == 'pvsg':
        if args.dataset_file == 'single':
            if image_set == 'train':
                return build_pvsg_single(image_set, args)
            else:
                return  build_pvsg_single_test(image_set, args)
    else:
        if args.dataset_file == 'single':
            return build_ag_image(image_set, args)
        elif args.dataset_file == 'multi-single':
            return build_ag_video_single(image_set, args)
        elif args.dataset_file == 'multi':
            return build_ag_video(image_set, args)
            
    raise ValueError(f'dataset {args.dataset_file} not supported')
