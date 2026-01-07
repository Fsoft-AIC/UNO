from .dsgg.ocdsg_sgdet import build as build_sgdet
from .dsgg.ocdsg_pointer_sgdet import build as build_pointer_sgdet
from .pvsg.ocdsg_pvsg import build as build_pvsg


def build_model(args):
    if args.dsgg_task == 'sgdet':
        if args.use_pointer_matching:
            return build_pointer_sgdet(args)
        else:
            return build_sgdet(args)
    elif args.dsgg_task == 'pvsg':
        # if args.only_seg:
        #     return build_seg(args)
        # else:
        return build_pvsg(args)
    
    # else:
    #     return build_xcls(args)
    # else:
    #     # if args.one_dec and args.dsgg_task == 'predcls':
    #     #     return buidl_single_predcls_one_dec(args)
    #     # else:
    #     return build_xcls(args)
