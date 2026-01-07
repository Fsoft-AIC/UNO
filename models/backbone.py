from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

from models.dinov2 import vision_transformers as vitsv2
from models.dino import vision_transformer as vitsv1

class FrozenBatchNorm2d(torch.nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            parameter.requires_grad_(False)
        self.body = backbone

    def forward(self, tensor_list: NestedTensor):
        x = self.body(tensor_list.tensors.squeeze())
        out: Dict[str, NestedTensor] = {}
        m = tensor_list.mask
        # mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        out = NestedTensor(x, m)
        return out


class MLPProjector(nn.Module):
    def __init__(self, intput_dim: int, out_dim: int) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(intput_dim, out_dim, bias=True),
            nn.GELU(),
            nn.Linear(out_dim, out_dim, bias=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projector(features)


class Backbone(BackboneBase):
    def __init__(self, args: str):
        backbone = build_dino(args)
        super().__init__(backbone)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    model = Backbone(args)
    return model


def build_dino(args):

    WEIGHT_PATH = {
        "v1": {
            "vit_tiny": None,
            "vit_small": None,
            "vit_base":"/your_dir/dino/weights/dino_vitbase8_pretrain.pth"
        },

        "v2":{
            "vit_small":"/your_dir/dinov2/weights/dinov2_vits14_pretrain.pth" ,
            "vit_base":"/your_dir/dinov2/weights/dinov2_vitb14_pretrain.pth" ,
            "vit_large":"/your_dir/dinov2/weights/dinov2_vitl14_pretrain.pth",
            "vit_giant":"/your_dir/dinov2/weights/dinov2_vitg14_pretrain.pth",
        }

    }

    is_strict = True
    if args.dino_version == "v1":
        model = vitsv1.__dict__[args.dino_type](patch_size=8,num_classes=0)
        model.load_state_dict(torch.load(WEIGHT_PATH[args.dino_version][args.dino_type]), strict=True)

    elif args.dino_version == "v2":
        model = vitsv2.__dict__[args.dino_type](img_size=518,
        patch_size=14,
        init_values=1.0,
        block_chunks=0,
        num_register_tokens=0)
        
        if args.dino_type == 'vit_giant':
            is_strict = False
            
        model.load_state_dict(torch.load(WEIGHT_PATH[args.dino_version][args.dino_type]), strict=is_strict)

    return model


