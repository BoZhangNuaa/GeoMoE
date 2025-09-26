import torch.nn as nn
from functools import partial
import models.geomoe as geomoe

def geomoe_base_patch16_dec512d8b(**kwargs):
    model = geomoe.GeoMoE(
        img_size=[224, 56, 28], patch_size=[4, 2, 2], embed_dim=[256, 384, 768], depth=[2, 2, 11], num_heads=12,
        decoder_embed_dim=512, decoder_num_heads=16, decoder_depth=8, moe_mlp_ratio=0.75, self_attn=False,
        mlp_ratio=[4, 4, 4], norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

GeoMoE = geomoe_base_patch16_dec512d8b