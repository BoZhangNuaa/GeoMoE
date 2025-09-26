# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from timm.models.layers import trunc_normal_
import torch
import torch.nn as nn
from models.GeoMoE_classify import MOEBlock, Block, get_2d_sincos_pos_embed
from timm.layers.helpers import to_2tuple


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x, random_sample=False):
        B, C, H, W = x.shape
        assert random_sample or (H == self.img_size[0] and W == self.img_size[1]), \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        # print('x1',x.shape)
        # x=x.flatten(2)
        # print('x2',x.shape )
        # x.transpose(1, 2)
        # print('x3', x.shape)
        # exit()
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class MoE(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=1024, depth=24, num_heads=16, self_attn=False, drop_rate=0., moe_mlp_ratio=0.75,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_det = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                          requires_grad=False)  # fixed sin-cos embedding

        dpr = [
            x.item() for x in torch.linspace(0, drop_rate, depth)
        ]  # stochastic depth decay rule
        blocks = [MOEBlock(embed_dim, num_heads, moe_mlp_ratio, qkv_bias=True, drop_path=dpr[i+1],
                           qk_scale=None, norm_layer=norm_layer) for i in range(depth-1)]
        blocks = [Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, drop_path=dpr[0],
                        qk_scale=None, norm_layer=norm_layer)] + blocks
        self.blocks = nn.ModuleList(blocks)
        self.fc_norm = norm_layer(embed_dim)
        self.head = nn.Linear(
            embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.initialize_weights()
        trunc_normal_(self.head.weight, std=2e-5)

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed_det = get_2d_sincos_pos_embed(self.pos_embed_det.shape[-1], int(self.patch_embed.num_patches ** .5),
                                                cls_token=True)
        self.pos_embed_det.data.copy_(
            torch.from_numpy(pos_embed_det).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed_det[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed_det[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        aux_losses = []
        x = self.blocks[0](x)
        for blk in self.blocks[1:]:
            x, aux_loss = blk(x)
            aux_losses.append(aux_loss)

        return x, torch.stack(aux_losses)

    def forward(self, imgs):
        # with torch.amp.autocast('cuda'):
        latent, aux_losses = self.forward_encoder(imgs)
        x = latent[:, 1:, :].mean(dim=1)  # global pool without cls token
        x = self.fc_norm(x)
        # return loss, pred, mask
        return self.head(x), aux_losses



def classify_moemae_vit_base_patch16_dec512d8b(imgsize=224, **kwargs):
    model = MoE(
        img_size=imgsize, patch_size=16, embed_dim=768, depth=12, num_heads=12, drop_rate=0.1,
        moe_mlp_ratio=0.75, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


MoE_classify = classify_moemae_vit_base_patch16_dec512d8b