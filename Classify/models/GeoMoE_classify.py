# Copyright (c) 2022 Alpha-VL
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial
import torch.nn.functional as F
import torch
import torch.nn as nn
from timm.layers import DropPath
from timm.models.layers import trunc_normal_
import numpy as np
torch_version = torch.__version__
is_torch2 = torch_version.startswith('2.')


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate(
            [np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MOEMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.experts_num = 24
        self.share_experts_num = 1
        self.unique_experts_num = 23
        self.act_experts_num = 3
        self.gate = nn.Linear(in_features, self.unique_experts_num)
        self.unique_experts = nn.ModuleList([
            Mlp(in_features=in_features, hidden_features=hidden_features,
                act_layer=act_layer, drop=drop)
            for _ in range(self.unique_experts_num)
        ])
        self.register_buffer(
            'bias', torch.zeros(self.unique_experts_num))
        self.share_experts = Mlp(in_features=in_features, hidden_features=hidden_features,
                                 act_layer=act_layer, drop=drop)

    def forward(self, x):
        o_shape = x.shape
        x = x.view(-1, o_shape[-1])
        gate_weights = torch.sigmoid(self.gate(x))
        top_indices = torch.topk(
            gate_weights + self.bias, self.act_experts_num, dim=-1, sorted=False).indices
        top_weight = torch.gather(gate_weights, dim=-1, index=top_indices)

        top_weight = top_weight / top_weight.sum(dim=-1, keepdim=True)
        top_indices = top_indices.flatten()
        top_weight = top_weight.flatten()

        expert_num = torch.bincount(
            top_indices, minlength=self.unique_experts_num)

        token_indices = torch.arange(
            x.shape[0], device=x.device).repeat_interleave(self.act_experts_num)
        perm = torch.argsort(top_indices)
        perm_token_indices = token_indices[perm]
        output = torch.split(x[perm_token_indices], expert_num.tolist(), dim=0)
        output = torch.cat([self.unique_experts[i](output[i])
                           for i in range(self.unique_experts_num)], dim=0)
        x = self.share_experts(x)
        x.index_add_(0, perm_token_indices, (output *
                     top_weight[perm].unsqueeze(1)).to(x.dtype))

        gate_weights = gate_weights / gate_weights.sum(dim=-1, keepdim=True)
        P = gate_weights.mean(dim=0)
        F = self.unique_experts_num * expert_num.float() / \
            (self.act_experts_num * x.shape[0])
        return x.view(o_shape), (P * F).sum()


class MOEBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=0.75, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MOEMlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x_t, aux_loss = self.mlp(self.norm2(x))
        x = x + self.drop_path(x_t)
        return x, aux_loss


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if is_torch2:
            self.attn_drop = attn_drop
        else:
            self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if is_torch2:
            attn = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop,
            )
            x = attn.transpose(1, 2).reshape(B, N, C)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CMlp(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        # x = self.drop(x)
        return x


class CBlock(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
#        self.attn = nn.Conv2d(dim, dim, 13, padding=6, groups=dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x + self.drop_path(self.conv2(self.attn(mask * self.conv1(
                self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()))))
        else:
            x = x + self.drop_path(self.conv2(self.attn(self.conv1(
                self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()))))
        x = x + \
            self.drop_path(
                self.mlp(self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        return self.act(x)


class GeoMoE(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16, moe_mlp_ratio=0.75,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, drop_rate=0., num_classes=1000):
        super().__init__()

        self.patch_embed1 = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size[0], in_chans=in_chans, embed_dim=embed_dim[0])
        self.patch_embed2 = PatchEmbed(
            img_size=img_size[1], patch_size=patch_size[1], in_chans=embed_dim[0], embed_dim=embed_dim[1])
        self.patch_embed3 = PatchEmbed(
            img_size=img_size[2], patch_size=patch_size[2], in_chans=embed_dim[1], embed_dim=embed_dim[2])

        self.patch_embed4 = nn.Linear(embed_dim[2], embed_dim[2])

        num_patches = self.patch_embed3.num_patches

        self.pos_embed_det = nn.Parameter(torch.zeros(
            1, num_patches, embed_dim[2]), requires_grad=False)

        dpr = [
            x.item() for x in torch.linspace(0, drop_rate, sum(depth))
        ]

        self.blocks1 = nn.ModuleList([
            CBlock(
                dim=embed_dim[0], num_heads=num_heads, mlp_ratio=mlp_ratio[0],  qkv_bias=True, qk_scale=None, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([
            CBlock(
                dim=embed_dim[1], num_heads=num_heads, mlp_ratio=mlp_ratio[1],  qkv_bias=True, qk_scale=None, drop_path=dpr[i + depth[0]], norm_layer=norm_layer)
            for i in range(depth[1])])
        blocks3 = [MOEBlock(embed_dim[2], num_heads, moe_mlp_ratio, qkv_bias=True, drop_path=dpr[depth[0] + depth[1] + i],
                            qk_scale=None, norm_layer=norm_layer) for i in range(depth[2])]
        self.blocks3 = nn.ModuleList(blocks3)

        self.fc_norm = norm_layer(embed_dim[-1])

        self.head = nn.Linear(
            embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.initialize_weights()
        trunc_normal_(self.head.weight, std=2e-5)

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed_det = get_2d_sincos_pos_embed(
            self.pos_embed_det.shape[-1], int(self.patch_embed3.num_patches**.5), cls_token=False)
        self.pos_embed_det.data.copy_(
            torch.from_numpy(pos_embed_det).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed3.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

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

        x = self.patch_embed1(x)
        for blk in self.blocks1:
            x = blk(x)

        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)

        x = self.patch_embed3(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.patch_embed4(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed_det

        # apply Transformer blocks
        aux_losses = []
        for blk in self.blocks3:
            x, aux_loss = blk(x)
            aux_losses.append(aux_loss)

        return x, torch.stack(aux_losses)

    def forward(self, imgs):
        # with torch.amp.autocast('cuda'):
        latent, aux_losses = self.forward_encoder(
            imgs)
        x = latent.mean(dim=1)
        x = self.fc_norm(x)
        return self.head(x), aux_losses


def classify_geomoe_base_patch16_dec512d8b(imgsize=224, **kwargs):

    img_size = [imgsize, imgsize // 4, imgsize // 8]
    model = GeoMoE(
        img_size=img_size, patch_size=[4, 2, 2], embed_dim=[256, 384, 768], depth=[2, 2, 11], num_heads=12, drop_rate=0.1,
        moe_mlp_ratio=0.75, mlp_ratio=[4, 4, 4], norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


GeoMoE_classify = classify_geomoe_base_patch16_dec512d8b
