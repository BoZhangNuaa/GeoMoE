# Copyright (c) 2022 Alpha-VL
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from utils.pos_embed import get_2d_sincos_pos_embed
from functools import partial
import torch
import torch.nn as nn
from timm.layers import DropPath
torch_version = torch.__version__
is_torch2 = torch_version.startswith('2.')
import torch.nn.functional as F

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

    def forward(self, x, bias_change=0.001):
        o_shape = x.shape
        x = x.view(-1, o_shape[-1])  # flatten the input
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
        return x.view(o_shape), (P * F).sum(), expert_num


"""
    def forward(self, x, bias_change):
        o_shape = x.shape
        x = x.view(-1, o_shape[-1])  # flatten the input
        gate_weights = torch.sigmoid(self.gate(x))
        top_indices = torch.topk(
            gate_weights + self.bias, self.act_experts_num, dim=-1, sorted=False).indices
        top_weight = torch.gather(gate_weights, dim=-1, index=top_indices)
        expert_num = torch.bincount(
            top_indices.flatten(), minlength=self.unique_experts_num)
        expert_mean = expert_num.float().mean()
        self.bias[expert_num < expert_mean] += bias_change
        self.bias[expert_num > expert_mean] -= bias_change
        output = torch.zeros_like(x, dtype=x.dtype, device=x.device)
        top_weight = top_weight / top_weight.sum(dim=-1, keepdim=True)
        for i in range(self.unique_experts_num):
            mask = (top_indices == i).any(dim=-1)
            if mask.any():
                output[mask] = output[mask] + self.unique_experts[i](
                    x[mask]) * top_weight[top_indices == i].unsqueeze(-1)
        x = self.share_experts(x) + output
        gate_weights = gate_weights / gate_weights.sum(dim=-1, keepdim=True)
        P = gate_weights.mean(dim=0)
        F = self.unique_experts_num * expert_num.float() / \
            (self.act_experts_num * x.shape[0])
        return x.view(o_shape), (P * F).sum()
"""


class MOEBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=0.75, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
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
        self.mlp = MOEMlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, bias_change=0.001):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x_t, aux_loss, expert_load = self.mlp(self.norm2(x), bias_change)
        x = x + self.drop_path(x_t)
        return x, aux_loss, expert_load
    
    
class CrossAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = decoder_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(decoder_dim, decoder_dim, bias=qkv_bias)
        self.kv = nn.Linear(encoder_dim, decoder_dim * 2, bias=qkv_bias)
        if is_torch2:
            self.attn_drop = attn_drop
        else:
            self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(decoder_dim, decoder_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        """
        query from decoder (x), key and value from encoder (y)
        """
        B, N, C = x.shape
        Ny = y.shape[1]
        q = self.q(x).reshape(B, N, self.num_heads, C //
                              self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(y).reshape(B, Ny, 2, self.num_heads, C //
                                self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

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


class CrossAttentionBlock(nn.Module):

    def __init__(self, encoder_dim, decoder_dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, self_attn=False):
        super().__init__()
        self.self_attn = self_attn
        if self.self_attn:
            self.norm0 = norm_layer(decoder_dim)
            self.self_attn = Attention(
                decoder_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm1 = norm_layer(decoder_dim)
        self.cross_attn = CrossAttention(
            encoder_dim, decoder_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(decoder_dim)
        mlp_hidden_dim = int(decoder_dim * mlp_ratio)
        self.mlp = Mlp(in_features=decoder_dim,
                       hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, y):
        """
        x: decoder feature; y: encoder feature (after layernorm)
        """
        if self.self_attn:
            x = x + self.drop_path(self.self_attn(self.norm0(x)))
        x = x + self.drop_path(self.cross_attn(self.norm1(x), y))
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
        #x = self.drop(x)
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
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, self_attn=False,):
        super().__init__()
        # --------------------------------------------------------------------------
        # ConvMAE encoder specifics
        self.patch_embed1 = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size[0], in_chans=in_chans, embed_dim=embed_dim[0])
        self.patch_embed2 = PatchEmbed(
            img_size=img_size[1], patch_size=patch_size[1], in_chans=embed_dim[0], embed_dim=embed_dim[1])
        self.patch_embed3 = PatchEmbed(
            img_size=img_size[2], patch_size=patch_size[2], in_chans=embed_dim[1], embed_dim=embed_dim[2])

        self.patch_embed4 = nn.Linear(embed_dim[2], embed_dim[2])
        self.stage1_output_decode = nn.Conv2d(
            embed_dim[0], embed_dim[2], 4, stride=4)
        self.stage2_output_decode = nn.Conv2d(
            embed_dim[1], embed_dim[2], 2, stride=2)

        num_patches = self.patch_embed3.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(
            1, num_patches, embed_dim[2]), requires_grad=False)
        self.blocks1 = nn.ModuleList([
            CBlock(
                dim=embed_dim[0], num_heads=num_heads, mlp_ratio=mlp_ratio[0],  qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([
            CBlock(
                dim=embed_dim[1], num_heads=num_heads, mlp_ratio=mlp_ratio[1],  qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth[1])])
        blocks3 = [MOEBlock(embed_dim[2], num_heads, moe_mlp_ratio, qkv_bias=True,
                            qk_scale=None, norm_layer=norm_layer) for i in range(depth[2])]
        self.blocks3 = nn.ModuleList(blocks3)
        self.norm = norm_layer(embed_dim[-1])

        # --------------------------------------------------------------------------
        # ConvMAE decoder specifics
        #self.decoder_embed = nn.Linear(
        #    embed_dim[-1], decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(
            1, num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            CrossAttentionBlock(embed_dim[2], decoder_embed_dim, decoder_num_heads, mlp_ratio[0], qkv_bias=True,
                                qk_scale=None, norm_layer=norm_layer, self_attn=self_attn)
            for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            # decoder to patch
            decoder_embed_dim, (patch_size[0] * patch_size[1] * patch_size[2])**2 * in_chans, bias=True)
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.patch_embed3.num_patches**.5), cls_token=False)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(self.patch_embed3.num_patches**.5), cls_token=False)
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed3.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
#        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

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

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = 16
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed1.patch_size[0] * self.patch_embed2.patch_size[0] * self.patch_embed3.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio, build_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N = x.shape[0]
        L = self.patch_embed3.num_patches
#        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        len_build = int(L * build_ratio)
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
#        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask_build = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask_build[:, :-len_build] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        mask_build = torch.gather(mask_build, dim=1, index=ids_restore)

        return ids_keep, mask, mask_build, ids_restore

    def forward_encoder(self, x, mask_ratio, build_ratio):
        # embed patches
        ids_keep, mask, mask_build, ids_restore = self.random_masking(x, mask_ratio, build_ratio)

        mask_for_patch1 = mask.reshape(-1, 14, 14).unsqueeze(-1).repeat(1, 1, 1, 16).reshape(
            -1, 14, 14, 4, 4).permute(0, 1, 3, 2, 4).reshape(x.shape[0], 56, 56).unsqueeze(1)
        mask_for_patch2 = mask.reshape(-1, 14, 14).unsqueeze(-1).repeat(1, 1, 1, 4).reshape(
            -1, 14, 14, 2, 2).permute(0, 1, 3, 2, 4).reshape(x.shape[0], 28, 28).unsqueeze(1)
        x = self.patch_embed1(x)
        for blk in self.blocks1:
            x = blk(x, 1 - mask_for_patch1)
        stage1_embed = self.stage1_output_decode(x).flatten(2).permute(0, 2, 1)

        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x, 1 - mask_for_patch2)
        stage2_embed = self.stage2_output_decode(x).flatten(2).permute(0, 2, 1)
        x = self.patch_embed3(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.patch_embed4(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed
        x = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))
        stage1_embed = torch.gather(
            stage1_embed, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, stage1_embed.shape[-1]))
        stage2_embed = torch.gather(
            stage2_embed, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, stage2_embed.shape[-1]))

        # apply Transformer blocks
        aux_losses = []
        expert_loads = []
        for blk in self.blocks3:
            x, aux_loss, expert_load = blk(x)
            aux_losses.append(aux_loss)
            expert_loads.append(expert_load)

        x = x + stage1_embed + stage2_embed
        x = self.norm(x)

        return x, mask, mask_build, ids_restore, torch.stack(aux_losses), torch.stack(expert_loads)

    def forward_decoder(self, y, ids_restore, mask, mask_build):
        # embed tokens latent, ids_restore, mask
        #y = self.decoder_embed(y)

        masked_indices = mask_build.nonzero(
            as_tuple=True)[1].reshape(y.shape[0], -1)
        masked_pos_embed = torch.gather(self.decoder_pos_embed.expand(y.shape[0], -1, -1),
                                        dim=1, index=masked_indices.unsqueeze(-1).expand(-1, -1, self.decoder_pos_embed.shape[-1]))
        num_masked = masked_indices.shape[1]
        queries = self.mask_token.repeat(y.shape[0], num_masked, 1)
        x = queries + masked_pos_embed
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, y)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        # x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        build_indices = mask.nonzero(as_tuple=True)[
            1].reshape(pred.shape[0], -1)
        build_target = torch.gather(target.expand(pred.shape[0], -1, -1),
                                         dim=1, index=build_indices.unsqueeze(-1).expand(-1, -1, pred.shape[-1]))

        loss = ((pred - build_target) ** 2).mean()

        return loss

    def forward(self, imgs, mask_ratio=0.75, kept_mask_ratio=0.25):
        with torch.amp.autocast('cuda'):
            latent, mask, mask_build, ids_restore, aux_losses, expert_loads = self.forward_encoder(
                imgs, mask_ratio, kept_mask_ratio)
            pred = self.forward_decoder(
                latent, ids_restore, mask, mask_build)  # [N, L, p*p*3]
            loss = self.forward_loss(imgs, pred, mask_build)
        return loss, aux_losses, expert_loads