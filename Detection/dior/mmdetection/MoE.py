from mmdet.registry import MODELS
import torch.nn.functional as F
from timm.layers import DropPath
from functools import partial
from timm.models.layers import trunc_normal_
import torch
import torch.nn as nn
import numpy as np
from timm.layers.helpers import to_2tuple
torch_version = torch.__version__
is_torch2 = torch_version.startswith('2.')


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
        return x.view(o_shape), (P * F).sum()


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
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

@MODELS.register_module()
class MoEDet(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, pretrained=None,
                 embed_dim=1024, depth=24, num_heads=16, drop_rate=0., moe_mlp_ratio=0.75,
                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), fpn_layers=None):
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
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            Norm2d(embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        )

        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        )

        self.fpn3 = nn.Identity()

        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fpn_layers = fpn_layers
        self.initialize_weights()
        self.load_pretrained_weights(pretrained)

    def load_pretrained_weights(self, weights):
        checkpoint = torch.load(
            weights, map_location='cpu', weights_only=False)
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']
        msg = self.load_state_dict(checkpoint, strict=False)

        ckpt_pos_embed = checkpoint['pos_embed']
        ckpt_W = int((ckpt_pos_embed.shape[1])**0.5)
        det_W = int((self.pos_embed_det.shape[1])**0.5)
        cls_pos_embed = None
        if ckpt_pos_embed.shape[1] == ckpt_W**2 + 1:
            cls_pos_embed = ckpt_pos_embed[:, :1, :]
            ckpt_pos_embed = ckpt_pos_embed[:, 1:, :]
        if ckpt_pos_embed.shape != self.pos_embed_det.shape and not det_W**2 + 1 == ckpt_W**2 + 1:
            print(
                f"ckpt: {ckpt_pos_embed.shape} det: {self.pos_embed_det.shape}\nPosition embedding shape mismatch, interpolate it.")
            ckpt_pos_embed = ckpt_pos_embed.reshape(
                -1, ckpt_W, ckpt_W, ckpt_pos_embed.shape[-1]).permute(0, 3, 1, 2)
            ckpt_pos_embed = torch.nn.functional.interpolate(
                ckpt_pos_embed, size=(det_W, det_W), mode='bicubic', align_corners=False)
            ckpt_pos_embed = ckpt_pos_embed.permute(
                0, 2, 3, 1).flatten(1, 2)
        else:
            print("Position embedding successfully loaded.")
        if self.pos_embed_det.shape[1] == det_W**2 + 1:
            if cls_pos_embed is not None:
                ckpt_pos_embed = torch.cat(
                    (cls_pos_embed, ckpt_pos_embed), dim=1)
                self.pos_embed_det.data.copy_(ckpt_pos_embed)
            else:
                self.pos_embed_det[:, 1:, :].data.copy_(ckpt_pos_embed)
        else:
            self.pos_embed_det.data.copy_(ckpt_pos_embed)

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
        fpn = []
        W, H = self.patch_embed.grid_size
        if 0 in self.fpn_layers:
            fpn.append(self.fpn1(x[:, 1:, :].permute(0, 2, 1).reshape(x.shape[0], -1, W, H)))
        for i, blk in enumerate(self.blocks[1:]):
            x, aux_loss = blk(x)
            aux_losses.append(aux_loss)
            if i+1 in self.fpn_layers:
                if len(fpn) == 0:
                    fpn.append(self.fpn1(x[:, 1:, :].permute(0, 2, 1).reshape(x.shape[0], -1, W, H)))
                elif len(fpn) == 1:
                    fpn.append(self.fpn2(x[:, 1:, :].permute(0, 2, 1).reshape(x.shape[0], -1, W, H)))
                elif len(fpn) == 2:
                    fpn.append(self.fpn3(x[:, 1:, :].permute(0, 2, 1).reshape(x.shape[0], -1, W, H)))
                elif len(fpn) == 3:
                    fpn.append(self.fpn4(x[:, 1:, :].permute(0, 2, 1).reshape(x.shape[0], -1, W, H)))

        return tuple(fpn), torch.stack(aux_losses)

    def forward(self, imgs):
        # with torch.amp.autocast('cuda'):
        latent, aux_losses = self.forward_encoder(imgs)
        return latent, aux_losses.sum() * 0.000005
