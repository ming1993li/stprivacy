import math
import logging
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    # patch models (my experiments)
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),

    # patch models (weights ported from official Google JAX impl)
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_base_patch32_224': _cfg(
        url='',  # no official model weights for this combo, only for in21k
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch32_224': _cfg(
        url='',  # no official model weights for this combo, only for in21k
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),

    # patch models, imagenet21k (weights ported from official Google JAX impl)
    'vit_base_patch16_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_base_patch32_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch32_224_in21k-8db57226.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch16_224_in21k-606da67d.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch32_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_huge_patch14_224_in21k': _cfg(
        hf_hub='timm/vit_huge_patch14_224_in21k',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),

    # hybrid models (weights ported from official Google JAX impl)
    'vit_base_resnet50_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_resnet50_224_in21k-6f7c7740.pth',
        num_classes=21843, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=0.9, first_conv='patch_embed.backbone.stem.conv'),
    'vit_base_resnet50_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_resnet50_384-9fd3c705.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0, first_conv='patch_embed.backbone.stem.conv'),

    # hybrid models (my experiments)
    'vit_small_resnet26d_224': _cfg(),
    'vit_small_resnet50d_s3_224': _cfg(),
    'vit_base_resnet26d_224': _cfg(),
    'vit_base_resnet50d_224': _cfg(),

    # deit models (FB weights)
    'vit_deit_tiny_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth'),
    'vit_deit_small_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth'),
    'vit_deit_base_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',),
    'vit_deit_base_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_deit_tiny_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth'),
    'vit_deit_small_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth'),
    'vit_deit_base_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth', ),
    'vit_deit_base_distilled_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
}


def batch_index_select(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError


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


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, N, _ = policy.size()
        B, H, N, N = attn.size()
        attn_policy = policy.reshape(B, 1, 1, N)  # * policy.reshape(B, 1, N, 1)
        eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(1, 1, N, N)
        attn_policy = attn_policy + (1.0 - attn_policy) * eye
        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att
        # attn = attn.exp_() * attn_policy
        # return attn / attn.sum(dim=-1, keepdim=True)

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps/N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    def forward(self, x, policy):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if policy is None:
            attn = attn.softmax(dim=-1)
        else:
            attn = self.softmax_with_policy(attn, policy)

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
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, policy=None):
        x = x + self.drop_path(self.attn(self.norm1(x), policy=policy))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim,
                            kernel_size = (self.tubelet_size,  patch_size[0],patch_size[1]),
                            stride=(self.tubelet_size,  patch_size[0],  patch_size[1]))

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.proj(x)
        x = rearrange(x, 'b c t h w -> b c t (h w)')
        x = rearrange(x, 'b c t n -> b c (t n)').transpose(1, 2)
        return x


class PredictorLG(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, embed_dim=384, num_tubelet=8):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )
        self.num_tubelet = num_tubelet
        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, policy):
        x = self.in_conv(x)
        B, N, C = x.size()
        t = self.num_tubelet
        x = rearrange(x, 'b (t n) c -> b t n c', t=t)
        policy = rearrange(policy, 'b (t n) c -> b t n c', t=t)
        local_x = x[:,:,:, :C//3]
        frame_x = (x[:,:,:, C//3:C//3*2] * policy).sum(dim=2, keepdim=True) / torch.sum(policy, dim=2, keepdim=True)
        video_x = (x[:,:,:, C//3*2:] * policy).sum(dim=(1, 2), keepdim=True) / torch.sum(policy, dim=(1, 2), keepdim=True)
        x = torch.cat([local_x, frame_x.expand(B, t, N//t, C//3), video_x.expand(B, t, N//t, C//3)], dim=-1)
        x = rearrange(x, 'b t n c -> b (t n) c')
        return self.out_conv(x)


class STPrivacy(nn.Module):
    def __init__(self, img_size=112, patch_size=16, tubelet_size=2, all_frames=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None, 
                 pruning_loc=None, token_ratio=None, distill=False):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames,
            tubelet_size=tubelet_size)
        num_patches = self.patch_embed.num_patches
        self.num_tubelet = all_frames // tubelet_size
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm_recon = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Reconstruction head
        num_classes = patch_size * patch_size * tubelet_size * 3
        self.head_recon = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        predictor_list = [PredictorLG(embed_dim, self.num_tubelet) for _ in range(len(pruning_loc))]

        self.score_predictor = nn.ModuleList(predictor_list)

        self.distill = distill

        self.pruning_loc = pruning_loc
        self.token_ratio = token_ratio

        trunc_normal_(self.pos_embed, std=.02)
        # trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        # cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        p_count = 0
        out_pred_prob = []
        init_n = self.patch_embed.num_patches
        preserve_index = torch.arange(init_n, dtype=torch.long, device=x.device).reshape(1, -1, 1).expand(B, init_n, 1)
        prev_decision = torch.ones(B, init_n, 1, dtype=x.dtype, device=x.device)
        policy = torch.ones(B, init_n, 1, dtype=x.dtype, device=x.device)
        for i, blk in enumerate(self.blocks):
            if i in self.pruning_loc:
                spatial_x = x
                pred_score = self.score_predictor[p_count](spatial_x, prev_decision).reshape(B, -1, 2)
                if self.training:
                    hard_keep_decision = F.gumbel_softmax(pred_score, hard=True)[:, :, 0:1] * prev_decision
                    out_pred_prob.append(rearrange(hard_keep_decision.reshape(B, init_n), 'b (t n) -> b t n', t=self.num_tubelet))
                    policy = hard_keep_decision
                    x = blk(x, policy=policy)
                    prev_decision = hard_keep_decision
                else:
                    score = rearrange(pred_score[:, :, 0], 'b (t n) -> b t n', t=self.num_tubelet)
                    num_keep_node_per_frame = int(init_n // self.num_tubelet * self.token_ratio[p_count])
                    keep_policy = torch.argsort(score, dim=2, descending=True)[:, :, :num_keep_node_per_frame]
                    now_policy = keep_policy
                    offset = torch.arange(self.num_tubelet, dtype=torch.long, device=x.device).view(1, self.num_tubelet, 1) * score.shape[-1]
                    now_policy = now_policy + offset
                    now_policy = rearrange(now_policy, 'b t c -> b (t c)')
                    x = batch_index_select(x, now_policy)
                    prev_decision = batch_index_select(prev_decision, now_policy)
                    preserve_index = batch_index_select(preserve_index, now_policy)
                    x = blk(x)
                p_count += 1
            else:
                if self.training:
                    x = blk(x, policy)
                else:
                    x = blk(x)

        x = self.norm_recon(x)
        x = self.head_recon(x)
        h = w = int(math.sqrt(self.patch_embed.num_patches // self.num_tubelet))
        if self.training:
            anony_video = rearrange(x, 'b (t h w) (p0 p1 p2 c) -> b c (t p0) (h p1) (w p2)', t=self.num_tubelet, h=h, w=w,
                          p0=self.tubelet_size, p1=self.patch_size, p2=self.patch_size)

            final_select = rearrange(out_pred_prob[-1], 'b t n -> b (t n)').unsqueeze(-1)
            return anony_video, final_select, out_pred_prob
        else:
            final_select = torch.zeros(B, init_n, dtype=x.dtype, device=x.device)
            offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * init_n
            preserve_index = preserve_index.squeeze(-1) + offset
            final_select = final_select.reshape(B * init_n)
            final_select[preserve_index.reshape(-1)] = 1.0
            final_select = final_select.reshape(B, -1).unsqueeze(-1)

            B, N, C = x.size()
            anony_video = torch.zeros(B, init_n, C, dtype=x.dtype, device=x.device)
            anony_video = anony_video.reshape(B * init_n, C)
            anony_video[preserve_index.reshape(-1)] = x.reshape(-1, C)
            anony_video = anony_video.reshape(B, init_n, C)
            anony_video = rearrange(anony_video, 'b (t h w) (p0 p1 p2 c) -> b c (t p0) (h p1) (w p2)', t=self.num_tubelet, h=h, w=w,
                          p0=self.tubelet_size, p1=self.patch_size, p2=self.patch_size)

            return anony_video, final_select, None


def get_num_layer_for_d(var_name, num_max_layer, pruning_loc):
    if var_name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("rel_pos_bias"):
        return num_max_layer - 1
    elif var_name.startswith("blocks"):
        layer_id = int(var_name.split('.')[1])
        return layer_id + 1
    elif var_name.startswith("score_predictor"):
        pruning_loc_id = int(var_name.split('.')[1])
        layer_id = pruning_loc[pruning_loc_id]
        return layer_id
    else:
        return num_max_layer - 1


def freeze_sparse_blocks_module_selection(model):
    for var_name, param in model.named_parameters():
        if var_name.startswith('module'):
            var_name = var_name[7:]
        freeze_or_not = False
        if var_name in ("cls_token", "mask_token", "pos_embed"):
            freeze_or_not = True
        elif var_name.startswith("patch_embed"):
            freeze_or_not = True
        elif var_name.startswith("rel_pos_bias"):
            freeze_or_not = True
        elif var_name.startswith("blocks"):
            layer_id = int(var_name.split('.')[1])
            if layer_id < 9:
                freeze_or_not = True
            else:
                freeze_or_not = False
        elif var_name.startswith("score_predictor"):
            freeze_or_not = True
        else:
            freeze_or_not = False
        if freeze_or_not:
            param.requires_grad = False
    return None


class LayerDecayValueAssignerforD(object):
    def __init__(self, values, pruning_loc):
        self.values = values
        self.pruning_loc = pruning_loc

    def get_scale(self, layer_id):
        return self.values[layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_d(var_name, len(self.values), self.pruning_loc)


def convert_conv(weight_2d, time_dim, strategy='averaging'):
    if strategy == 'central':
        weight_3d = torch.zeros_like(weight_2d)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    elif strategy == 'averaging':
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim
    else:
        raise ValueError('The weight converting strategy is not known!')
    return weight_3d


def convert_pos_embed(pos_embed_2d, time_dim, new_size):
    pos_embed_2d = pos_embed_2d[:, 1:, :]
    B, N, C = pos_embed_2d.shape
    orig_size = int(math.sqrt(N))
    # B, L, C -> B, H, W, C
    pos_embed_2d = pos_embed_2d.reshape(B, orig_size, orig_size, C)
    if orig_size != new_size:
        print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
        # B, H, W, C -> B, C, H, W
        pos_embed_2d = pos_embed_2d.permute(0, 3, 1, 2)
        pos_embed_2d = torch.nn.functional.interpolate(
            pos_embed_2d, size=(new_size, new_size), mode='bicubic', align_corners=False)
        # B, C, H, W -> B, H, W, C
        pos_embed_2d = pos_embed_2d.permute(0, 2, 3, 1)

    pos_embed_3d = pos_embed_2d.unsqueeze(1).repeat(1, time_dim, 1, 1, 1)
    pos_embed_3d = rearrange(pos_embed_3d, 'b t h w c -> b t (h w) c')
    pos_embed_3d = rearrange(pos_embed_3d, 'b t n c -> b (t n) c')
    return pos_embed_3d


def checkpoint_image2video(state_dict, args, model):
    """ convert image pretrained weights to video ones"""
    out_dict = {}
    dim_t = model.num_tubelet
    dim_s = int(math.sqrt(model.patch_embed.num_patches // model.num_tubelet))
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if k == 'patch_embed.proj.weight':
            v = convert_conv(v, args.tubelet_size, args.d_image2video_init)
        elif k == 'patch_embed.proj.bias':
            pass
        elif k == 'pos_embed':
            # To resize pos embedding when using model at different size from pretrained weights
            v = convert_pos_embed(v, dim_t, dim_s)
        out_dict[k] = v
    return out_dict


def remove_cls_from_pos_embed(state_dict):
    """ convert image pretrained weights to video ones"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if k == 'pos_embed':
            v = v[:, 1:]
        out_dict[k] = v
    return out_dict


def get_d_param_groups(model, weight_decay):
    decay = []
    no_decay = []
    predictor = []
    for name, param in model.named_parameters():
        if 'predictor' in name:
            predictor.append(param)
        elif not param.requires_grad:
            continue  # frozen weights
        elif 'cls_token' in name or 'pos_embed' in name:
            # continue  # frozen weights
            no_decay.append(param)
        elif len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': predictor, 'weight_decay': weight_decay, 'name': 'predictor'},
        {'params': no_decay, 'weight_decay': 0., 'name': 'base_no_decay'},
        {'params': decay, 'weight_decay': weight_decay, 'name': 'base_decay'}
        ]


def create_optimizer_d(args, model, get_num_layer=None, get_layer_scale=None, filter_bias_and_bn=True, skip_list=None):
    opt_lower = args.opt.lower()
    weight_decay = args.d_weight_decay
    if weight_decay and filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters = get_parameter_groups_for_d_with_layer_decay(model, weight_decay, skip, get_num_layer, get_layer_scale)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    opt_args = dict(lr=args.d_lr, weight_decay=weight_decay)
    if hasattr(args, 'd_opt_eps') and args.d_opt_eps is not None:
        opt_args['eps'] = args.d_opt_eps
    if hasattr(args, 'd_opt_betas') and args.d_opt_betas is not None:
        opt_args['betas'] = args.d_opt_betas

    print("optimizer settings:", opt_args)
    optimizer = torch.optim.AdamW(parameters, **opt_args)
    return optimizer


def get_parameter_groups_for_d_with_layer_decay(model, weight_decay=1e-5, skip_list=(), get_num_layer=None, get_layer_scale=None):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if 'predictor' in name:
            general_name = "predictor"
            this_weight_decay = weight_decay
        elif len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            general_name = "base_no_decay"
            this_weight_decay = 0.
        else:
            general_name = "base_decay"
            this_weight_decay = weight_decay

        if get_num_layer is not None:
            layer_id = get_num_layer(name)
            group_name = "layer_%d_%s" % (layer_id, general_name)
        else:
            layer_id = None
            group_name = general_name

        if group_name not in parameter_group_names:
            if get_layer_scale is not None:
                scale = get_layer_scale(layer_id)
            else:
                scale = 1.

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
                'name': general_name,
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "lr_scale": scale,
                'name': general_name,
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def adjust_d_learning_rate(param_groups, init_lr, min_lr, step, max_step, warming_up_step=2, warmup_predictor=False, base_multi=0.1):
    cos_lr = (math.cos(step / max_step * math.pi) + 1) * 0.5
    cos_lr = min_lr + cos_lr * (init_lr - min_lr)
    if warmup_predictor and step < 1:
        cos_lr = init_lr * 0.01
    if step < warming_up_step:
        backbone_lr = 0
    else:
        backbone_lr = min(init_lr * base_multi, cos_lr)
    print('## Using lr  %.7f for BACKBONE, cosine lr = %.7f for PREDICTOR' % (backbone_lr, cos_lr))
    for param_group in param_groups:
        if param_group['name'] == 'predictor':
            param_group['lr'] = cos_lr * param_group["lr_scale"]
        else:
            param_group['lr'] = backbone_lr * param_group["lr_scale"] # init_lr * 0.01 # cos_lr * base_multi


def cosine_adjust(min_vaule, max_value, step, max_step, increase=True):
    if increase:
        cos = (math.cos(step / max_step * math.pi - math.pi) + 1) * 0.5
    else:
        cos = (math.cos(step / max_step * math.pi) + 1) * 0.5
    cos = min_vaule + cos * (max_value - min_vaule)
    return cos