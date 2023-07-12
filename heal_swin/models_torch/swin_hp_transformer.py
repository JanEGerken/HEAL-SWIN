"""This implementation of the HEAL-SWIN-UNet was adapted from
https://github.com/HuCaoFighting/Swin-Unet/blob/1c8b3e860dfaa89c98fa8e5ad1d4abd2251744f9/networks/swin_transformer_unet_skip_expand_decoder_sys.py
"""

import math
from dataclasses import dataclass, field
from typing import Optional, List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_

from heal_swin.data.segmentation.data_spec import DataSpec
from heal_swin.models_torch import hp_shifting
from heal_swin.models_torch.hp_windowing import window_partition, window_reverse, get_nest_win_idcs


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
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


class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (int): Number of pixels in the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        use_cos_attn (bool): Whether to use cosine attention as in version 2 of swin transformer
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        rel_pos_bias=None,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_cos_attn=False,
    ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.use_cos_attn = use_cos_attn
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.rel_pos_bias = rel_pos_bias

        if self.use_cos_attn:
            self.logit_scale = nn.Parameter(
                torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
            )

        if self.rel_pos_bias == "flat":

            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (int((2 * window_size**0.5 - 1) * (2 * window_size**0.5 - 1)), num_heads)
                )
            )  # 2*sqrt(Ws)-1 * 2*sqrt(Ws)-1, nH

            # get pair-wise relative position index for each token inside the window
            coords = torch.arange(window_size**0.5)
            coords = torch.stack(torch.meshgrid([coords, coords]))  # 2, sqrt(Ws), sqrt(Ws)
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Ws, Ws
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Ws, Ws, 2
            relative_coords[:, :, 0] += window_size**0.5 - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size**0.5 - 1
            relative_coords[:, :, 0] *= 2 * window_size**0.5 - 1
            relative_position_index = relative_coords.sum(-1).long()  # Ws, Ws

            # translate from nested index scheme into Cartesian coordinates
            nest_idcs = get_nest_win_idcs(window_size)  # sqrt(Ws), sqrt(Ws)
            nest_idcs_inv = nest_idcs.flatten().argsort()
            relative_position_index = relative_position_index[nest_idcs_inv]  # Ws, Ws
            relative_position_index = relative_position_index[:, nest_idcs_inv]  # Ws, Ws
            self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, window_size, window_size) or None
        """
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        if self.use_cos_attn:
            attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
            logit_scale = torch.clamp(
                self.logit_scale, max=torch.log(torch.tensor(1.0 / 0.01))
            ).exp()
            attn = attn * logit_scale
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)

        if self.rel_pos_bias is not None:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index
            ]  # Ws,Ws,nH
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1
            ).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r"""Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (int): Number of input pixels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        use_v2_norm_placement (bool): Whether to use changed norm layer placement from version 2
        use_cos_attn (bool): Whether to use cosine attention as in version 2 of swin transformer
    """

    def __init__(
        self,
        dim,
        input_resolution,
        base_pix,
        num_heads,
        window_size=4,
        shift_size=0,
        shift_strategy="nest_roll",
        rel_pos_bias=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_v2_norm_placement=False,
        use_cos_attn=False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_v2_norm_placement = use_v2_norm_placement
        if self.input_resolution <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = self.input_resolution

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=window_size,
            num_heads=num_heads,
            rel_pos_bias=rel_pos_bias,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_cos_attn=use_cos_attn,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        # get nside parameter of current resolution
        nside = math.sqrt(input_resolution // base_pix)
        assert nside % 1 == 0, "nside has to be an integer in every layer"
        nside = int(nside)
        # shifter classes and arguments for their init functions
        # separate this so only the needed class gets instantiated
        shifters = {
            "nest_roll": (
                hp_shifting.NestRollShift,
                {
                    "shift_size": self.shift_size,
                    "input_resolution": self.input_resolution,
                    "window_size": self.window_size,
                },
            ),
            "nest_grid_shift": (
                hp_shifting.NestGridShift,
                {"nside": nside, "base_pix": base_pix, "window_size": self.window_size},
            ),
            "ring_shift": (
                hp_shifting.RingShift,
                {
                    "nside": nside,
                    "base_pix": base_pix,
                    "window_size": self.window_size,
                    "shift_size": self.shift_size,
                },
            ),
        }

        if self.shift_size > 0:
            self.shifter = shifters[shift_strategy][0](**shifters[shift_strategy][1])
        else:
            self.shifter = hp_shifting.NoShift()

        attn_mask = self.shifter.get_mask()

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        N = self.input_resolution
        B, N, C = x.shape

        shortcut = x
        if not self.use_v2_norm_placement:
            x = self.norm1(x)

        # cyclic shift
        shifted_x = self.shifter.shift(x)

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, C
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size, C

        # merge windows
        shifted_x = window_reverse(attn_windows, self.window_size, N)  # B N' C

        # reverse cyclic shift
        x = self.shifter.shift_back(shifted_x)

        # FFN
        if self.use_v2_norm_placement:
            x = shortcut + self.drop_path(self.norm1(x))
            x = x + self.drop_path(self.norm2(self.mlp(x)))
        else:
            x = shortcut + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads},"
            f" window_size={self.window_size}, shift_size={self.shift_size}"
            f", mlp_ratio={self.mlp_ratio}"
        )

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size[0] / self.window_size[1]
        flops += nW * self.attn.flops(self.window_size[0] * self.window_size[1])
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r"""Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, dim_scale * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, N, C
        """
        B, N, C = x.shape
        assert N % 4 == 0, f"x size {N} is not divisible by 4 as necessary for patching."

        x0 = x[:, 0::4, :]  # B N/4 C
        x1 = x[:, 1::4, :]  # B N/4 C
        x2 = x[:, 2::4, :]  # B N/4 C
        x3 = x[:, 3::4, :]  # B N/4 C
        # concatenate the patches per merge-window channel-wise
        x = torch.cat([x0, x1, x2, x3], -1)  # B N/4 patch_size*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        N = self.input_resolution
        flops = N * self.dim
        flops += (N // 2) * self.patch_size * self.dim * 2 * self.dim
        return flops


class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        """
        dim: input channels
        dim_scale: upscaling factor for channels before patch expansion
        """
        super().__init__()
        self.dim = dim
        self.expand = (
            nn.Linear(dim, dim_scale * dim, bias=False) if dim_scale != 1 else nn.Identity()
        )
        self.norm = norm_layer(dim * dim_scale // 4)

    def forward(self, x):
        """
        x: B, N, dim
        """
        x = self.expand(x)
        B, N, C = x.shape

        x = rearrange(x, "b n (p c)-> b (n p) c", p=4, c=C // 4)
        x = self.norm(x)

        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, patch_size, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.expand = nn.Linear(dim, patch_size * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, N, dim
        """
        x = self.expand(x)
        B, N, C = x.shape

        x = rearrange(x, "b n (p c)-> b (n p) c", p=self.patch_size, c=C // self.patch_size)
        x = self.norm(x)

        return x


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (int): Number of pixels in input.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        use_v2_norm_placement (bool): Whether to use changed norm layer placement from version 2
        use_cos_attn (bool): Whether to use cosine attention as in version 2 of swin transformer
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        base_pix,
        shift_size,
        shift_strategy,
        rel_pos_bias,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        use_v2_norm_placement=False,
        use_cos_attn=False,
    ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    base_pix=base_pix,
                    shift_size=0 if (i % 2 == 0) else shift_size,
                    shift_strategy=shift_strategy,
                    rel_pos_bias=rel_pos_bias,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_v2_norm_placement=use_v2_norm_placement,
                    use_cos_attn=use_cos_attn,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class BasicLayer_up(nn.Module):
    """A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (int): Number of input pixels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        use_v2_norm_placement (bool): Whether to use changed norm layer placement from version 2
        use_cos_attn (bool): Whether to use cosine attention as in version 2 of swin transformer
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        base_pix,
        shift_size,
        shift_strategy,
        rel_pos_bias,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        upsample=None,
        use_checkpoint=False,
        use_v2_norm_placement=False,
        use_cos_attn=False,
    ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    base_pix=base_pix,
                    shift_size=0 if (i % 2 == 0) else shift_size,
                    shift_strategy=shift_strategy,
                    rel_pos_bias=rel_pos_bias,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_v2_norm_placement=use_v2_norm_placement,
                    use_cos_attn=use_cos_attn,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class PatchEmbed(nn.Module):
    r"""Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, config, data_spec):
        super().__init__()
        assert config.patch_size % 4 == 0, "required for valid nside in deeper layers"

        self.config = config
        self.data_spec = data_spec
        self.num_patches = data_spec.dim_in // config.patch_size

        self.proj = nn.Conv1d(
            data_spec.f_in,
            config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        if config.patch_embed_norm_layer is not None:
            self.norm = config.patch_embed_norm_layer
        else:
            self.norm = None

    def forward(self, x):
        B, C, N = x.shape
        assert (
            N == self.data_spec.dim_in
        ), f"Input image size ({N}) doesn't match model ({self.data_spec.dim_in})."
        x = self.proj(x).transpose(1, 2)  # B Pn C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        N = self.num_patches
        flops = N * self.config.embed_dim * self.data_spec.f_in * self.patch_size
        if self.norm is not None:
            flops += N * self.embed_dim
        return flops


class UnetDecoder(nn.Module):
    def __init__(self, config, data_spec, dpr):
        super().__init__()
        self.config = config
        self.num_layers = len(config.depths)
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))
        num_patches = data_spec.dim_in // config.patch_size
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            down_idx = self.num_layers - 1 - i_layer
            concat_out = int(config.embed_dim * 2**down_idx)
            concat_in = 2 * concat_out
            concat_linear = nn.Linear(concat_in, concat_out) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(
                    dim=concat_out,
                    dim_scale=2,
                    norm_layer=config.norm_layer,
                )
            else:
                layer_up = BasicLayer_up(
                    dim=concat_out,
                    input_resolution=num_patches // (4**down_idx),
                    depth=config.depths[down_idx],
                    num_heads=config.num_heads[down_idx],
                    window_size=config.window_size,
                    base_pix=data_spec.base_pix,
                    shift_size=config.shift_size,
                    shift_strategy=config.shift_strategy,
                    rel_pos_bias=config.rel_pos_bias,
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=config.qkv_bias,
                    qk_scale=config.qk_scale,
                    use_cos_attn=config.use_cos_attn,
                    drop=config.drop_rate,
                    attn_drop=config.attn_drop_rate,
                    drop_path=dpr[
                        sum(config.depths[:down_idx]) : sum(config.depths[: down_idx + 1])
                    ],
                    norm_layer=config.norm_layer,
                    use_v2_norm_placement=config.use_v2_norm_placement,
                    upsample=PatchExpand if down_idx > 0 else None,
                    use_checkpoint=config.use_checkpoint,
                )
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.up = FinalPatchExpand_X4(
            patch_size=config.patch_size,
            dim=config.embed_dim,
        )
        self.output = nn.Conv1d(
            in_channels=config.embed_dim,
            out_channels=data_spec.f_out,
            kernel_size=1,
            bias=False,
        )

        self.norm_up = config.norm_layer(config.embed_dim)

    def forward(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
                if self.config.dev_mode:
                    print(f"feature shape after PatchExpand: {x.size()}")
            else:
                x = torch.cat([x, x_downsample[self.num_layers - 1 - inx]], -1)
                if self.config.dev_mode:
                    print(f"feature shape after concatenation: {x.size()}")
                x = self.concat_back_dim[inx](x)
                if self.config.dev_mode:
                    print(f"feature shape after linear layer: {x.size()}")
                x = layer_up(x)
                if self.config.dev_mode:
                    print("feature shape after basic layer up", inx, ": ", x.size())
        x = self.norm_up(x)  # B L C
        x = self.up(x)
        if self.config.dev_mode:
            print("feature shape after FinalPatchExpand_X4: ", x.size())
        x = x.permute(0, 2, 1)  # B,C,N
        if self.config.dev_mode:
            print("feature shape after permutation: ", x.size())
        x = self.output(x)
        if self.config.dev_mode:
            print("feature shape after 1x1 convolution: ", x.size())
        return x


@dataclass
class SwinHPTransformerConfig:
    patch_size: int = 4
    window_size: int = 4
    shift_size: int = 2
    shift_strategy: Literal["nest_roll", "nest_grid_shift", "ring_shift"] = "nest_roll"
    rel_pos_bias: Optional[Literal["flat"]] = None
    embed_dim: int = 96
    patch_embed_norm_layer: Optional[Literal[nn.LayerNorm]] = None
    depths: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    num_heads: List[int] = field(default_factory=lambda: [3, 6, 12, 24])
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    qk_scale: Optional[float] = None
    use_cos_attn: bool = False
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    norm_layer: Literal[nn.LayerNorm] = nn.LayerNorm
    use_v2_norm_placement: bool = False
    ape: bool = False
    patch_norm: bool = True
    use_checkpoint: bool = False
    dev_mode: bool = False  # Developer mode for printing extra information
    decoder_class: Literal[UnetDecoder] = UnetDecoder


class SwinHPTransformerSys(nn.Module):
    r"""Swin Transformer A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer
        using Shifted Windows` - https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        use_v2_norm_placement (bool): Whether to use changed norm layer placement from version 2
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        use_cos_attn (bool): Whether to use cosine attention as in version 2 of swin transformer

    """

    def __init__(self, config: SwinHPTransformerConfig, data_spec: DataSpec, **kwargs):
        super().__init__()

        self.config = config
        self.data_spec = data_spec

        self.num_layers = len(config.depths)
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(config.embed_dim * 2)

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(config, data_spec=data_spec)
        num_patches = self.patch_embed.num_patches

        # absolute position embedding
        if config.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=config.drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))
        ]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if config.decoder_class == UnetDecoder:
                downsample = PatchMerging if (i_layer < self.num_layers - 1) else None

            layer = BasicLayer(
                dim=int(config.embed_dim * 2**i_layer),
                input_resolution=num_patches // (4**i_layer),
                depth=config.depths[i_layer],
                num_heads=config.num_heads[i_layer],
                window_size=config.window_size,
                base_pix=data_spec.base_pix,
                shift_size=config.shift_size,
                shift_strategy=config.shift_strategy,
                rel_pos_bias=config.rel_pos_bias,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias,
                qk_scale=config.qk_scale,
                use_cos_attn=config.use_cos_attn,
                drop=config.drop_rate,
                attn_drop=config.attn_drop_rate,
                drop_path=dpr[sum(config.depths[:i_layer]) : sum(config.depths[: i_layer + 1])],
                norm_layer=config.norm_layer,
                use_v2_norm_placement=config.use_v2_norm_placement,
                downsample=downsample,
                use_checkpoint=config.use_checkpoint,
            )
            self.layers.append(layer)

        self.decoder = config.decoder_class(config, data_spec, dpr)

        out_channels = self.num_features * (1 if config.decoder_class == UnetDecoder else 2)
        self.norm = config.norm_layer(out_channels)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    # Encoder and Bottleneck
    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.config.dev_mode:
            print("feature shape after patch embedding: ", x.size())
        if self.config.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []

        for k, layer in enumerate(self.layers):
            x_downsample.append(x)
            x = layer(x)
            if self.config.dev_mode:
                print("feature shape after basic layer", k, ": ", x.size())

        x = self.norm(x)  # B N C
        return x, x_downsample

    def forward(self, x):
        if self.config.dev_mode:
            print("feature shape at input: ", x.size())
        x, x_downsample = self.forward_features(x)
        x = self.decoder(x, x_downsample)
        if self.config.dev_mode:
            print("feature shape at output: ", x.size())
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += (
            self.num_features
            * self.patches_resolution[0]
            * self.patches_resolution[1]
            // (2**self.num_layers)
        )
        flops += self.num_features * self.data_spec.f_out
        return flops
