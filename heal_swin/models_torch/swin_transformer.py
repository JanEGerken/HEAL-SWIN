"""This implementation of the SWIN-UNet was adapted from
https://github.com/HuCaoFighting/Swin-Unet/blob/1c8b3e860dfaa89c98fa8e5ad1d4abd2251744f9/networks/swin_transformer_unet_skip_expand_decoder_sys.py
"""

from dataclasses import dataclass, field
from typing import Optional, List, Literal, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_

from heal_swin.data.segmentation.data_spec import DataSpec


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


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(
        B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  Whether to add a learnable bias. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        use_cos_attn (bool): Use cosine attention as in v2 of swin transformer. Default: False
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_cos_attn=False,
        use_rel_pos_bias=True,
    ):

        super().__init__()
        self.use_rel_pos_bias = use_rel_pos_bias
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.use_cos_attn = use_cos_attn
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        if self.use_cos_attn:
            self.logit_scale = nn.Parameter(
                torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
            )

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
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
            attn = q @ k.transpose(-2, -1).contiguous()

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        if self.use_rel_pos_bias:
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).contiguous().reshape(B_, N, C)
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
        input_resolution (tuple[int]): Input resulotion.
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
        use_masking (bool): Use masked self attention. Default: True
        use_v2_norm_placement (bool): Use changed norm layer placement as in swin v2. Default: False
        use_cos_attn (bool): Use cosine attention as in swin v2. Default: False
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=[4, 4],
        shift_size=-1,  # Workaround, see self.shift_size.
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_masking=True,
        use_cos_attn=False,
        use_v2_norm_placement=False,
        use_rel_pos_bias=True,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = (
            tuple([window_size[0] // 2, window_size[1] // 2]) if shift_size == -1 else shift_size
        )
        self.mlp_ratio = mlp_ratio
        self.use_v2_norm_placement = use_v2_norm_placement
        if (
            self.input_resolution[0] <= self.window_size[0]
            or self.input_resolution[1] <= self.window_size[1]
        ):
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = [0, 0]
            self.window_size = self.input_resolution

        error_str = "Shift size and window size must satisfy 0 <= shift_size[{i}] < "
        error_str += "window_size[{i}], got shift_size[{i}]={ss} and window_size[{i}]={ws}"
        assert 0 <= self.shift_size[0] < self.window_size[0], error_str.format(
            i=0, ss=self.shift_size[0], ws=self.window_size[0]
        )
        assert 0 <= self.shift_size[1] < self.window_size[1], error_str.format(
            i=1, ss=self.shift_size[1], ws=self.window_size[1]
        )

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_rel_pos_bias=use_rel_pos_bias,
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

        if use_masking and (self.shift_size[0] > 0 or self.shift_size[1] > 0):
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            # JG: These slices select points inside full windows and partly split windows,
            # cf. Figure 4 in the SWIN paper
            h_slices = (
                slice(0, -self.window_size[0]),
                slice(-self.window_size[0], -self.shift_size[0]),
                slice(-self.shift_size[0], None),
            )
            w_slices = (
                slice(0, -self.window_size[1]),
                slice(-self.window_size[1], -self.shift_size[1]),
                slice(-self.shift_size[1], None),
            )
            # JG: In this loop, all the different subwindows A, B, ... from Figure 4 are given
            # different numbers 0...8
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(
                img_mask, self.window_size
            )  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1])
            # JG: The following gives a tensor of shape (nW, Wh*Ww, Wh*Ww) it computes inside each
            # window (0th index) for each pixel pair if they lie in the same subwindow (=0) or not
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            # JG: pixel pairs with non-matching subwindows are set to -100, matching pairs to 0
            # inside the attention module, this gets added to the argument of softmax, so for -100
            # the attention output is 0
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
                attn_mask == 0, float(0.0)
            )
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        if not self.use_v2_norm_placement:
            x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size[0] > 0 or self.shift_size[1] > 0:  # FIXME Change or to and?
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size[0], -self.shift_size[0]), dims=(1, 2)
            )
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(
            -1, self.window_size[0] * self.window_size[1], C
        )  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

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
            f" window_size={self.window_size}, shift_size={self.shift_size},"
            f" mlp_ratio={self.mlp_ratio}"
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
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.patch_size = 4
        self.reduction = nn.Linear(self.patch_size * dim, 2 * dim, bias=False)
        self.norm = norm_layer(self.patch_size * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 patch_size*C
        x = x.view(B, -1, self.patch_size * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * self.patch_size * self.dim * 2 * self.dim
        return flops


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)
        self.dim_scale = 4

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(
            x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=2, p2=2, c=C // self.dim_scale
        )  # Changed 4 to self.dim_scale
        x = x.view(B, -1, C // self.dim_scale)  # Changed 4 to self.dim_scale
        x = self.norm(x)

        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, patch_size, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.patch_size = patch_size
        self.L = self.input_resolution[0] * self.input_resolution[1]
        self.expand = nn.Linear(dim, (self.patch_size[0] * self.patch_size[1]) * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(
            x,
            "b h w (p1 p2 c)-> b (h p1) (w p2) c",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            c=C // (self.patch_size[0] * self.patch_size[1]),
        )
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)

        return x


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
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
        use_masking (bool): Whether to use masked self-attention. Default: True.
        use_v2_norm_placement (bool): Use changed norm layer placement as in swin v2. Default: False
        use_cos_attn (bool): Use cosine attention as in swin v2. Default: False
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        shift_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        use_masking=True,
        use_cos_attn=False,
        use_v2_norm_placement=False,
        use_rel_pos_bias=True,
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
                    shift_size=[0, 0] if (i % 2 == 0) else shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_masking=use_masking,
                    use_rel_pos_bias=use_rel_pos_bias,
                    use_v2_norm_placement=use_v2_norm_placement,
                    use_cos_attn=use_cos_attn,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
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
        input_resolution (tuple[int]): Input resolution.
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
        use_masking (bool): Whether to use masked self-attention. Default: True.
        use_v2_norm_placement (bool): Use changed norm layer placement as in swin v2. Default: False
        use_cos_attn (bool): Use cosine attention as in swin v2. Default: False
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        shift_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        upsample=None,
        use_checkpoint=False,
        use_masking=True,
        use_cos_attn=False,
        use_v2_norm_placement=False,
        use_rel_pos_bias=True,
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
                    shift_size=[0, 0] if (i % 2 == 0) else shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_masking=use_masking,
                    use_rel_pos_bias=use_rel_pos_bias,
                    use_cos_attn=use_cos_attn,
                    use_v2_norm_placement=use_v2_norm_placement,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(
                input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer
            )
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
        self.config = config
        self.data_spec = data_spec
        self.patches_resolution = [
            data_spec.dim_in[0] // config.patch_size[0],
            data_spec.dim_in[1] // config.patch_size[1],
        ]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.proj = nn.Conv2d(
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
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert (
            H == self.data_spec.dim_in[0] and W == self.data_spec.dim_in[1]
        ), f"Input image size {H}*{W} doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = (
            Ho
            * Wo
            * self.config.embed_dim
            * self.data_spec.f_in
            * (self.patch_size[0] * self.patch_size[1])
        )
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


@dataclass
class SwinTransformerConfig:
    patch_size: Union[int, Tuple[int, int]] = (4, 4)
    window_size: Union[int, Tuple[int, int]] = (4, 4)
    shift_size: Union[int, Tuple[int, int]] = -1
    embed_dim: int = 96
    patch_embed_norm_layer: Optional[str] = None
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
    final_upsample: Literal["expand_first"] = "expand_first"
    use_masking: bool = True
    use_rel_pos_bias: bool = True
    dev_mode: bool = False  # Developer mode for printing extra information


class SwinTransformerSys(nn.Module):
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
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        use_v2_norm_placement (bool): Use changed norm layer placement as in swin v2. Default: False
        use_cos_attn (bool): Use cosine attention as in swin v2. Default: False

    """

    def __init__(self, config: SwinTransformerConfig, data_spec: DataSpec, **kwargs):
        super().__init__()
        self.config = config
        self.data_spec = data_spec

        self.num_layers = len(config.depths)
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(config.embed_dim * 2)

        # checks
        H, W = data_spec.dim_in[0], data_spec.dim_in[1]
        if isinstance(config.patch_size, int):
            config.patch_size = [config.patch_size, config.patch_size]
        else:
            if len(config.patch_size) == 1:
                config.patch_size = [config.patch_size[0], config.patch_size[0]]

        patch_height, patch_width = config.patch_size[0], config.patch_size[1]

        if isinstance(config.window_size, int):
            config.window_size = [config.window_size, config.window_size]
        else:
            if len(config.window_size) == 1:
                config.window_size = [config.window_size[0], config.window_size[0]]

        window_height, window_width = config.window_size[0], config.window_size[1]

        merge_factor = 2 ** (self.num_layers - 1)

        def get_error_str(HW, f1, f2, f3, corr):
            error_str = "{HW} must be divisible by {f1}*{f2}*{f3}, got {HW}={HW_val}, "
            error_str += "{f1}*{f2}*{f3}={f1_val}*{f2_val}*{f3_val}={prod_val}."
            corr_str = " Correct by {corr_val_1} or {corr_val_2}."
            prod_val = eval(f"{f1}*{f2}*{f3}")
            result = error_str.format(
                HW=HW,
                f1=f1,
                f2=f2,
                f3=f3,
                HW_val=eval(HW),
                f1_val=eval(f1),
                f2_val=eval(f2),
                f3_val=eval(f3),
                prod_val=prod_val,
            )
            if corr:
                result += corr_str.format(
                    corr_val_1=eval(f"-{HW}%{prod_val}"),
                    corr_val_2=eval(f"-(({HW}%{prod_val})-{prod_val})"),
                )
            return result

        assert (H / (merge_factor * patch_height * window_height)) % 1 == 0, get_error_str(
            HW="H", f1="merge_factor", f2="patch_height", f3="window_height", corr=True
        )
        assert (W / (merge_factor * patch_width * window_width)) % 1 == 0, get_error_str(
            HW="W", f1="merge_factor", f2="patch_width", f3="window_width", corr=True
        )
        assert (H * W / (merge_factor**2 * patch_height * patch_width)) % 1 == 0, get_error_str(
            HW="H*W", f1="merge_factor**2", f2="patch_height", f3="patch_width", corr=False
        )

        # Workaround to get correct default values
        if config.shift_size == -1:
            self.shift_size = tuple([config.window_size[0] // 2, config.window_size[1] // 2])
        else:
            if isinstance(config.shift_size, int):
                config.shift_size = [config.shift_size, config.shift_size]

            self.shift_size = config.shift_size

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(config, data_spec=data_spec)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

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
            layer = BasicLayer(
                dim=int(config.embed_dim * 2**i_layer),
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=config.depths[i_layer],
                num_heads=config.num_heads[i_layer],
                window_size=config.window_size,
                shift_size=self.shift_size,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias,
                qk_scale=config.qk_scale,
                use_cos_attn=config.use_cos_attn,
                drop=config.drop_rate,
                attn_drop=config.attn_drop_rate,
                drop_path=dpr[sum(config.depths[:i_layer]) : sum(config.depths[: i_layer + 1])],
                norm_layer=config.norm_layer,
                use_v2_norm_placement=config.use_v2_norm_placement,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=config.use_checkpoint,
                use_masking=config.use_masking,
                use_rel_pos_bias=config.use_rel_pos_bias,
            )
            self.layers.append(layer)

        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = (
                nn.Linear(
                    2 * int(config.embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    int(config.embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                )
                if i_layer > 0
                else nn.Identity()
            )
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(
                        patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    dim=int(config.embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    dim_scale=2,
                    norm_layer=config.norm_layer,
                )
            else:
                layer_up = BasicLayer_up(
                    dim=int(config.embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    input_resolution=(
                        patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    depth=config.depths[(self.num_layers - 1 - i_layer)],
                    num_heads=config.num_heads[(self.num_layers - 1 - i_layer)],
                    window_size=config.window_size,
                    shift_size=self.shift_size,
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=config.qkv_bias,
                    qk_scale=config.qk_scale,
                    use_cos_attn=config.use_cos_attn,
                    drop=config.drop_rate,
                    attn_drop=config.attn_drop_rate,
                    drop_path=dpr[
                        sum(config.depths[: (self.num_layers - 1 - i_layer)]) : sum(
                            config.depths[: (self.num_layers - 1 - i_layer) + 1]
                        )
                    ],
                    norm_layer=config.norm_layer,
                    use_v2_norm_placement=config.use_v2_norm_placement,
                    upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=config.use_checkpoint,
                    use_masking=config.use_masking,
                    use_rel_pos_bias=config.use_rel_pos_bias,
                )
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = config.norm_layer(self.num_features)
        self.norm_up = config.norm_layer(config.embed_dim)

        if config.final_upsample == "expand_first":
            self.up = FinalPatchExpand_X4(
                input_resolution=(
                    data_spec.dim_in[0] // config.patch_size[0],
                    data_spec.dim_in[1] // config.patch_size[1],
                ),
                patch_size=config.patch_size,
                dim=config.embed_dim,
            )
            self.output = nn.Conv2d(
                in_channels=config.embed_dim,
                out_channels=data_spec.f_out,
                kernel_size=1,
                bias=False,
            )

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
            print(f"forward_features after patch_embed: {x.size()}")
        if self.config.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        if self.config.dev_mode:
            print(f"forward_features after pos_drop: {x.size()}")
        x_downsample = []

        k = 0
        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)
            if self.config.dev_mode:
                print(f"forward_features after layer {k}: {x.size()}")
            k = k + 1

        x = self.norm(x)  # B L C
        if self.config.dev_mode:
            print(f"forward_features after norm: {x.size()}")
        return x, x_downsample

    # Decoder and Skip connection
    def forward_up_features(self, x, x_downsample):
        k = 0
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[self.num_layers - 1 - inx]], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)
            if self.config.dev_mode:
                print(f"forward_up_features after layer {k}: {x.size()}")
            k = k + 1
        x = self.norm_up(x)  # B L C

        return x

    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.config.final_upsample == "expand_first":
            x = self.up(x)  # Adds factor 16 upsampling
            if self.config.dev_mode:
                print(f"up_x4 after  self.up: {x.size()}")
            x = x.view(B, self.config.patch_size[0] * H, self.config.patch_size[1] * W, -1)
            if self.config.dev_mode:
                print(f"up_x4 after x.view: {x.size()}")
            x = x.permute(0, 3, 1, 2).contiguous()  # B,C,H,W
            if self.config.dev_mode:
                print(f"up_x4 after permute: {x.size()}")
            x = self.output(x)

        return x

    def forward(self, x):
        if self.config.dev_mode:
            print(f"forward before forward_features: {x.size()}")

        x, x_downsample = self.forward_features(x)
        if self.config.dev_mode:
            print(f"forward after downsample: {x.size()}")
        x = self.forward_up_features(x, x_downsample)
        if self.config.dev_mode:
            print(f"forward after forward_up_features: {x.size()}")
        x = self.up_x4(x)
        if self.config.dev_mode:
            print(f"forward after up_x4: {x.size()}")

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
