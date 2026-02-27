# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle import ParamAttr
import os
import sys
from ppocr.modeling.necks.intracl import IntraCLBlock

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../../..')))

from ppocr.modeling.backbones.det_mobilenet_v3 import SEModule


class AVG(nn.Layer):
    def __init__(self, down_n=2):
        super().__init__()
        self.down_n = down_n

    def forward(self, x):
        B, C, H, W = x.shape
        H = int(H / self.down_n)
        W = int(W / self.down_n)
        x = F.adaptive_avg_pool2d(x, (H, W))
        return x


class DilatedReparamBlock(nn.Layer):
    """
    Dilated Reparam Block from UniRepLKNet.
    Reference: https://github.com/AILab-CVC/UniRepLKNet

    Training: uses multiple parallel dilated depthwise convolutions + BN
    Inference: all branches merge into a single large-kernel depthwise conv

    For kernel_size=9, the branches are:
      - origin: 9x9 DW Conv (dil=1)
      - branch1: 5x5 DW Conv (dil=1, equiv RF=5)
      - branch2: 5x5 DW Conv (dil=2, equiv RF=9)
      - branch3: 3x3 DW Conv (dil=3, equiv RF=7)
      - branch4: 3x3 DW Conv (dil=4, equiv RF=9)
    """

    def __init__(self, channels, kernel_size=9, deploy=False):
        super(DilatedReparamBlock, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.deploy = deploy

        if kernel_size == 9:
            self.kernel_sizes = [5, 5, 3, 3]
            self.dilates = [1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3]
            self.dilates = [1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 5, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        else:
            raise ValueError(
                'DilatedReparamBlock requires kernel_size in [5,7,9,11,13], '
                'but got {}'.format(kernel_size))

        if not deploy:
            self.lk_origin = nn.Conv2D(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                groups=channels,
                bias_attr=False)
            self.origin_bn = nn.BatchNorm2D(channels)

            for k, r in zip(self.kernel_sizes, self.dilates):
                equiv_ks = r * (k - 1) + 1
                p = equiv_ks // 2
                conv = nn.Conv2D(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=k,
                    stride=1,
                    padding=p,
                    dilation=r,
                    groups=channels,
                    bias_attr=False)
                bn = nn.BatchNorm2D(channels)
                setattr(self, 'dil_conv_k{}_{}'.format(k, r), conv)
                setattr(self, 'dil_bn_k{}_{}'.format(k, r), bn)
        else:
            self.lk_origin = nn.Conv2D(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                groups=channels,
                bias_attr=True)

    def forward(self, x):
        if self.deploy:
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = getattr(self, 'dil_conv_k{}_{}'.format(k, r))
            bn = getattr(self, 'dil_bn_k{}_{}'.format(k, r))
            out = out + bn(conv(x))
        return out

    @staticmethod
    def _fuse_bn(conv, bn):
        """Fuse Conv2D + BatchNorm2D into a single Conv2D (weight, bias)."""
        kernel = conv.weight
        gamma = bn.weight
        beta = bn.bias
        running_mean = bn._mean
        running_var = bn._variance
        eps = bn._epsilon
        std = paddle.sqrt(running_var + eps)
        fused_weight = kernel * (gamma / std).reshape([-1, 1, 1, 1])
        fused_bias = beta - running_mean * gamma / std
        return fused_weight, fused_bias

    @staticmethod
    def _convert_dilated_to_nondilated(kernel, dilate_rate):
        """Convert dilated conv kernel to equivalent non-dilated (sparse) kernel
        by inserting zeros between kernel elements using transposed convolution."""
        if dilate_rate == 1:
            return kernel
        identity = paddle.ones(shape=[1, 1, 1, 1], dtype=kernel.dtype)
        # F.conv2d_transpose with stride=dilate_rate inserts zeros
        # Process each channel independently
        C = kernel.shape[0]
        result_list = []
        for i in range(C):
            k_i = kernel[i:i + 1]  # (1, 1, kH, kW)
            dilated = F.conv2d_transpose(k_i, identity, stride=dilate_rate)
            result_list.append(dilated)
        return paddle.concat(result_list, axis=0)

    @staticmethod
    def _merge_dilated_into_large_kernel(large_kernel, dilated_kernel,
                                          dilated_r):
        """Pad dilated equivalent kernel to large kernel size and add."""
        large_k = large_kernel.shape[2]
        dilated_k = dilated_kernel.shape[2]
        equiv_ks = dilated_r * (dilated_k - 1) + 1
        equiv_kernel = DilatedReparamBlock._convert_dilated_to_nondilated(
            dilated_kernel, dilated_r)
        rows_to_pad = large_k // 2 - equiv_ks // 2
        if rows_to_pad > 0:
            merged = large_kernel + F.pad(
                equiv_kernel,
                [rows_to_pad, rows_to_pad, rows_to_pad, rows_to_pad])
        else:
            merged = large_kernel + equiv_kernel
        return merged

    def merge_dilated_branches(self):
        """Merge all parallel branches into a single large-kernel DW conv.
        Call this before switching to deploy/inference mode."""
        if not hasattr(self, 'origin_bn'):
            return  # already merged
        origin_k, origin_b = self._fuse_bn(self.lk_origin, self.origin_bn)
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = getattr(self, 'dil_conv_k{}_{}'.format(k, r))
            bn = getattr(self, 'dil_bn_k{}_{}'.format(k, r))
            branch_k, branch_b = self._fuse_bn(conv, bn)
            origin_k = self._merge_dilated_into_large_kernel(
                origin_k, branch_k, r)
            origin_b = origin_b + branch_b

        merged_conv = nn.Conv2D(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            groups=self.channels,
            bias_attr=True)
        merged_conv.weight.set_value(origin_k)
        merged_conv.bias.set_value(origin_b)
        self.lk_origin = merged_conv
        self.deploy = True

        delattr(self, 'origin_bn')
        for k, r in zip(self.kernel_sizes, self.dilates):
            delattr(self, 'dil_conv_k{}_{}'.format(k, r))
            delattr(self, 'dil_bn_k{}_{}'.format(k, r))


class DilatedReparamConv(nn.Layer):
    """
    A drop-in replacement for standard Conv2D (in_ch → out_ch, large kernel)
    using DilatedReparamBlock (depthwise) + 1x1 pointwise convolution.

    Architecture:
      input(in_ch) → DilatedReparamBlock(in_ch, DW, kernel_size) → 1x1 Conv(in_ch→out_ch) → BN

    This decomposition replaces a single large standard conv with DW + PW,
    drastically reducing parameters while maintaining the large receptive field.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=9,
                 deploy=False,
                 **kwargs):
        super(DilatedReparamConv, self).__init__()
        weight_attr = paddle.nn.initializer.KaimingUniform() if 'weight_attr' not in kwargs else kwargs['weight_attr']
        self.dw = DilatedReparamBlock(
            channels=in_channels, kernel_size=kernel_size, deploy=deploy)
        self.pw = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.bn = nn.BatchNorm2D(out_channels)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return x


class CoordAtt(nn.Layer):
    """Coordinate Attention (Hou et al., CVPR 2021).

    Encodes channel relationships and long-range spatial dependencies
    via separate H-direction and W-direction average pooling, then
    generates two 1-D attention maps that are broadcast-multiplied back.

    Reference: https://github.com/houqb/CoordAttention
    """

    def __init__(self, channels, reduction=32):
        super().__init__()
        mid = max(8, channels // reduction)
        self.conv1 = nn.Conv2D(channels, mid, 1)
        self.bn1 = nn.BatchNorm2D(mid)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2D(mid, channels, 1)
        self.conv_w = nn.Conv2D(mid, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        # Encode position along H and W separately
        x_h = F.adaptive_avg_pool2d(x, (H, 1))             # (B, C, H, 1)
        x_w = F.adaptive_avg_pool2d(x, (1, W))              # (B, C, 1, W)
        x_w = x_w.transpose([0, 1, 3, 2])                   # (B, C, W, 1)

        # Shared bottleneck transform
        y = paddle.concat([x_h, x_w], axis=2)               # (B, C, H+W, 1)
        y = self.act(self.bn1(self.conv1(y)))                # (B, mid, H+W, 1)

        # Split and generate separate attention maps
        x_h, x_w = y[:, :, :H, :], y[:, :, H:, :]
        x_w = x_w.transpose([0, 1, 3, 2])                   # (B, mid, 1, W)
        a_h = F.sigmoid(self.conv_h(x_h))                    # (B, C, H, 1)
        a_w = F.sigmoid(self.conv_w(x_w))                    # (B, C, 1, W)

        return x * a_h * a_w


class GPAN(nn.Layer):
    def __init__(self, in_channels, out_channels, reduction=32, **kwargs):
        super(GPAN, self).__init__()
        weight_attr = paddle.nn.initializer.KaimingUniform()
        ch2, ch3, ch4, ch5 = in_channels
        oc = out_channels

        # 1×1 projections: only reduce when backbone ch > oc, else Identity
        self.proj_c2 = nn.Conv2D(ch2, oc, 1, weight_attr=ParamAttr(
            initializer=weight_attr), bias_attr=False) if ch2 > oc else nn.Identity()
        self.proj_c3 = nn.Conv2D(ch3, oc, 1, weight_attr=ParamAttr(
            initializer=weight_attr), bias_attr=False) if ch3 > oc else nn.Identity()
        self.proj_c4 = nn.Conv2D(ch4, oc, 1, weight_attr=ParamAttr(
            initializer=weight_attr), bias_attr=False) if ch4 > oc else nn.Identity()
        self.proj_c5 = nn.Conv2D(ch5, oc, 1, weight_attr=ParamAttr(
            initializer=weight_attr), bias_attr=False) if ch5 > oc else nn.Identity()

        # Effective channels after projection
        pc2 = oc if ch2 > oc else ch2
        pc3 = oc if ch3 > oc else ch3
        pc4 = oc if ch4 > oc else ch4
        pc5 = oc if ch5 > oc else ch5

        self.Down_c4 = AVG()
        self.Down_c3 = AVG()
        self.Down_c2 = AVG()

        self.Up_c5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.Up_c4 = nn.Upsample(scale_factor=2, mode='nearest')

        # Phase 1: concat channels depend on actual projected sizes
        cat_c5 = pc4 + pc5
        cat_c4 = pc3 + pc4 + pc5 + oc
        cat_c3 = pc2 + pc3 + pc4 + oc
        cat_c2 = pc2 + oc

        self.attn_c5 = CoordAtt(cat_c5, reduction)
        self.attn_c4 = CoordAtt(cat_c4, reduction)
        self.attn_c3 = CoordAtt(cat_c3, reduction)
        self.attn_c2 = CoordAtt(cat_c2, reduction)
        self.fconv_c5 = DilatedReparamConv(cat_c5, oc, kernel_size=9)
        self.fconv_c4 = DilatedReparamConv(cat_c4, oc, kernel_size=7)
        self.fconv_c3 = DilatedReparamConv(cat_c3, oc, kernel_size=5)
        self.fconv_c2 = DilatedReparamConv(cat_c2, oc, kernel_size=5)

        self.Up_f5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.Up_f4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.Up_f3 = nn.Upsample(scale_factor=2, mode='nearest')

        self.Down_f3 = AVG()
        self.Down_f4 = AVG()
        self.Down_p3 = nn.Conv2D(oc, oc, 3, 2, 1,
                                 weight_attr=ParamAttr(initializer=weight_attr))
        self.Down_p4 = nn.Conv2D(oc, oc, 3, 2, 1,
                                 weight_attr=ParamAttr(initializer=weight_attr))

        # Phase 2: concat uses projected backbone features (pc*) + oc-channel fused features
        cat_p3 = oc + oc + pc3
        cat_p4 = oc + oc + oc + pc4 + oc
        cat_p5 = oc + oc + pc5 + oc

        self.attn_p3 = CoordAtt(cat_p3, reduction)
        self.attn_p4 = CoordAtt(cat_p4, reduction)
        self.attn_p5 = CoordAtt(cat_p5, reduction)
        self.pconv_f3 = DilatedReparamConv(cat_p3, oc, kernel_size=5)
        self.pconv_f4 = DilatedReparamConv(cat_p4, oc, kernel_size=7)
        self.pconv_f5 = DilatedReparamConv(cat_p5, oc, kernel_size=9)

        self.attn_fuse = CoordAtt(oc * 4, reduction)
        self.fuse_conv = DilatedReparamConv(oc * 4, oc, kernel_size=5)

        self.out_channels = out_channels

    def forward(self, x):
        c2, c3, c4, c5 = x

        # Project backbone features to unified out_channels
        pc2 = self.proj_c2(c2)
        pc3 = self.proj_c3(c3)
        pc4 = self.proj_c4(c4)
        pc5 = self.proj_c5(c5)

        # Phase 1: Top-down FPN — concat → CoordAtt → DilatedReparamConv
        f5 = self.fconv_c5(self.attn_c5(paddle.concat([self.Down_c4(pc4), pc5], 1)))
        up_f5 = self.Up_f5(f5)
        f4 = self.fconv_c4(self.attn_c4(paddle.concat([self.Down_c3(pc3), pc4, self.Up_c5(pc5), up_f5], 1)))
        up_f4 = self.Up_f4(f4)
        f3 = self.fconv_c3(self.attn_c3(paddle.concat([self.Down_c2(pc2), pc3, self.Up_c4(pc4), up_f4], 1)))
        up_f3 = self.Up_f3(f3)
        f2 = self.fconv_c2(self.attn_c2(paddle.concat([pc2, up_f3], 1)))

        # Phase 2: Bottom-up PAN — concat → CoordAtt → DilatedReparamConv
        p3 = self.pconv_f3(self.attn_p3(paddle.concat([up_f4, f3, pc3], 1)))
        p4 = self.pconv_f4(self.attn_p4(paddle.concat([self.Down_f3(f3), up_f5, f4, pc4, self.Down_p3(p3)], 1)))
        p5 = self.pconv_f5(self.attn_p5(paddle.concat([self.Down_f4(f4), f5, pc5, self.Down_p4(p4)], 1)))

        # Upsample all to P2 resolution and concat
        p5 = F.upsample(p5, scale_factor=8, mode="nearest", align_mode=1)
        p4 = F.upsample(p4, scale_factor=4, mode="nearest", align_mode=1)
        p3 = F.upsample(p3, scale_factor=2, mode="nearest", align_mode=1)

        fuse = paddle.concat([p5, p4, p3, f2], axis=1)
        fuse = self.fuse_conv(self.attn_fuse(fuse))
        return fuse

    def merge_reparam_blocks(self):
        """Merge all DilatedReparamBlock branches for inference deployment."""
        for conv in [self.fconv_c5, self.fconv_c4, self.fconv_c3, self.fconv_c2,
                     self.pconv_f3, self.pconv_f4, self.pconv_f5, self.fuse_conv]:
            conv.dw.merge_dilated_branches()


class GFPN(nn.Layer):
    """Top-down only FPN with cross-scale concat and CoordAtt gating.

    Compared to GPAN, this removes the bottom-up PAN path, reducing
    parameters and latency for student / mobile models.

    Architecture:
      backbone → proj(1×1) → cross-scale FPN (4 levels) → upsample+concat → fuse
      Each FPN node: concat multi-scale features → CoordAtt → DilatedReparamConv
    """

    def __init__(self, in_channels, out_channels, reduction=32, **kwargs):
        super(GFPN, self).__init__()
        weight_attr = paddle.nn.initializer.KaimingUniform()
        ch2, ch3, ch4, ch5 = in_channels
        oc = out_channels

        # Conditional 1×1 projections
        self.proj_c2 = nn.Conv2D(ch2, oc, 1, weight_attr=ParamAttr(
            initializer=weight_attr), bias_attr=False) if ch2 > oc else nn.Identity()
        self.proj_c3 = nn.Conv2D(ch3, oc, 1, weight_attr=ParamAttr(
            initializer=weight_attr), bias_attr=False) if ch3 > oc else nn.Identity()
        self.proj_c4 = nn.Conv2D(ch4, oc, 1, weight_attr=ParamAttr(
            initializer=weight_attr), bias_attr=False) if ch4 > oc else nn.Identity()
        self.proj_c5 = nn.Conv2D(ch5, oc, 1, weight_attr=ParamAttr(
            initializer=weight_attr), bias_attr=False) if ch5 > oc else nn.Identity()

        # Effective channels after projection
        pc2 = oc if ch2 > oc else ch2
        pc3 = oc if ch3 > oc else ch3
        pc4 = oc if ch4 > oc else ch4
        pc5 = oc if ch5 > oc else ch5

        # Cross-scale up/down sampling
        self.Down_c4 = AVG()
        self.Down_c3 = AVG()
        self.Down_c2 = AVG()
        self.Up_c5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.Up_c4 = nn.Upsample(scale_factor=2, mode='nearest')

        # Top-down FPN: concat → CoordAtt → DilatedReparamConv
        cat_c5 = pc4 + pc5
        cat_c4 = pc3 + pc4 + pc5 + oc
        cat_c3 = pc2 + pc3 + pc4 + oc
        cat_c2 = pc2 + oc

        self.attn_c5 = CoordAtt(cat_c5, reduction)
        self.attn_c4 = CoordAtt(cat_c4, reduction)
        self.attn_c3 = CoordAtt(cat_c3, reduction)
        self.attn_c2 = CoordAtt(cat_c2, reduction)
        self.fconv_c5 = DilatedReparamConv(cat_c5, oc, kernel_size=9)
        self.fconv_c4 = DilatedReparamConv(cat_c4, oc, kernel_size=7)
        self.fconv_c3 = DilatedReparamConv(cat_c3, oc, kernel_size=5)
        self.fconv_c2 = DilatedReparamConv(cat_c2, oc, kernel_size=5)

        self.Up_f5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.Up_f4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.Up_f3 = nn.Upsample(scale_factor=2, mode='nearest')

        # Final fuse
        self.attn_fuse = CoordAtt(oc * 4, reduction)
        self.fuse_conv = DilatedReparamConv(oc * 4, oc, kernel_size=5)

        self.out_channels = out_channels

    def forward(self, x):
        c2, c3, c4, c5 = x

        pc2 = self.proj_c2(c2)
        pc3 = self.proj_c3(c3)
        pc4 = self.proj_c4(c4)
        pc5 = self.proj_c5(c5)

        # Top-down FPN with cross-scale concat
        f5 = self.fconv_c5(self.attn_c5(paddle.concat([self.Down_c4(pc4), pc5], 1)))
        up_f5 = self.Up_f5(f5)
        f4 = self.fconv_c4(self.attn_c4(paddle.concat([self.Down_c3(pc3), pc4, self.Up_c5(pc5), up_f5], 1)))
        up_f4 = self.Up_f4(f4)
        f3 = self.fconv_c3(self.attn_c3(paddle.concat([self.Down_c2(pc2), pc3, self.Up_c4(pc4), up_f4], 1)))
        up_f3 = self.Up_f3(f3)
        f2 = self.fconv_c2(self.attn_c2(paddle.concat([pc2, up_f3], 1)))

        # Upsample all to P2 resolution and fuse
        f5 = F.upsample(f5, scale_factor=8, mode="nearest", align_mode=1)
        f4 = F.upsample(f4, scale_factor=4, mode="nearest", align_mode=1)
        f3 = F.upsample(f3, scale_factor=2, mode="nearest", align_mode=1)

        fuse = paddle.concat([f5, f4, f3, f2], axis=1)
        fuse = self.fuse_conv(self.attn_fuse(fuse))
        return fuse

    def merge_reparam_blocks(self):
        """Merge all DilatedReparamBlock branches for inference deployment."""
        for conv in [self.fconv_c5, self.fconv_c4, self.fconv_c3, self.fconv_c2,
                     self.fuse_conv]:
            conv.dw.merge_dilated_branches()
