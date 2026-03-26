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

import math
import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle import ParamAttr
from ppocr.modeling.backbones.det_mobilenet_v3 import ConvBNLayer


def get_bias_attr(k):
    stdv = 1.0 / math.sqrt(k * 1.0)
    initializer = paddle.nn.initializer.Uniform(-stdv, stdv)
    bias_attr = ParamAttr(initializer=initializer)
    return bias_attr


class Head(nn.Layer):
    def __init__(self, in_channels, kernel_list=[3, 2, 2], **kwargs):
        super(Head, self).__init__()

        self.conv1 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels // 4,
            kernel_size=kernel_list[0],
            padding=int(kernel_list[0] // 2),
            weight_attr=ParamAttr(),
            bias_attr=False)
        self.conv_bn1 = nn.BatchNorm(
            num_channels=in_channels // 4,
            param_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1.0)),
            bias_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1e-4)),
            act='relu')

        self.conv2 = nn.Conv2DTranspose(
            in_channels=in_channels // 4,
            out_channels=in_channels // 4,
            kernel_size=kernel_list[1],
            stride=2,
            weight_attr=ParamAttr(
                initializer=paddle.nn.initializer.KaimingUniform()),
            bias_attr=get_bias_attr(in_channels // 4))
        self.conv_bn2 = nn.BatchNorm(
            num_channels=in_channels // 4,
            param_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1.0)),
            bias_attr=ParamAttr(
                initializer=paddle.nn.initializer.Constant(value=1e-4)),
            act="relu")
        self.conv3 = nn.Conv2DTranspose(
            in_channels=in_channels // 4,
            out_channels=1,
            kernel_size=kernel_list[2],
            stride=2,
            weight_attr=ParamAttr(
                initializer=paddle.nn.initializer.KaimingUniform()),
            bias_attr=get_bias_attr(in_channels // 4), )
        self.is_repped = False

    def forward(self, x, return_f=False):
        x = self.conv1(x)
        x = self.conv_bn1(x)
        x = self.conv2(x)
        x = self.conv_bn2(x)
        if return_f is True:
            f = x
        x = self.conv3(x)
        x = F.sigmoid(x)
        if return_f is True:
            return x, f
        return x

    @paddle.no_grad()
    def rep(self):
        """Fuse Conv+BN and ConvTranspose+BN pairs for deployment."""
        if self.is_repped:
            return

        # conv1 (Conv2D, no bias) + conv_bn1 (BatchNorm, act=relu)
        self.conv1 = self._fuse_conv_bn(self.conv1, self.conv_bn1)
        self.conv_bn1 = nn.ReLU()

        # conv2 (Conv2DTranspose, has bias) + conv_bn2 (BatchNorm, act=relu)
        self.conv2 = self._fuse_convtranspose_bn(self.conv2, self.conv_bn2)
        self.conv_bn2 = nn.ReLU()

        self.is_repped = True

    @staticmethod
    @paddle.no_grad()
    def _fuse_conv_bn(conv, bn):
        """Fuse Conv2D + BatchNorm into Conv2D with bias."""
        gamma = bn.weight
        std = paddle.sqrt(bn._variance + bn._epsilon)
        scale = gamma / std

        w = conv.weight * scale[:, None, None, None]
        b = bn.bias - bn._mean * scale

        fused = nn.Conv2D(
            conv._in_channels, conv._out_channels, conv._kernel_size,
            stride=conv._stride, padding=conv._padding,
            dilation=conv._dilation, groups=conv._groups)
        fused.weight.set_value(w)
        fused.bias.set_value(b)
        return fused

    @staticmethod
    @paddle.no_grad()
    def _fuse_convtranspose_bn(conv, bn):
        """Fuse Conv2DTranspose + BatchNorm into Conv2DTranspose with bias.

        Conv2DTranspose weight shape: [in_ch, out_ch/groups, kH, kW]
        BN scale applies on out_ch, i.e. axis=1.
        """
        gamma = bn.weight
        std = paddle.sqrt(bn._variance + bn._epsilon)
        scale = gamma / std

        # axis=1 for ConvTranspose (output channel dimension)
        w = conv.weight * scale[None, :, None, None]
        b = bn.bias - bn._mean * scale
        if conv.bias is not None:
            b = b + conv.bias * scale

        fused = nn.Conv2DTranspose(
            conv._in_channels, conv._out_channels, conv._kernel_size,
            stride=conv._stride, padding=conv._padding,
            dilation=conv._dilation, groups=conv._groups)
        fused.weight.set_value(w)
        fused.bias.set_value(b)
        return fused


class DBHead(nn.Layer):
    """
    Differentiable Binarization (DB) for text detection:
        see https://arxiv.org/abs/1911.08947
    args:
        params(dict): super parameters for build DB network
    """

    def __init__(self, in_channels, k=50, aux_in_channels=0, lite_head=False, **kwargs):
        super(DBHead, self).__init__()
        self.k = k
        self.is_repped = False
        HeadCls = LiteHead if lite_head else Head
        self.binarize = HeadCls(in_channels, **kwargs)
        self.thresh = HeadCls(in_channels, **kwargs)
        self.aux_in_channels = aux_in_channels

        if aux_in_channels > 0:
            self._aux_upsample_scale = {
                'aux_p4': 4,   # 1/16 -> 1/4
                'aux_p3': 2,   # 1/8  -> 1/4
                'aux_p2': 1,   # 1/4  -> 1/4 (no-op)
            }
            # 每个尺度独立创建 binarize + thresh Head 对
            self.aux_binarize_p4 = HeadCls(aux_in_channels, **kwargs)
            self.aux_thresh_p4 = HeadCls(aux_in_channels, **kwargs)
            self.aux_binarize_p3 = HeadCls(aux_in_channels, **kwargs)
            self.aux_thresh_p3 = HeadCls(aux_in_channels, **kwargs)
            self.aux_binarize_p2 = HeadCls(aux_in_channels, **kwargs)
            self.aux_thresh_p2 = HeadCls(aux_in_channels, **kwargs)

    def step_function(self, x, y):
        return paddle.reciprocal(1 + paddle.exp(-self.k * (x - y)))

    def forward(self, x, targets=None):
        # 兼容 neck 返回 dict（训练）或 tensor（推理）
        if isinstance(x, dict):
            fuse = x['fuse']
            aux_feats = {k: x[k] for k in ('aux_p4', 'aux_p3', 'aux_p2') if k in x}
        else:
            fuse = x
            aux_feats = {}

        shrink_maps = self.binarize(fuse)
        if not self.training:
            return {'maps': shrink_maps}

        threshold_maps = self.thresh(fuse)
        binary_maps = self.step_function(shrink_maps, threshold_maps)
        y = paddle.concat([shrink_maps, threshold_maps, binary_maps], axis=1)
        result = {'maps': y}

        if self.aux_in_channels > 0 and aux_feats:
            for key, feat in aux_feats.items():
                scale = self._aux_upsample_scale[key]
                if scale > 1:
                    feat = F.interpolate(
                        feat, scale_factor=scale,
                        mode='bilinear', align_corners=False)
                level = key[4:]  # 'p4', 'p3', 'p2'
                aux_binarize = getattr(self, 'aux_binarize_' + level)
                aux_thresh_head = getattr(self, 'aux_thresh_' + level)
                aux_shrink = aux_binarize(feat)
                aux_thresh = aux_thresh_head(feat)
                aux_binary = self.step_function(aux_shrink, aux_thresh)
                result['aux_maps_' + level] = paddle.concat(
                    [aux_shrink, aux_thresh, aux_binary], axis=1)

        return result

    def rep(self):
        """Fuse reparam structures in all sub-modules for deployment."""
        if self.is_repped:
            return
        for layer in self.sublayers():
            if isinstance(layer, (Head, LiteHead)):
                layer.rep()
        self.is_repped = True


class LiteHead(nn.Layer):
    """Lightweight Head: upsample + DW smooth, BN-fusible.

    Improvements over Head:
      1. Conv2DTranspose → nearest upsample + DW Conv (faster, no checkerboard)
      2. BatchNorm2D fused into Conv at deploy via rep()

    Structure (train):
      conv1:   Conv2D(in_ch→mid, k=3, pad=1)     → channel reduction
      bn1 + ReLU
      ↑2× nearest upsample
      conv2_dw: DW Conv2D(mid, k=3)               → smooth
      bn2 + ReLU
      ↑2× nearest upsample
      conv3:   Conv2D(mid→1, k=1)                 → predict
      sigmoid

    Structure (deploy, after rep()):
      conv1:   Conv2D 3×3 (BN absorbed, with bias) + ReLU
      ↑2× nearest upsample
      conv2_dw: DW Conv2D 3×3 (BN absorbed, with bias) + ReLU
      ↑2× nearest upsample
      conv3:   Conv2D 1×1 + sigmoid
    """

    def __init__(self, in_channels, kernel_list=[3, 2, 2], **kwargs):
        super(LiteHead, self).__init__()
        mid = in_channels // 4
        self.is_repped = False

        # Stage 1: channel reduction
        self.conv1 = nn.Conv2D(
            in_channels, mid,
            kernel_size=kernel_list[0],
            padding=int(kernel_list[0] // 2),
            weight_attr=ParamAttr(),
            bias_attr=False)
        self.bn1 = nn.BatchNorm2D(mid)

        # Stage 2: 2× upsample + DW 3×3 smooth
        self.conv2_dw = nn.Conv2D(
            mid, mid, 3, padding=1, groups=mid,
            weight_attr=ParamAttr(
                initializer=paddle.nn.initializer.KaimingUniform()),
            bias_attr=False)
        self.bn2 = nn.BatchNorm2D(mid)

        # Stage 3: 2× upsample + 1×1 predict
        self.conv3 = nn.Conv2D(
            mid, 1, 1,
            weight_attr=ParamAttr(
                initializer=paddle.nn.initializer.KaimingUniform()),
            bias_attr=get_bias_attr(mid))

    def forward(self, x, return_f=False):
        x = self.conv1(x)
        if self.is_repped:
            x = F.relu(x)
        else:
            x = F.relu(self.bn1(x))

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv2_dw(x)
        if self.is_repped:
            x = F.relu(x)
        else:
            x = F.relu(self.bn2(x))

        if return_f is True:
            f = x

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv3(x)
        x = F.sigmoid(x)

        if return_f is True:
            return x, f
        return x

    @staticmethod
    @paddle.no_grad()
    def _fuse_conv_bn(conv, bn):
        """Fuse Conv2D + BatchNorm2D into a single Conv2D with bias."""
        gamma = bn.weight
        beta = bn.bias
        mean = bn._mean
        var = bn._variance
        eps = bn._epsilon

        std = paddle.sqrt(var + eps)
        scale = gamma / std

        w = conv.weight * scale.reshape([-1, 1, 1, 1])
        b = beta - mean * scale
        if conv.bias is not None:
            b = b + conv.bias * scale

        fused = nn.Conv2D(
            conv._in_channels, conv._out_channels, conv._kernel_size,
            stride=conv._stride, padding=conv._padding,
            dilation=conv._dilation, groups=conv._groups)
        fused.weight.set_value(w)
        fused.bias.set_value(b)
        return fused

    @paddle.no_grad()
    def rep(self):
        """Fuse all BN into preceding Conv for deployment."""
        if self.is_repped:
            return
        self.conv1 = self._fuse_conv_bn(self.conv1, self.bn1)
        self.conv2_dw = self._fuse_conv_bn(self.conv2_dw, self.bn2)
        del self.bn1
        del self.bn2
        self.is_repped = True


class LocalModule(nn.Layer):
    def __init__(self, in_c, mid_c, use_distance=True):
        super(self.__class__, self).__init__()
        self.last_3 = ConvBNLayer(in_c + 1, mid_c, 3, 1, 1, act='relu')
        self.last_1 = nn.Conv2D(mid_c, 1, 1, 1, 0)

    def forward(self, x, init_map, distance_map):
        outf = paddle.concat([init_map, x], axis=1)
        # last Conv
        out = self.last_1(self.last_3(outf))
        return out


class PFHeadLocal(DBHead):
    def __init__(self, in_channels, k=50, mode='small', **kwargs):
        super(PFHeadLocal, self).__init__(in_channels, k, **kwargs)
        self.mode = mode

        self.up_conv = nn.Upsample(scale_factor=2, mode="nearest", align_mode=1)
        if self.mode == 'large':
            self.cbn_layer = LocalModule(in_channels // 4, in_channels // 4)
        elif self.mode == 'small':
            self.cbn_layer = LocalModule(in_channels // 4, in_channels // 8)

    def forward(self, x, targets=None):
        shrink_maps, f = self.binarize(x, return_f=True)
        base_maps = shrink_maps
        cbn_maps = self.cbn_layer(self.up_conv(f), shrink_maps, None)
        cbn_maps = F.sigmoid(cbn_maps)
        if not self.training:
            return {'maps': 0.5 * (base_maps + cbn_maps), 'cbn_maps': cbn_maps}

        threshold_maps = self.thresh(x)
        binary_maps = self.step_function(shrink_maps, threshold_maps)
        y = paddle.concat([cbn_maps, threshold_maps, binary_maps], axis=1)
        return {'maps': y, 'distance_maps': cbn_maps, 'cbn_maps': binary_maps}
