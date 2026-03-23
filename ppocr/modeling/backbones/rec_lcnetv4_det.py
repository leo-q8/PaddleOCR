# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import Constant, KaimingNormal
from paddle.nn import (
    AdaptiveAvgPool2D,
    BatchNorm2D,
    Conv2D,
    GELU,
    Hardsigmoid,
    Hardswish,
    ReLU,
)
from paddle.regularizer import L2Decay


NET_CONFIG_V4 = {
    "blocks2":
    [[3, 96, 96, 1, True]],
    "blocks3": [
        [3, 96, 96, 1, False],
        [3, 96, 96, 1, False],
    ],
    "blocks4": [
        [3, 96, 192, (2, 1), False],
        [3, 192, 192, 1, True],
        [3, 192, 192, 1, False],
        [3, 192, 192, 1, True],
        [3, 192, 192, 1, False],
        [3, 192, 192, 1, True],
        [3, 192, 192, 1, False],
    ],
    "blocks5": [
        [3, 192, 384, (2, 1), False],
        [3, 384, 384, 1, True],
        [3, 384, 384, 1, False],
    ],
    "blocks6": [],
}


NET_CONFIG_V4_DET = [
    # k, in_c, out_c, s, use_se
    # 轻量版: 48 -> 48 -> 96 -> 192 -> 384
    [3, 48, 48, 1, True],
    [3, 48, 48, 1, False],
    [3, 48, 96, 2, False],
    [3, 96, 96, 1, True],
    [3, 96, 96, 1, False],
    [3, 96, 192, 2, False],
    [3, 192, 192, 1, True],
    [3, 192, 192, 1, False],
    [3, 192, 192, 1, True],
    [3, 192, 192, 1, False],
    [3, 192, 384, 2, False],
    [3, 384, 384, 1, True],
    [3, 384, 384, 1, False],
]

NET_CONFIG_V4_DET_REP = [
    # k, in_c, out_c, s, use_se
    # 与识别对齐版: 96 -> 96 -> 192 -> 384
    [3, 96, 96, 1, True],
    [3, 96, 96, 1, False],
    [3, 96, 96, 1, False],
    [3, 96, 192, 2, False],
    [3, 192, 192, 1, True],
    [3, 192, 192, 1, False],
    [3, 192, 192, 1, True],
    [3, 192, 192, 1, False],
    [3, 192, 192, 1, True],
    [3, 192, 192, 1, False],
    [3, 192, 384, 2, False],
    [3, 384, 384, 1, True],
    [3, 384, 384, 1, False],
]


class Conv2D_BN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, groups=1, bn_weight_init=1.0):
        super().__init__()
        self.add_sublayer(
            "conv",
            Conv2D(
                in_channels, out_channels, kernel_size, stride, padding,
                groups=groups, bias_attr=False
            )
        )
        bn = BatchNorm2D(out_channels)
        if bn_weight_init == 1.0:
            nn.initializer.Constant(1.0)(bn.weight)
        else:
            nn.initializer.Constant(0.0)(bn.weight)
        nn.initializer.Constant(0.0)(bn.bias)
        self.add_sublayer("bn", bn)

    @paddle.no_grad()
    def fuse(self):
        c, bn = self.conv, self.bn
        w = bn.weight / (bn._variance + bn._epsilon) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn._mean * bn.weight / (bn._variance + bn._epsilon) ** 0.5
        m = Conv2D(
            w.shape[1] * c._groups,
            w.shape[0],
            w.shape[2:],
            stride=c._stride,
            padding=c._padding,
            groups=c._groups,
        )
        m.weight.set_value(w)
        m.bias.set_value(b)
        return m


class ConvBNAct(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=1,
        use_act=True,
        lr_mult=1.0,
    ):
        super().__init__()
        self.use_act = use_act
        self.conv = Conv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=padding if isinstance(padding, str) else (kernel_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=False,
        )
        self.bn = BatchNorm2D(
            out_channels,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0), learning_rate=lr_mult),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0), learning_rate=lr_mult),
        )
        if self.use_act:
            self.act = ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.use_act:
            x = self.act(x)
        return x


class StemBlock(nn.Layer):
    def __init__(
        self,
        in_channels=3,
        mid_channels=48,
        out_channels=96,
        lr_mult=1.0,
    ):
        super().__init__()
        self.stem1 = ConvBNAct(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=2,
            use_act=True,
            lr_mult=lr_mult,
        )
        self.stem2a = ConvBNAct(
            in_channels=mid_channels,
            out_channels=mid_channels // 2,
            kernel_size=2,
            stride=1,
            padding="SAME",
            use_act=True,
            lr_mult=lr_mult,
        )
        self.stem2b = ConvBNAct(
            in_channels=mid_channels // 2,
            out_channels=mid_channels,
            kernel_size=2,
            stride=1,
            padding="SAME",
            use_act=True,
            lr_mult=lr_mult,
        )
        self.stem3 = ConvBNAct(
            in_channels=mid_channels * 2,
            out_channels=mid_channels,
            kernel_size=3,
            stride=2,
            use_act=True,
            lr_mult=lr_mult,
        )
        self.stem4 = ConvBNAct(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            use_act=True,
            lr_mult=lr_mult,
        )
        self.pool = nn.MaxPool2D(
            kernel_size=2, stride=1, ceil_mode=True, padding="SAME"
        )

    def forward(self, x):
        x = self.stem1(x)
        x2 = self.stem2a(x)
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = paddle.concat([x1, x2], axis=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class SELayer(nn.Layer):
    def __init__(self, channel, reduction=4, lr_mult=1.0):
        super().__init__()
        self.avg_pool = None

        self.conv1 = Conv2D(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=ParamAttr(learning_rate=lr_mult),
        )
        self.relu = ReLU()
        self.conv2 = Conv2D(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=ParamAttr(learning_rate=lr_mult),
        )
        self.hardsigmoid = Hardsigmoid()

    def forward(self, x):
        identity = x
        x = x.mean(axis=[2, 3], keepdim=True)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        x = paddle.multiply(x=identity, y=x)
        return x


class RepDWConv(nn.Layer):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        padding = (kernel_size - 1) // 2

        self.conv = Conv2D_BN(channels, channels, kernel_size, 1, padding, groups=channels)
        self.conv1 = Conv2D(channels, channels, 1, 1, 0, groups=channels, bias_attr=False)
        self.bn = BatchNorm2D(channels)
        nn.initializer.Constant(1.0)(self.bn.weight)
        nn.initializer.Constant(0.0)(self.bn.bias)

        self.is_repped = False
        self.reparam_conv = None

    def forward(self, x):
        if self.is_repped:
            return self.reparam_conv(x)
        return self.bn(self.conv(x) + self.conv1(x) + x)

    def rep(self, fuse_lab=None):
        if self.is_repped:
            return

        fused_conv = self._fuse_conv()

        padding = (self.kernel_size - 1) // 2
        self.reparam_conv = Conv2D(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=padding,
            groups=self.channels,
        )
        self.reparam_conv.weight.set_value(fused_conv.weight)
        self.reparam_conv.bias.set_value(fused_conv.bias)

        self.__delattr__('conv')
        self.__delattr__('conv1')
        self.__delattr__('bn')

        self.is_repped = True

    @paddle.no_grad()
    def _fuse_conv(self):
        conv = self.conv.fuse()
        conv1 = self.conv1

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight

        pad_size = self.kernel_size // 2
        conv1_w = F.pad(conv1_w, [pad_size, pad_size, pad_size, pad_size])

        identity = F.pad(
            paddle.ones([conv1_w.shape[0], conv1_w.shape[1], 1, 1]),
            [pad_size, pad_size, pad_size, pad_size]
        )

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b

        conv.weight.set_value(final_conv_w)
        conv.bias.set_value(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn._variance + bn._epsilon) ** 0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn._mean) * bn.weight / (bn._variance + bn._epsilon) ** 0.5

        conv.weight.set_value(w)
        conv.bias.set_value(b)
        return conv

    def fuse(self):
        return self._fuse_conv()


class LCNetV4Block(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        dw_size,
        use_se=False,
        lr_mult=1.0,
        expand_ratio=2,
    ):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.has_residual = (in_channels == out_channels and stride == 1)
        self.use_rep_dw = (stride == 1 and in_channels == out_channels)

        if self.use_rep_dw:
            self.token_mixer = nn.Sequential()
            self.token_mixer.add_sublayer("rep_dw", RepDWConv(in_channels, dw_size))
            if use_se:
                self.token_mixer.add_sublayer(
                    "se",
                    SELayer(in_channels, lr_mult=lr_mult)
                )
        else:
            padding = (dw_size - 1) // 2
            self.token_mixer = nn.Sequential()
            self.token_mixer.add_sublayer(
                "dw_conv",
                Conv2D_BN(in_channels, in_channels, dw_size, stride, padding, groups=in_channels)
            )
            if use_se:
                self.token_mixer.add_sublayer(
                    "se",
                    SELayer(in_channels, lr_mult=lr_mult)
                )

        hidden_channels = int(in_channels * expand_ratio)

        compress_bn_init = 0.0 if self.has_residual else 1.0
        self.channel_mixer = nn.Sequential()
        self.channel_mixer.add_sublayer(
            "expand",
            Conv2D_BN(in_channels, hidden_channels, 1, 1, 0)
        )
        self.channel_mixer.add_sublayer("act", GELU())
        self.channel_mixer.add_sublayer(
            "compress",
            Conv2D_BN(hidden_channels, out_channels, 1, 1, 0, bn_weight_init=compress_bn_init)
        )

    def forward(self, x):
        x = self.token_mixer(x)
        if self.has_residual:
            return x + self.channel_mixer(x)
        else:
            return self.channel_mixer(x)

    def rep(self, fuse_lab=None):
        if hasattr(self, 'is_repped') and self.is_repped:
            return
        if self.use_rep_dw and hasattr(self.token_mixer, 'rep_dw'):
            self.token_mixer.rep_dw.rep(fuse_lab=fuse_lab)
        self.is_repped = True


class PPLCNetV4(nn.Layer):
    def __init__(
        self,
        scale=1.0,
        lr_mult_list=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        **kwargs,
    ):
        super().__init__()
        self.scale = scale
        self.lr_mult_list = lr_mult_list
        self.net_config = NET_CONFIG_V4

        assert isinstance(
            self.lr_mult_list, (list, tuple)
        ), "lr_mult_list should be in (list, tuple) but got {}".format(
            type(self.lr_mult_list)
        )
        assert (
            len(self.lr_mult_list) == 6
        ), "lr_mult_list length should be 6 but got {}".format(len(self.lr_mult_list))

        self.conv1 = StemBlock(
            in_channels=3,
            mid_channels=48,
            out_channels=96,
            lr_mult=lr_mult_list[0],
        )
        self._stem_out_channels = 96

        self.blocks2 = self._make_stage("blocks2", 1)
        self.blocks3 = self._make_stage("blocks3", 2)
        self.blocks4 = self._make_stage("blocks4", 3)
        self.blocks5 = self._make_stage("blocks5", 4)
        self.blocks6 = self._make_stage("blocks6", 5)

        self.out_channels = 384

    def _make_stage(self, stage_name, lr_mult_idx):
        blocks = []
        stage_config = self.net_config.get(stage_name, [])

        for config in stage_config:
            k, in_c, out_c, s, se = config
            blocks.append(
                LCNetV4Block(
                    in_channels=in_c,
                    out_channels=out_c,
                    dw_size=k,
                    stride=s,
                    use_se=se,
                    lr_mult=self.lr_mult_list[lr_mult_idx],
                    expand_ratio=2,
                )
            )
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)
        x = self.blocks5(x)
        x = self.blocks6(x)

        if self.training:
            x = F.adaptive_avg_pool2d(x, [1, 40])
        else:
            assert x.shape[2] >= 3, (
                f"Feature height {x.shape[2]} < pool kernel 3. "
                f"Check spatial downsampling config with current stem."
            )
            x = F.avg_pool2d(x, [3, 2])
        return x

    def rep(self, fuse_lab=None):
        for blocks in [self.blocks2, self.blocks3, self.blocks4, self.blocks5, self.blocks6]:
            for block in blocks:
                if hasattr(block, 'rep'):
                    block.rep(fuse_lab=fuse_lab)


class PPLCNetV4_det(nn.Layer):
    def __init__(
        self,
        in_channels=3,
        out_indices=[2, 5, 10, 13],
        **kwargs,
    ):
        super().__init__()
        self.out_indices = out_indices

        self.features = nn.LayerList()

        stem = StemBlock(
            in_channels=in_channels,
            mid_channels=24,
            out_channels=48,
            lr_mult=1.0,
        )
        self.features.append(stem)

        for config in NET_CONFIG_V4_DET:
            k, in_c, out_c, s, se = config
            self.features.append(
                LCNetV4Block(in_c, out_c, s, k, se, expand_ratio=2)
            )

        self.out_channels = []
        for idx in out_indices:
            if idx == 0:
                self.out_channels.append(48)
            else:
                self.out_channels.append(NET_CONFIG_V4_DET[idx - 1][2])

    def forward(self, x):
        outs = []
        for i, f in enumerate(self.features):
            x = f(x)
            if i in self.out_indices:
                outs.append(x)
        return outs

    def rep(self, fuse_lab=None):
        for f in self.features:
            if hasattr(f, 'rep'):
                f.rep(fuse_lab=fuse_lab)


class PPLCNetV4_det_rep(nn.Layer):
    def __init__(
        self,
        in_channels=3,
        out_indices=[3, 10, 13],
        **kwargs,
    ):
        super().__init__()
        self.out_indices = out_indices

        self.features = nn.LayerList()

        stem = StemBlock(
            in_channels=in_channels,
            mid_channels=48,
            out_channels=96,
            lr_mult=1.0,
        )
        self.features.append(stem)

        for config in NET_CONFIG_V4_DET_REP:
            k, in_c, out_c, s, se = config
            self.features.append(
                LCNetV4Block(in_c, out_c, s, k, se, expand_ratio=2)
            )

        self.out_channels = []
        for idx in out_indices:
            if idx == 0:
                self.out_channels.append(96)
            else:
                self.out_channels.append(NET_CONFIG_V4_DET_REP[idx - 1][2])

    def forward(self, x):
        outs = []
        for i, f in enumerate(self.features):
            x = f(x)
            if i in self.out_indices:
                outs.append(x)
        return outs

    def rep(self, fuse_lab=None):
        for f in self.features:
            if hasattr(f, 'rep'):
                f.rep(fuse_lab=fuse_lab)
