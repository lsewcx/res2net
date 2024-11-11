# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmengine.model import ModuleList, Sequential

from mmpretrain.registry import MODELS
from mmpretrain.models.backbones.resnet import Bottleneck as _Bottleneck
from mmpretrain.models.backbones.resnet import ResNet

class HierarchicalLocalAttention(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=3):
        super(HierarchicalLocalAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, padding=0)
        
        # 局部上下文信息提取的层次化卷积层
        self.local_conv1 = nn.Conv2d(in_channels // reduction, in_channels // reduction, kernel_size=kernel_size, padding=kernel_size//2, groups=in_channels // reduction)
        self.local_conv2 = nn.Conv2d(in_channels // reduction, in_channels // reduction, kernel_size=kernel_size, padding=kernel_size//2, groups=in_channels // reduction)
        self.local_conv3 = nn.Conv2d(in_channels // reduction, in_channels // reduction, kernel_size=kernel_size, padding=kernel_size//2, groups=in_channels // reduction)
        
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        # 动态卷积核生成
        self.dynamic_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        identity = x

        # 动态卷积核生成
        dynamic_kernel = self.dynamic_conv(x)
        dynamic_kernel = self.softmax(dynamic_kernel)

        # 局部上下文信息提取
        out = self.conv1(x)
        out = self.local_conv1(out)
        out = self.local_conv2(out)
        out = self.local_conv3(out)
        out = self.conv2(out)
        out = self.sigmoid(out)

        # 应用动态卷积核
        out = F.conv2d(out, dynamic_kernel, padding=1, groups=x.size(1))

        return identity * out

class Bottle2neck(_Bottleneck):
    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels,
                 scales=4,
                 base_width=26,
                 base_channels=64,
                 stage_type='normal',
                 hla_reduction=16,  # 新增参数
                 hla_kernel_size=3,  # 新增参数
                 **kwargs):
        """Bottle2neck block for Res2Net."""
        super(Bottle2neck, self).__init__(in_channels, out_channels, **kwargs)
        assert scales > 1, 'Res2Net degenerates to ResNet when scales = 1.'

        mid_channels = out_channels // self.expansion
        width = int(math.floor(mid_channels * (base_width / base_channels)))

        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, width * scales, postfix=1)
        self.norm3_name, norm3 = build_norm_layer(
            self.norm_cfg, self.out_channels, postfix=3)

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            self.in_channels,
            width * scales,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)

        if stage_type == 'stage':
            self.pool = nn.AvgPool2d(
                kernel_size=3, stride=self.conv2_stride, padding=1)

        self.convs = ModuleList()
        self.bns = ModuleList()
        for i in range(scales - 1):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(width, width, kernel_size=3, stride=self.conv2_stride, padding=self.dilation, dilation=self.dilation, groups=width, bias=False),  # 深度可分离卷积
                    nn.Conv2d(width, width, kernel_size=1, bias=False),  # 逐点卷积
                    build_norm_layer(self.norm_cfg, width, postfix=i + 1)[1]
                )
            )
            self.bns.append(
                build_norm_layer(self.norm_cfg, width, postfix=i + 1)[1])

        self.conv3 = build_conv_layer(
            self.conv_cfg,
            width * scales,
            self.out_channels,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.hla = HierarchicalLocalAttention(self.out_channels, reduction=hla_reduction, kernel_size=hla_kernel_size)  # 引入Hierarchical Local Attention模块

        self.stage_type = stage_type
        self.scales = scales
        self.width = width
        delattr(self, 'conv2')
        delattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            spx = torch.split(out, self.width, 1)
            sp = self.convs[0](spx[0].contiguous())
            sp = self.relu(self.bns[0](sp))
            out = sp
            for i in range(1, self.scales - 1):
                if self.stage_type == 'stage':
                    sp = spx[i]
                else:
                    sp = sp + spx[i]
                sp = self.convs[i](sp.contiguous())
                sp = self.relu(self.bns[i](sp))
                out = torch.cat((out, sp), 1)

            if self.stage_type == 'normal' and self.scales != 1:
                out = torch.cat((out, spx[self.scales - 1]), 1)
            elif self.stage_type == 'stage' and self.scales != 1:
                out = torch.cat((out, self.pool(spx[self.scales - 1])), 1)

            out = self.conv3(out)
            out = self.norm3(out)

            out = self.hla(out)  # 应用Hierarchical Local Attention模块

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class Res2Layer(Sequential):
    """Res2Layer to build Res2Net style backbone.

    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottle2neck. Defaults to True.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        scales (int): Scales used in Res2Net. Default: 4
        base_width (int): Basic width of each scale. Default: 26
        drop_path_rate (float or np.ndarray): stochastic depth rate.
            Default: 0.
    """

    def __init__(self,
                 block,
                 in_channels,
                 out_channels,
                 num_blocks,
                 stride=1,
                 avg_down=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 scales=4,
                 base_width=26,
                 drop_path_rate=0.0,
                 **kwargs):
        self.block = block

        if isinstance(drop_path_rate, float):
            drop_path_rate = [drop_path_rate] * num_blocks

        assert len(drop_path_rate
                   ) == num_blocks, 'Please check the length of drop_path_rate'

        downsample = None
        if stride != 1 or in_channels != out_channels:
            if avg_down:
                downsample = nn.Sequential(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False),
                    build_conv_layer(
                        conv_cfg,
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=1,
                        bias=False),
                    build_norm_layer(norm_cfg, out_channels)[1],
                )
            else:
                downsample = nn.Sequential(
                    build_conv_layer(
                        conv_cfg,
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=stride,
                        bias=False),
                    build_norm_layer(norm_cfg, out_channels)[1],
                )

        layers = []
        layers.append(
            block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                downsample=downsample,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                scales=scales,
                base_width=base_width,
                stage_type='stage',
                drop_path_rate=drop_path_rate[0],
                **kwargs))
        in_channels = out_channels
        for i in range(1, num_blocks):
            layers.append(
                block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    scales=scales,
                    base_width=base_width,
                    drop_path_rate=drop_path_rate[i],
                    **kwargs))
        super(Res2Layer, self).__init__(*layers)


@MODELS.register_module()
class Res2Net(ResNet):
    """Res2Net backbone.

    A PyTorch implement of : `Res2Net: A New Multi-scale Backbone
    Architecture <https://arxiv.org/pdf/1904.01169.pdf>`_

    Args:
        depth (int): Depth of Res2Net, choose from {50, 101, 152}.
        scales (int): Scales used in Res2Net. Defaults to 4.
        base_width (int): Basic width of each scale. Defaults to 26.
        in_channels (int): Number of input image channels. Defaults to 3.
        num_stages (int): Number of Res2Net stages. Defaults to 4.
        strides (Sequence[int]): Strides of the first block of each stage.
            Defaults to ``(1, 2, 2, 2)``.
        dilations (Sequence[int]): Dilation of each stage.
            Defaults to ``(1, 1, 1, 1)``.
        out_indices (Sequence[int]): Output from which stages.
            Defaults to ``(3, )``.
        style (str): "pytorch" or "caffe". If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer. Defaults to "pytorch".
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv.
            Defaults to True.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottle2neck. Defaults to True.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Defaults to ``dict(type='BN', requires_grad=True)``.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Defaults to False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Defaults to True.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.

    Example:
        >>> from mmpretrain.models import Res2Net
        >>> import torch
        >>> model = Res2Net(depth=50,
        ...                 scales=4,
        ...                 base_width=26,
        ...                 out_indices=(0, 1, 2, 3))
        >>> model.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = model.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 256, 8, 8)
        (1, 512, 4, 4)
        (1, 1024, 2, 2)
        (1, 2048, 1, 1)
    """

    arch_settings = {
        50: (Bottle2neck, (3, 4, 6, 3)),
        101: (Bottle2neck, (3, 4, 23, 3)),
        152: (Bottle2neck, (3, 8, 36, 3))
    }

    def __init__(self,
                 scales=4,
                 base_width=26,
                 style='pytorch',
                 deep_stem=True,
                 avg_down=True,
                 init_cfg=None,
                 **kwargs):
        self.scales = scales
        self.base_width = base_width
        super(Res2Net, self).__init__(
            style=style,
            deep_stem=deep_stem,
            avg_down=avg_down,
            init_cfg=init_cfg,
            **kwargs)

    def make_res_layer(self, **kwargs):
        return Res2Layer(
            scales=self.scales,
            base_width=self.base_width,
            base_channels=self.base_channels,
            **kwargs)