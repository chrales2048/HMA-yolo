import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from models.common import *
from models.extra_modules.conv import *
from models.extra_modules.down import *
from models.extra_modules.attention import *
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.autograd import Function

try:
    from timm.models.layers import trunc_normal_
    from timm.models.layers import to_2tuple
    from timm.models import checkpoint_seq
    from timm.models.layers import DropPath
except:
    pass
try:
    from typing import Optional, Union, Sequence
    from einops import rearrange
    from models.extra_modules.ops_dcnv3.modules import DCNv3
except:
    pass


try:
    import numbers
    import pywt
except:
    pass
try:
    from mmengine.model import BaseModule, constant_init
    from mmcv.cnn import ConvModule, build_norm_layer
except:
    pass


__all__ = [ 'GhostInceptionV2c', 'GhostInceptionV2a', 'GhostInceptionV2b', 'C2f_MLKA_Ablation', 'C3_MLKA_Ablatio',  'AFDM',
           'InceptionNeXtBlock' ]



class GhostInceptionV2a(InceptionV2a):
    def __init__(self, c1, c2, s=1, act=True):
        super().__init__(c1, c2, s, act)
        c_ = c2 // 8 * 2
        c_out = c2 - 3*c_
        self.conv1 = Conv(c1, c_out, 1, s=s, act=act)
        self.conv2 = nn.Sequential(GhostConv(c1, c_, 1, s=1, act=act), GhostConv(c_, c_, 3, s=s, act=act))
        self.conv3 = nn.Sequential(GhostConv(c1, c_, 1, s=1, act=act), GhostConv(c_out, c_out, 3, s=s, act=act),
                                   GhostConv(c_out, c_out, 3, s=1, act=act))
        self.pool = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=s, padding=1), GhostConv(c1, c_, 1, s=1, act=act))
class GhostInceptionV2b(InceptionV2b):
    def __init__(self, c1, c2, s=1, act=True):
        super().__init__(c1, c2, s, act)
        c_ = c2 // 8 * 2
        c_out = c2 - 3*c_
        self.conv1 = Conv(c1, c_out, 1, s=s, act=act)
        self.conv2 = nn.Sequential(GhostConv(c1, c_, 1, s=s, act=act), GhostConv(c_, c_, (3, 1), s=1, act=act),
                                   GhostConv(c_, c_, (1, 3), s=1, act=act))
        self.conv3 = nn.Sequential(GhostConv(c1, c_, 1, s=s, act=act), GhostConv(c_, c_, (3, 1), s=1, act=act),
                                   GhostConv(c_, c_, (1, 3), s=1, act=act), GhostConv(c_, c_, (3, 1), s=1, act=act),
                                   GhostConv(c_, c_, (1, 3), s=1, act=act))
        self.pool = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=s, padding=1), GhostConv(c1, c_, 1, s=1, act=act))
class GhostInceptionV2c(InceptionV2c):
    def __init__(self, c1, c2, s=1, act=True):
        super().__init__(c1, c2, s, act)
        c_ = c2 // 12 * 2
        c_out = c2 - 5 * c_
        self.conv1 = Conv(c1, c_out, 1, s=s, act=act)
        self.conv2 = GhostConv(c1, c_, 1, s=s, act=act)
        self.conv2_1 = GhostConv(c_, c_, (3, 1), s=1, act=act)
        self.conv2_2 = GhostConv(c_, c_, (1, 3), s=1, act=act)
        self.conv3 = nn.Sequential(GhostConv(c1, c_, 1, s=1, act=act), GhostConv(c_, c_, 3, s=s, act=act))
        self.conv3_1 = GhostConv(c_, c_, (3, 1), s=1, act=act)
        self.conv3_2 = GhostConv(c_, c_, (1, 3), s=1, act=act)
        self.pool = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=s, padding=1), GhostConv(c1, c_, 1, s=1, act=act))
class BottleneckByMLKA_Ablation(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.attention = RCBv6(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.attention(self.cv2(self.cv1(x))) if self.add else self.attention(self.cv2(self.cv1(x)))


class C2f_MLKA_Ablation(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)

        self.m = nn.ModuleList(
            BottleneckByMLKA_Ablation(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))


class C3_MLKA_Ablatio(C3):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(BottleneckByMLKA_Ablation(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

class AFDM(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super().__init__()

        main = []
        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        main.append(UN(out_channels))



        self.main = nn.Sequential(*main)

    def forward(self, feat):
        return self.main(feat)
