import torch
from torch import nn
import torch.nn.functional as F


class VanillaUNetUnit(nn.Module):
    """conv -> batchNorm -> ReLU
    -> conv -> batchNorm -> ReLU
    """

    def __init__(self, in_channels, out_channels):
        super(VanillaUNetUnit, self).__init__()

        self.net = nn.Sequential()
        
        self.net.add_module('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        self.net.add_module('batch1', nn.BatchNorm2d(out_channels))
        self.net.add_module('ReLU1', nn.ReLU(inplace=True))

        self.net.add_module('conv2', nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        self.net.add_module('batch2', nn.BatchNorm2d(out_channels))
        self.net.add_module('ReLU2', nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class VanillaUNetDownLayer(nn.Module):
    """| :maxpool 2 * 2
       v 
       |->|->| :VanillaUNetUnit
    """

    def __init__(self, in_channels, out_channels):
        super(VanillaUNetDownLayer, self).__init__()

        self.net = nn.Sequential()

        self.net.add_module('maxpool', nn.MaxPool2d(2))
        self.net.add_module('VanilaUNetUnit', VanillaUNetUnit(in_channels, out_channels))

    def forward(self, x):
        return self.net(x)


class VanillaUNetUpLayer(nn.Module):
    """|->|->| :VanillaUNetUnit
       ^
       | :Upsample
    """

    def __init__(self, in_channels, out_channels, up_sample_params={'scale_factor': 2, 'mode': 'bilinear', 'align_corners': True}):
        super(VanillaUNetUpLayer, self).__init__()

        self.up = nn.Upsample(**up_sample_params)
        self.conv = VanillaUNetUnit(in_channels, out_channels)

    def padding_x1(self, x1, height, width):
        diff_height = height - x1.size()[2]
        diff_width = width - x1.size()[3]
        padding_left = diff_width // 2
        padding_right = diff_width - padding_left
        padding_top = diff_height // 2
        padding_bottom = diff_height - padding_top
        padding_rect = [padding_left, padding_right, padding_top, padding_bottom]
        return F.pad(x1, padding_rect)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.padding_x1(x1, x2.size()[2], x2.size()[3])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class VanillaUNetOutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VanillaUNetOutputLayer, self).__init__()
        self.net = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.net(x)
