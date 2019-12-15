from torch import nn
from .unet_parts import VanillaUNetUnit, VanillaUNetDownLayer, VanillaUNetUpLayer, VanillaUNetOutputLayer


class VanillaUNet(nn.Module):

    def __init__(self, in_channel_num, out_channel_num):
        super(VanillaUNet, self).__init__()

        self.input_layer = VanillaUNetUnit(in_channel_num, 64)
        self.down1 = VanillaUNetDownLayer(64, 128)
        self.down2 = VanillaUNetDownLayer(128, 256)
        self.down3 = VanillaUNetDownLayer(256, 512)
        self.down4 = VanillaUNetDownLayer(512, 512)
        self.up1 = VanillaUNetUpLayer(1024, 256)
        self.up2 = VanillaUNetUpLayer(512, 128)
        self.up3 = VanillaUNetUpLayer(256, 64)
        self.up4 = VanillaUNetUpLayer(128, 64)
        self.output_layer = VanillaUNetOutputLayer(64, out_channel_num)

    def forward(self, x):
        x1 = self.input_layer(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.output_layer(x)
