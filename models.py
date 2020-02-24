from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn


class WNetDownConvBlock(nn.Module):
    r"""Performs two 3x3 2D convolutions, each followed by a ReLU and batch norm. Ends with a 2D max-pool operation."""

    def __init__(self, in_features: int, out_features: int):
        r"""
        :param in_features: Number of feature channels in the incoming data
        :param out_features: Number of feature channels in the outgoing data
        """
        super(WNetDownConvBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_features, out_features, 3),
            nn.ReLU(),
            nn.BatchNorm2d(out_features),
            nn.ReplicationPad2d(1),
            nn.Conv2d(out_features, out_features, 3),
            nn.ReLU(),
            nn.BatchNorm2d(out_features),
            nn.ReplicationPad2d(1),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: Tensor) -> Tensor:
        """Pushes a set of inputs (x) through the network.

        :param x: Input values
        :return: Module outputs
        """
        feature_map = self.layers(x)
        return self.pool(feature_map), feature_map


class WNetUpConvBlock(nn.Module):
    r"""Performs two 3x3 2D convolutions, each followed by a ReLU and batch norm. Ends with a transposed convolution with a stride of 2 on the last layer. Halves features at first and third convolutions"""

    def __init__(self, in_features: int, mid_features: int, out_features: int):
        r"""
        :param in_features: Number of feature channels in the incoming data
        :param out_features: Number of feature channels in the outgoing data
        """
        super(WNetUpConvBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 3),
            nn.BatchNorm2d(mid_features),
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid_features, mid_features, 3),
            nn.BatchNorm2d(mid_features),
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.ConvTranspose2d(mid_features, out_features, 2, stride=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Pushes a set of inputs (x) through the network.

        :param x: Input values
        :return: Module outputs
        """
        return self.layers(x)


class WNetOutputBlock(nn.Module):
    r"""Performs two 3x3 2D convolutions, each followed by a ReLU and batch Norm.
    Ending with a 1x1 convolution to map features to classes."""

    def __init__(self, in_features: int, num_classes: int):
        r"""
        :param in_features: Number of feature channels in the incoming data
        :param num_classes: Number of feature channels in the outgoing data
        """
        super(WNetOutputBlock, self).__init__()
        mid_features = int(in_features / 2)
        self.layers = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 3),
            nn.BatchNorm2d(mid_features),
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid_features, mid_features, 3),
            nn.BatchNorm2d(mid_features),
            nn.ReLU(),
            nn.ReplicationPad2d(1),

            # 1x1 convolution to map features to classes
            nn.Conv2d(mid_features, num_classes, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Pushes a set of inputs (x) through the network.

        :param x: Input values
        :return: Module outputs
        """
        return self.layers(x)

# TODO: seperable convolutions

class UNetAuto(nn.Module):
    r"""UNet based architecture for image auto encoding"""

    def __init__(self, num_channels: int = 3, num_out_channels: int = 3, max_features: int = 1024):
        r"""
        :param num_channels: Number of channels in the raw image data
        :param num_out_channels: Number of channels in the output data
        """
        super(UNetAuto, self).__init__()
        if max_features not in [1024, 512, 256]:
            print('Max features restricted to [1024, 512, 256]')
            max_features = 1024
        features_4 = max_features // 2
        features_3 = features_4 // 2
        features_2 = features_3 // 2
        features_1 = features_2 // 2

        self.conv_block1 = WNetDownConvBlock(num_channels, features_1)
        self.conv_block2 = WNetDownConvBlock(features_1, features_2)
        self.conv_block3 = WNetDownConvBlock(features_2, features_3)
        self.conv_block4 = WNetDownConvBlock(features_3, features_4)

        self.deconv_block1 = WNetUpConvBlock(features_4, max_features, features_4)
        self.deconv_block2 = WNetUpConvBlock(max_features, features_4, features_3)
        self.deconv_block3 = WNetUpConvBlock(features_4, features_3, features_2)
        self.deconv_block4 = WNetUpConvBlock(features_3, features_2, features_1)

        self.output_block = WNetOutputBlock(features_2, num_out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Pushes a set of inputs (x) through the network.

        :param x: Input values
        :return: Network output Tensor
        """
        # print(f'Block: 0 Curr shape: {x.shape}')
        x, c1 = self.conv_block1(x)
        # print(f'Block: 1 Out shape: {x.shape}; features shape: {c1.shape}')
        x, c2 = self.conv_block2(x)
        # print(f'Block: 2 Out shape: {x.shape}; features shape: {c2.shape}')
        x, c3 = self.conv_block3(x)
        # print(f'Block: 3 Out shape: {x.shape}; features shape: {c3.shape}')
        x, c4 = self.conv_block4(x)
        # print(f'Block: 4 Out shape: {x.shape}; features shape: {c4.shape}')
        d1 = self.deconv_block1(x)
        # print(f'Block: 5 Out shape: {d1.shape}')
        d2 = self.deconv_block2(torch.cat((c4, d1), dim=1))
        # print(f'Block: 6 Out shape: {d2.shape}')
        d3 = self.deconv_block3(torch.cat((c3, d2), dim=1))
        # print(f'Block: 7 Out shape: {d3.shape}')
        d4 = self.deconv_block4(torch.cat((c2, d3), dim=1))
        # print(f'Block: 8 Out shape: {d4.shape}')
        out = self.output_block(torch.cat((c1, d4), dim=1))
        # print(f'Block: 9 Out shape: {out.shape}')

        return out


if __name__ == "__main__":
    model = UNetAuto(max_features=512)
    if torch.cuda.is_available():
        model.to('cuda')
    try:
        from torchsummary import summary
        summary(model, input_size=(3, 224, 256))
        print('torchsummary')
    except:
        print(model)
        print('no torchsummary found: pip install torchsummary')
