import numpy as np
import torch.nn as nn
from src.AIDomains.concrete_layers import Normalization


class myNet(nn.Module):
    def __init__(self, device, dataset, n_class=10, input_size=32, input_channel=3, conv_widths=None,
                 kernel_sizes=None, linear_sizes=None, depth_conv=None, paddings=None, strides=None,
                 dilations=None, pool=False, net_dim=None, bn=False, bn2=False, max=False, scale_width=True, mean=0, sigma=1):
        super(myNet, self).__init__()
        if kernel_sizes is None:
            kernel_sizes = [3]
        if conv_widths is None:
            conv_widths = [2]
        if linear_sizes is None:
            linear_sizes = [200]
        if paddings is None:
            paddings = [1]
        if strides is None:
            strides = [2]
        if dilations is None:
            dilations = [1]
        if net_dim is None:
            net_dim = input_size

        if len(conv_widths) != len(kernel_sizes):
            kernel_sizes = len(conv_widths) * [kernel_sizes[0]]
        if len(conv_widths) != len(paddings):
            paddings = len(conv_widths) * [paddings[0]]
        if len(conv_widths) != len(strides):
            strides = len(conv_widths) * [strides[0]]
        if len(conv_widths) != len(dilations):
            dilations = len(conv_widths) * [dilations[0]]

        self.n_class=n_class
        self.input_size=input_size
        self.input_channel=input_channel
        self.conv_widths=conv_widths
        self.kernel_sizes=kernel_sizes
        self.paddings=paddings
        self.strides=strides
        self.dilations = dilations
        self.linear_sizes=linear_sizes
        self.depth_conv=depth_conv
        self.net_dim = net_dim
        self.bn=bn
        self.bn2=bn2
        self.max=max

        if dataset == "fashionmnist":
            mean = 0.1307
            sigma = 0.3081
        elif dataset == "cifar10":
            mean = [0.4914, 0.4822, 0.4465]
            sigma = [0.2023, 0.1994, 0.2010]
        elif dataset == "tinyimagenet":
            mean = [0.4802, 0.4481, 0.3975]
            sigma = [0.2302, 0.2265, 0.2262]

        layers = []
        layers += [Normalization((input_channel,input_size,input_size),mean, sigma)]

        N = net_dim
        n_channels = input_channel
        self.dims = [(n_channels,N,N)]

        for width, kernel_size, padding, stride, dilation in zip(conv_widths, kernel_sizes, paddings, strides, dilations):
            if scale_width:
                width *= 16
            N = int(np.floor((N + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))
            layers += [nn.Conv2d(n_channels, int(width), kernel_size, stride=stride, padding=padding, dilation=dilation)]
            if self.bn:
                layers += [nn.BatchNorm2d(int(width))]
            if self.max:
                layers += [nn.MaxPool2d(int(width))]
            layers += [nn.ReLU((int(width), N, N))]
            n_channels = int(width)
            self.dims += 2*[(n_channels,N,N)]

        if depth_conv is not None:
            layers += [nn.Conv2d(n_channels, depth_conv, 1, stride=1, padding=0),
                       nn.ReLU((n_channels, N, N))]
            n_channels = depth_conv
            self.dims += 2*[(n_channels,N,N)]

        if pool:
            layers += [nn.GlobalAvgPool2d()]
            self.dims += 2 * [(n_channels, 1, 1)]
            N=1

        layers += [nn.Flatten()]
        N = n_channels * N ** 2
        self.dims += [(N,)]

        for width in linear_sizes:
            if width == 0:
                continue
            layers += [nn.Linear(int(N), int(width))]
            if self.bn2:
                layers += [nn.BatchNorm1d(int(width))]
            layers += [nn.ReLU(width)]
            N = width
            self.dims+=2*[(N,)]

        layers += [nn.Linear(N, n_class)]
        self.dims+=[(n_class,)]

        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.blocks(x)


class FFNN(myNet):
    def __init__(self, device, dataset, sizes, n_class=10, input_size=32, input_channel=3, net_dim=None):
        super(FFNN, self).__init__(device, dataset, n_class, input_size, input_channel, conv_widths=[],
                                  linear_sizes=sizes, net_dim=net_dim)


def ConvMed_tiny(dataset, bn=False, bn2=False, device="cuda"):
    in_ch, in_dim, n_class = get_dataset_info(dataset)
    return myNet(device, dataset, n_class, in_dim, in_ch, conv_widths=[1,2], kernel_sizes=[5,4],
                 linear_sizes=[50],  strides=[2,2], paddings=[1,1], net_dim=None, bn=bn, bn2=bn2)


def CNN7(dataset, bn, bn2, device="cuda"):
    in_ch, in_dim, n_class = get_dataset_info(dataset)
    return myNet(device, dataset, n_class, in_dim, in_ch,
                                   conv_widths=[4, 4, 8, 8, 8], kernel_sizes=[3, 3, 3, 3, 3],
                                   linear_sizes=[512], strides=[1, 1, 2, 1, 1], paddings=[1, 1, 1, 1, 1],
                                   net_dim=None, bn=bn, bn2=bn2)


def CNN7_narrow(dataset, bn, bn2, device="cuda"):
    in_ch, in_dim, n_class = get_dataset_info(dataset)
    return myNet(device, dataset, n_class, in_dim, in_ch,
                                   conv_widths=[2, 2, 4, 4, 4], kernel_sizes=[3, 3, 3, 3, 3],
                                   linear_sizes=[216], strides=[1, 1, 2, 1, 1], paddings=[1, 1, 1, 1, 1],
                                   net_dim=None, bn=bn, bn2=bn2)


def CNN7_wide(dataset, bn, bn2, device="cuda"):
    in_ch, in_dim, n_class = get_dataset_info(dataset)
    return myNet(device, dataset, n_class, in_dim, in_ch,
                                   conv_widths=[6, 6, 12, 12, 12], kernel_sizes=[3, 3, 3, 3, 3],
                                   linear_sizes=[512], strides=[1, 1, 2, 1, 1], paddings=[1, 1, 1, 1, 1],
                                   net_dim=None, bn=bn, bn2=bn2)


def get_dataset_info(dataset):
    if dataset == "mnist":
        return 1, 28, 10
    elif dataset == "emnist":
        return 1, 28, 10
    elif dataset == "fashionmnist":
        return 1, 28, 10
    if dataset == "svhn":
        return 3, 32, 10
    elif dataset == "cifar10":
        return 3, 32, 10
    elif dataset == "tinyimagenet":
        return 3, 56, 200
    else:
        raise ValueError(f"Dataset {dataset} not available")


Models = {
    'ConvMed_tiny': ConvMed_tiny,
    'CNN7': CNN7,
    'CNN7_narrow': CNN7_narrow,
    'CNN7_wide': CNN7_wide,
}
