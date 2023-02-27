import torch
import torch.nn as nn

class Bias(nn.Module):
    def __init__(self, in_dim=None, bias=None):
        super().__init__()
        assert in_dim is not None or bias is not None
        in_dim = list(bias.shape) if in_dim is None else in_dim
        self.out_dim = in_dim if isinstance(in_dim, list) else [in_dim]
        if bias is not None:
            self.bias = bias
        else:
            self.bias = nn.Parameter(torch.zeros(in_dim))

    def forward(self, x):
        return x + self.bias


class Scale(nn.Module):
    def __init__(self, in_dim=None, scale=None):
        super().__init__()
        assert in_dim is not None
        self.out_dim = in_dim if isinstance(in_dim, list) else [in_dim]
        if scale is not None:
            self.scale = scale
        else:
            self.scale = nn.Parameter(torch.ones(in_dim))

    def forward(self, x):
        return x * self.scale


class Normalization(nn.Module):
    def __init__(self, in_dim=None, mean=None, std=None):
        super().__init__()
        assert in_dim is not None
        self.mean = torch.nn.Parameter(torch.tensor(0.) if mean is None else torch.tensor(mean, dtype=torch.float), requires_grad=False)
        self.std = torch.nn.Parameter(torch.tensor(0.) if std is None else torch.tensor(std, dtype=torch.float), requires_grad=False)

        if len(in_dim) in [3, 4]:
            self.mean.data = self.mean.data.view(1, -1, 1, 1)
            self.std.data = self.std.data.view(1, -1, 1, 1)
        elif len(in_dim) in [1, 2]:
            self.mean.data = self.mean.data.view(1, -1)
            self.std.data = self.std.data.view(1, -1)
        else:
            assert False

        self.out_dim = in_dim if isinstance(in_dim, list) else [in_dim]

    def forward(self, x):
        return (x - self.mean) / self.std


class DeNormalization(nn.Module):
    def __init__(self, in_dim=None, mean=None, std=None):
        super().__init__()
        assert in_dim is not None
        self.mean = torch.nn.Parameter(torch.tensor(0.) if mean is None else torch.tensor(mean, dtype=torch.float), requires_grad=False)
        self.std = torch.nn.Parameter(torch.tensor(0.) if std is None else torch.tensor(std, dtype=torch.float), requires_grad=False)

        if len(in_dim) in [3, 4]:
            self.mean.data = self.mean.data.view(1, -1, 1, 1)
            self.std.data = self.std.data.view(1, -1, 1, 1)
        elif len(in_dim) in [1, 2]:
            self.mean.data = self.mean.data.view(1, -1)
            self.std.data = self.std.data.view(1, -1)
        else:
            assert False

        self.out_dim = in_dim if isinstance(in_dim, list) else [in_dim]

    def forward(self, x):
        return x * self.std + self.mean


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn=True, kernel=3, in_dim=-1):
        super(BasicBlock, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
        self.bn = bn
        self.kernel = kernel
        self.in_dim = in_dim

        kernel_size = kernel
        assert kernel_size in [1,2,3], "kernel not supported!"
        p_1 = 1 if kernel_size > 1 else 0
        p_2 = 1 if kernel_size > 2 else 0

        layers_b = []
        layers_b.append(nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=p_1, bias=(not bn)))
        _,_, in_dim = getShapeConv((in_planes, in_dim, in_dim), (self.in_planes, kernel_size, kernel_size), stride=stride, padding=p_1)

        if bn:
            layers_b.append(nn.BatchNorm2d(planes))
        layers_b.append(nn.ReLU())
        layers_b.append(nn.Conv2d(planes, self.expansion * planes, kernel_size=kernel_size, stride=1, padding=p_2, bias=(not bn)))
        _,_, in_dim = getShapeConv((planes, in_dim, in_dim), (self.in_planes, kernel_size, kernel_size), stride=1, padding=p_2)
        if bn:
            layers_b.append(nn.BatchNorm2d(self.expansion * planes))
        self.path_b = nn.Sequential(*layers_b)

        layers_a = [torch.nn.Identity()]
        if stride != 1 or in_planes != self.expansion*planes:
            layers_a.append(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=(not bn)))
            if bn:
                layers_a.append(nn.BatchNorm2d(self.expansion * planes))
        self.path_a = nn.Sequential(*layers_a)
        self.out_dim=in_dim

    def forward(self, x):
        out = self.path_a(x) + self.path_b(x)
        return out


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn=True, kernel=3, in_dim=-1):
        super(PreActBasicBlock, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
        self.bn = bn
        self.kernel = kernel
        self.in_dim = in_dim

        kernel_size = kernel
        assert kernel_size in [1,2,3], "kernel not supported!"
        p_1 = 1 if kernel_size > 1 else 0
        p_2 = 1 if kernel_size > 2 else 0

        layers_b = []
        if bn:
            layers_b.append(nn.BatchNorm2d(in_planes))
        layers_b.append(nn.ReLU())
        layers_b.append(nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=p_1, bias=(not bn)))
        _,_, in_dim = getShapeConv((in_planes, in_dim, in_dim), (self.in_planes, kernel_size, kernel_size), stride=stride, padding=p_1)
        if bn:
            layers_b.append(nn.BatchNorm2d(planes))
        layers_b.append(nn.ReLU())
        layers_b.append(nn.Conv2d(planes, self.expansion * planes, kernel_size=kernel_size, stride=1, padding=p_2, bias=(not bn)))
        _,_, in_dim = getShapeConv((planes, in_dim, in_dim), (self.in_planes, kernel_size, kernel_size), stride=1, padding=p_2)
        self.path_b = nn.Sequential(*layers_b)

        layers_a = [torch.nn.Identity()]
        # TODO check if this conv needed?
        if stride != 1 or in_planes != self.expansion*planes:
            layers_a.append(nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=(not bn)))
            if bn:
                layers_a.append(nn.BatchNorm2d(self.expansion * planes))
        self.path_a = nn.Sequential(*layers_a)
        self.out_dim=in_dim

    def forward(self, x):
        out = self.path_a(x) + self.path_b(x)
        return out


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def getShapeConv(in_shape, conv_shape, stride = 1, padding = 0):
    inChan, inH, inW = in_shape
    outChan, kH, kW = conv_shape[:3]

    outH = 1 + int((2 * padding + inH - kH) / stride)
    outW = 1 + int((2 * padding + inW - kW) / stride)
    return (outChan, outH, outW)
