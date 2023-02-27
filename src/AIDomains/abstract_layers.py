import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from functools import reduce
from typing import Optional, List, Tuple, Union

from src.AIDomains.zonotope import HybridZonotope
from src.AIDomains.ai_util import AbstractElement
import src.AIDomains.concrete_layers as concrete_layers


class AbstractModule(nn.Module):
    def __init__(self, save_bounds=True, detach_bounds=True):
        super(AbstractModule, self).__init__()
        self.save_bounds = save_bounds
        self.detach_bounds = detach_bounds
        self.bounds = None
        self.dim = None

    def update_bounds(self, bounds):
        lb, ub = bounds

        if self.detach_bounds:
            lb, ub = lb.detach(), ub.detach()

        if self.dim is not None:
            lb = lb.view(-1, *self.dim)
            ub = ub.view(-1, *self.dim)

        if self.bounds is None:
            self.bounds = (lb, ub)
        else:
            self.bounds = (torch.maximum(lb, self.bounds[0]), torch.minimum(ub, self.bounds[1]))

    def reset_bounds(self):
        self.bounds = None

    def set_detach_bounds(self, detach_bounds):
        self.detach_bounds = detach_bounds

    def reset_dim(self):
        self.dim = None

    def set_dim(self, x):
        self.dim = torch.tensor(x.shape[1:])
        x = self.forward(x)
        if hasattr(self, "track_running_stats"):
            if hasattr(self, "training"):
                assert (not getattr(self, "training")) or (not getattr(self, "track_running_stats")), "Setting dimensions while statistics are tracked can have unintended consequences. Set eval mode."
            else:
                assert not getattr(self, "track_running_stats"), "Setting dimensions while statistics are tracked can have unintended consequences. Set eval mode."
        return x

    def get_lambda(self, filtered=False):
        return None

    def get_crossing(self):
        return 0

    def get_dead(self):
        return 0

    def get_alive(self):
        return 0

    def get_layers(self):
        return self


class Sequential(AbstractModule):
    def __init__(self, *layers):
        super(Sequential, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.dim = None
        self.epoch = None
        self.loss_weights = None
        self.output_dim = None

    @classmethod
    def from_concrete_network(
            cls,
            network: nn.Sequential,
            input_dim: Tuple[int, ...],
            disconnect: Optional[bool] = False
    ) -> "Sequential":
        abstract_layers: List[AbstractModule] = []
        for i, layer in enumerate(network.children()):
            if i == 0:
                current_layer_input_dim = input_dim
            else:
                current_layer_input_dim = abstract_layers[-1].output_dim

            if isinstance(layer, nn.Sequential):
                abstract_layers.append(Sequential.from_concrete_network(layer, current_layer_input_dim, disconnect))
            elif isinstance(layer, nn.Linear):
                abstract_layers.append(Linear.from_concrete_layer(layer, current_layer_input_dim, disconnect))
            elif isinstance(layer, concrete_layers.Bias):
                abstract_layers.append(Bias.from_concrete_layer(layer, current_layer_input_dim, disconnect))
            elif isinstance(layer, concrete_layers.Normalization):
                abstract_layers.append(Normalization.from_concrete_layer(layer, current_layer_input_dim, disconnect))
            elif isinstance(layer, concrete_layers.DeNormalization):
                abstract_layers.append(DeNormalization.from_concrete_layer(layer, current_layer_input_dim, disconnect))
            elif isinstance(layer, nn.ReLU):
                abstract_layers.append(ReLU.from_concrete_layer(layer, current_layer_input_dim, disconnect))
            elif isinstance(layer, nn.Conv2d):
                abstract_layers.append(Conv2d.from_concrete_layer(layer, current_layer_input_dim, disconnect))
            elif isinstance(layer, nn.Flatten):
                abstract_layers.append(Flatten.from_concrete_layer(layer, current_layer_input_dim, disconnect))
            elif isinstance(layer, nn.AvgPool2d):
                abstract_layers.append(AvgPool2d.from_concrete_layer(layer, current_layer_input_dim, disconnect))
            elif isinstance(layer, nn.MaxPool2d):
                abstract_layers.append(MaxPool2d.from_concrete_layer(layer, current_layer_input_dim, disconnect))
            elif isinstance(layer, nn.Identity):
                abstract_layers.append(Identity(current_layer_input_dim, disconnect))
            elif isinstance(layer, nn.BatchNorm1d):
                abstract_layers.append(BatchNorm1d.from_concrete_layer(layer, current_layer_input_dim, disconnect))
            elif isinstance(layer, nn.BatchNorm2d):
                abstract_layers.append(BatchNorm2d.from_concrete_layer(layer, current_layer_input_dim, disconnect))
            elif isinstance(layer, concrete_layers.BasicBlock):
                abstract_layers.append(BasicBlock.from_concrete_layer(layer, current_layer_input_dim, disconnect))
            elif isinstance(layer, concrete_layers.PreActBasicBlock):
                abstract_layers.append(PreActBasicBlock.from_concrete_layer(layer, current_layer_input_dim, disconnect))
            else:
                raise NotImplementedError(f"Unsupported layer type: {type(layer)}")
            abstract_layer = Sequential(*abstract_layers)
            abstract_layer.output_dim = abstract_layers[-1].output_dim
        return abstract_layer

    def forward_between(self, i_from, i_to, x, C=None):
        for i, layer in enumerate(self.layers[i_from:i_to]):
            if isinstance(x, AbstractElement) and layer.save_bounds:
                layer.update_bounds(x.concretize())
            if C is not None and i == i_to-i_from-1 and isinstance(layer, Linear):
                x = layer(x, C)
            elif isinstance(layer,Sequential):
                x = layer(x, C)
            else:
                x = layer(x)
        return x

    def forward_until(self, i, x, C=None):
        return self.forward_between(0, i+1, x, C)

    def forward_from(self, i, x, C=None):
        return self.forward_between(i+1, len(self.layers), x, C)

    def forward(self, x, C=None):
        return self.forward_from(-1, x, C)

    def reset_bounds(self, i_from=0, i_to=-1):
        self.bounds = None
        i_to = i_to+1 if i_to != -1 else len(self.layers)
        for layer in self.layers[i_from:i_to]:
            layer.reset_bounds()

    def set_detach_bounds(self, detach_bounds, i_from=0, i_to=-1):
        self.detach_bounds = detach_bounds
        i_to = i_to + 1 if i_to != -1 else len(self.layers)
        for layer in self.layers[i_from:i_to]:
            layer.set_detach_bounds(detach_bounds)

    def set_use_old_train_stats(self, old_train_stats, i_from=0, i_to=-1):
        i_to = i_to + 1 if i_to != -1 else len(self.layers)
        for layer in self.layers[i_from:i_to]:
            if isinstance(layer, BatchNorm1d) or isinstance(layer, BatchNorm2d) or isinstance(layer, Sequential):
                layer.set_use_old_train_stats(old_train_stats)

    def set_track_running_stats(self, track_running_stats, i_from=0, i_to=-1):
        i_to = i_to + 1 if i_to != -1 else len(self.layers)
        for layer in self.layers[i_from:i_to]:
            if isinstance(layer, BatchNorm2d) or isinstance(layer, BatchNorm1d):
                layer.track_running_stats = track_running_stats

    def reset_dim(self, i_from=0, i_to=-1):
        i_to = i_to+1 if i_to != -1 else len(self.layers)
        for layer in self.layers[i_from:i_to]:
            layer.reset_dim()

    def set_dim(self, x):
        self.dim = torch.tensor(x.shape[1:])
        for layer in self.layers:
            x = layer.set_dim(x)
        self.output_dim = torch.tensor(x.shape[1:])
        return x

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_epoch(self):
        return self.epoch

    def set_loss_weights(self, loss_weights):
        self.loss_weights = loss_weights

    def get_lambda(self, filtered=False):
        lambdas = []
        for layer in self.layers:
            lambda_layer = layer.get_lambda(filtered=filtered)
            if lambda_layer is not None:
                lambdas += lambda_layer
        return lambdas

    def get_crossing(self):
        return sum([layer.get_crossing() for layer in self.layers])

    def get_dead(self):
        return sum([layer.get_dead() for layer in self.layers])

    def get_alive(self):
        return sum([layer.get_alive() for layer in self.layers])

    def get_layers(self):
        layers = []
        for layer in self.layers:
            if isinstance(layer, Sequential):
                layers += layer.get_layers()
            else:
                layers.append(layer.get_layers())
        return layers

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, i):
        return self.layers[i]


class Conv2d(nn.Conv2d, AbstractModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 dim=None):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.dim = dim

    def forward(self, x):
        if isinstance(x, AbstractElement):
            return x.conv2d(self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            return super(Conv2d, self).forward(x)

    @classmethod
    def from_concrete_layer(
        cls, layer: nn.Conv2d, input_dim: Tuple[int, ...], disconnect: Optional[bool] = False
    ) -> "Conv2d":
        abstract_layer = cls(
            layer.in_channels,
            layer.out_channels,
            layer.kernel_size,
            layer.stride,
            layer.padding,
            layer.dilation,
            layer.groups,
            layer.bias is not None,
        )
        if disconnect:
            abstract_layer.weight.data = layer.weight.data.clone()
            if layer.bias is not None:
                abstract_layer.bias.data = layer.bias.data.clone()
        else:
            abstract_layer.weight = layer.weight
            if layer.bias is not None:
                abstract_layer.bias = layer.bias

        abstract_layer.output_dim = abstract_layer.getShapeConv(input_dim)

        return abstract_layer

    def getShapeConv(self, input_dim):
        if len(input_dim) == 4:
            inBs, inChan, inH, inW = input_dim
        else:
            inChan, inH, inW = input_dim
        kH, kW = self.kernel_size

        outH = 1 + int((2 * self.padding[0] + inH - kH) / self.stride[0])
        outW = 1 + int((2 * self.padding[1] + inW - kW) / self.stride[1])
        return ((inBs,) if len(input_dim) == 4 else ()) + (self.out_channels, outH, outW)

class Upsample(nn.Upsample, AbstractModule):
    def __init__(self, size, mode="nearest", align_corners=False, consolidate_errors=False):
        align_corners = None if mode in ["nearest", "area"] else align_corners
        super(Upsample, self).__init__(size=size, mode=mode, align_corners=align_corners)
        self.consolidate_errors = consolidate_errors

    def forward(self, x):
        if isinstance(x, AbstractElement):
            x = x.upsample(size=self.size, mode=self.mode, align_corners=self.align_corners, consolidate_errors=self.consolidate_errors)
        else:
            return super(Upsample, self).forward(x)
        return x


class ReLU(nn.ReLU, AbstractModule):
    def __init__(self, dim: Optional[Tuple]=None) -> None:
        super(ReLU, self).__init__()
        self.deepz_lambda = None if dim is None else nn.Parameter(-torch.ones(dim, dtype=torch.float))
        self.num_unsound = 0
        self.num_crossing = 0
        self.num_dead = 0
        self.num_alive = 0
        self.num_positive = 0
        self.num_errors = 0
        self.scaling_c = None
        #self.is_dead = None

    def get_neurons(self) -> int:
        return reduce(lambda a, b: a * b, self.dim)

    def forward(self, x) -> Union[AbstractElement,Tensor]:
        if isinstance(x, AbstractElement):
            out, deepz_lambda, num_unsound, num_crossing, num_dead, num_alive, num_positive, num_errors, scaling_c = x.relu(self.deepz_lambda, self.bounds)
            if deepz_lambda is not None and (self.deepz_lambda < 0).any():
                self.deepz_lambda.data = deepz_lambda
            self.num_unsound = num_unsound
            self.num_crossing = num_crossing
            self.num_dead = num_dead
            self.num_alive = num_alive
            self.num_positive = num_positive
            self.num_errors = num_errors
            self.scaling_c = scaling_c
            #self.is_dead = is_dead
            #print("relu dim")
            #print(self.dim)
            return out
        else:
            return super(ReLU, self).forward(x)

    @classmethod
    def from_concrete_layer(
        cls, layer: nn.ReLU, input_dim: Tuple[int, ...], disconnect: Optional[bool] = False
    ) -> "ReLU":
        abstract_layer = cls(input_dim)
        abstract_layer.output_dim = input_dim
        return abstract_layer

    def get_relu_states(self):
        return self.num_unsound, self.num_crossing, self.num_dead, self.num_alive, self.num_positive, self.num_errors

    def get_crossing(self):
        return self.num_crossing

    def get_dead(self):
        return self.num_dead

    def get_alive(self):
        return self.num_alive

    def get_scaling_c(self):
        return self.scaling_c

    def get_lambda(self, filtered=False):
        crossing = self.get_crossing_bounds()
        if not filtered or (crossing is None or crossing > 0):
            return [self.deepz_lambda]
        else:
            return None

    def get_crossing_bounds(self):
        if self.bounds is None:
            return None
        else:
            return (self.bounds[0] < 0).__and__(self.bounds[1] > 0).sum()


class MaxPool2d(nn.MaxPool2d, AbstractModule):
    def __init__(self, k:int, s:Optional[int]=None, p:Optional[int]=0, d:Optional[int]=1,dim: Optional[Tuple]=None) -> None:
        super(MaxPool2d, self).__init__(kernel_size=k, stride=s, padding=p, dilation=d)
        self.dim=dim

    def get_neurons(self) -> int:
        return reduce(lambda a, b: a * b, self.dim)

    def forward(self, x) -> Union[AbstractElement,Tensor]:
        if isinstance(x, AbstractElement):
            assert self.padding == 0
            assert self.dilation == 1
            out = x.max_pool2d(self.kernel_size, self.stride)
            return out
        else:
            return super(MaxPool2d, self).forward(x)

    @classmethod
    def from_concrete_layer(
        cls, layer: nn.MaxPool2d, input_dim: Tuple[int, ...], disconnect: Optional[bool] = False
    ) -> "ReLU":
        abstract_layer = cls(layer.kernel_size, layer.stride, layer.padding, layer.dilation, input_dim)
        abstract_layer.output_dim = abstract_layer.getShapeConv(input_dim)

        return abstract_layer

    def getShapeConv(self, input_dim):
        if len(input_dim) == 4:
            inBs, inChan, inH, inW = input_dim
        else:
            inChan, inH, inW = input_dim

        kH = kW = self.kernel_size

        outH = 1 + int((2 * self.padding + inH - kH) / self.stride)
        outW = 1 + int((2 * self.padding + inW - kW) / self.stride)
        return ((inBs,) if len(input_dim) == 4 else ()) + (inChan, outH, outW)

class Identity(nn.Identity, AbstractModule):
    def __init__(self, input_dim: Tuple[int, ...]) -> None:
        super(Identity, self).__init__()
        self.output_dim = input_dim

    def forward(self, x: Union[AbstractElement, Tensor]) -> Union[AbstractElement, Tensor]:
        return x

class Log(AbstractModule):
    def __init__(self, dim:Optional[Tuple]=None):
        super(Log, self).__init__()
        self.deepz_lambda = nn.Parameter(-torch.ones(dim, dtype=torch.float))

    def get_neurons(self) -> int:
        return reduce(lambda a, b: a * b, self.dim)

    def forward(self, x) -> Union[AbstractElement, Tensor]:
        if isinstance(x, AbstractElement):
            out, deepz_lambda = x.log(self.deepz_lambda, self.bounds)
            if self.deepz_lambda is not None and deepz_lambda is not None and (self.deepz_lambda < 0).any():
                self.deepz_lambda.data = deepz_lambda
            return out
        return x.log()

class Exp(AbstractModule):
    def __init__(self, dim:Optional[Tuple]=None):
        super(Exp, self).__init__()
        self.deepz_lambda = nn.Parameter(-torch.ones(dim, dtype=torch.float))

    def get_neurons(self):
        return reduce(lambda a, b: a * b, self.dim)

    def forward(self, x) -> Union[AbstractElement, Tensor]:
        if isinstance(x, AbstractElement):
            out, deepz_lambda = x.exp(self.deepz_lambda, self.bounds)
            if self.deepz_lambda is not None and deepz_lambda is not None and (self.deepz_lambda < 0).any():
                self.deepz_lambda.data = deepz_lambda
            return out
        return x.exp()


class Inv(AbstractModule):
    def __init__(self, dim:Optional[Tuple]=None):
        super(Inv, self).__init__()
        self.deepz_lambda = nn.Parameter(-torch.ones(dim, dtype=torch.float))

    def get_neurons(self):
        return reduce(lambda a, b: a * b, self.dim)

    def forward(self, x) -> Union[AbstractElement, Tensor]:
        if isinstance(x, AbstractElement):
            out, deepz_lambda = x.inv(self.deepz_lambda, self.bounds)
            if self.deepz_lambda is not None and deepz_lambda is not None and (self.deepz_lambda < 0).any():
                self.deepz_lambda.data = deepz_lambda
            return out
        return 1./x


class LogSumExp(AbstractModule):
    def __init__(self, dim=None):
        super(LogSumExp, self).__init__()
        self.dims = dim
        self.exp = Exp(dim)
        self.log = Log(1)
        self.c = None # for MILP verificiation

    def get_neurons(self):
        return reduce(lambda a, b: a * b, self.dim)

    def reset_bounds(self):
        self.log.bounds = None
        self.bounds = None
        self.exp.bounds = None
        self.c = None

    def set_bounds(self, x):
        self.dim = torch.tensor(x.shape[1:])
        exp_sum = self.exp.set_dim(x).sum(dim=1).unsqueeze(dim=1)
        log_sum = self.log.set_dim(exp_sum)
        return log_sum

    def forward(self, x) -> Union[AbstractElement, Tensor]:
        if isinstance(x, AbstractElement):
            max_head = x.max_center().detach()
            self.c = max_head
            x_temp = x - max_head
            if self.save_bounds:
                self.exp.update_bounds(x_temp.concretize())
            exp_sum = self.exp(x_temp).sum(dim=1)
            if self.save_bounds:
                self.log.update_bounds(exp_sum.concretize())
            log_sum = self.log(exp_sum)
            return log_sum+max_head
        max_head = x.max(dim=1)[0].unsqueeze(1)
        self.c = max_head
        x_tmp = x-max_head
        exp_sum = x_tmp.exp().sum(dim=1).unsqueeze(dim=1)
        log_sum = exp_sum.log() + max_head
        return log_sum


class Entropy(AbstractModule):
    def __init__(self, dim:Optional[Tuple]=None, low_mem:bool=False, neg:bool=False):
        super(Entropy, self).__init__()
        self.exp = Exp(dim)
        self.log_sum_exp = LogSumExp(dim)
        self.low_mem = low_mem
        self.out_features = 1
        self.neg = neg

    def get_neurons(self):
        return reduce(lambda a, b: a * b, self.dims)

    def reset_bounds(self):
        self.log_sum_exp.reset_bounds()
        self.bounds = None
        self.exp.bounds = None

    def set_bounds(self, x):
        self.dim = torch.tensor(x.shape[1:])
        log_sum = self.log_sum_exp.set_dim(x)
        softmax = self.exp.set_dim(x - log_sum)
        prob_weighted_act = (softmax * x).sum(dim=1).unsqueeze(dim=1)
        entropy = log_sum - prob_weighted_act
        return entropy

    def forward(self, x:Union[AbstractElement, Tensor]) -> Union[AbstractElement, Tensor]:
        if isinstance(x, HybridZonotope):
            log_sum = self.log_sum_exp(x)
            x_temp = x.add(-log_sum, shared_errors=None if x.errors is None else x.errors.size(0))
            if self.save_bounds:
                self.exp.update_bounds(x_temp.concretize())
            softmax = self.exp(x_temp)
            prob_weighted_act = softmax.prod(x, None if x.errors is None else x.errors.size(0), low_mem=self.low_mem).sum(dim=1)
            entropy = log_sum.add(-prob_weighted_act, shared_errors=None if log_sum.errors is None else log_sum.errors.size(0))
            return entropy * torch.FloatTensor([1-2*self.neg]).to(entropy.head.device)
        log_sum = self.log_sum_exp(x)
        softmax = (x-log_sum).exp()
        prob_weighted_act = (softmax*x).sum(dim=1).unsqueeze(dim=1)
        entropy = log_sum - prob_weighted_act
        return entropy * (1-2*self.neg)


class Flatten(AbstractModule):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x: Union[AbstractElement, Tensor]) -> Union[AbstractElement, Tensor]:
        return x.view((x.size()[0], -1))

    @classmethod
    def from_concrete_layer(
        cls, layer: nn.Flatten, input_dim: Tuple[int, ...], disconnect: Optional[bool] = False
    ) -> "Flatten":
        abstract_layer = cls()
        abstract_layer.output_dim = (input_dim[0], np.prod(input_dim[1:]))
        return abstract_layer


class Linear(nn.Linear, AbstractModule):
    def __init__(self, in_features:int, out_features:int, bias:bool=True) -> None:
        super(Linear, self).__init__(in_features, out_features, bias)

    def forward(self, x:Union[AbstractElement, Tensor], C:Optional[Tensor] = None) -> Union[AbstractElement, Tensor]:
        if isinstance(x, AbstractElement):
            return x.linear(self.weight, self.bias, C)
        if C is None:
            return super(Linear, self).forward(x)
        else:
            return torch.bmm(C, super(Linear, self).forward(x).unsqueeze(-1)).squeeze(-1)

    @classmethod
    def from_concrete_layer(
        cls, layer: nn.Linear, input_dim: Tuple[int, ...], disconnect: Optional[bool] = False
    ) -> "Linear":
        abstract_layer = cls(
            layer.in_features,
            layer.out_features,
            layer.bias is not None,
        )

        if disconnect:
            abstract_layer.weight.data = layer.weight.data.clone()
            if layer.bias is not None:
                abstract_layer.bias.data = layer.bias.data.clone()
        else:
            abstract_layer.weight = layer.weight
            if layer.bias is not None:
                abstract_layer.bias = layer.bias

        abstract_layer.output_dim = input_dim[:-1] + (layer.out_features,)

        return abstract_layer


class _BatchNorm(nn.modules.batchnorm._BatchNorm, AbstractModule):
    def __init__(self, out_features:int, dimensions:int, affine:bool=False, momentum=0.1):
        super(_BatchNorm, self).__init__(out_features, affine=affine, momentum=momentum)
        # self.running_mean = None
        # self.running_var = None
        self.current_mean = None
        self.current_var = None
        self.affine = affine
        self.use_old_train_stats = False
        if not self.affine:
            self.weight = nn.Parameter(torch.ones(1))
            self.bias = nn.Parameter(torch.zeros(1))
        if dimensions == 1:
            self.mean_dim = [0]
            self.view_dim = (1, -1)
        if dimensions == 2:
            self.mean_dim = [0, 2, 3]
            self.view_dim = (1, -1, 1, 1)

    def set_use_old_train_stats(self, old_train_stats):
        self.use_old_train_stats = old_train_stats

    def forward(self, x:Union[AbstractElement, Tensor]) -> Union[AbstractElement, Tensor]:
        if isinstance(x, AbstractElement):
            return x.batch_norm(self)
        if self.training:
            if self.use_old_train_stats:
                view_dim_list = [1, -1] + (x.dim() - 2) * [1]
                c = (self.weight / torch.sqrt(self.current_var + self.eps))
                b = (-self.current_mean * c + self.bias)
                output = x * c.view(*view_dim_list) + b.view(*view_dim_list)
                return output
            self.current_mean = x.mean(dim=self.mean_dim)
            self.current_var = x.var(unbiased=False, dim=self.mean_dim)
        else:
            self.current_mean = self.running_mean.data
            self.current_var = self.running_var.data
        output = F.batch_norm(x, self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None, self.weight, self.bias, self.training, self.momentum, self.eps)
        return output

    @classmethod
    def from_concrete_layer(
        cls, layer: Union[nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d], input_dim: Tuple[int, ...], disconnect: Optional[bool] = False
    ) -> "_BatchNorm":
        abstract_layer = cls(
            layer.num_features,
            layer.affine,
        )
        if disconnect:
            abstract_layer.running_var.data = layer.running_var.data.clone()
            abstract_layer.running_mean.data = layer.running_mean.data.clone()
            if layer.affine:
                abstract_layer.weight.data = layer.weight.data.clone()
                abstract_layer.bias.data = layer.bias.data.clone()
        else:
            abstract_layer.running_var = layer.running_var
            abstract_layer.running_mean = layer.running_mean
            if layer.affine:
                abstract_layer.weight = layer.weight
                abstract_layer.bias = layer.bias
        abstract_layer.momentum = layer.momentum

        abstract_layer.track_running_stats = layer.track_running_stats
        abstract_layer.num_batches_tracked = layer.num_batches_tracked
        abstract_layer.training = False
        abstract_layer.output_dim = input_dim

        return abstract_layer


class BatchNorm1d(_BatchNorm):
    def __init__(self, out_features:int, affine:bool=False):
        super(BatchNorm1d, self).__init__(out_features, 1, affine)


class BatchNorm2d(_BatchNorm):
    def __init__(self, out_features:int, affine:bool=False):
        super(BatchNorm2d, self).__init__(out_features, 2, affine)


class AvgPool2d(nn.AvgPool2d, AbstractModule):
    def __init__(self, kernel_size:int, stride:Optional[int]=None, padding:int=0):
        super(AvgPool2d, self).__init__(kernel_size, stride, padding)

    def forward(self, x:Union[AbstractElement, Tensor]) -> Union[AbstractElement, Tensor]:
        if isinstance(x, AbstractElement):
            assert self.padding == 0
            return x.avg_pool2d(self.kernel_size, self.stride)
        return super(AvgPool2d, self).forward(x)

    @classmethod
    def from_concrete_layer(
        cls, layer: Union[nn.AvgPool2d], input_dim: Tuple[int, ...], disconnect: Optional[bool] = False
    ) -> "AvgPool2d":
        abstract_layer = cls(
            layer.kernel_size,
            layer.stride,
            layer.padding,
        )
        abstract_layer.output_dim = input_dim

        return abstract_layer


class GlobalAvgPool2d(nn.AdaptiveAvgPool2d, AbstractModule):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__(1)

    def forward(self, x: Union[AbstractElement, Tensor]) -> Union[AbstractElement, Tensor]:
        if isinstance(x, AbstractElement):
            return x.global_avg_pool2d()
        return super(GlobalAvgPool2d, self).forward(x)


class Bias(AbstractModule):
    def __init__(self, bias=0, fixed=False):
        super(Bias, self).__init__()
        self.bias = nn.Parameter(bias*torch.ones(1))
        self.bias.requires_grad_(not fixed)

    def forward(self, x:Union[AbstractElement, Tensor]) -> Union[AbstractElement, Tensor]:
        return x + self.bias

    @classmethod
    def from_concrete_layer(
        cls, layer: Union[concrete_layers.Bias], input_dim: Tuple[int, ...], disconnect: Optional[bool] = False
    ) -> "Bias":
        abstract_layer = cls()
        if disconnect:
            abstract_layer.bias.data = layer.bias.data.clone()
        else:
            abstract_layer.bias = layer.bias
        abstract_layer.output_dim = input_dim
        return abstract_layer


class Scale(AbstractModule):
    def __init__(self, scale=1, fixed=False):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(scale*torch.ones(1))
        self.scale.requires_grad_(not fixed)

    def forward(self, x: Union[AbstractElement, Tensor]) -> Union[AbstractElement, Tensor]:
        return x * self.scale

    @classmethod
    def from_concrete_layer(cls, layer: Union[concrete_layers.Scale], input_dim: Tuple[int, ...], disconnect: Optional[bool] = False) -> "Scale":
        abstract_layer = cls()
        if disconnect:
            abstract_layer.scale.data = layer.scale.data.clone()
        else:
            abstract_layer.scale = layer.scale
        abstract_layer.output_dim = input_dim
        return abstract_layer


class Normalization(AbstractModule):
    def __init__(self, mean, sigma):
        super(Normalization, self).__init__()
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.sigma = nn.Parameter(sigma, requires_grad=False)
        self.mean.requires_grad_(False)
        self.sigma.requires_grad_(False)

    def forward(self, x: Union[AbstractElement, Tensor]) -> Union[AbstractElement, Tensor]:
        target_shape = [1,-1] + (x.dim()-2) * [1]
        # if isinstance(x, AbstractElement):
        #     return x.normalize(self.mean.view(target_shape), self.sigma.view(target_shape))
        return (x - self.mean.view(target_shape)) / self.sigma.view(target_shape)

    @classmethod
    def from_concrete_layer(
        cls, layer: concrete_layers.Normalization, input_dim: Tuple[int, ...], disconnect: Optional[bool] = False
    ) -> "Normalization":
        abstract_layer = cls(layer.mean, layer.std)
        abstract_layer.output_dim = input_dim
        return abstract_layer


class DeNormalization(AbstractModule):
    def __init__(self, mean, sigma):
        super(DeNormalization, self).__init__()
        self.mean = nn.Parameter(mean)
        self.sigma = nn.Parameter(sigma)
        self.mean.requires_grad_(False)
        self.sigma.requires_grad_(False)

    def forward(self, x: Union[AbstractElement, Tensor]) -> Union[AbstractElement, Tensor]:
        target_shape = [1,-1] + (x.dim()-2) * [1]
        # if isinstance(x, AbstractElement):
        #     return x.denormalize(self.mean.view(target_shape), self.sigma.view(target_shape))
        return x * self.sigma.view(target_shape) + self.mean.view(target_shape)

    @classmethod
    def from_concrete_layer(
        cls, layer: concrete_layers.DeNormalization, input_dim: Tuple[int, ...], disconnect: Optional[bool] = False
    ) -> "DeNormalization":
        abstract_layer = cls(layer.mean, layer.std)
        abstract_layer.output_dim = input_dim
        return abstract_layer

class DomainEnforcing(Sequential):
    def __init__(self, input_dim:Union[Tensor, List[int]],
                 mean: Optional[Union[Tensor, List[float], float]] = None,
                 std: Optional[Union[Tensor, List[float], float]] = None) -> "DomainEnforcing":
        layers = []
        if mean is not None or std is not None:
            layers += [DeNormalization(mean, std)]
        layers += [ReLU()]
        layers += [Normalization(torch.tensor([1.]), torch.tensor([-1.]))]
        layers += [ReLU()]
        layers += [DeNormalization(torch.tensor([1.]), torch.tensor([-1.]))]
        if mean is not None or std is not None:
            layers += [Normalization(mean, std)]

        super(DomainEnforcing, self).__init__(*layers)

    @classmethod
    def enforce_domain(cls, x, mean: Optional[Union[Tensor, List[float], float]] = None,
                 std: Optional[Union[Tensor, List[float], float]] = None):
        input_dim = torch.tensor(x.shape).numpy().tolist()
        layer = cls(input_dim, mean, std).to(device=x.device)
        return layer(x)

class ResBlock(AbstractModule):
    def __init__(self, dim, in_planes, planes, stride=1, downsample=None, mode="standard"):
        super(ResBlock, self).__init__()

        self.residual = self.get_residual_layers(mode, in_planes, planes, stride, dim)
        self.downsample = downsample
        self.relu_final = ReLU((planes, dim//stride, dim//stride)) if mode in ["standard"] else None
        self.output_dim = (planes, dim//stride, dim//stride) if mode in ["standard"] else None # TODO check

    def forward(self, x):
        identity = x.clone()
        out = self.residual(x)

        if self.downsample is not None:
            if not isinstance(self.downsample, Sequential):
                if isinstance(x, AbstractElement) and self.downsample.save_bounds:
                    self.downsample.update_bounds(x.concretize(), detach=True)
            identity = self.downsample(x)

        if isinstance(out, HybridZonotope):
            out = out.add(identity, shared_errors=None if identity.errors is None else identity.errors.size(0))
        elif isinstance(out, AbstractElement):
            out = out.add(identity)
        else:
            out += identity
        if self.relu_final is not None:
            if isinstance(out, AbstractElement) and self.relu_final.save_bounds:
                self.relu_final.update_bounds(out.concretize(), detach=True)
            out = self.relu_final(out)
        return out

    def set_dim(self, x):
        self.dim = torch.tensor(x.shape[1:])

        identity = x.clone()
        out = self.residual.set_dim(x)
        if self.downsample is not None:
            out += self.downsample.set_dim(identity)
        else:
            out += identity
        if self.relu_final is not None:
            out = self.relu_final.set_dim(out)
        return out

    def reset_bounds(self):
        self.bounds = None
        if self.downsample is not None:
            self.downsample.reset_bounds()

        if self.relu_final is not None:
            self.relu_final.reset_bounds()

        self.residual.reset_bounds()

    def get_residual_layers(self, mode, in_planes, out_planes, stride, dim):
        if mode == "standard":
            residual = Sequential(
                Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
                BatchNorm2d(out_planes),
                ReLU((out_planes, dim // stride, dim // stride)),
                Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
                BatchNorm2d(out_planes),
            )
        elif mode == "wide":
            residual = Sequential(
                BatchNorm2d(in_planes),
                ReLU((in_planes, dim, dim)),
                Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True),
                BatchNorm2d(out_planes),
                ReLU((out_planes, dim, dim)),
                Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True),
            )
        elif mode == "fixup":
            residual = Sequential(
                Bias(),
                Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
                Bias(),
                ReLU((out_planes, dim // stride, dim // stride)),
                Bias(),
                Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
                Scale(),
                Bias(),
            )
        elif mode =="test":
            residual = Sequential(
                # Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
                BatchNorm2d(out_planes),
                ReLU((out_planes, dim // stride, dim // stride)),
                Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
            )
        else:
            raise RuntimeError(f"Unknown layer mode {mode:%s}")

        return residual


class PreActBasicBlock(Sequential):
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
            layers_b.append(BatchNorm2d(in_planes))
        layers_b.append(ReLU())
        layers_b.append(
            Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=p_1, bias=(not bn)))
        _, _, in_dim = concrete_layers.getShapeConv((in_planes, in_dim, in_dim), (self.in_planes, kernel_size, kernel_size),
                                                    stride=stride, padding=p_1)
        if bn:
            layers_b.append(BatchNorm2d(planes))
        layers_b.append(ReLU())
        layers_b.append(
            Conv2d(planes, self.expansion * planes, kernel_size=kernel_size, stride=1, padding=p_2, bias=(not bn)))
        _, _, in_dim = concrete_layers.getShapeConv((planes, in_dim, in_dim), (self.in_planes, kernel_size, kernel_size), stride=1,
                                                    padding=p_2)
        self.path_b = Sequential(*layers_b)

        layers_a = [Identity(in_planes)]
        # TODO check if this conv needed?
        if stride != 1 or in_planes != self.expansion * planes:
            layers_a.append(Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=(not bn)))
            if bn:
                layers_a.append(BatchNorm2d(self.expansion * planes))
        self.path_a = Sequential(*layers_a)
        self.out_dim = in_dim

    def forward(self, x):
        out_a = self.path_a(x)
        out_b = self.path_b(x)
        if isinstance(x, HybridZonotope):
            out = out_a.add(out_b, shared_errors=None if out_b.errors is None else out_b.errors.size(0))
        elif isinstance(x, AbstractElement):
            out = out_a.add(out_b)
        else:
            out = out_a + out_b
        return out

    @classmethod
    def from_concrete_layer(cls, layer:  concrete_layers.BasicBlock, input_dim: Tuple[int, ...],
                            disconnect: Optional[bool] = False) -> "BasicBlock":
        abstract_layer = cls(layer.in_planes, layer.planes, layer.stride, layer.bn, layer.kernel, layer.in_dim)
        for i in range(len(abstract_layer.path_a)):
            if isinstance(abstract_layer.path_a[i], Conv2d):
                if disconnect:
                    abstract_layer.path_a[i].weight.data = layer.path_a[i].weight.data.clone()
                    if layer.path_a[i].bias is not None:
                        abstract_layer.path_a[i].bias.data = layer.path_a[i].bias.data.clone()
                else:
                    abstract_layer.path_a[i].weight = layer.path_a[i].weight
                    if layer.path_a[i].bias is not None:
                        abstract_layer.path_a[i].bias = layer.path_a[i].bias
                # TODO for batchnorm?
        for i in range(len(abstract_layer.path_b)):
            if isinstance(abstract_layer.path_b[i], Conv2d):
                if disconnect:
                    abstract_layer.path_b[i].weight.data = layer.path_b[i].weight.data.clone()
                    if layer.path_b[i].bias is not None:
                        abstract_layer.path_b[i].bias.data = layer.path_b[i].bias.data.clone()
                else:
                    abstract_layer.path_b[i].weight = layer.path_b[i].weight
                    if layer.path_b[i].bias is not None:
                        abstract_layer.path_b[i].bias = layer.path_b[i].bias

        return abstract_layer

    def reset_bounds(self):
        self.bounds = None
        for i in range(len(abstract_layer.path_a)):
            i.reset_bounds()

        for i in range(len(abstract_layer.path_b)):
            i.reset_bounds()


class BasicBlock(Sequential):
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
        layers_b.append(Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=p_1, bias=(not bn)))
        _,_, in_dim = concrete_layers.getShapeConv((in_planes, in_dim, in_dim), (self.in_planes, kernel_size, kernel_size), stride=stride, padding=p_1)

        if bn:
            layers_b.append(BatchNorm2d(planes))
        layers_b.append(ReLU())
        layers_b.append(Conv2d(planes, self.expansion * planes, kernel_size=kernel_size, stride=1, padding=p_2, bias=(not bn)))
        _,_, in_dim = concrete_layers.getShapeConv((planes, in_dim, in_dim), (self.in_planes, kernel_size, kernel_size), stride=1, padding=p_2)
        if bn:
            layers_b.append(BatchNorm2d(self.expansion * planes))
        self.path_b = Sequential(*layers_b)

        layers_a = [Identity(in_planes)]
        if stride != 1 or in_planes != self.expansion*planes:
            layers_a.append(Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=(not bn)))
            if bn:
                layers_a.append(BatchNorm2d(self.expansion * planes))
        self.path_a = Sequential(*layers_a)
        self.output_dim=in_dim

    def forward(self, x):
        out_a = self.path_a(x)
        out_b = self.path_b(x)
        if isinstance(x, HybridZonotope):
            out = out_a.add(out_b, shared_errors=None if out_b.errors is None else out_b.errors.size(0))
        elif isinstance(x, AbstractElement):
            out = out_a.add(out_b)
        else:
            out = out_a + out_b
        return out

    @classmethod
    def from_concrete_layer(cls, layer:  concrete_layers.BasicBlock, input_dim: Tuple[int, ...],
                            disconnect: Optional[bool] = False) -> "BasicBlock":
        abstract_layer = cls(layer.in_planes, layer.planes, layer.stride, layer.bn, layer.kernel, layer.in_dim)
        for i in range(len(abstract_layer.path_a)):
            if isinstance(abstract_layer.path_a[i], Conv2d):
                if disconnect:
                    abstract_layer.path_a[i].weight.data = layer.path_a[i].weight.data.clone()
                    if layer.path_a[i].bias is not None:
                        abstract_layer.path_a[i].bias.data = layer.path_a[i].bias.data.clone()
                else:
                    abstract_layer.path_a[i].weight = layer.path_a[i].weight
                    if layer.path_a[i].bias is not None:
                        abstract_layer.path_a[i].bias = layer.path_a[i].bias
                # TODO for batchnorm?
        for i in range(len(abstract_layer.path_b)):
            if isinstance(abstract_layer.path_b[i], Conv2d):
                if disconnect:
                    abstract_layer.path_b[i].weight.data = layer.path_b[i].weight.data.clone()
                    if layer.path_b[i].bias is not None:
                        abstract_layer.path_b[i].bias.data = layer.path_b[i].bias.data.clone()
                else:
                    abstract_layer.path_b[i].weight = layer.path_b[i].weight
                    if layer.path_b[i].bias is not None:
                        abstract_layer.path_b[i].bias = layer.path_b[i].bias

        return abstract_layer

    def reset_bounds(self):
        self.bounds = None
        for i in range(len(self.path_a)):
            self.path_a[i].reset_bounds()

        for i in range(len(self.path_b)):
            self.path_b[i].reset_bounds()


"""
class BasicBlock(ResBlock):
    def __init__(self, dim, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__(dim, in_planes, planes, stride=stride, downsample=downsample, mode="standard")

    @classmethod
    def from_concrete_layer(
            cls, layer: concrete_layers.BasicBlock, input_dim: Tuple[int, ...], disconnect: Optional[bool] = False
    ) -> "BasicBlock":
        abstract_layer = cls(layer.out_dim, layer.in_planes, layer.planes, layer.stride)
        return abstract_layer
"""

class TestBlock(ResBlock):
    def __init__(self, dim, in_planes, planes, stride=1, downsample=None):
        super(TestBlock, self).__init__(dim, in_planes, planes, stride=stride, downsample=downsample, mode="test")


class WideBlock(ResBlock):
    def __init__(self, dim, in_planes, planes, stride=1, downsample=None):
        super(WideBlock, self).__init__(dim, in_planes, planes, stride=stride, downsample=downsample, mode="wide")


class FixupBasicBlock(ResBlock):
    def __init__(self, dim, in_planes, planes, stride=1, downsample=None):
        super(FixupBasicBlock, self).__init__(dim, in_planes, planes, stride=stride, downsample=downsample, mode="fixup")

        if downsample is not None:
            self.downsample = Sequential(
                self.residual.layers[0],
                downsample)


def add_bounds(lidx, zono, bounds=None, layer=None):
    lb_new, ub_new = zono.concretize()
    if layer is not None:
        if layer.bounds is not None:
            lb_old, ub_old = layer.bounds
            lb_new, ub_new = torch.max(lb_old, lb_new).detach(), torch.min(ub_old, ub_new).detach()
        layer.bounds = (lb_new, ub_new)
    if bounds is not None:
        bounds[lidx] = (lb_new, ub_new)
        return bounds
