import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple, Union
from torch import Tensor

from src.AIDomains.abstract_layers import Normalization, Linear, ReLU, Conv2d, Flatten, GlobalAvgPool2d, AvgPool2d, Upsample, _BatchNorm, Bias, Scale, ResBlock, Sequential
from src.AIDomains.ai_util import get_neg_pos_comp


class DeepPoly:
    def __init__(self, x_l_coef: Optional[Tensor]=None, x_u_coef: Optional[Tensor]=None, x_l_bias: Optional[Tensor]=None,
                 x_u_bias: Optional[Tensor]=None, expr_coef: Optional[Tensor]=None) -> None:
        '''
        expr_coeff is used for the initialization to define the linear expression to be bounded
        '''
        if expr_coef is None and (x_l_coef is None or x_u_coef is None):
            return
        assert expr_coef is None or isinstance(expr_coef, torch.Tensor)
        self.device = x_l_coef.device if expr_coef is None else expr_coef.device

        self.x_l_coef = expr_coef if x_l_coef is None else x_l_coef
        self.x_u_coef = expr_coef if x_u_coef is None else x_u_coef
        self.x_l_bias = torch.tensor(0) if x_l_bias is None else x_l_bias
        self.x_u_bias = torch.tensor(0) if x_u_bias is None else x_u_bias

    def clone(self) -> "DeepPoly":
        return DeepPoly(self.x_l_coef.clone(), self.x_u_coef.clone(), self.x_l_bias.clone(), self.x_u_bias.clone())

    def detach(self) -> "DeepPoly":
        x_l_coef = self.x_l_coef.detach()
        x_u_coef = self.x_u_coef.detach()
        x_l_bias = self.x_l_bias.detach()
        x_u_bias = self.x_u_bias.detach()
        return DeepPoly(x_l_coef, x_u_coef, x_l_bias, x_u_bias)

    def dp_linear(self, weight: Tensor, bias: Tensor) -> "DeepPoly":
        x_l_bias = self.x_l_bias + (0 if bias is None else self.x_l_coef.matmul(bias))
        x_u_bias = self.x_u_bias + (0 if bias is None else self.x_u_coef.matmul(bias))

        x_l_coef = self.x_l_coef.matmul(weight)
        x_u_coef = self.x_u_coef.matmul(weight)

        return DeepPoly(x_l_coef, x_u_coef, x_l_bias, x_u_bias)

    def dp_bias(self, bias: Tensor) -> "DeepPoly":
        view_dim = (1, 1) + (bias.shape)

        x_l_bias = self.x_l_bias + (self.x_l_coef*bias.view(view_dim)).sum(tuple(range(2-self.x_l_coef.dim(),0)))
        x_u_bias = self.x_u_bias + (self.x_u_coef*bias.view(view_dim)).sum(tuple(range(2-self.x_l_coef.dim(),0)))
        return DeepPoly(self.x_l_coef, self.x_u_coef, x_l_bias, x_u_bias)

    def dp_scale(self, scale: Tensor) -> "DeepPoly":
        view_dim = (1, 1) + (scale.shape)
        x_l_coef = self.x_l_coef*scale.view(view_dim)
        x_u_coef = self.x_u_coef*scale.view(view_dim)
        return DeepPoly(x_l_coef, x_u_coef, self.x_l_bias, self.x_u_bias)

    def dp_add(self, other: "DeepPoly") -> "DeepPoly":
        x_l_coef = self.x_l_coef + other.x_l_coef
        x_u_coef = self.x_u_coef + other.x_u_coef
        x_l_bias = self.x_l_bias + other.x_l_bias
        x_u_bias = self.x_u_bias + other.x_u_bias
        return DeepPoly(x_l_coef, x_u_coef, x_l_bias, x_u_bias)

    def dp_global_avg_pool2d(self, preconv_wh: Union[Tensor, torch.Size]) -> "DeepPoly":
        sz = self.x_l_coef.shape
        input_spatial_size = np.prod(preconv_wh[-2:])
        dtype=self.x_l_coef.dtype
        device=self.x_l_coef.device

        x_l_coef = self.x_l_coef * torch.ones((1,1,1,*preconv_wh[-2:]), dtype=dtype, device=device)/input_spatial_size
        x_u_coef = self.x_u_coef * torch.ones((1,1,1,*preconv_wh[-2:]), dtype=dtype, device=device)/input_spatial_size

        return DeepPoly(x_l_coef, x_u_coef, self.x_l_bias, self.x_u_bias)

    def dp_avg_pool2d(self, preconv_wh: Union[Tensor, torch.Size], kernel_size: Union[Tuple[int,int],int],
                      stride: Union[Tuple[int,int],int], padding: Union[Tuple[int,int],int]) -> "DeepPoly":
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(stride, int):
            stride = (stride, stride)
        dtype = self.x_l_coef.dtype
        device = self.x_l_coef.device

        w_padding = (preconv_wh[1] + 2 * padding[0] - kernel_size[0]) % stride[0]
        h_padding = (preconv_wh[2] + 2 * padding[1] - kernel_size[1]) % stride[1]
        output_padding = (w_padding, h_padding)

        sz = self.x_l_coef.shape

        weight = 1/(np.prod(kernel_size)) * torch.ones((preconv_wh[0],1,*kernel_size), dtype=dtype, device=device)

        new_x_l_coef = F.conv_transpose2d(self.x_l_coef.view((sz[0] * sz[1], *sz[2:])), weight, None, stride, padding,
                                           output_padding, preconv_wh[0], 1)
        new_x_u_coef = F.conv_transpose2d(self.x_u_coef.view((sz[0] * sz[1], *sz[2:])), weight, None, stride, padding,
                                           output_padding, preconv_wh[0], 1)
        x_l_coef = new_x_l_coef.view((sz[0], sz[1], *new_x_l_coef.shape[1:]))
        x_u_coef = new_x_u_coef.view((sz[0], sz[1], *new_x_u_coef.shape[1:]))

        return DeepPoly(x_l_coef, x_u_coef, self.x_l_bias, self.x_u_bias)

    def dp_normalize(self, mean: Tensor, sigma: Tensor) -> "DeepPoly":
        req_shape = [1] * self.x_l_coef.dim()
        req_shape[2] = mean.shape[1]

        x_l_bias = self.x_l_bias + (self.x_l_coef * (-mean / sigma).view(req_shape)).view(*self.x_l_coef.size()[:2], -1).sum(2)
        x_u_bias = self.x_u_bias + (self.x_u_coef * (-mean / sigma).view(req_shape)).view(*self.x_u_coef.size()[:2], -1).sum(2)

        x_l_coef = self.x_l_coef / sigma.view(req_shape)
        x_u_coef = self.x_u_coef / sigma.view(req_shape)

        return DeepPoly(x_l_coef, x_u_coef, x_l_bias, x_u_bias)

    def dp_relu(self, bounds: Tuple[Tensor, Tensor], it: int, dp_lambda:Optional[Tensor]=None) -> "DeepPoly":
        x_lb, x_ub = bounds
        device = x_lb.device

        lambda_l = torch.where(x_ub < -x_lb, torch.zeros(x_lb.size(), device=device), torch.ones(x_lb.size(), device=device))
        lambda_u = x_ub / (x_ub - x_lb + 1e-15)

        if dp_lambda is not None:
            if it == 0:
                dp_lambda.data[0] = lambda_l.data
            lambda_l = dp_lambda[0].view(lambda_l.shape)

        # stably inactive
        lambda_l = torch.where(x_ub < 0, torch.zeros(x_lb.size(), device=device), lambda_l)
        lambda_u = torch.where(x_ub < 0, torch.zeros(x_ub.size(), device=device), lambda_u)

        # stably active
        lambda_l = torch.where(x_lb > 0, torch.ones(x_lb.size(), device=device), lambda_l)
        lambda_u = torch.where(x_lb > 0, torch.ones(x_ub.size(), device=device), lambda_u)

        mu_l = torch.zeros(x_lb.size(), device=device)
        mu_u = torch.where((x_lb < 0) & (x_ub > 0), -x_ub * x_lb / (x_ub - x_lb + 1e-15),
                           torch.zeros(x_lb.size(), device=device))  # height of upper bound intersection with y axis

        lambda_l, lambda_u = lambda_l.unsqueeze(1), lambda_u.unsqueeze(1)
        mu_l, mu_u = mu_l.unsqueeze(1), mu_u.unsqueeze(1)

        neg_x_l_coef, pos_x_l_coef = get_neg_pos_comp(self.x_l_coef)
        neg_x_u_coef, pos_x_u_coef = get_neg_pos_comp(self.x_u_coef)

        x_l_coef = pos_x_l_coef * lambda_l + neg_x_l_coef * lambda_u
        new_x_l_bias = pos_x_l_coef * mu_l + neg_x_l_coef * mu_u
        x_u_coef = pos_x_u_coef * lambda_u + neg_x_u_coef * lambda_l
        new_x_u_bias = pos_x_u_coef * mu_u + neg_x_u_coef * mu_l

        if len(new_x_l_bias.size()) == 3:
            new_x_l_bias = new_x_l_bias.sum(2)
            new_x_u_bias = new_x_u_bias.sum(2)
        else:
            new_x_l_bias = new_x_l_bias.sum((2, 3, 4))
            new_x_u_bias = new_x_u_bias.sum((2, 3, 4))

        x_l_bias = self.x_l_bias + new_x_l_bias
        x_u_bias = self.x_u_bias + new_x_u_bias

        return DeepPoly(x_l_coef, x_u_coef, x_l_bias, x_u_bias)

    def dp_conv(self, preconv_wh: Union[Tensor, torch.Size], weight: Tensor, bias: Tensor,
                stride: Union[Tuple[int,int],int], padding: Union[Tuple[int,int],int], groups: int,
                dilation: Union[Tuple[int,int],int]) -> "DeepPoly":
        kernel_wh = weight.shape[-2:]
        w_padding = (preconv_wh[1] + 2 * padding[0] - 1 - dilation[0] * (kernel_wh[0] - 1)) % stride[0]
        h_padding = (preconv_wh[2] + 2 * padding[1] - 1 - dilation[1] * (kernel_wh[1] - 1)) % stride[1]
        output_padding = (w_padding, h_padding)

        sz = self.x_l_coef.shape

        # process reference
        x_l_bias = self.x_l_bias + (0 if bias is None else (self.x_l_coef.sum((3, 4)) * bias).sum(2))
        x_u_bias = self.x_u_bias + (0 if bias is None else (self.x_u_coef.sum((3, 4)) * bias).sum(2))

        new_x_l_coef = F.conv_transpose2d(self.x_l_coef.view((sz[0] * sz[1], *sz[2:])), weight, None, stride, padding,
                                           output_padding, groups, dilation)
        new_x_u_coef = F.conv_transpose2d(self.x_u_coef.view((sz[0] * sz[1], *sz[2:])), weight, None, stride, padding,
                                           output_padding, groups, dilation)
        #F.pad(new_x_l_coef, (0, 0, w_padding, h_padding), "constant", 0)
        x_l_coef = new_x_l_coef.view((sz[0], sz[1], *new_x_l_coef.shape[1:]))
        x_u_coef = new_x_u_coef.view((sz[0], sz[1], *new_x_u_coef.shape[1:]))

        return DeepPoly(x_l_coef, x_u_coef, x_l_bias, x_u_bias)

    def dp_flatten(self, input_size: Union[torch.Size, List[int]]) -> "DeepPoly":
        x_l_coef = self.x_l_coef.view(*self.x_l_coef.size()[:2], *input_size)
        x_u_coef = self.x_u_coef.view(*self.x_u_coef.size()[:2], *input_size)

        return DeepPoly(x_l_coef, x_u_coef, self.x_l_bias, self.x_u_bias)

    def dp_concretize(self, bounds: Optional[Tuple[Tensor]]=None, abs_input: Optional["HybridZonotope"]=None) -> "DeepPoly":
        assert not (bounds is None and abs_input is None)
        if abs_input is not None:
            if abs_input.errors is None:
                bounds = abs_input.concretize()
            else:
                abs_lb = abs_input.flatten().linear(self.x_l_coef.view(-1, abs_input.head.numel()), bias=self.x_l_bias.flatten()).view(self.x_l_bias.shape).concretize()[0]
                abs_ub = abs_input.flatten().linear(self.x_u_coef.view(-1, abs_input.head.numel()), bias=self.x_u_bias.flatten()).view(self.x_l_bias.shape).concretize()[1]
                return abs_lb, abs_ub
        if bounds is None:
            bounds = abs_input.concretize()

        lb_x, ub_x = bounds
        lb_x, ub_x = lb_x.unsqueeze(1), ub_x.unsqueeze(1)

        neg_x_l_coef, pos_x_l_coef = get_neg_pos_comp(self.x_l_coef)
        neg_x_u_coef, pos_x_u_coef = get_neg_pos_comp(self.x_u_coef)

        x_l_bias = self.x_l_bias + (pos_x_l_coef * lb_x + neg_x_l_coef * ub_x).view(lb_x.size()[0], self.x_l_coef.size()[1], -1).sum(2)
        x_u_bias = self.x_u_bias + (pos_x_u_coef * ub_x + neg_x_u_coef * lb_x).view(lb_x.size()[0], self.x_l_coef.size()[1], -1).sum(2)
        return x_l_bias, x_u_bias

    def dp_upsample(self, pre_sample_size:Union[Tensor, torch.Size], mode:str, align_corners:bool):
        sz = self.x_l_coef.shape

        new_x_l_coef = F.interpolate(self.x_l_coef.view((-1, *sz[-3:])), size=pre_sample_size, mode=mode,
                                     align_corners=align_corners)
        new_x_u_coef = F.interpolate(self.x_u_coef.view((-1, *sz[-3:])), size=pre_sample_size, mode=mode,
                                     align_corners=align_corners)

        x_l_coef = new_x_l_coef.view((sz[0], sz[1], *new_x_l_coef.shape[1:]))
        x_u_coef = new_x_u_coef.view((sz[0], sz[1], *new_x_u_coef.shape[1:]))

        return DeepPoly(x_l_coef, x_u_coef, self.x_l_bias, self.x_u_bias)

    def dp_batch_norm(self, current_mean: Tensor, current_var: Tensor, weight: Tensor, bias: Tensor) -> "DeepPoly":
        D = 1e-5
        c = (weight / torch.sqrt(current_var + D))
        b = -current_mean * c + (0 if bias is None else bias)
        view_dim = (1, 1, -1) + (self.x_l_coef.dim()-3)*(1,)

        if self.x_l_coef.dim() == 3: #1d
            x_l_bias = self.x_l_bias + self.x_l_coef.matmul(b)
            x_u_bias = self.x_u_bias + self.x_u_coef.matmul(b)
        elif self.x_l_coef.dim() == 5: #2d
            x_l_bias = self.x_l_bias + (self.x_l_coef*b.view(view_dim)).sum((-1,-2,-3))
            x_u_bias = self.x_u_bias + (self.x_u_coef*b.view(view_dim)).sum((-1,-2,-3))
        else:
            raise NotImplementedError

        x_l_coef = self.x_l_coef*c.view(view_dim)
        x_u_coef = self.x_u_coef*c.view(view_dim)

        return DeepPoly(x_l_coef, x_u_coef, x_l_bias, x_u_bias)

    def dp_res_block(self, residual, downsample, relu_final, it, dp_lambda):
        in_dp_elem = self

        if relu_final is not None:
            in_dp_elem = in_dp_elem.dp_relu(relu_final.bounds, it, dp_lambda["relu_final"] if dp_lambda is not None and relu_final in dp_lambda else None)

        id_dp_elem = DeepPoly(in_dp_elem.x_l_coef, in_dp_elem.x_u_coef)

        res_dp_elem = backprop_dp(residual, in_dp_elem, it, dp_lambda["residual"] if dp_lambda is not None and "residual" in dp_lambda else None)

        if downsample is not None:
            id_dp_elem = backprop_dp(downsample, id_dp_elem, it, dp_lambda["downsample"] if dp_lambda is not None and "downsample" in dp_lambda else None)

        out_dp_elem = id_dp_elem.dp_add(res_dp_elem)

        return out_dp_elem


def backprop_dp(layer, abs_dp_element, it, lambda_layer=None):
    if isinstance(layer,Sequential):
        for j in range(len(layer.layers)-1, -1, -1):
            sub_layer = layer.layers[j]
            abs_dp_element = backprop_dp(sub_layer, abs_dp_element, it, lambda_layer=lambda_layer[j] if lambda_layer is not None and j in lambda_layer else None)
    elif isinstance(layer, Linear):
        abs_dp_element = abs_dp_element.dp_linear(layer.weight, layer.bias)
    elif isinstance(layer, Flatten):
        abs_dp_element = abs_dp_element.dp_flatten(layer.dim)
    elif isinstance(layer, Normalization):
        abs_dp_element = abs_dp_element.dp_normalize(layer.mean, layer.sigma)
    elif isinstance(layer, ReLU):
        abs_dp_element = abs_dp_element.dp_relu(layer.bounds, it, lambda_layer)
    elif isinstance(layer, Conv2d):
        abs_dp_element = abs_dp_element.dp_conv(layer.dim, layer.weight, layer.bias, layer.stride, layer.padding, layer.groups, layer.dilation)
    elif isinstance(layer, GlobalAvgPool2d):
        abs_dp_element = abs_dp_element.dp_global_avg_pool2d(layer.bounds[0].shape)
    elif isinstance(layer, AvgPool2d):
        abs_dp_element = abs_dp_element.dp_avg_pool2d(layer.dim, layer.kernel_size, layer.stride, layer.padding)
    elif isinstance(layer, Upsample):
        abs_dp_element = abs_dp_element.dp_upsample(layer.dim[-2:], layer.mode, layer.align_corners)
    elif isinstance(layer, _BatchNorm):
        abs_dp_element = abs_dp_element.dp_batch_norm(layer.current_mean, layer.current_var, layer.weight, layer.bias)
    elif isinstance(layer, Bias):
        abs_dp_element = abs_dp_element.dp_bias(layer.bias)
    elif isinstance(layer, Scale):
        abs_dp_element = abs_dp_element.dp_scale(layer.scale)
    elif isinstance(layer, ResBlock):
        abs_dp_element = abs_dp_element.dp_res_block(layer.residual, layer.downsample, layer.relu_final, it, lambda_layer)
    else:
        raise RuntimeError(f'Unknown layer type: {type(layer)}')
    return abs_dp_element

def backward_deeppoly(net, layer_idx, abs_dp_element, it, dp_lambda=None, use_intermediate=False, abs_inputs=None):
    x_u_bias, x_l_bias = None, None

    for j in range(layer_idx, -1, -1):
        layer = net.layers[j]
        abs_dp_element = backprop_dp(layer, abs_dp_element, it, dp_lambda[j] if dp_lambda is not None and j in dp_lambda else None)
        if j == 0 or (use_intermediate and layer.bounds is not None):
            x_l_bias_tmp, x_u_bias_tmp = abs_dp_element.dp_concretize(layer.bounds if j > 0 else None, None if j > 0 else abs_inputs)
            if x_u_bias is not None:
                x_l_bias = torch.maximum(x_l_bias, x_l_bias_tmp)
                x_u_bias = torch.minimum(x_u_bias, x_u_bias_tmp)
            else:
                x_l_bias = x_l_bias_tmp
                x_u_bias = x_u_bias_tmp

    return x_l_bias, x_u_bias


def get_layer_sizes(net, x):
    layer_sizes = {}
    for i, layer in enumerate(net.blocks):
        layer_sizes[i] = x.size()
        x = layer(x)
    layer_sizes[i+1] = x.size()
    return layer_sizes


def compute_dp_relu_bounds(net, max_layer_id, abs_input, it, dp_lambda, recompute_bounds=True, use_intermediate=False):
    x = abs_input.head
    device = x.device

    if max_layer_id == 0:
        x_l_bias, x_u_bias = abs_input.concretize()
    else:
        for i, layer in enumerate(net.layers[:max_layer_id]):
            x = layer(x)
            if isinstance(layer, ReLU):
                if layer.bounds is None or recompute_bounds:
                    compute_dp_relu_bounds(net, i, abs_input, it, dp_lambda, use_intermediate=use_intermediate)

        k = int(np.prod(x[0].size()))
        expr_coef = torch.eye(k, device=device).view(-1, *x[0].size()).unsqueeze(0)

        abs_dp_element = DeepPoly(expr_coef=expr_coef)
        x_l_bias, x_u_bias = backward_deeppoly(net, max_layer_id - 1, abs_dp_element, it, dp_lambda, use_intermediate, abs_input)

    net.layers[max_layer_id].update_bounds((x_l_bias, x_u_bias))


def forward_deeppoly(net, abs_input, expr_coef=None, it=0, dp_lambda=None, recompute_bounds=False, use_intermediate=True):
    net.set_dim(abs_input.concretize()[0][0:1])
    x = net(abs_input)

    if recompute_bounds:
        compute_dp_relu_bounds(net, len(net.layers)-1, abs_input, it, dp_lambda, use_intermediate=use_intermediate)

    if expr_coef is None:
        k = int(np.prod(x[0].size()))
        abs_dp_element = DeepPoly(expr_coef=torch.eye(k).view(-1, *x[0].size()).unsqueeze(0).to(abs_input.head.device))
    else:
        abs_dp_element = DeepPoly(expr_coef=expr_coef)

    x_l_bias, x_u_bias = backward_deeppoly(net, len(net.layers) - 1, abs_dp_element, it, dp_lambda, use_intermediate,
                                           abs_input)

    if expr_coef is None:
        x_l_bias = x_l_bias.view(-1, *x.size()[1:])
        x_u_bias = x_u_bias.view(-1, *x.size()[1:])

    return x_l_bias, x_u_bias

