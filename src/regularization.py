import torch
import torch.nn.functional as F
from src.AIDomains.abstract_layers import ReLU, Normalization


def compute_bound_reg(model, eps, max_eps, reference = 0.5, reg_lambda=0.5):
    reg = torch.zeros((), device=model[0][1].weight.device)
    layers = model.get_layers()
    relu_layers = [layer for layer in layers if isinstance(layer, ReLU)]
    first_layer = [layer for layer in layers if not isinstance(layer, Normalization)][0]

    if first_layer.bounds is None:
        return reg

    reg_tightness, reg_std, reg_relu = (reg.clone() for _ in range(3))

    input_radius = ((first_layer.bounds[1] - first_layer.bounds[0]) / 2).mean()
    relu_cnt = len(relu_layers)
    for layer in relu_layers:
        lb, ub = layer.bounds
        center = (ub + lb) / 2
        radius = ((ub - lb) / 2).mean()
        mean_ = center.mean()
        std_ = center.std()            

        reg_tightness += F.relu(reference - input_radius / radius.clamp(min=1e-12)) / reference
        reg_std += F.relu(reference - std_) / reference

        # L_{relu}
        mask_act, mask_inact = lb > 0, ub < 0
        mean_act = (center * mask_act).mean()
        mean_inact = (center * mask_inact).mean()
        delta = (center - mean_)**2
        var_act = (delta * mask_act).sum()
        var_inact = (delta * mask_inact).sum()

        mean_ratio = mean_act / -mean_inact
        var_ratio = var_act / var_inact
        mean_ratio = torch.min(mean_ratio, 1 / mean_ratio.clamp(min=1e-12))
        var_ratio = torch.min(var_ratio, 1 / var_ratio.clamp(min=1e-12))
        reg_relu_ = (F.relu(reference - mean_ratio) + F.relu(reference - var_ratio)) / reference
        if not torch.isnan(reg_relu_) and not torch.isinf(reg_relu_):
            reg_relu += reg_relu_

    reg = (reg_tightness + reg_relu) / relu_cnt
    reg *= reg_lambda * (1 - eps / max_eps)

    return reg


def compute_IBP_reg(model, batch_size, reg_lambda):
    # TODO add option for with masking?
    layers = model.get_layers()
    reg = torch.zeros((), device=model[0][1].weight.device)
    train_cross_relu = 0
    relu_layers = [layer for layer in layers if isinstance(layer, ReLU)]
    for layer in relu_layers:
        lb, ub = layer.bounds
        is_cross = (lb < 0) & (ub > 0)
        train_cross_relu += is_cross.float().sum() / batch_size
        stable_loss = (torch.clamp(-lb, min=0) * torch.clamp(ub, min=0)).sum() / batch_size
        reg += 0.5 * reg_lambda * stable_loss
    return reg
