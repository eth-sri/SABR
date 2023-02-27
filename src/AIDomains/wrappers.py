import torch
import numpy as np

from src.AIDomains.deeppoly import DeepPoly, backward_deeppoly, forward_deeppoly
from src.AIDomains.ai_util import construct_C

def propagate_abs(net_abs, domain, data_abs, y, get_bounds_only=False):
    net_abs.reset_bounds()
    if get_bounds_only:
        # construct a querry matrix as identity matrix
        bs = y.shape[0]
        device = y.device
        C = -torch.eye(net_abs.output_dim[-1], device=device).repeat(bs, 1, 1)
        C[np.arange(bs), :, y] += 1.
    else:
        # construct a querry matrix of target - adversary class
        C = construct_C(net_abs.output_dim[-1], y)

    if domain == "box_naive":
        out_box = net_abs(data_abs)
        lb = out_box.get_wc_logits(y)
        return lb, y

    if domain == "box":
        out_box = net_abs(data_abs, C=C)
        lb, ub = out_box.concretize()
    elif domain == "deeppoly_box":
        out_box = net_abs(data_abs, C=C)
        lb_box = out_box.concretize()[0]
        abs_dp_element = DeepPoly(expr_coef=C)
        lb, ub = backward_deeppoly(net_abs[0], len(net_abs[0].layers) - 1, abs_dp_element, it=0, dp_lambda=None,
                                   use_intermediate=False,
                                   abs_inputs=data_abs)
        # assert (ub + 1e-5 > lb_box).all(), "Soundnes violation in CIBP prop"
        lb = torch.maximum(lb, lb_box)
    elif domain == "deeppoly":
        lb, ub = forward_deeppoly(net_abs, data_abs, expr_coef=C, recompute_bounds=True, use_intermediate=True)
    else:
        assert False, f"Unknown domain {domain} encountered"

    lb_padded = torch.cat((torch.zeros(size=(lb.size(0), 1), dtype=lb.dtype, device=lb.device), lb), dim=1)
    fake_labels = torch.zeros(size=(lb.size(0),), dtype=torch.int64, device=lb.device)
    if get_bounds_only:
        return lb, ub
    else:
        return -lb_padded, fake_labels