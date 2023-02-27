import numpy as np
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

from src.AIDomains.zonotope import HybridZonotope


def margin_loss(logits, y, y_target=None):
    logit_org = logits.gather(1, y.view(-1, 1))
    if y_target is None:
        y_target = (logits - torch.eye(logits.shape[1])[y].to("cuda") * 9999).argmax(1, keepdim=True)
    else:
        y_target = torch.Tensor([[y_target]]).to(device=y.device).long()
    logit_target = logits.gather(1, y_target)
    loss = -logit_org + logit_target
    loss = loss.view(-1)
    return loss


class step_lr_scheduler:
    def __init__(self, initial_step_size, gamma=0.1, interval=10):
        self.initial_step_size = initial_step_size
        self.gamma = gamma
        self.interval = interval
        self.current_step = 0

    def step(self, k=1):
        self.current_step += k

    def get_lr(self):
        if isinstance(self.interval, int):
            return self.initial_step_size * self.gamma**(np.floor(self.current_step/self.interval))
        else:
            phase = len([x for x in self.interval if self.current_step>=x])
            return self.initial_step_size * self.gamma**(phase)


def adv_whitebox(model, X, y, tau, eps, n_steps=200, step_size=0.2, data_range=(0, 1), loss_function="CE",
                 ODI_num_steps=10, ODI_step_size=1., restarts=1, train=True, rand_init=True, shift=True,
                 early_stopping=True,  y_target=None, dimwise_scaling=False):
    large_box = HybridZonotope.construct_from_noise(x=X, eps=eps, domain="box", data_range=data_range)
    specLB, specUB = large_box.concretize()

    if dimwise_scaling:
        tau_dim = torch.minimum(0.5 * (specUB - specLB), torch.tensor(tau))
    else:
        tau_dim = tau

    device = X.device
    out_X = model(X).detach()
    adex = X.detach().clone()
    adex_found = torch.zeros(X.shape[0], dtype=torch.bool, device=device)
    best_loss = torch.ones(X.shape[0], device=device)*(-np.inf)
    gama_lambda_orig = 1
    loss_FN = attack_loss(loss_function, n_steps, out_X, gama_lambda_orig=gama_lambda_orig, extra_regularization=True)

    with torch.enable_grad():
        for _ in range(restarts):
            if adex_found.all():
                break

            X_pgd = Variable(X.data, requires_grad=True).to(device)
            randVector_ = torch.ones_like(model(X_pgd)).uniform_(-1, 1)
            random_noise = torch.ones_like(X_pgd).uniform_(-0.5, 0.5)*(specUB-specLB) if rand_init else torch.zeros_like(X_pgd)
            X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)
            X_pgd.data[adex_found] = adex.data[adex_found] # retain once we have found and adv example

            lr_scale = torch.max((specUB-specLB)/2)
            lr_scheduler = step_lr_scheduler(step_size, gamma=0.1, interval=[np.ceil(0.5*n_steps), np.ceil(0.8*n_steps), np.ceil(0.9*n_steps)])

            for i in range(ODI_num_steps + n_steps+1):
                opt = optim.SGD([X_pgd], lr=1e-3)
                opt.zero_grad()

                with torch.enable_grad():
                    out = model(X_pgd)

                    adex_found[~torch.argmax(out.detach(), dim=1).eq(y)] = True
                    adex[adex_found] = X_pgd[adex_found].detach()
                    if early_stopping and (adex_found.all() or (i == ODI_num_steps + n_steps)):
                        break

                    if i < ODI_num_steps:
                        loss = (out * randVector_).sum(-1)
                        regularization = 0.0
                    else:
                        loss, regularization = loss_FN.compute(out, y, i, y_target=y_target)
                        improvement_idx = loss > best_loss
                        best_loss[improvement_idx] = loss[improvement_idx].detach()
                        if train:
                            adex[improvement_idx] = X_pgd[improvement_idx].detach()

                loss = loss + regularization

                if not train:
                    loss[adex_found] = 0.0
                loss.sum().backward(retain_graph=False)

                if i < ODI_num_steps:
                    eta = ODI_step_size * lr_scale * X_pgd.grad.data.sign()
                else:
                    eta = lr_scheduler.get_lr() * lr_scale * X_pgd.grad.data.sign()
                    lr_scheduler.step()

                X_pgd = Variable(torch.minimum(torch.maximum(X_pgd.data + eta, specLB), specUB), requires_grad=True)

    if tau == 0:
        return adex, None
    else:
        if shift:
            midpoints = torch.clamp(adex, specLB + tau_dim, specUB - tau_dim)
        else:
            midpoints = adex
        return adex, HybridZonotope.construct_from_noise(x=midpoints, eps=tau_dim, domain="box", data_range=data_range)


class attack_loss:
    def __init__(self, mode, num_steps, ref_out=None, gama_lambda_orig=None, extra_regularization=False):
        self.mode = mode
        self.num_steps = num_steps
        self.ref_out = ref_out
        self.gama_lambda_orig = gama_lambda_orig
        self.extra_regularization = extra_regularization

        assert mode in ["CE", "margin", "GAMA"]

        if mode == "GAMA":
            assert gama_lambda_orig is not None

    def compute(self, out, labels, step=0, y_target=None):
        regularization = 0
        if self.mode == 'CE':
            loss = F.cross_entropy(out, labels, reduction="none")
        elif self.mode == "margin":
            wc_logits = torch.softmax(out, 1)
            loss =  margin_loss(wc_logits, labels, y_target=y_target)
        elif self.mode == "GAMA":
            wc_logits = torch.softmax(out, 1)
            loss = margin_loss(wc_logits, labels)
            gama_lambda = max((1 - step / (self.num_steps * 0.8)), 0) * self.gama_lambda_orig
            regularization = (gama_lambda * (self.ref_out - wc_logits[:, :]) ** 2).sum(dim=1)

        if self.extra_regularization:
            return loss, regularization
        else:
            return loss + regularization
