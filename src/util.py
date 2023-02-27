import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import math
import random
import os
import sys
from typing import Optional
try:
    from pip._internal.operations import freeze
except ImportError: # pip < 10.0
    from pip.operations import freeze



from src.AIDomains.zonotope import HybridZonotope
from src.AIDomains.ai_util import clamp_image
from src.AIDomains.abstract_layers import Conv2d, Flatten, ReLU, Normalization, Linear, BatchNorm2d, BatchNorm1d


def seed_everything(seed: Optional[int] = None) -> None:
    if seed is not None:
        os.environ["PL_GLOBAL_SEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False


class Logger(object):
    def __init__(self, filename, stdout):
        self.terminal = stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

    def _get_writer(self, verbose):
        def write(str):
            if verbose:
                print(str)
            else:
                self.log.write(str+"\n")
        return write

    def log_default(self, args):
        self.log_env(verbose=False)
        self.log_args(args, verbose=False)
        print("")

    def log_env(self, verbose=False):
        write = self._get_writer(verbose)
        write("\nEnvironment Info:")
        pkgs = freeze.freeze()
        for pkg in pkgs:
            write(pkg)

    def log_args(self, args, verbose=False):
        write = self._get_writer(verbose)
        write("\nArgs:")
        for key in dir(args):
            if key.startswith("_"): continue
            write(f"{key}: {getattr(args,key)}")


class Scheduler:
    def __init__(self, start_epoch, end_epoch, start_value, end_value, mode="linear", c=0.4, e=3, s=500, beta=4.0, midpoint=0.25):
        assert end_epoch >= start_epoch
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.start_value = start_value
        self.end_value = end_value
        self.mode = mode
        self.c = c
        self.e = e
        self.s = s
        self.beta = beta
        self.mid_point = midpoint
        assert e >= 1, "please choose an exponent >= 1"
        assert 0 < c < 0.5, "please choose c in the range (0,0.5)"

    def getcurrent(self, epoch):
        if epoch < self.start_epoch:
            return self.start_value
        if epoch >= self.end_epoch:
            return self.end_value

        if self.mode == "linear":
            current = self.start_value + (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch) * \
                  (self.end_value - self.start_value)
        elif self.mode == "smooth":
            c = self.c # portion with cubic growth at beginning and end
            e = self.e
            width = self.end_epoch - self.start_epoch
            offset = c * width
            d = self.end_value - self.start_value
            a = d/((2 - 2 * e) * c ** e + e * c ** (e-1))
            b = e * a * c ** (e-1)
            d1 = a * c ** e
            if epoch - self.start_epoch < offset:
                ### first high order section
                current = self.start_value + a * ((epoch - self.start_epoch) / width) ** e
            elif self.end_epoch - epoch > offset:
                ### linear section
                current = self.start_value + d1 + b * ((epoch - self.start_epoch - offset) / width)
            else:
                ### second high order section
                current = self.end_value - a * ((self.end_epoch - epoch) / width) ** e
        elif self.mode == "step":
            n_steps = int((self.end_epoch - self.start_epoch) / self.s)
            delta = (self.end_value - self.start_value) / n_steps
            current = np.ceil((epoch - self.start_epoch + 0.1) / (self.end_epoch - self.start_epoch) * n_steps) * delta + self.start_value
        elif self.mode == "exp":
            ### Code is based on auto_LIRPA
            ### https://github.com/KaidiXu/auto_LiRPA
            beta = self.beta
            init_step = self.start_epoch - 1
            final_step = self.end_epoch
            init_value = self.start_value
            final_value = self.end_value
            # Batch number for switching from exponential to linear schedule
            mid_step = int((final_step - init_step) * self.mid_point) + init_step
            t = (mid_step - init_step) ** (beta - 1.)
            # find coefficient for exponential growth, such that at mid point the gradient is the same as a linear ramp to final value
            alpha = (final_value - init_value) / (
                    (final_step - mid_step) * beta * t + (mid_step - init_step) * t)
            # value at switching point
            mid_value = init_value + alpha * (mid_step - init_step) ** beta
            # linear schedule after mid step
            exp_value = init_value + alpha * float(epoch - init_step) ** beta
            linear_value = min(
                mid_value + (final_value - mid_value) * (epoch - mid_step) / (final_step - mid_step),
                final_value)
            current = exp_value if epoch <= mid_step else linear_value
        else:
            raise NotImplementedError
        return current


class CyclicScheduler:
    def __init__(self, min_value, max_value, half_cycle, start_epoch=0):
        assert max_value >= min_value
        self.max_value = max_value
        self.min_value = min_value
        self.half_cycle = half_cycle
        self.start_epoch = start_epoch

    def getcurrent(self, epoch):
        # starts with max value and goes down first
        curr_epoch = epoch - self.start_epoch
        if int(curr_epoch / self.half_cycle) % 2 == 0:
            # in downward part
            curr_val = self.max_value - (self.max_value - self.min_value) * \
                       (curr_epoch % self.half_cycle) / self.half_cycle
        else:
            # in upward part
            curr_val = self.min_value + (self.max_value - self.min_value) * \
                       (curr_epoch % self.half_cycle) / self.half_cycle
        return curr_val


"""
below function is based on https://github.com/shizhouxing/Fast-Certified-Robust-Training/blob/main/manual_init.py
"""

def get_params(model):
    weights = []
    biases = []
    for p in model.named_parameters():
        if 'weight' in p[0]:
            weights.append(p)
        elif 'bias' in p[0]:
            biases.append(p)
        else:
            print('Skipping parameter {}'.format(p[0]))
    return weights, biases

def ibp_init(model_ori):
    weights, biases = get_params(model_ori)
    for i in range(len(weights)-1):
        if weights[i][1].ndim == 1:
            # skip BN layers
            continue
        weight = weights[i][1]
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(weight)
        std = math.sqrt(2 * math.pi / (fan_in**2))
        std_before = weight.std().item()
        torch.nn.init.normal_(weight, mean=0, std=std)
        print(f'Reinitialize {weights[i][0]}, std before {std_before:.5f}, std now {weight.std():.5f}')


class WeightInit:
    def __init__(self, approach):
        self.approach = approach

    def init_weight(self, m):
        if self.approach == "IBP":
            """
            IBP weight init suggested in
            https://proceedings.neurips.cc/paper/2021/hash/988f9153ac4fd966ea302dd9ab9bae15-Abstract.html
            """
            ibp_init(m)
            # TODO check if below also correct?
            #if isinstance(m, nn.Linear):
            #    torch.nn.init.normal_(m.weight, 0.0, torch.sqrt(2 * torch.tensor(math.pi)) / m.weight.shape[1])
            #    m.bias.data.fill_(0.01)
            #if isinstance(m, nn.Conv2d):
            #    n_i = m.weight.shape[-1] ** 2 * m.weight.shape[1]
            #    torch.nn.init.normal_(m.weight, 0.0, torch.sqrt(2 * torch.tensor(math.pi)) / n_i)
        elif self.approach == "xavier_uni":
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
        elif self.approach == "xavier_normal":
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.01)
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
        elif self.approach == "he_uni":
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight)
        elif self.approach == "he_normal":
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.01)
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)


def regularize(m, p):
    # loss to regularize parameters of model m based on norm p
    reg_loss = 0
    for param in m.parameters():
         reg_loss += param.norm(p)
    return reg_loss


def get_random_patch_corners(image_size: int, patch_size: int, batch_size: int):
    # get random upper left corners of patch
    x = [random.randrange(0, image_size - patch_size) for b in range(batch_size)]
    y = [random.randrange(0, image_size - patch_size) for b in range(batch_size)]
    return torch.Tensor([x, y]).transpose(0, 1).int()


def get_sensitivity_patches(model, X, y, patch_size, eps):
    with torch.enable_grad():
        X_sens = Variable(X.data, requires_grad=True).to(X.device)

        adv_box = HybridZonotope.construct_from_noise(x=X_sens, eps=eps, domain='box')
        out = model(adv_box)
        loss = out.ce_loss(y)
        loss.sum().backward()
    in_channels = X.shape[1]
    grads = torch.abs(X_sens.grad.data)  # sensitivity = sum(abs(grads))
    # TODO: check why grad sometimes everywhere 0.0
    #conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=patch_size)
    #conv.weight.data.fill_(1.0)
    #conv.bias.data.fill_(0.0)
    #conv.to(X.device)
    max = nn.MaxPool2d(kernel_size=patch_size, stride=1)
    patch_grads = max(grads) - max(-grads)

    return patch_grads


def get_scaled_sensitivity_patches(model, X, y, patch_size, eps):
    with torch.enable_grad():
        X_sens = Variable(X.data, requires_grad=True).to(X.device)

        adv_box = HybridZonotope.construct_from_noise(x=X_sens, eps=eps, domain='box')
        out = model(adv_box)
        loss = out.ce_loss(y)
        loss.sum().backward()
    in_channels = X.shape[1]
    grads = X_sens.grad.data
    # m = torch.nn.ReLU()
    # scaling_factor = torch.abs(X - m(torch.sign(grads)))
    # scaling_factor = 1.0 - m(eps - torch.abs(X_sens - m(torch.sign(grads))))
    # scaling_factor = torch.clamp(torch.abs(X_sens - m(torch.sign(grads))), max=eps)
    # grads_scaled = scaling_factor * torch.abs(grads)
    delta_pos = torch.clamp(torch.abs(X_sens - 1.0), max=eps)
    delta_neg = torch.clamp(X_sens, max=eps)
    grads_scaled = torch.abs(grads) * (delta_pos - delta_neg)  # try with and without abs?
    print(torch.mean(delta_pos))
    print(torch.mean(delta_neg))
    print(torch.mean(delta_pos - delta_neg))
    conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=patch_size)
    conv.weight.data.fill_(1.0)
    conv.bias.data.fill_(0.0)
    conv.to(X.device)
    patch_grads_scaled = conv(grads_scaled) * 5.0
    return patch_grads_scaled


def get_sensitive_patch_corners(model, X, y, patch_size, eps):
    patch_grads = get_sensitivity_patches(model, X, y, patch_size, eps)
    max_grads = torch.amax(patch_grads, (2, 3))  # TODO: sensitivity only in direction where loss worse? or take abs?
    max_grads_index = torch.eq(patch_grads[:, 0, :, :], max_grads[:, :, None]).nonzero()[:, 1:]
    if X.shape[0] != max_grads_index.shape[0]:
        # TODO: select from randomly from max_grads_index per sample
        max_grads_index = get_random_patch_corners(X.shape[-1], patch_size, X.shape[0]).to(X.device)
    return max_grads_index


def get_sensitivity_pixels(model, X, y, eps):
    with torch.enable_grad():
        X_sens = Variable(X.data, requires_grad=True).to(X.device)
        adv_box = HybridZonotope.construct_from_noise(x=X_sens, eps=eps, domain='box')
        out = model(adv_box)
        loss = out.ce_loss(y)
        loss.sum().backward()
    grads = torch.sum(torch.abs(X_sens.grad.data), dim=1)
    return grads


def get_sensitive_pixels(model, X, y, dim_werror, eps):
    # select the dim_werror most sensitive pixels
    grads = get_sensitivity_pixels(model, X, y, eps)
    grads = torch.flatten(grads, start_dim=1)
    _, sens_index_flat = torch.topk(grads, dim_werror, dim=1)  # TODO: sensitivity only in direction where loss worse? or take abs?
    image_size = X.shape[-1]
    x = sens_index_flat % image_size
    y = (sens_index_flat / image_size).int()
    sens_index = torch.stack((x, y), dim=-1)  # [batchsize, dim_werror, 2]
    return sens_index


def sample_sensitive_pixels(model, X, y, dim_werror, eps):
    # sample dim_werror pixels depending on sensitivity
    grads = get_sensitivity_pixels(model, X, y, eps)
    grads = torch.flatten(grads, start_dim=1)
    prob = nn.functional.normalize(grads, p=1)
    #prob = nn.functional.softmax(grads, dim=1) # grads scales vary a lot
    prob_scaled = torch.clamp(dim_werror * prob, min=0.0, max=1.0)
    samples = torch.bernoulli(prob_scaled)
    image_size = X.shape[-1]

    while not torch.all(torch.sum(samples, dim=1) == dim_werror):
        diff = dim_werror - torch.sum(samples, dim=1)
        prob[diff == 0, :] = 0.0
        prob[diff > 0, :] = torch.clamp(diff[diff > 0][:, None] * (prob[diff > 0]-samples[diff > 0]), min=0.0, max=1.0)
        prob[diff < 0, :] = samples[diff < 0] * (-diff[diff < 0][:, None]) / torch.sum(samples[diff < 0], dim=1)[:, None]
        samples = samples + diff[:, None].sign() * torch.bernoulli(prob)
    index_flatt = samples.nonzero()
    index_1d = index_flatt[:, 1].view((X.shape[0], dim_werror))
    x = index_1d % image_size
    y = (index_1d / image_size).int()
    index = torch.stack((x,y), dim=-1)  # [batchsize, dim_werror, 2]
    """
    index_flatt = samples.nonzero()
    x = index_flatt[:, 1] % image_size
    y = (index_flatt[:, 1] / image_size).int()
    index_flatt_extended = torch.stack((index_flatt[:, 0], x, y), dim=1)
    """
    return index_flatt_extended


def get_eps_greedy_patches(model, X, y, patch_size, eps, eps_greedy):
    image_size = X.shape[-1]
    batch_size = X.shape[0]
    num_random = int(batch_size * eps_greedy)
    random_corners = get_random_patch_corners(image_size, patch_size, num_random)
    indices = [False] * num_random + [True] * (batch_size - num_random)
    random.shuffle(indices)
    sensitive_corners = get_sensitive_patch_corners(model, X, y, patch_size, eps)
    sensitive_corners = sensitive_corners[indices]
    # TODO: pass indices only, max for these
    corners = torch.zeros((batch_size, 2)).int().to(X.device)  # TODO not needed
    corners[indices] = sensitive_corners.int()
    corners[np.logical_not(indices)] = random_corners.to(X.device)
    return corners


class IndexDataset(Dataset):
    def __init__(self, dataset):
        if dataset == "mnist":
            train_data = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
        elif args.dataset == "cifar10":
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            train_data = datasets.CIFAR10("./data", train=True, download=True, transform=transform_train)

        self.input_dim = train_data[0][0].size()

        #train_data_extended = [[train_data[0][0], train_data[0][0], train_data[0][1]]]
        #for i in range(2, len(train_data)):
        #    train_data_extended.append([train_data[i][0], train_data[i][0], train_data[i][1]])
        self.dataset = train_data  #_extended
        self.midpoints = train_data.data[:, None, :, :].type(dtype=torch.float32) / 255.

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def update_midpoints(self, index, midpoints):
        for i in range(len(index)):
            self.dataset[index[i]][1] = midpoints[i]

    def set_midpoints(self, index, midpoints):
        self.midpoints[index] = midpoints.cpu().detach()

    def get_midpoints(self, index):
        return self.midpoints[index, :, :, :]

    def __len__(self):
        return len(self.dataset)

    def get_input_dim(self):
        return self.input_dim


"""
Below function for random cauchy projections based on https://github.com/eth-sri/colt/blob/master/code/attacks.py
"""


def compute_bounds_approx(eps, blocks, inputs, k=50, domain='zono', c_min=None):
    bounds_approx = {}
    batch_size = inputs.shape[0]
    curr_head, A_0 = clamp_image(inputs, eps)
    curr_cauchy = A_0.unsqueeze(0) * torch.clamp(torch.FloatTensor(k, *inputs.size()).to(inputs.device).cauchy_(), -1e10, 1e10)
    n_cauchy = 1
    num_crossing_tot = []
    num_unsound_tot = []
    num_alive_tot = []
    num_dead_tot = []
    num_positive_tot = []
    for j in range(len(blocks)):
        num_crossing = 0
        num_unsound = 0
        num_alive = 0
        num_dead = 0
        num_positive = 0
        layer = blocks[j]
        if isinstance(layer, Normalization):
            curr_head = (curr_head - layer.mean) / layer.sigma
            curr_cauchy /= layer.sigma.unsqueeze(0)
        elif isinstance(layer, Conv2d):
            #conv = layer.conv
            #curr_head = conv(curr_head)
            curr_head = layer(curr_head)
            tmp_cauchy = curr_cauchy.view(-1, *curr_cauchy.size()[2:])
            #tmp_cauchy = F.conv2d(tmp_cauchy, conv.weight, None, conv.stride, conv.padding, conv.dilation, conv.groups)
            tmp_cauchy = F.conv2d(tmp_cauchy, layer.weight, None, layer.stride, layer.padding, layer.dilation, layer.groups)
            curr_cauchy = tmp_cauchy.view(-1, batch_size, *tmp_cauchy.size()[1:])
        elif isinstance(layer, ReLU):
            D = 1e-6
            lb, ub = bounds_approx[j]
            is_cross = (lb < 0) & (ub > 0)
            num_crossing = is_cross.int().sum()
            is_dead = (ub <= 0)
            num_dead = is_dead.int().sum()
            is_alive = (lb >= 0) & (ub > 0)
            num_alive = is_alive.int().sum()
            relu_lambda = torch.where(is_cross, ub/(ub-lb+D), (lb >= 0).type(ub.dtype))  # TODO add +D to denominator?
            if domain == 'zono':
                relu_mu = torch.where(is_cross, -0.5*ub*lb/(ub-lb+D), torch.zeros(lb.size()).to(inputs.device))
            elif 'random' in domain:
                assert c_min is not None
                ub_crossing = ub[is_cross]
                lb_crossing = lb[is_cross]

                # E
                # p_sound = 0.5 * (ub_crossing - lb_crossing) / (torch.abs(ub_crossing - lb_crossing) + 1)

                # F
                width = ub_crossing - lb_crossing
                p_sound = torch.sigmoid(c_min * torch.log(width)) * \
                          2.0 * (width / (2.0 * torch.maximum(ub_crossing, -lb_crossing)) - 0.5)
                use_sound = torch.bernoulli(p_sound)

                num_unsound = use_sound.shape[0] - use_sound.sum()
                num_unsound = num_unsound.detach()

                relu_mu = torch.zeros(lb.size()).to(inputs.device)

                if domain == 'zono_random':
                    # if num_crossing > 0:
                    #    print("num unsound of num_crossing")
                    #    print(num_unsound, " of ", num_crossing)

                    p_positive = ub_crossing / (ub_crossing - lb_crossing)  # probability to pick positive case
                    use_positive = torch.bernoulli(p_positive)
                    num_positive = ((torch.ones(use_sound.shape).to(use_sound.device) - use_sound) *
                                    use_positive).sum().detach()

                    relu_lambda[is_cross] = torch.where(use_sound.bool(),
                                                        ub_crossing / (ub_crossing - lb_crossing + D),
                                                        use_positive)
                    relu_mu[is_cross] = torch.where(use_sound.bool(),
                                                    (-0.5 * ub_crossing * lb_crossing / (ub_crossing - lb_crossing + D)).type(relu_mu.dtype),
                                                    torch.zeros(lb_crossing.size()).to(inputs.device).type(relu_mu.dtype)
                                                    )
                elif domain == "zono_random2":
                    p_ub = 0.5 * torch.ones(ub_crossing.shape)
                    use_ub = torch.bernoulli(p_ub)

                    relu_mu[is_cross] = torch.where(use_sound.bool(),
                                                    -0.5 * ub_crossing * lb_crossing / (ub_crossing - lb_crossing + D),
                                                    relu_mu[is_cross])
                    use_ub_unsound = use_ub.to(use_sound.device) * (
                            torch.ones(use_sound.shape).to(use_sound.device) - use_sound)
                    relu_mu[is_cross] = torch.where(use_ub_unsound.bool(),
                                                    -ub_crossing * lb_crossing / (ub_crossing - lb_crossing + D),
                                                    relu_mu[is_cross])
                elif domain == "zono_random3":
                    p_positive = ub_crossing / (ub_crossing - lb_crossing)  # probability to pick positive case
                    use_positive = torch.bernoulli(p_positive)
                    num_positive = ((torch.ones(use_sound.shape).to(use_sound.device) - use_sound) *
                                    use_positive).sum().detach()

                    relu_lambda[is_cross] = torch.where(use_sound.bool(),
                                                        ub_crossing / (ub_crossing - lb_crossing + D),
                                                        use_positive.type(ub_crossing.dtype) *
                                                        ub_crossing / (ub_crossing - lb_crossing + D))
                    relu_mu[is_cross] = torch.where(use_sound.bool(),
                                                    -0.5 * ub_crossing * lb_crossing / (ub_crossing - lb_crossing + D),
                                                    use_positive.type(ub_crossing.dtype) * ub_crossing *
                                                    lb_crossing / (ub_crossing - lb_crossing + D))

            elif domain == 'zono_uns_line':
                ub_crossing = ub[is_cross]
                lb_crossing = lb[is_cross]
                relu_lambda[is_cross] = (7 * ub_crossing ** 4 - 3 * (ub_crossing * lb_crossing) ** 2
                                         - ub_crossing ** 3 * lb_crossing) / (
                                                    7 * ub_crossing ** 4 + 4 * lb_crossing ** 4 -
                                                    6 * (ub_crossing * lb_crossing) ** 2
                                                    - 4 * ub_crossing * lb_crossing ** 3
                                                    - ub_crossing ** 3 * lb_crossing)
                relu_mu = torch.zeros(lb.size()).to(inputs.device)
                relu_mu[is_cross] = (2 * ub_crossing ** 3 - relu_lambda[is_cross] *
                                     (ub_crossing ** 3 - lb_crossing ** 3)) / \
                                    (3 * (ub_crossing ** 2 - lb_crossing ** 2))
                num_unsound = num_crossing

            curr_head = curr_head * relu_lambda + relu_mu
            curr_cauchy = curr_cauchy * relu_lambda.unsqueeze(0)
            new_cauchy = relu_mu.unsqueeze(0) * torch.clamp(torch.FloatTensor(k, *curr_head.size()).to(inputs.device).cauchy_(), -1e10, 1e10)
            curr_cauchy = torch.cat([curr_cauchy, new_cauchy], dim=0)
            n_cauchy += 1
            num_unsound_tot.append(num_unsound)
            num_crossing_tot.append(num_crossing)
            num_dead_tot.append(num_dead)
            num_alive_tot.append(num_alive)
            num_positive_tot.append(num_positive)
        elif isinstance(layer, Flatten):
            curr_head = curr_head.view(batch_size, -1)
            curr_cauchy = curr_cauchy.view(curr_cauchy.size()[0], batch_size, -1)
        elif isinstance(layer, Linear):
            curr_head = layer(curr_head)
            curr_cauchy = torch.matmul(curr_cauchy, layer.weight.t())
        elif isinstance(layer, BatchNorm2d):
            layer.set_use_old_train_stats(True)
            curr_head = layer(curr_head)
            tmp_cauchy = curr_cauchy.view(-1, *curr_cauchy.size()[2:])
            tmp_cauchy = layer(tmp_cauchy)
            layer.set_use_old_train_stats(False)
            curr_cauchy = tmp_cauchy.view(-1, batch_size, *tmp_cauchy.size()[1:]) - layer.bias.view((1, 1, -1, 1, 1))
        elif isinstance(layer, BatchNorm1d):
            layer.set_use_old_train_stats(True)
            curr_head = layer(curr_head)
            tmp_cauchy = curr_cauchy.view(-1, *curr_cauchy.size()[2:])
            tmp_cauchy = layer(tmp_cauchy)
            layer.set_use_old_train_stats(False)
            curr_cauchy = tmp_cauchy.view(-1, batch_size, *tmp_cauchy.size()[1:]) - layer.bias.view((1, 1, -1))
        else:
            assert False, 'Unknown layer type!'

        #if j+1 < len(blocks) and isinstance(blocks[j+1], ReLU):
        l1_approx = 0
        for i in range(n_cauchy):
            l1_approx += torch.median(curr_cauchy[i*k:(i+1)*k].abs(), dim=0)[0]
        lb = curr_head - l1_approx
        ub = curr_head + l1_approx
        bounds_approx[j+1] = (lb, ub)

    return bounds_approx, num_unsound_tot, num_crossing_tot, num_dead_tot, num_alive_tot, num_positive_tot


def get_wc_logits_from_bounds(bounds, targets):
    (lb, ub) = bounds[len(bounds)]
    wc_logits = ub
    wc_logits[:, targets] = lb[:, targets]
    return wc_logits


def get_margin(logits, y, n_class=10):
    logit_org = logits.gather(1, y.view(-1, 1))
    logit_target = logits.gather(1, (logits - torch.eye(n_class)[y].to("cuda") * 9999).argmax(1, keepdim=True))
    margin = logit_org - logit_target
    return margin
