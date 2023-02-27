"""
Based on HybridZonotope from DiffAI (https://github.com/eth-sri/diffai/blob/master/ai.py)
"""
import numpy as np
import random
import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union
from torch import Tensor

from src.AIDomains.ai_util import clamp_image, clamp_image_const_error, head_from_bounds, AbstractElement, complete_basis


class HybridZonotope(AbstractElement):
    def __init__(self, head: Tensor, beta: Optional[Tensor], errors: Optional[Tensor], domain: str,
                 c: float=None) -> None:
        super(HybridZonotope,self).__init__()
        self.head = head
        self.beta = beta
        self.errors = errors
        self.domain = domain
        if self.domain == "zono_random" or "shrinking" in self.domain:
            assert c is not None
        self.c = c
        self.device = self.head.device
        assert not torch.isnan(self.head).any()
        assert self.beta is None or (not torch.isnan(self.beta).any())
        assert self.errors is None or (not torch.isnan(self.errors).any())

    @classmethod
    def construct_from_noise(cls, x: Tensor, eps: Union[float,Tensor], domain: str, dtype: torch.dtype=torch.float32,
                             data_range: Tuple[float,float]=(0, 1), c: float=None) -> "HybridZonotope":
        # compute center and error terms from input, perturbation size and data range
        assert data_range[0] < data_range[1]
        x_center, x_beta = clamp_image(x, eps, data_range[0], data_range[1])
        x_center, x_beta = x_center.to(dtype=dtype), x_beta.to(dtype=dtype)
        return cls.construct(x_center, x_beta, domain=domain, c=c)

    @classmethod
    def construct_from_noise_const_eps(cls, x: Tensor, eps: Union[float, Tensor], domain: str,
                                       dtype: torch.dtype = torch.float32, data_range: Tuple[float, float] = (0, 1),
                                       c: float = None) -> "HybridZonotope":
        # compute center and error terms from input, perturbation size and data range
        assert data_range[0] < data_range[1]
        x_center, x_beta = clamp_image_const_error(x, eps, data_range[0], data_range[1])
        x_center, x_beta = x_center.to(dtype=dtype), x_beta.to(dtype=dtype)
        return cls.construct(x_center, x_beta, domain=domain, c=c)


    @classmethod
    def construct_from_bounds(cls, min_x: Tensor, max_x: Tensor, dtype: torch.dtype=torch.float32,
                              domain: str="box", c: float=None) -> "HybridZonotope":
        # compute center and error terms from elementwise bounds
        assert min_x.shape == max_x.shape
        x_center, x_beta = head_from_bounds(min_x,max_x)
        x_center, x_beta = x_center.to(dtype=dtype), x_beta.to(dtype=dtype)
        return cls.construct(x_center, x_beta, domain=domain, c=c)

    @classmethod
    def construct_max_patch(cls, x: Tensor, corners: Tensor, patch_size: int, domain: str,
                        data_range: Tuple[float, float]=(0, 1), c: float=None) -> "HybridZonotope":
        # set bounds in patch to max of datarange, otherwise beta is zero
        assert data_range[0] < data_range[1]
        x_center = x
        device = x_center.device
        x_beta = torch.zeros(x.shape).to(device)
        half = (data_range[0] + data_range[1]) / 2
        for b in range(x.shape[0]):
            x_center[b, :, corners[b, 0]:corners[b, 0] + patch_size,
            corners[b, 1]:corners[b, 1] + patch_size] = half
            x_beta[b, :, corners[b, 0]:corners[b, 0] + patch_size,
            corners[b, 0]:corners[b, 1] + patch_size] = half
        return cls.construct(x_center, x_beta, domain=domain, c=c, sparse=True)

    @classmethod
    def construct_patch(cls, x: Tensor, corners: Tensor, patch_size: int, domain: str, eps: float,
                        data_range: Tuple[float, float]=(0, 1), c:float = None) -> "HybridZonotope":
        # set bounds in patch to max epsilon, otherwise beta is zero
        assert data_range[0] < data_range[1]
        x_center = x
        device = x_center.device
        x_beta = torch.zeros(x_center.shape).to(device)
        for b in range(x.shape[0]):
            x_center_patch, x_beta_patch = clamp_image(x[b, :, corners[b, 0]:corners[b, 0] + patch_size,
                                           corners[b, 1]:corners[b, 1] + patch_size],
                                           eps, data_range[0], data_range[1])
            x_center[b, :, corners[b, 0]:corners[b, 0] + patch_size, corners[b, 1]:corners[b, 1] + patch_size] = x_center_patch
            x_beta[b, :, corners[b, 0]:corners[b, 0] + patch_size, corners[b, 1]:corners[b, 1] + patch_size] = x_beta_patch
        return cls.construct(x_center, x_beta, domain=domain, c=c, sparse=True)

    @classmethod
    def construct_random_low_dim(cls, x: Tensor, dim_werror: int, domain: str, eps: float,
                                 data_range: Tuple[float, float]=(0, 1), c:float = None) -> "HybridZonotope":
        # sets dim_werror pixels at random to max epsilon, otherwise beta is zero
        assert data_range[0] < data_range[1]
        x_center = x
        device = x_center.device
        x_beta = torch.zeros(x_center.shape).to(device)
        image_size = x.shape[-1]
        for b in range(x.shape[0]):
            ran_x = [random.randrange(0, image_size) for r in range(dim_werror)]
            ran_y = [random.randrange(0, image_size) for r in range(dim_werror)]
            x_center[b, :, ran_x, ran_y], x_beta[b, :, ran_x, ran_y] = clamp_image(x_center[b, :, ran_x, ran_y], eps,
                                                                                   data_range[0], data_range[1])
        return cls.construct(x_center, x_beta, domain=domain, c=c, sparse=True)


    @classmethod
    def construct_sens_low_dim(cls, x: Tensor, sens_pixels: Tensor, domain: str, eps: float,
                               data_range: Tuple[float, float]=(0, 1), c:float = None) -> "HybridZonotope":
        # sets error of sens_pixels to max epsilon, otherwise beta is zero
        assert data_range[0] < data_range[1]
        x_center = x
        device = x_center.device
        x_beta = torch.zeros(x_center.shape).to(device)
        for b in range(x.shape[0]):
            curr_x = sens_pixels[b, :, 0]
            curr_y = sens_pixels[b, :, 1]
            x_center[b, :, curr_x, curr_y], x_beta[b, :, curr_x, curr_y] = clamp_image(x_center[b, :, curr_x, curr_y],
                                                                                       eps, data_range[0],
                                                                                       data_range[1])
        return cls.construct(x_center, x_beta, domain=domain, c=c, sparse=True)


    @classmethod
    def construct_sampled_low_dim(cls, x: Tensor, sampled_pixels: Tensor, domain: str, eps: float,
                               data_range: Tuple[float, float] = (0, 1), c: float = None) -> "HybridZonotope":
        # sets error of sampled_pixels to max epsilon, otherwise beta is zero
        assert data_range[0] < data_range[1]
        x_center = x
        device = x_center.device
        x_beta = torch.zeros(x_center.shape).to(device)
        for b in range(x.shape[0]):
            curr_x = sampled_pixels[b, :, 0]
            curr_y = sampled_pixels[b, :, 1]
            """
            curr_idx = (sampled_pixels[:, 0] == b)
            curr_x = sampled_pixels[curr_idx, 1]
            curr_y = sampled_pixels[curr_idx, 2]
            """
            x_center[b, :, curr_x, curr_y], x_beta[b, :, curr_x, curr_y] = clamp_image(x_center[b, :, curr_x, curr_y],
                                                                                       eps, data_range[0],
                                                                                       data_range[1])
        # TODO enable sparse (compensate for unequale number of non zero errors)
        return cls.construct(x_center, x_beta, domain=domain, c=c, sparse=True)


    @staticmethod
    def construct(x_center: Tensor, x_beta: Tensor, domain: str="box", c: float=None, sparse=False) -> "HybridZonotope":
        device = x_center.device
        if domain == 'box' or 'shrinking_box' in domain:
            return HybridZonotope(x_center, x_beta, None, domain, c)
        elif domain in ['zono', 'zono_iter', 'zono_random', 'zono_random2', 'zono_random3', 'zono_reduced', 'zono_reduced_sound', 'hbox', 'zono_uns_line']:
            batch_size = x_center.size()[0]
            n_elements = x_center[0].numel()
            # construct error coefficient matrix
            # ei = torch.eye(n_elements).expand(batch_size, n_elements, n_elements).permute(1, 0, 2).to(device)
            # if len(x_center.size()) > 2:
            #     ei = ei.contiguous().view(n_elements, *x_center.size())
            ei = HybridZonotope.get_error_matrix(x_center)
            # update beta tensor to account for errors captures by error coefficients
            new_beta = None if "zono" in domain else torch.zeros(x_beta.shape).to(device=device, dtype=torch.float32)

            errors = ei * x_beta.unsqueeze(0)

            # remove zero entries to reduce the number of error terms
            if sparse:
                errors = torch.flatten(errors.transpose(0, 1), 0, 1)
                nnz = torch.sum(errors, (1, 2, 3)) > 0
                errors = errors[nnz, ...].reshape(batch_size, -1, x_beta.shape[-3], x_beta.shape[-2], x_beta.shape[-1]).transpose(0, 1).contiguous()

            return HybridZonotope(x_center, new_beta, errors, domain, c)
        else:
            raise RuntimeError('Unsupported HybridZonotope domain: {}'.format(domain))

    def cut_inside_other(self, outer) -> None:
        # this function is only valid, if both sets are in the "box" domain
        # cut inner box, so that it is everywhere inside outer box (might make inner box smaller)
        assert self.domain == "box" and outer.domain == "box"
        lb_inner, ub_inner = self.concretize()
        lb_outer, ub_outer = outer.concretize()
        lb_inner = torch.where(lb_outer <= lb_inner, lb_inner, lb_outer)
        ub_inner = torch.where(ub_outer >= ub_inner, ub_inner, ub_outer)
        self.head = 0.5 * (lb_inner + ub_inner)
        self.beta = 0.5 * (ub_inner - lb_inner)

    def move_inside_other(self, outer) -> None:
        # this function is only valid, if both sets are in the "box" domain
        # moves the inner box, so that it is fully inside the outer box (keep size)
        assert self.domain == "box" and outer.domain == "box"
        #assert (self.beta > outer.beta).sum() == 0
        lb_inner, ub_inner = self.concretize()
        lb_outer, ub_outer = outer.concretize()
        self.head = torch.where(lb_outer <= lb_inner, self.head, lb_outer + self.beta)
        #lb_inner, ub_inner = self.concretize()
        self.head = torch.where(ub_outer >= ub_inner, self.head, ub_outer - self.beta)
        #lb_inner, ub_inner = self.concretize()

    @staticmethod
    def get_error_matrix(x, error_idx=None):
        batch_size = x.size()[0]
        if error_idx is None:
            n_elements_e = x[0].numel()
            ei = torch.eye(n_elements_e, dtype=x.dtype, device=x.device).expand(batch_size, n_elements_e, n_elements_e).permute(1, 0, 2)
        else:
            assert batch_size == 1
            n_elements_e = int(error_idx.sum())
            n_elements_x = x[0].numel()
            ei = torch.zeros((n_elements_e, n_elements_x), dtype=x.dtype, device=x.device)
            ei[torch.arange(n_elements_e).view(-1,1), error_idx.flatten().nonzero()] = 1
            ei = ei.expand(batch_size, n_elements_e, n_elements_x).permute(1, 0, 2)
        if len(x.size()) > 2:
            ei = ei.contiguous().view(n_elements_e, *x.size())
        return ei

    @staticmethod
    def get_new_errs(approx_indicator: torch.Tensor, x_center_new: torch.Tensor,
                     x_beta_new: torch.Tensor) -> torch.Tensor:
        device = x_center_new.device
        dtype = x_center_new.dtype
        batch_size, center_shape = x_center_new.size()[0], x_center_new.size()[1:]

        # accumulate error position over batch dimension
        new_err_pos = (approx_indicator.long().sum(dim=0) > 0).nonzero()
        num_new_errs = new_err_pos.size()[0]
        err_idx_dict = {tuple(pos.cpu().numpy()): idx for pos, idx in zip(new_err_pos,range(num_new_errs))}

        nnz = approx_indicator.nonzero()
        # extract error sizes
        beta_values = x_beta_new[tuple(nnz[:, i] for i in range((nnz.shape[1])))]
        # generate new error matrix portion
        new_errs = torch.zeros((num_new_errs, batch_size,) + center_shape).to(device, dtype=dtype)
        new_errs[([err_idx_dict[tuple(key[1:].cpu().numpy())] for key in nnz],) + tuple(nnz[:, i] for i in range((nnz.shape[1])))]= beta_values
        return new_errs

    def dim(self):
        return self.head.dim()

    @staticmethod
    def join(x: List["HybridZonotope"], trunk_errors: Union[None,Tensor,int]=None, dim: Union[Tensor,int]=0,
             mode: str="cat") -> "HybridZonotope":
        # x is list of HybridZonotopes
        # trunk_errors is number of last shared error between all Hybrid zonotopes, usually either number of initial
        # errors or number of errors at point where a split between network branches occured
        device = x[0].head.device
        if mode not in ["cat","stack"]:
            raise RuntimeError(f"Unkown join mode : {mode:}")

        if mode == "cat":
            new_head = torch.cat([x_i.head for x_i in x], dim=dim)
        elif mode == "stack":
            new_head = torch.stack([x_i.head for x_i in x], dim=dim)

        if all([x_i.beta is None for x_i in x]):
            new_beta = None
        elif any([x_i.beta is None for x_i in x]):
            assert False, "Mixed HybridZonotopes can't be joined"
        else:
            if mode == "cat":
                new_beta = torch.cat([x_i.beta for x_i in x], dim=dim)
            elif mode == "stack":
                new_beta = torch.stack([x_i.beta for x_i in x], dim=dim)

        if all([x_i.errors is None for x_i in x]):
            new_errors = None
        elif any([x_i.errors is None for x_i in x]):
            assert False, "Mixed HybridZonotopes can't be joined"
        else:
            if trunk_errors is None:
                trunk_errors = [0 for x_i in x]
            exit_errors = [x_i.errors.size()[0]-trunk_errors[i] for i, x_i in enumerate(x)] # number of additional errors for every Hybrid zonotope
            tmp_errors= [None for _ in x]
            for i, x_i in enumerate(x):
                tmp_errors[i] = torch.cat([x_i.errors[:trunk_errors[i]],
                                   torch.zeros([max(trunk_errors) - trunk_errors[i] + sum(exit_errors[:i])]
                                               + list(x_i.errors.size()[1:])).to(device),
                                   x_i.errors[trunk_errors[i]:],
                                   torch.zeros([sum(exit_errors[i + 1:])] + list(x_i.errors.size()[1:])).to(device)],
                                   dim=0)

            if mode == "cat":
                new_errors = torch.cat(tmp_errors, dim=dim + 1)
            elif mode == "stack":
                new_errors = torch.stack(tmp_errors, dim=dim+1)

        return HybridZonotope(new_head,
                              new_beta,
                              new_errors,
                              x[0].domain,
                              x[0].c)

    def size(self, idx=None):
        if idx is None:
            return self.head.size()
        else:
            return self.head.size(idx)

    def view(self, size):
        return HybridZonotope(self.head.view(*size),
                              None if self.beta is None else self.beta.view(size),
                              None if self.errors is None else self.errors.view(self.errors.size()[0], *size),
                              self.domain, self.c)

    def flatten(self):
        bsize = self.head.size(0)
        return self.view((bsize, -1))

    def normalize(self, mean: Tensor, sigma: Tensor) -> "HybridZonotope":
        return (self - mean) / sigma

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self) -> "HybridZonotope":
        new_head = -self.head
        new_beta = None if self.beta is None else self.beta
        new_errors = None if self.errors is None else -self.errors
        return HybridZonotope(new_head, new_beta, new_errors, self.domain, self.c)

    def __add__(self, other: Union[Tensor, float, int, "HybridZonotope"]) -> "HybridZonotope":
        if isinstance(other, torch.Tensor) or isinstance(other, float) or isinstance(other, int):
            return HybridZonotope(self.head + other, self.beta, self.errors, self.domain, self.c)
        elif isinstance(other, HybridZonotope):
            assert self.domain == other.domain
            return self.add(other, shared_errors=0)
        else:
            assert False, 'Unknown type of other object'

    def __truediv__(self, other: Union[Tensor, int, float]) -> "HybridZonotope":
        if isinstance(other, torch.Tensor) or isinstance(other, float) or isinstance(other, int) or isinstance(other, torch.Tensor):
            return HybridZonotope(self.head / other,
                                  None if self.beta is None else self.beta / abs(other),
                                  None if self.errors is None else self.errors / other,
                                  self.domain, self.c)
        else:
            assert False, 'Unknown type of other object'

    def __mul__(self, other: Union[Tensor, int, float]) -> "HybridZonotope":
        if isinstance(other, int) or isinstance(other, float) or isinstance(other, int) or (isinstance(other, torch.Tensor)):
            d = self.head.device
            return HybridZonotope((self.head * other).to(d),
                                    None if self.beta is None else (self.beta * abs(other)).to(d),
                                    None if self.errors is None else (self.errors * other).to(d),
                                    self.domain, self.c)
        else:
            assert False, 'Unknown type of other object'

    def __rmul__(self, other):  # Assumes associativity
        return self.__mul__(other)

    def __getitem__(self, indices) -> "HybridZonotope":
        if not isinstance(indices, tuple):
            indices = tuple([indices])
        return HybridZonotope(self.head[indices],
                              None if self.beta is None else self.beta[indices],
                              None if self.errors is None else self.errors[(slice(None), *indices)],
                              self.domain, self.c)

    def clone(self) -> "HybridZonotope":
        return HybridZonotope(self.head.clone(),
                              None if self.beta is None else self.beta.clone(),
                              None if self.errors is None else self.errors.clone(),
                              self.domain, self.c)

    def detach(self) -> "HybridZonotope":
        return HybridZonotope(self.head.detach(),
                              None if self.beta is None else self.beta.detach(),
                              None if self.errors is None else self.errors.detach(),
                              self.domain, self.c)

    def max_center(self) -> Tensor:
        return self.head.max(dim=1)[0].unsqueeze(1)

    def avg_pool2d(self, kernel_size: int, stride:int) -> "HybridZonotope":
        new_head = F.avg_pool2d(self.head, kernel_size, stride)
        new_beta = None if self.beta is None else F.avg_pool2d(self.beta.view(-1, *self.head.shape[1:]), kernel_size, stride)
        new_errors = None if self.errors is None else F.avg_pool2d(self.errors.view(-1, *self.head.shape[1:]), kernel_size, stride).view(-1, *new_head.shape)
        return HybridZonotope(new_head, new_beta, new_errors, self.domain, self.c)

    def global_avg_pool2d(self) -> "HybridZonotope":
        new_head = F.adaptive_avg_pool2d(self.head, 1)
        new_beta = None if self.beta is None else F.adaptive_avg_pool2d(self.beta.view(-1, *self.head.shape[1:]), 1)
        new_errors = None if self.errors is None else F.adaptive_avg_pool2d(self.errors.view(-1, *self.head.shape[1:]), 1).view(-1, *new_head.shape)
        return HybridZonotope(new_head, new_beta, new_errors, self.domain, self.c)

    def max_pool2d(self, kernel_size, stride):
        if self.errors is not None:
            assert False, "MaxPool for Zono not Implemented"
        lb, ub = self.concretize()
        new_lb = F.max_pool2d(lb, kernel_size, stride)
        new_ub = F.max_pool2d(ub, kernel_size, stride)
        return HybridZonotope.construct_from_bounds(new_lb, new_ub, self.dtype ,self.domain, self.c)

    def conv2d(self, weight:Tensor, bias:Tensor, stride:int, padding:int, dilation:int, groups:int) -> "HybridZonotope":
        new_head = F.conv2d(self.head, weight, bias, stride, padding, dilation, groups)
        new_beta = None if self.beta is None else F.conv2d(self.beta, weight.abs(), None, stride, padding, dilation, groups)
        if self.errors is not None:
            errors_resized = self.errors.view(-1, *self.errors.size()[2:])
            new_errors = F.conv2d(errors_resized, weight, None, stride, padding, dilation, groups)
            new_errors = new_errors.view(self.errors.size()[0], self.errors.size()[1], *new_errors.size()[1:])
        else:
            new_errors = None
        return HybridZonotope(new_head, new_beta, new_errors, self.domain, self.c)

    def convtranspose2d(self, weight:Tensor, bias:Tensor, stride:int, padding:int,  output_padding:int, dilation:int, groups:int) -> "HybridZonotope":
        new_head = F.conv_transpose2d(self.head, weight, bias, stride, padding, output_padding,  dilation, groups)
        new_beta = None if self.beta is None else F.conv_transpose2d(self.beta, weight.abs(), None, stride, padding, output_padding,  dilation, groups)
        if self.errors is not None:
            errors_resized = self.errors.view(-1, *self.errors.size()[2:])
            new_errors = F.conv_transpose2d(errors_resized, weight, None, stride, padding, output_padding, dilation, groups)
            new_errors = new_errors.view(self.errors.size()[0], self.errors.size()[1], *new_errors.size()[1:])
        else:
            new_errors = None
        return HybridZonotope(new_head, new_beta, new_errors, self.domain, self.c)

    def contains(self, other: "HybridZonotope", verbose:Optional[bool]=False):
        assert(self.head.size(0) == 1)
        if self.errors is None and other.errors is None: #interval
            lb, ub = self.concretize()
            other_lb, other_ub = other.concretize()
            contained = (lb <= other_lb) & (other_ub <= ub)
            cont_factor = 2*torch.max(((other_ub-self.head)/(ub-lb+1e-16)).abs().max(), ((other_lb-self.head)/(ub-lb+1e-16)).abs().max())
            return contained.all(), cont_factor
        else:
            dtype = self.head.dtype
            device = self.head.device

            # Solve the LGS that we get when representing the "other" zonotope in the "self" basis

            # System Ax = B
            # NOTE: This gives us eps parameterized vectors in x space (i.e. shape 40x824)
            A = self.errors.flatten(start_dim=1).T  # containing init errors
            B = other.errors.flatten(start_dim=1).T  # contained init errors

            if not hasattr(self, "errors_inv"):
                self.errors_inv = None

            if A.shape[-1] == A.shape[-2] and self.errors_inv is None:
                try:
                    self.errors_inv = torch.inverse(A)
                except:
                    print("Failed to invert error matrix")

            if self.errors_inv is None:
                if A.shape[0] != A.shape[1]:
                    sol = np.linalg.lstsq(A.cpu().numpy(), B.cpu().numpy(), rcond=None)
                    x = torch.tensor(sol[0], dtype=dtype, device=device)
                elif float(torch.__version__[:-2]) < 1.9:
                    x = torch.solve(B, A).solution
                else:
                    x = torch.linalg.solve(A, B)
            else:
                x = torch.matmul(self.errors_inv, B)

            # Note sometimes we dont have full rank for A (check sol[1]) - the solution however has no residuals
            # Here x contains the coordinates of the inner zonotope in the outer basis -> 824 x 824

            if not torch.isclose(torch.matmul(A, x), B, atol=1e-7,
                                 rtol=1e-6).all():  # , f"Projection of contained errors into base of containing errors failed"
                uncaptured_errors = torch.abs(B - torch.matmul(A, x)).sum(axis=1)
                print("Projection of contained errors into base of containing errors failed")
                # assert False
            else:
                uncaptured_errors = torch.zeros_like(self.head)

            # Sum the absolutes row-wise to get the scaling factor for the containing error coefficients to overapproximated the contained ones
            abs_per_orig_vector = torch.sum(torch.abs(x), axis=1)
            max_sp = torch.max(abs_per_orig_vector).cpu().item()

            if max_sp > 1 or str(max_sp) == "nan":
                if verbose:
                    print(f"Containment of errors failed with {max_sp}")
                return False, max_sp

            # Here would could probably do some smarter combination i.e. we could compensate the worst errors of the init errors in the differences in the merge errors
            # However this is generally hard (I believe) - a greedy solution should work

            # Here we adjust for the head displacement
            diff = torch.abs(self.head - other.head).detach().view(-1)

            # Here we adjust for errors not captured by the intial matching due to differences in spanned space:
            diff += uncaptured_errors.view(-1)

            # Here we just do a basic check on the "independent" merge errors
            if other.beta is None:
                max_sp_merge = 0
                merge_cont = True
            elif self.beta is None:
                max_sp_merge = "nan"
                merge_cont = False
            else:
                merge_cont = True
                # ei = self.get_error_matrix(self.head)
                for key, val in self.beta.items():
                    # Ensure that all beta of the outer zono actually induce a box
                    if not ((val != 0).sum(0) <= 1).all():  #
                        merge_cont = False
                        max_sp_merge = "nan"

                # Check that merge errors (or rather their difference) is diagonal
                if merge_cont:
                    self_beta = self.beta.detach()
                    other_beta = other.beta.detach()

                    merge_diff = (self_beta - other_beta).view(-1)
                    merge_cont = (
                                merge_diff >= 0).all()  # Ensure that the box errors of other can be contained with the box errors of selfe
                    max_sp_merge = torch.max(other_beta / (self_beta + 1e-8)).cpu().item()

                    # When the merge errors of the containing zono are larger than that of the contained one, we can use this extra to compensate for some of the difference in the heads
                    # diff = torch.maximum(diff - torch.diagonal(merge_diff), torch.tensor(0)).detach()
                    diff = torch.maximum(diff - merge_diff, torch.zeros_like(diff)).detach()

            if not merge_cont:
                if verbose:
                    print(f"Containment of merge errors failed")
                return False, max_sp_merge

            # This projects the remaining difference between the heads into the error coefficient matrix
            diff = torch.diag(diff.view(-1))
            if self.zono_errors_inv is None:
                if A.shape[0] != A.shape[1]:
                    sol_diff = np.linalg.lstsq(A.cpu().numpy(), diff.cpu().numpy(), rcond=None)
                    x_diff = torch.tensor(sol_diff[0], dtype=dtype, device=device)
                elif float(torch.__version__[:-2]) < 1.9:
                    x_diff = torch.solve(diff, A).solution
                else:
                    x_diff = torch.linalg.solve(A, diff)
            else:
                x_diff = torch.matmul(self.zono_errors_inv, diff)

            if not torch.isclose(torch.matmul(A, x_diff), diff, atol=1e-7, rtol=1e-6).all():
                # f"Projection of head difference into base of containing errors failed"
                return False, np.inf

            abs_per_orig_vector_diff = abs_per_orig_vector + torch.abs(x_diff).sum(axis=1)
            max_sp_diff = torch.max(abs_per_orig_vector_diff).cpu().item()

            # Check if with this additional component, we are still contained
            if max_sp_diff > 1 or str(max_sp_diff) == "nan":
                if verbose:
                    print(f"Containment of head differences failed with {max_sp_diff}")
                return False, max_sp_diff

            if verbose:
                print(f"Containment with {max_sp_diff}")

            return True, max(max_sp_merge, max_sp_diff, max_sp)

    @property
    def shape(self):
        return self.head.shape

    @property
    def dtype(self):
        return self.head.dtype

    def dim(self):
        return self.head.dim()

    def linear(self, weight:Tensor, bias:Union[Tensor,None], C:Union[Tensor,None]=None) -> "HybridZonotope":
        if C is None:
            if bias is None:
                return self.matmul(weight.t())
            else:
                return self.matmul(weight.t()) + bias.unsqueeze(0)
        else:
            if bias is None:
                return self.unsqueeze(-1).rev_matmul(C.matmul(weight)).squeeze()
            else:
                return self.unsqueeze(-1).rev_matmul(C.matmul(weight)).squeeze() + C.matmul(bias)

    def matmul(self, other: Tensor) -> "HybridZonotope":
        return HybridZonotope(self.head.matmul(other),
                              None if self.beta is None else self.beta.matmul(other.abs()),
                              None if self.errors is None else self.errors.matmul(other),
                              self.domain, self.c)

    def bmm(self, other: Tensor) -> "HybridZonotope":
        if self.dim != 3:
            self = self.unsqueeze(1)
            unsqueezed = True
        else:
            unsqueezed = False

        new_head = torch.bmm(self.head, other)
        new_beta = None if self.beta is None else torch.bmm(self.beta.view(-1, *self.head.shape[1:]), other)
        new_errors = None if self.errors is None else torch.matmul(self.errors, other)

        if unsqueezed:
            new_head = new_head.squeeze(1)
            new_beta = None if self.beta is None else new_beta.squeeze(1)
            new_errors = None if self.errors is None else new_errors.squeeze(1 + 1)

        return HybridZonotope(new_head, new_beta, new_errors, self.domain, self.c)

    def rev_matmul(self, other: Tensor) -> "HybridZonotope":
        return HybridZonotope(other.matmul(self.head),
                              None if self.beta is None else other.abs().matmul(self.beta),
                              None if self.errors is None else other.matmul(self.errors),
                              self.domain, self.c)

    def fft(self):
        assert self.beta is None
        return HybridZonotope(torch.fft.fft2(self.head).real,
                         None,
                         None if self.errors is None else torch.fft.fft2(self.errors).real,
                         self.domain, self.c)

    def batch_norm(self, bn) -> "HybridZonotope":
        view_dim_list = [1, -1] + (self.head.dim()-2)*[1]
        # self_stat_dim_list = [0, 2, 3] if self.head.dim()==4 else [0]
        # if bn.training:
        #     momentum = 1 if bn.momentum is None else bn.momentum
        #     mean = self.head.mean(dim=self_stat_dim_list).detach()
        #     var = self.head.var(unbiased=False, dim=self_stat_dim_list).detach()
        #     if bn.running_mean is not None and bn.running_var is not None and bn.track_running_stats:
        #         bn.running_mean = bn.running_mean * (1 - momentum) + mean * momentum
        #         bn.running_var = bn.running_var * (1 - momentum) + var * momentum
        #     else:
        #         bn.running_mean = mean
        #         bn.running_var = var

        if bn.training:
            mean = bn.current_mean
            var = bn.current_var
        else:
            mean = bn.running_mean
            var = bn.running_var

        c = (bn.weight / torch.sqrt(var + bn.eps))
        b = (-mean * c + bn.bias)
        new_head = self.head*c.view(*view_dim_list)+b.view(*view_dim_list)
        new_errors = None if self.errors is None else self.errors * c.view(*([1]+view_dim_list))
        new_beta = None if self.beta is None else self.beta * c.abs().view(*view_dim_list)
        return HybridZonotope(new_head, new_beta, new_errors, self.domain, self.c)

    # def batch_norm(self, bn, mean, var):
    #     view_dim_list = [1, -1]+(self.head.dim()-2)*[1]
    #     assert mean is not None and var is not None
    #     c = (bn.weight / torch.sqrt(var + bn.eps))
    #     b = (-mean*c + bn.bias)
    #     new_head = self.head*c.view(*view_dim_list)+b.view(*view_dim_list)
    #     new_errors = None if self.errors is None else self.errors * c.view(*([1]+view_dim_list))
    #     new_beta = None if self.beta is None else self.beta * c.abs().view(*view_dim_list)
    #     return HybridZonotope(new_head, new_beta, new_errors, self.domain)

    @staticmethod
    def cat(zonos, dim=0):
        new_head = torch.cat([x.head for x in zonos], dim)
        new_beta = torch.cat([x.beta if x.beta is not None else torch.zeros_like(x.head) for x in zonos], dim)
        dtype = zonos[0].head.dtype
        device = zonos[0].head.device

        errors = [zono.errors for zono in zonos if zono.errors is not None]
        if len(errors)>0:
            n_err = [x.shape[0] for x in errors]

            new_errors = torch.zeros((n_err, *new_head.shape), dtype=dtype, device=device).transpose(1, dim+1)

            i=0
            j=0
            for error in errors:
                error = error.transpose(1, dim+1)
                new_errors[i:i+error.shape[0], j:j+error.shape[1+dim]] = error
                i += error.shape[0]
                j += error.shape[1+dim]
            new_errors = new_errors.transpose(1, dim + 1)

        else:
            new_errors = None

        return HybridZonotope(new_head, new_beta, new_errors, zonos[0].domain, self.c)

    def relu(self, deepz_lambda:Union[None,Tensor]=None, bounds:Union[None,Tensor]=None) -> Tuple["HybridZonotope",Union[Tensor,None]]:
        lb, ub = self.concretize()
        D = 1e-6
        is_under = (ub <= 0)
        is_above = (ub > 0) & (lb >= 0)
        is_cross = (ub > 0) & (lb < 0)
        num_crossing = is_cross.float().sum()
        num_dead = is_under.float().sum()
        num_alive = is_above.float().sum()

        if self.domain == "box" or "shrinking_box" in self.domain:
            if "shrinking_box" in self.domain:
                shrinking_ratio = self.c
                width = ub - lb
                if "all" in self.domain:
                    use_shrinking = torch.ones_like(is_cross)
                elif "cross" in self.domain:
                    use_shrinking = is_cross
                elif "above" in self.domain:
                    use_shrinking = is_above
                elif "wide" in self.domain:
                    shrinking_fraction = int(0.2 * is_cross.shape[0])
                    max_widths, _ = torch.topk(torch.flatten(width, start_dim=1), shrinking_fraction)
                    max_widths, _ = max_widths.min(dim=1)
                    max_widths = max_widths[:, None]
                    if not max_widths.dim() == width.dim():
                        max_widths = max_widths[:, :, None, None]
                    use_shrinking = torch.where(width >= max_widths, torch.ones_like(is_cross), torch.zeros_like(is_cross))
                if "random_loc_shrinking_box" in self.domain:
                    total_delta = shrinking_ratio * width[use_shrinking]
                    rand_ratio = torch.rand_like(total_delta)
                    ub_delta = width[use_shrinking] * rand_ratio
                    lb_delta = total_delta - ub_delta # TODO
                    ub[use_shrinking] = ub[use_shrinking] - ub_delta
                    lb[use_shrinking] = lb[use_shrinking] + lb_delta
                elif "std_shrinking_box" in self.domain:
                    ub[use_shrinking] = ub[use_shrinking] - width[use_shrinking] * (shrinking_ratio / 2.0)
                    lb[use_shrinking] = lb[use_shrinking] + width[use_shrinking] * (shrinking_ratio / 2.0)
                elif "to_zero_shrinking_box" in self.domain:
                    ub[use_shrinking] = ub[use_shrinking] * (1 - (shrinking_ratio / 2.0))
                elif "std_upper_shrinking_box" in self.domain:
                    ub[use_shrinking] = ub[use_shrinking] - width[use_shrinking] * (shrinking_ratio / 2.0)
                else:
                    assert False, f"shrinking_domain {self.domain} not recognized"
            min_relu, max_relu = F.relu(lb), F.relu(ub)
            return HybridZonotope(0.5 * (max_relu + min_relu), 0.5 * (max_relu - min_relu), None, self.domain, self.c), None, None, num_crossing, num_dead, num_alive, None, None, None
        elif self.domain == "hbox":

            new_head = self.head.clone()
            new_beta = self.beta.clone()
            new_errors = self.errors.clone()

            ub_half = ub / 2

            new_head[is_under] = 0
            new_head[is_cross] = ub_half[is_cross]

            new_beta[is_under] = 0
            new_beta[is_cross] = ub_half[is_cross]

            new_errors[:, ~is_above] = 0

            return HybridZonotope(new_head, new_beta, new_errors, self.domain, self.c), None, None, num_crossing, num_dead, num_alive, None, None, None
        elif "zono" in self.domain:
            if bounds is not None:
                lb_refined, ub_refined = bounds
                lb = torch.max(lb_refined, lb)
                ub = torch.min(ub_refined, ub)

            relu_lambda = torch.where(is_cross, ub/(ub-lb+D), (lb >= 0).float())
            if self.domain == 'zono_iter':
                if deepz_lambda is not None:
                    # assert (deepz_lambda >= 0).all() and (deepz_lambda <= 1).all()
                    if not ((deepz_lambda >= 0).all() and (deepz_lambda <= 1).all()):
                        deepz_lambda = relu_lambda
                    relu_lambda_cross = deepz_lambda
                else:
                    relu_lambda_cross = relu_lambda
                relu_mu_cross = torch.where(relu_lambda_cross < relu_lambda, 0.5*ub*(1-relu_lambda_cross), -0.5*relu_lambda_cross*lb)
                relu_lambda = torch.where(is_cross, relu_lambda_cross, (lb >= 0).float())
                relu_mu = torch.where(is_cross, relu_mu_cross, torch.zeros(lb.size()).to(self.device))

                num_unsound = None
                num_positive = None

            elif "random" in self.domain:
                # For crossing ReLUs picks randomly between a sound and unsound transformer
                # If unsound, then picks randomly between activation states of ReLU

                ub_crossing = ub[is_cross]
                lb_crossing = lb[is_cross]

                # B: probability for unsound transformer increases with bound asymmetry
                # p_sound = 2.0 * torch.sqrt(-ub_crossing * lb_crossing) / (ub_crossing - lb_crossing)

                # C: probability to ensure specific ratio of inputs captured by transformer
                # p_sound = torch.clip(1. - (1. - self.c) * (ub_crossing - lb_crossing) ** 2. / \
                #          (1. - self.c - 2. * ub_crossing * lb_crossing), 0.0, 1.0)
                # TODO can get negative?
                # TODO 1 if ub and lb both very small and close to 0

                # D
                # 0 < c <= 1, tuning parameter, higher c --> more sound
                # p_sound = -4.0 * self.c * ub_crossing * lb_crossing / (ub_crossing - lb_crossing) ** 2.0

                # E
                # p_sound = 0.5 * (ub_crossing - lb_crossing) / (torch.abs(ub_crossing - lb_crossing) + 1)

                # F
                width = ub_crossing - lb_crossing
                p_sound = torch.sigmoid(self.c * torch.log(width + 1.0)) * \
                          2.0 * (width / (2.0 * torch.maximum(ub_crossing, -lb_crossing)) - 0.5)

                use_sound = torch.bernoulli(p_sound)

                num_unsound = use_sound.shape[0] - use_sound.sum()
                num_unsound = num_unsound.detach()

                relu_mu = torch.zeros(lb.size()).to(self.device)
                deepz_lambda = None

                if self.domain == 'zono_random':
                    # pick positive and negative case at random
                    p_positive = ub_crossing / (ub_crossing - lb_crossing)  # probability to pick positive case
                    use_positive = torch.bernoulli(p_positive)
                    num_positive = ((torch.ones(use_sound.shape).to(use_sound.device) - use_sound) *
                                    use_positive).sum().detach()

                    relu_lambda[is_cross] = torch.where(use_sound.bool(),
                                                        ub_crossing / (ub_crossing - lb_crossing + D),
                                                        use_positive)
                    relu_mu[is_cross] = torch.where(use_sound.bool(),
                                                    -0.5 * ub_crossing * lb_crossing / (ub_crossing - lb_crossing + D),
                                                    torch.zeros(lb_crossing.size()).to(self.device)
                                                    )

                elif self.domain == "zono_random2":
                    # for unsound: pick ub or lb of sound case at random
                    p_ub = 0.5 * torch.ones(ub_crossing.shape)
                    use_ub = torch.bernoulli(p_ub)
                    relu_mu[is_cross] = torch.where(use_sound.bool(),
                                                    -0.5 * ub_crossing * lb_crossing / (ub_crossing - lb_crossing + D),
                                                    relu_mu[is_cross])
                    use_ub_unsound = use_ub.to(use_sound.device) * (
                                torch.ones(use_sound.shape).to(use_sound.device) - use_sound)
                    relu_mu[is_cross] = torch.where(use_ub_unsound.bool(),
                                                    ub_crossing * lb_crossing / (ub_crossing - lb_crossing + D),
                                                    relu_mu[is_cross])
                    num_positive = torch.zeros(num_crossing.shape)

                elif self.domain == "zono_random3":
                    # as zono_random but pick ub of sound case for positive case
                    p_positive = ub_crossing / (ub_crossing - lb_crossing)  # probability to pick positive case
                    #use_positive = torch.ones(p_positive.shape).to(use_sound.device) # TODO only for testing
                    use_positive = torch.bernoulli(p_positive)
                    num_positive = ((torch.ones(use_sound.shape).to(use_sound.device) - use_sound) *
                                    use_positive).sum().detach()
                    relu_lambda[is_cross] = torch.where(use_sound.bool(),
                                                        ub_crossing / (ub_crossing - lb_crossing + D),
                                                        use_positive.type(ub_crossing.dtype) *
                                                        ub_crossing / (ub_crossing - lb_crossing + D))
                    relu_mu[is_cross] = torch.where(use_sound.bool(),
                                                    -0.5 * ub_crossing * lb_crossing / (ub_crossing - lb_crossing + D),
                                                    -use_positive.type(ub_crossing.dtype) * ub_crossing *
                                                    lb_crossing / (ub_crossing - lb_crossing + D))

            elif self.domain == "zono_uns_line":
                deepz_lambda = None
                ub_crossing = ub[is_cross]
                lb_crossing = lb[is_cross]
                relu_lambda[is_cross] = (7 * ub_crossing ** 4 - 3 * (ub_crossing * lb_crossing) ** 2
                               - ub_crossing ** 3 * lb_crossing) / (7 * ub_crossing ** 4 + 4 * lb_crossing ** 4 -
                                                                    6 *(ub_crossing * lb_crossing) ** 2
                                                                    - 4 * ub_crossing * lb_crossing ** 3
                                                                    - ub_crossing ** 3 * lb_crossing)
                relu_mu = torch.zeros(lb.size()).to(self.device)
                relu_mu[is_cross] = (2 * ub_crossing ** 3 - relu_lambda[is_cross] *
                                     (ub_crossing ** 3 - lb_crossing ** 3)) / \
                                    (3 * (ub_crossing ** 2 - lb_crossing ** 2))
                num_unsound = num_crossing
                num_positive = torch.zeros(num_crossing.shape)
                deepz_lambda = None
            else:
                relu_mu = torch.where(is_cross, -0.5*ub*lb/(ub-lb+D), torch.zeros(lb.size()).to(self.device))
                deepz_lambda = None
                num_unsound = torch.zeros(num_crossing.shape)
                num_positive = torch.zeros(num_crossing.shape)

            assert (not torch.isnan(relu_mu).any()) and (not torch.isnan(relu_lambda).any())

            new_head = self.head * relu_lambda + relu_mu
            old_errs = self.errors * relu_lambda
            if self.domain == "zono_uns_line":
                new_errors = old_errs
            elif "random" in self.domain:
                is_sound_crossing = is_cross.clone()
                is_sound_crossing[is_cross] = use_sound.to(dtype=is_cross.dtype)
                new_errs = self.get_new_errs(is_sound_crossing, new_head, relu_mu)
                new_errors = torch.cat([old_errs, new_errs], dim=0)
                #breakpoint()
            else:
                new_errs = self.get_new_errs(is_cross, new_head, relu_mu)
                #print("old: ", old_errs.shape)
                #print("new: ", new_errs.shape)
                new_errors = torch.cat([old_errs, new_errs], dim=0)
                #breakpoint()
            #print(new_errors.shape)
            if self.domain == "zono_reduced":
                num_unsound = num_crossing  # TODO: wrong! only if num_crossing > 0
                if new_errors.shape[0] > torch.prod(torch.Tensor([new_errors.shape[2:]])):
                    new_errors, scaling_c = self.move_to_basis(new_errors, sound=False)
                    breakpoint()
                else:
                    scaling_c = torch.Tensor([1.0])
            elif self.domain == "zono_reduced_sound":
                num_unsound = torch.zeros(num_crossing.shape)
                if new_errors.shape[0] > torch.prod(torch.Tensor([new_errors.shape[2:]])):
                    new_errors, scaling_c = self.move_to_basis(new_errors, sound=True)
                else:
                    scaling_c = torch.Tensor([1.0])
            else:
                scaling_c = None
            assert (not torch.isnan(new_head).any()) and (not torch.isnan(new_errors).any())
            num_errors = new_errors.shape[0]

            return HybridZonotope(new_head, None, new_errors, self.domain, self.c), deepz_lambda, num_unsound, num_crossing, num_dead, num_alive, num_positive, num_errors, scaling_c
        else:
            raise RuntimeError("Error applying ReLU with unkown domain: {}".format(self.domain))

    def log(self, deepz_lambda:Union[None,Tensor]=None, bounds:Union[None,Tensor]=None) -> Tuple["HybridZonotope",Union[Tensor,None]]:
        lb, ub = self.concretize()
        D = 1e-6

        if bounds is not None:
            lb_refined, ub_refined = bounds
            lb = torch.max(lb_refined, lb)
            ub = torch.min(ub_refined, ub)

        assert (lb >= 0).all()

        if self.domain in ["box", "hbox"]:
            min_log, max_log = lb.log(), ub.log()
            return HybridZonotope(0.5 * (max_log + min_log), 0.5 * (max_log - min_log), None, "box", self.c), None
        assert self.beta is None

        is_tight = (ub == lb)

        log_lambda_s = (ub.log() - (lb+D).log()) / (ub - lb + D)
        if self.domain == 'zono_iter' and deepz_lambda is not None:
            if not ((deepz_lambda >= 0).all() and (deepz_lambda <= 1).all()):
                deepz_lambda = ((log_lambda_s - (1/ub)) / (1/(lb+D)-1/ub)).detach().requires_grad_(True)
            log_lambda = deepz_lambda * (1 / (lb+D)) + (1 - deepz_lambda) * (1 / ub)
        else:
            log_lambda = log_lambda_s

        log_lb_0 = torch.where(log_lambda < log_lambda_s, ub.log() - ub * log_lambda, lb.log() - lb * log_lambda)
        log_mu = 0.5 * (-log_lambda.log() - 1 + log_lb_0)
        log_delta = 0.5 * (-log_lambda.log() - 1 - log_lb_0)

        log_lambda = torch.where(is_tight, torch.zeros_like(log_lambda), log_lambda)
        log_mu = torch.where(is_tight, ub.log(), log_mu)
        log_delta = torch.where(is_tight, torch.zeros_like(log_delta), log_delta)

        assert (not torch.isnan(log_mu).any()) and (not torch.isnan(log_lambda).any())

        new_head = self.head * log_lambda + log_mu
        old_errs = self.errors * log_lambda
        new_errs = self.get_new_errs(~is_tight, new_head, log_delta)
        new_errors = torch.cat([old_errs, new_errs], dim=0)
        assert (not torch.isnan(new_head).any()) and (not torch.isnan(new_errors).any())
        return HybridZonotope(new_head, None, new_errors, self.domain, self.c), deepz_lambda

    def exp(self, deepz_lambda:Union[None,Tensor]=None, bounds:Union[None,Tensor]=None) -> Tuple["HybridZonotope",Union[Tensor,None]]:
        lb, ub = self.concretize()
        D = 1e-6

        if bounds is not None:
            lb_refined, ub_refined = bounds
            lb = torch.max(lb_refined, lb)
            ub = torch.min(ub_refined, ub)

        if self.domain in ["box", "hbox"]:
            min_exp, max_exp = lb.exp(), ub.exp()
            return HybridZonotope(0.5 * (max_exp + min_exp), 0.5 * (max_exp - min_exp), None, "box", self.c), None

        assert self.beta is None

        is_tight = (ub.exp() - lb.exp()).abs() < 1e-15

        exp_lambda_s = (ub.exp() - lb.exp()) / (ub - lb + D)
        if self.domain == 'zono_iter' and deepz_lambda is not None:
            if not ((deepz_lambda >= 0).all() and (deepz_lambda <= 1).all()):
                deepz_lambda = ((exp_lambda_s -lb.exp()) / (ub.exp()-lb.exp()+D)).detach().requires_grad_(True)
            exp_lambda = deepz_lambda * torch.min(ub.exp(), (lb + 1 - 0.05).exp()) + (1 - deepz_lambda) * lb.exp()
        else:
            exp_lambda = exp_lambda_s
        exp_lambda = torch.min(exp_lambda, (lb + 1 - 0.05).exp()) # ensure non-negative output only

        for _ in range(2): #First go with minimum area (while non negative lb)/ provided slopes. If this leads to negative/zero lower bounds, fall back to minimum slope
            exp_ub_0 = torch.where(exp_lambda > exp_lambda_s, lb.exp() - exp_lambda * lb, ub.exp() - exp_lambda * ub)
            exp_mu = 0.5 * (exp_lambda * (1 - exp_lambda.log()) + exp_ub_0)
            exp_delta = 0.5 * (exp_ub_0 - exp_lambda * (1 - exp_lambda.log()))

            exp_lambda = torch.where(is_tight, torch.zeros_like(exp_lambda), exp_lambda)
            exp_mu = torch.where(is_tight, ub.exp(), exp_mu)
            exp_delta = torch.where(is_tight, torch.zeros_like(exp_delta), exp_delta)

            assert (not torch.isnan(exp_mu).any()) and (not torch.isnan(exp_lambda).any())

            new_head = self.head * exp_lambda + exp_mu
            old_errs = self.errors * exp_lambda
            new_errs = self.get_new_errs(~is_tight, new_head, exp_delta)
            new_errors = torch.cat([old_errs, new_errs], dim=0)
            assert (not torch.isnan(new_head).any()) and (not torch.isnan(new_errors).any())
            new_zono = HybridZonotope(new_head, None, new_errors, self.domain, self.c)
            if (new_zono.concretize()[0] <= 0).any():
                exp_lambda = torch.where(new_zono.concretize()[0] <= 0, lb.exp(), exp_lambda) #torch.zeros_like(exp_lambda)
            else:
                break

        return HybridZonotope(new_head, None, new_errors, self.domain, self.c), deepz_lambda

    # def inv(self, deepz_lambda, bounds):
    #     lb, ub = self.concretize()
    #     assert (lb > 0).all()
    #
    #     if self.domain in ["box", "hbox"]:
    #         min_inv, max_inv = 1 / ub, 1 / lb
    #         return HybridZonotope(0.5 * (max_inv + min_inv), 0.5 * (max_inv - min_inv), None, "box"), None
    #
    #     assert self.beta is None
    #
    #     if bounds is not None:
    #         lb_refined, ub_refined = bounds
    #         lb = torch.max(lb_refined, lb)
    #         ub = torch.min(ub_refined, ub)
    #
    #     assert (lb > 0).all()
    #
    #     is_tight = (ub == lb)
    #
    #     inv_lambda_s = -1 / (ub * lb)
    #     if self.domain == 'zono_iter' and deepz_lambda is not None:
    #         # assert (deepz_lambda >= 0).all() and (deepz_lambda <= 1).all()
    #         if not ((deepz_lambda >= 0).all() and (deepz_lambda <= 1).all()):
    #             deepz_lambda = (-ub*lb+lb**2)/(lb**2-ub**2)
    #         inv_lambda = deepz_lambda * (-1 / lb**2) + (1 - deepz_lambda) * (-1 / ub**2)
    #     else:
    #         inv_lambda = inv_lambda_s
    #
    #
    #     inv_ub_0 = torch.where(inv_lambda > inv_lambda_s, 1 / lb - inv_lambda * lb, 1 / ub - inv_lambda * ub)
    #     inv_mu = 0.5 * (2 * (-inv_lambda).sqrt() + inv_ub_0)
    #     inv_delta = 0.5 * (inv_ub_0 - 2 * (-inv_lambda).sqrt())
    #
    #     # inv_mu = torch.where(inv_lambda > inv_lambda_s, 0.5 * (2 * (-inv_lambda).sqrt() + 1 / lb - inv_lambda * lb),
    #     #                      0.5 * (2 * (-inv_lambda).sqrt() + 1 / ub - inv_lambda * ub))
    #
    #     inv_lambda = torch.where(is_tight, torch.zeros_like(inv_lambda), inv_lambda)
    #     inv_mu = torch.where(is_tight, 1/ub, inv_mu)
    #     inv_delta = torch.where(is_tight, torch.zeros_like(inv_delta), inv_delta)
    #
    #     assert (not torch.isnan(inv_mu).any()) and (not torch.isnan(inv_lambda).any())
    #
    #     new_head = self.head * inv_lambda + inv_mu
    #     old_errs = self.errors * inv_lambda
    #     new_errs = self.get_new_errs(~is_tight, new_head, inv_delta)
    #     new_errors = torch.cat([old_errs, new_errs], dim=0)
    #     assert (not torch.isnan(new_head).any()) and (not torch.isnan(new_errors).any())
    #     return HybridZonotope(new_head, None, new_errors, self.domain), deepz_lambda

    def sum(self, dim:int, reduce_dim=False) -> "HybridZonotope":
        new_head = self.head.sum(dim=dim)
        new_beta = None if self.beta is None else self.beta.abs().sum(dim=dim)
        new_errors = None if self.errors is None else self.errors.sum(dim=dim+1)

        if not reduce_dim:
            new_head = new_head.unsqueeze(dim)
            new_beta = None if new_beta is None else new_beta.unsqueeze(dim)
            new_errors = None if new_errors is None else new_errors.unsqueeze(dim+1)

        assert not torch.isnan(new_head).any()
        assert new_beta is None or not torch.isnan(new_beta).any()
        assert new_errors is None or not torch.isnan(new_errors).any()
        return HybridZonotope(new_head, new_beta, new_errors, self.domain, self.c)

    def unsqueeze(self, dim:int) -> "HybridZonotope":
        new_head = self.head.unsqueeze(dim)
        new_beta = None if self.beta is None else self.beta.unsqueeze(dim)
        new_errors = None if self.errors is None else self.errors.unsqueeze(dim+1)

        assert not torch.isnan(new_head).any()
        assert new_beta is None or not torch.isnan(new_beta).any()
        assert new_errors is None or not torch.isnan(new_errors).any()
        return HybridZonotope(new_head, new_beta, new_errors, self.domain, self.c)

    def squeeze(self, dim:Union[None,int]=None) -> "HybridZonotope":
        if dim is None:
            new_head = self.head.squeeze()
            new_beta = None if self.beta is None else self.beta.squeeze()
            new_errors = None if self.errors is None else self.errors.squeeze()
        else:
            new_head = self.head.squeeze(dim)
            new_beta = None if self.beta is None else self.beta.squeeze(dim)
            new_errors = None if self.errors is None else self.errors.squeeze(dim+1)

        assert not torch.isnan(new_head).any()
        assert new_beta is None or not torch.isnan(new_beta).any()
        assert new_errors is None or not torch.isnan(new_errors).any()
        return HybridZonotope(new_head, new_beta, new_errors, self.domain, self.c)

    def add(self, summand_zono:"HybridZonotope", shared_errors:int=0) -> "HybridZonotope":
        assert all([x == y or x == 1 or y == 1 for x, y in zip(self.head.shape[::-1], summand_zono.head.shape[::-1])])
        dtype = self.head.dtype
        device = self.head.device

        new_head = self.head + summand_zono.head
        if self.beta is None and summand_zono.beta is None:
            new_beta = None
        elif self.beta is not None and summand_zono.beta is not None:
            new_beta = self.beta.abs() + summand_zono.beta.abs()
        else:
            new_beta = self.beta if self.beta is not None else summand_zono.beta

        if self.errors is None:
            new_errors = None
        elif self.errors is not None and summand_zono.errors is not None:
            if shared_errors < 0:
                shared_errors = self.errors.size(0) if self.errors.size(0) == summand_zono.errors.size(0) else 0

            #Shape cast errors to output shape
            self_errors = torch.cat([self.errors
                                     * torch.ones((self.errors.size(0),) + tuple(summand_zono.head.shape),
                                                  dtype=dtype, device=device),
                                     torch.zeros((summand_zono.errors.size(0) - shared_errors,)+ tuple(self.head.shape),
                                                   dtype=dtype, device=device)
                                     * torch.ones((summand_zono.errors.size(0) - shared_errors,) + tuple(summand_zono.head.shape),
                                                   dtype=dtype, device=device)], dim=0)
            summand_errors = torch.cat([summand_zono.errors[:shared_errors] * torch.ones_like(self.errors[:shared_errors]),
                                       torch.zeros((self.errors.size(0) - shared_errors,) + tuple(self.head.shape), dtype=dtype, device=device)
                                       * torch.ones((self.errors.size(0) - shared_errors,) + tuple(summand_zono.head.shape), dtype=dtype, device=device),
                                       summand_zono.errors[shared_errors:]
                                       * torch.ones((summand_zono.errors.size(0) - shared_errors,) + tuple(self.head.shape),
                                                     dtype=dtype, device=device)], dim=0)

            new_errors = self_errors + summand_errors
        else:
            new_errors = self.errors if self.errors is not None else summand_zono.errors

        assert not torch.isnan(new_head).any()
        assert new_beta is None or not torch.isnan(new_beta).any()
        assert new_errors is None or not torch.isnan(new_errors).any()
        new_domain = summand_zono.domain if new_beta is None else ("hbox" if new_errors is not None else "box")
        return HybridZonotope(new_head, new_beta, new_errors, new_domain, self.c)

    def prod(self, factor_zono:"HybridZonotope", shared_errors:Union[int, None]=None, low_mem:bool=False) -> "HybridZonotope":
        dtype = self.head.dtype
        device = self.head.device
        lb_self, ub_self = self.concretize()
        lb_other, ub_other = factor_zono.concretize()

        if self.domain == factor_zono.domain:
            domain = self.domain
        elif "box" in [self.domain, factor_zono.domain]:
            domain = "box"
        elif "hbox" in [self.domain, factor_zono.domain]:
            domain = "hbox"
        else:
            assert False

        if domain in ["box", "hbox"] or low_mem:
            min_prod = torch.min(torch.min(torch.min(lb_self*lb_other, lb_self*ub_other), ub_self*lb_other), ub_self*ub_other)
            max_prod = torch.max(torch.max(torch.max(lb_self*lb_other, lb_self*ub_other), ub_self*lb_other), ub_self*ub_other)
            return HybridZonotope(0.5 * (max_prod + min_prod), 0.5 * (max_prod - min_prod), None, "box", self.c)
        assert self.beta is None

        assert all([x==y or x==1 or y ==1 for x,y in zip(self.head.shape[::-1],factor_zono.head.shape[::-1])])
        if shared_errors is None:
            shared_errors = 0
        if shared_errors == -1:
            shared_errors = self.errors.size(0) if self.errors.size(0) == factor_zono.errors.size(0) else 0

        #Shape cast to output shape
        self_errors = torch.cat([self.errors
                                 * torch.ones((self.errors.size(0),) + tuple(factor_zono.head.shape),
                                              dtype=dtype, device=device),
                                 torch.zeros((factor_zono.errors.size(0) - shared_errors,)+ tuple(self.head.shape),
                                               dtype=dtype, device=device)
                                 * torch.ones((factor_zono.errors.size(0) - shared_errors,) + tuple(factor_zono.head.shape),
                                               dtype=dtype, device=device)], dim=0)
        factor_errors = torch.cat([factor_zono.errors[:shared_errors]
                                   * torch.ones_like(self.errors[:shared_errors]),
                                   torch.zeros((self.errors.size(0) - shared_errors,) + tuple(self.head.shape),
                                                 dtype=dtype, device=device)
                                   * torch.ones((self.errors.size(0) - shared_errors,) + tuple(factor_zono.head.shape),
                                                 dtype=dtype, device=device),
                                   factor_zono.errors[shared_errors:]
                                   * torch.ones((factor_zono.errors.size(0) - shared_errors,) + tuple(self.head.shape),
                                                 dtype=dtype, device=device)], dim=0)

        lin_err = self.head.unsqueeze(dim=0) * factor_errors + factor_zono.head.unsqueeze(dim=0) * self_errors
        quadr_const = (self_errors * factor_errors)
        quadr_error_tmp = self_errors.unsqueeze(1) * factor_errors.unsqueeze(0)
        quadr_error_tmp = 1./2. * (quadr_error_tmp + quadr_error_tmp.transpose(1, 0)).abs().sum(dim=1).sum(dim=0)\
                            - 1./2. * quadr_const.abs().sum(dim=0)

        new_head = self.head * factor_zono.head + 1. / 2. * quadr_const.sum(dim=0)
        old_errs = lin_err
        new_errs = self.get_new_errs(torch.ones(self.head.shape), new_head, quadr_error_tmp)
        new_errors = torch.cat([old_errs, new_errs], dim=0)
        assert (not torch.isnan(new_head).any()) and (not torch.isnan(new_errors).any())
        return HybridZonotope(new_head, None, new_errors, self.domain, self.c)

    def upsample(self, size:int, mode:str, align_corners:bool, consolidate_errors:bool=True) -> "HybridZonotope":
        new_head = F.interpolate(self.head, size=size, mode=mode, align_corners=align_corners)
        delta = 0
        assert mode in ["nearest","linear","bilinear","trilinear"], f"Upsample"

        if self.beta is not None:
            new_beta = F.interpolate(self.beta, size=size, mode=mode, align_corners=align_corners)
            delta = delta + new_beta
        else:
            new_beta = None

        if self.errors is not None:
            errors_resized = self.errors.view(-1, *self.head.shape[1:])
            new_errors = F.interpolate(errors_resized, size=size, mode=mode, align_corners=align_corners)
            new_errors = new_errors.view(-1, *new_head.shape)
            delta = delta + new_errors.abs().sum(0)
        else:
            new_errors = None

        if consolidate_errors:
            return HybridZonotope.construct_from_bounds(new_head-delta, new_head+delta, domain=self.domain, c=self.c)
        else:
            return HybridZonotope(new_head, new_beta, new_errors, self.domain, self.c)

    def beta_to_error(self):
        if self.beta is None:
            return HybridZonotope(self.head, None, self.errors, self.domain)
        new_errors = self.get_error_matrix(self.head, error_idx=self.beta != 0) * self.beta.unsqueeze(0)
        new_errors = torch.cat(([] if self.errors is None else [self.errors]) + [new_errors], dim=0)
        return HybridZonotope(self.head, None, new_errors, self.domain, self.c)

    def concretize(self) -> Tuple[Tensor,Tensor]:
        delta = 0
        if self.beta is not None:
            delta = delta + self.beta
        if self.errors is not None:
            delta = delta + self.errors.abs().sum(0)
        return self.head - delta, self.head + delta

    def avg_width(self) -> Tensor:
        lb, ub = self.concretize()
        return (ub - lb).mean()

    def is_greater(self, i:int, j:int, threshold_min:Union[Tensor,float]=0) -> Tuple[Tensor,bool]:
        diff_head = self.head[:, i] - self.head[:, j]
        delta = diff_head
        if self.errors is not None:
            diff_errors = (self.errors[:, :, i] - self.errors[:, :, j]).abs().sum(dim=0)
            delta -= diff_errors
        if self.beta is not None:
            diff_beta = (self.beta[:, i] + self.beta[:, j]).abs()
            delta -= diff_beta
        return delta, delta > threshold_min

    def get_min_diff(self, i, j):
        """ returns minimum of logit[i] - logit[j] """
        return self.is_greater(i, j)[0]

    def verify(self, targets: Tensor, threshold_min:Union[Tensor,float]=0, corr_only:bool=False) -> Tuple[Tensor,Tensor,Tensor]:
        n_class = self.head.size()[1]
        verified = torch.zeros(targets.size(), dtype=torch.uint8).to(self.head.device)
        verified_corr = torch.zeros(targets.size(), dtype=torch.uint8).to(self.head.device)
        if n_class == 1:
            # assert len(targets) == 1
            verified_list = torch.cat([self.concretize()[1] < threshold_min, self.concretize()[0] >= threshold_min], dim=1)
            verified[:] = torch.any(verified_list, dim=1)
            verified_corr[:] = verified_list.gather(dim=1,index=targets.long().unsqueeze(dim=1)).squeeze(1)
            threshold = torch.cat(self.concretize(),1).gather(dim=1, index=(1-targets).long().unsqueeze(dim=1)).squeeze(1)
        else:
            threshold = np.inf * torch.ones(targets.size(), dtype=torch.float).to(self.head.device)
            for i in range(n_class):
                if corr_only and i not in targets:
                    continue
                isg = torch.ones(targets.size(), dtype=torch.uint8).to(self.head.device)
                margin = np.inf * torch.ones(targets.size(), dtype=torch.float).to(self.head.device)
                for j in range(n_class):
                    if i != j and isg.any():
                        margin_tmp, ok = self.is_greater(i, j, threshold_min)
                        margin = torch.min(margin, margin_tmp)
                        isg = isg & ok.byte()
                verified = verified | isg
                verified_corr = verified_corr | (targets.eq(i).byte() & isg)
                threshold = torch.where(targets.eq(i).byte(), margin, threshold)
        return verified, verified_corr, threshold

    def get_wc_logits(self, targets:Tensor, use_margins:bool=False)->Tensor:
        n_class = self.size(-1)
        device = self.head.device

        if use_margins:
            def get_c_mat(n_class, target):
                return torch.eye(n_class, dtype=torch.float32)[target].unsqueeze(dim=0) \
                       - torch.eye(n_class, dtype=torch.float32)
            if n_class > 1:
                c = torch.stack([get_c_mat(n_class,x) for x in targets], dim=0)
                self = -(self.unsqueeze(dim=1) * c.to(device)).sum(dim=2, reduce_dim=True)
        batch_size = targets.size()[0]
        lb, ub = self.concretize()
        if n_class == 1:
            wc_logits = torch.cat([ub, lb],dim=1)
            wc_logits = wc_logits.gather(dim=1, index=targets.long().unsqueeze(1))
        else:
            # takes upper bound for all classes except target
            wc_logits = ub#.clone()
            wc_logits[np.arange(batch_size), targets] = lb[np.arange(batch_size), targets]
        return wc_logits

    def ce_loss(self, targets:Tensor, get_weights: bool = False) -> Tensor:
        wc_logits = self.get_wc_logits(targets)
        if wc_logits.size(1) == 1:
            return F.binary_cross_entropy_with_logits(wc_logits.squeeze(1), targets.float(), reduction="none")
        else:
            if get_weights:
                _, pred = torch.max(wc_logits, 1)
                weights = torch.ones(targets.shape).to(targets.device) - (pred == targets).int()
                return F.cross_entropy(wc_logits, targets.long(), reduction="none"), weights
            else:
                return F.cross_entropy(wc_logits, targets.long(), reduction="none")

    def to(self, device):
        return HybridZonotope(self.head.to(device),
                          None if self.beta is None else self.beta.to(device),
                          None if self.errors is None else self.errors.to(device),
                          self.domain, self.c)
    """
    def basis_transform(self, new_basis: Tensor, ERROR_EPS_ADD: Optional[float]=0., ERROR_EPS_MUL:Optional[float]=0., sound=True, errors_only=False) -> Tuple["HybridZonotope", Tensor]:
        # We solve for the coordinates (x) of curr_basis (B) in new_basis (A)
        # I.e. we solve Ax=b
        A = new_basis
        B = torch.flatten(self.errors, start_dim=1).T

        device = self.device
        dtype = self.dtype

        if A.shape[0] < 500 or A.shape[0] != A.shape[1]:
            # depending on the size of the matrices different methods are faster
            if isinstance(A, torch.Tensor):
                A = A.cpu().detach().numpy()
            B = B.cpu().detach().numpy()
            sol = np.linalg.lstsq(A, B, rcond=None)[0]
            sol = torch.tensor(sol, dtype=dtype, devive=device)
        else:
            if not isinstance(A, torch.Tensor):
                A = torch.tensor(A)
            sol = torch.solve(B, A).solution

        assert torch.isclose(np.matmul(A, sol), B, atol=1e-7, rtol=1e-6).all(), f"Projection into new base errors failed"

        # We add the component ERROR_EPS_ADD to ensure the resulting error matrix has full rank and to compensate for potential numerical errors
        if sound:
            x = torch.sum(sol.abs(), dim=1) * (1 + ERROR_EPS_MUL) + ERROR_EPS_ADD
        else:
            x = torch.sum(sol, dim=1) * (1 + ERROR_EPS_MUL) + ERROR_EPS_ADD

        new_errors = torch.tensor(x.reshape(1, -1) * new_basis, dtype=dtype).T.unsqueeze(1).view(-1, *self.head.shape)

        if not errors_only:
            return HybridZonotope(self.head, self.beta, new_errors, self.domain, self.c), x
        else:
            new_errors
        """

    def concretize_into_basis(self, basis):
        shp = self.head.shape
        all_as_errors = self.beta_to_error()
        delta = all_as_errors.basis_transform(basis)[1]

        if isinstance(basis, torch.Tensor):
            A = basis.cpu().detach().numpy()
        B = torch.flatten(self.head, start_dim=1).cpu().detach().numpy().T
        sol = np.linalg.lstsq(A, B, rcond=None)[0].T

        new_head = torch.tensor(sol, dtype=self.dtype, device=self.device)

        return (new_head - delta).view(shp), (new_head + delta).view(shp)

    def get_head(self):
        return self.head

    def get_new_basis(self, method:Optional[str]="pca", errors_to_get_basis_from: Optional[Tensor]=None):
        """
        Compute a bais of error directions from errors_to_get_basis_from
        :param errors_to_get_basis_from: Error matrix to be overapproximated
        :param method: "pca" or "pca_zero_mean"
        :return: a basis of error directions
        """
        if errors_to_get_basis_from is None:
            errors_to_get_basis_from = self.errors

        if method == "pca":
            U, S, Vt = np.linalg.svd((errors_to_get_basis_from - errors_to_get_basis_from.mean(1, keepdim=True)).cpu(),
                                     full_matrices=False)  # TODO very slow
            max_abs_cols = np.argmax(np.abs(U), axis=1)
            signs = np.sign(U[torch.arange(U.shape[0]).unsqueeze(-1), max_abs_cols, range(U.shape[2])])

            Vt *= signs[:, :, np.newaxis]
            new_basis_vectors = Vt.transpose((0, 2, 1))
            ### Torch version is much (factor 6) slower despite avoiding move of data to cpu
            # U, S, V = torch.svd(errors_to_get_basis_from - errors_to_get_basis_from.mean(0), some=True)
            # max_abs_cols = U.abs().argmax(0)
            # signs = U[max_abs_cols, range(U.shape[1])].sign()
            # new_basis_vectors_2 = V*signs.unsqueeze(0)

        elif method == "pca_zero_mean":
            U, S, Vt = np.linalg.svd(errors_to_get_basis_from.cpu(), full_matrices=False)
            max_abs_cols = np.argmax(np.abs(U), axis=1)
            signs = np.sign(U[torch.arange(U.shape[0]).unsqueeze(-1), max_abs_cols, range(U.shape[2])])
            Vt *= signs[:, :, np.newaxis]
            new_basis_vectors = Vt.transpose((0, 2, 1))

        #if new_basis_vectors.shape[0] > new_basis_vectors.shape[1]:
        #    # not a full basis
        #    new_basis_vectors = complete_basis(new_basis_vectors)
        #    print("used complete basis")
        # TODO check

        return new_basis_vectors

    def basis_transform(self, new_basis, old_vectors, ERROR_EPS_ADD=0.0, ERROR_EPS_MUL=0.0, return_full_res=False,
                        dtype=torch.float64, sound=True):
        # We solve for the coordinates (x) of curr_basis (B) in new_basis (A)
        # I.e. we solve Ax=b
        A = torch.tensor(new_basis).to(device=old_vectors.device)
        B = old_vectors

        #if A.shape[0] < 500 or A.shape[0] != A.shape[1]:
            # depending on the size of the matrices different methods are faster
            #sol = np.linalg.lstsq(A, B, rcond=None)[0]
            #sol = torch.linalg.lstsq(A, B, rcond=None)[0]
        #else:
        #sol = torch.solve(torch.tensor(B), torch.tensor(A)).solution #.cpu().numpy()
        sol = torch.linalg.solve(A, B)

        # assert torch.isclose(torch.matmul(A, sol), B, atol=1e-7, rtol=1e-6).all(), f"Projection into new base errors failed" # TODO in torch

        # We add the component ERROR_EPS_ADD to ensure the resulting error matrix has full rank and to compensate for potential numerical errors
        # Also this is how widening is implemented
        if sound:
            #x = np.sum(np.abs(sol), axis=1) * (1 + ERROR_EPS_MUL) + ERROR_EPS_ADD
            x = torch.sum(torch.abs(sol), axis=2) * (1 + ERROR_EPS_MUL) + ERROR_EPS_ADD # TODO old axis = 1
        else:
            #x = np.sum(sol, axis=1) * (1 + ERROR_EPS_MUL) + ERROR_EPS_ADD
            x = torch.sum(sol, axis=2) * (1 + ERROR_EPS_MUL) + ERROR_EPS_ADD # TODO old axis = 1
        breakpoint()
        res = (x.reshape(sol.shape[0], 1, -1) * A).transpose(1, 2) #.T.unsqueeze(1) # TODO check axis

        if return_full_res:  # sometimes we also want the see how the consolidation coefficients were obtained
            return res, sol
        else:
            return (x.reshape(sol.shape[0], 1, -1) * A).transpose(1, 2), torch.mean(x) #T.unsqueeze(1) # TODO check axis

    def move_to_basis(self, old_error_terms, ERROR_EPS_ADD=0.0, ERROR_EPS_MUL=0.0, basis_transform_method="pca", sound=True,
                      dtype: torch.dtype=torch.float32):
        # This moves the init errors of a zonotope into a new basis (and essentially reduces the number of error terms via over-approximation)
        dtype = self.head.dtype
        device = self.head.device
        breakpoint()
        shp = old_error_terms.shape
        old_error_terms = old_error_terms.view((shp[1], shp[0], -1))

        # TODO: with torch.no_grad() ?
        new_basis = self.get_new_basis(method=basis_transform_method,
                                       errors_to_get_basis_from=old_error_terms.detach())  # TODO: speed up?        new_basis = torch.Tensor(new_basis).to(old_error_terms.device)
        #if self.zono_errors is None:
        #    return LBZonotope(self.head, None, self.box_errors, self.domain, -1)  # TODO check for Hybrid
        #else:
        # Old basis is the concatenation of all zono_errors
        #shp = old_error_terms.shape

        #old_error_terms = old_error_terms.view(shp[0], -1).T #.cpu().numpy().T
        new_zono_error_terms, x = self.basis_transform(new_basis, old_error_terms.transpose(1, 2), ERROR_EPS_ADD, ERROR_EPS_MUL,
                                               return_full_res=False, sound=sound)
        new_zono_error_terms = new_zono_error_terms.to(device)
        #new_zono_error_terms.view(-1, 1, *shp[1:])
        return new_zono_error_terms.view(-1, shp[1], *shp[2:]).to(dtype=dtype), x

