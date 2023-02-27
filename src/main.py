import os
import random
import sys

import numpy as np
import torch
from datetime import datetime
import json
from torch.utils.tensorboard import SummaryWriter
from bunch import Bunch

from src.AIDomains.abstract_layers import Sequential
from src.parse_args import parse_args
from src.datasets import get_data_loader
from src.networks import Models
from src.util import Scheduler, WeightInit, seed_everything, Logger
from src.train import train_net
from src.convert_to_dict_mnbab import convert_state_dict


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main():
    args = parse_args()
    use_cuda = torch.cuda.is_available()

    save_dir = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"))

    # Set random seed
    seed_everything(args.seed)

    # Configure Logging
    exp_name = f"{args.dataset}__{args.net}__bn_{1 if args.bn else 0}_{1 if args.bn2 else 0}__eps_{args.eps_end}__lambda_{args.lambda_ratio}"
    if args.experiment_key is not None:
        exp_name += f"__{args.experiment_key}"
    date = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    file_name = exp_name + "__" + date
    dir_name = os.path.join(save_dir, args.dataset, file_name)

    exp_file_name = os.path.join(dir_name, file_name + '_args.txt')
    log_file_name = os.path.join(dir_name, file_name + '_log.txt')
    dir_name = os.path.dirname(exp_file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
    with open(exp_file_name, 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    writer = SummaryWriter(dir_name)
    logger = Logger(log_file_name, sys.stdout)
    sys.stdout = logger
    logger.log_default(args)

    # Initialize network
    net = Models[args.net](args.dataset, args.bn, args.bn2, "cuda")
    w_init = WeightInit(args.weight_init)
    net.apply(w_init.init_weight)

    # Load dataset
    if args.debug_ds is None:
        num_workers = 1
    else:
        num_workers = 0
    train_loader, test_loader, n_class, input_dim, data_range, eps_test = get_data_loader(args.dataset,
                                                                                          args.bs,
                                                                                          num_workers,
                                                                                          use_cuda,
                                                                                          args.debug_ds,
                                                                                          args.data_augmentation)

    if args.eps_test is None:
        eps_test = eps_test  # set eps_test to default of dataset
    else:
        eps_test = args.eps_test

    if args.eps_test > args.eps_end:
        print(f"Warning: eps_test f{eps_test} is bigger than eps_end f{args.eps_end} for training.")

    # Translate Net
    net_abs = Sequential.from_concrete_network(net, input_dim) if args.saved_net is None else torch.load(args.saved_net)
    net_abs.to("cuda" if use_cuda else "cpu")
    if args.cert_reg:
        net_abs.set_detach_bounds(False)

    # Define Optimizer
    if args.opt == 'SGD':
        net_abs.optimizer = torch.optim.SGD(net_abs.parameters(), lr=args.lr, momentum=args.momentum,
                                            weight_decay=args.wd)
    elif args.opt == 'Adam':
        net_abs.optimizer = torch.optim.Adam(net_abs.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        raise RuntimeError('Unsupported Optimizer: {}'.format(args.opt))

    # Define training schedules
    n_samples = len(train_loader)
    kappa_scheduler = Scheduler(args.start_epoch_kappa*n_samples,
                                args.end_epoch_kappa*n_samples,
                                args.kappa_start,
                                args.kappa_end)
    eps_scheduler = Scheduler(args.start_epoch_eps*n_samples,
                              args.end_epoch_eps*n_samples,
                              args.eps_start,
                              args.eps_end,
                              args.eps_scheduler_mode,
                              s=len(train_loader))
    if args.start_anneal_lambda is not None:
        lambda_scheduler = Scheduler(args.start_anneal_lambda * n_samples,
                                     args.end_anneal_lambda * n_samples,
                                     args.lambda_ratio,
                                     args.end_lambda_ratio)
    else:
        lambda_scheduler = None
    adv_steps_scheduler = Scheduler(args.start_epoch_adv,
                                    args.end_epoch_adv,
                                    0,
                                    1)
    if args.end_clip_norm is not None:
        clip_norm_scheduler = Scheduler(args.start_epoch_clip_norm,
                                        args.end_epoch_clip_norm,
                                        args.clip_norm,
                                        args.end_clip_norm)
    else:
        clip_norm_scheduler = None

    if args.lr_schedule == "MultiStepLR":
        default_schedule = [200, 250, 300] if args.dataset == "cifar10" else [100, 150]
        schedule = args.custom_schedule if args.custom_schedule is not None else default_schedule
        net_abs.lrschedule = torch.optim.lr_scheduler.MultiStepLR(net_abs.optimizer, gamma=args.lr_decay_factor,
            milestones=schedule)
    elif args.lr_schedule == "CyclicLR":
        cyclic_lr = eval(args.cyclic_lr)
        net_abs.lrschedule = torch.optim.lr_scheduler.CyclicLR(net_abs.optimizer, cyclic_lr[0], cyclic_lr[1], step_size_up=args.cycle_len, cycle_momentum= args.opt != 'Adam')
    else:
        assert False, f"Unknown learning rate schedule {args.lr_schedule}"

    if args.saved_net is None or args.eval_only:
        start_epoch = 1
    else:
        if args.start_epoch is None:
            start_epoch = net_abs.get_epoch() + 1
        else:
            start_epoch = args.start_epoch

    # Do eval/training
    if args.eval_only:
        with torch.no_grad():
            # Test
            train_net(net_abs, 0, False, args, test_loader, input_dim, data_range, eps_test, use_cuda,
                  adv_steps_scheduler, eps_scheduler, clip_norm_scheduler, lambda_scheduler, kappa_scheduler, writer)

    else:
        for epoch in range(start_epoch, args.epochs + 1):
            net_abs.lrschedule.step()

            # Do an epoch of training
            train_net(net_abs, epoch, True, args, train_loader, input_dim, data_range, eps_test, use_cuda, adv_steps_scheduler,
                  eps_scheduler, clip_norm_scheduler, lambda_scheduler, kappa_scheduler, writer)

            # Test
            if epoch == args.epochs or not args.no_test:
                with torch.no_grad():
                    train_net(net_abs, epoch, False, args, test_loader if args.debug_ds is None else train_loader, input_dim,
                          data_range, eps_test, use_cuda, adv_steps_scheduler, eps_scheduler, clip_norm_scheduler,
                          lambda_scheduler, kappa_scheduler, writer)

            torch.save(net_abs, os.path.join(dir_name, file_name + "_epoch_current.pynet"))

            if epoch % args.save_freq == 0:
                torch.save(net_abs,os.path.join(dir_name, file_name + "_epoch_" + str(epoch) + ".pynet"))

        # save final model
        pynet_path = os.path.join(dir_name, file_name + ".pynet")
        onnx_path = pynet_path[:-6] + ".onnx"

        torch.save(net_abs, pynet_path)
        writer.close()
        net_abs.eval()
        torch.onnx.export(net_abs.to("cpu"), torch.tensor(torch.rand((1,*input_dim)), device="cpu", dtype=torch.float32),
                          onnx_path, verbose=True, input_names=["input"], output_names=["output"])
        # Convert model for MN-BaB
        convers_args = Bunch(model_path=pynet_path, net=args.net, bn=args.bn, bn2=args.bn2, dataset=args.dataset, no_bn_merge=False)
        convert_state_dict(convers_args)

        print("End time:", datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p"), "File_name :", file_name)


if __name__ == "__main__":
    main()
