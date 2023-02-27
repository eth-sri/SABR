import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=300, help='number of training epochs')
    parser.add_argument('--seed', type=int, default=12, help='save random seed')

    parser.add_argument('--not_save', default=False, action='store_true')
    parser.add_argument('--no_test', default=False, action='store_true')
    parser.add_argument('--save_freq', type=int, default=50)
    parser.add_argument('--eval_only', default=False, action='store_true')
    parser.add_argument("--experiment_key", default=None, type=str, help="short name to identify experiment")

    parser.add_argument('--net', type=str, default='ConvMedBig', help='network to train')
    parser.add_argument('--saved_net', type=str, default=None)
    parser.add_argument('--start_epoch', type=int, default=None)
    parser.add_argument('--bn', default=False, action='store_true', help="Use BN after Conv layers")
    parser.add_argument('--bn2', default=False, action='store_true', help='Use BN after linear layer for conv nets')
    parser.add_argument('--adv_bn', default=False, action='store_true', help="Use adversarial instead of clean statistics for propagation during training")

    parser.add_argument('--opt', type=str, default="Adam", choices=["SGD", "Adam"], help="optimizer")
    parser.add_argument('--momentum', type=float, default=0.09, help='momentum')
    parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
    parser.add_argument('--l1', type=float, default=None, help='l1 regularization')
    parser.add_argument('--weight_init', type=str, default='IBP', choices=["IBP", "xavier_uni", "xavier_normal", "he_uni", "he_normal", "default"], help='choose weight initialization approach')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_schedule', type=str, default="MultiStepLR", help='learning rate scheduler')
    parser.add_argument('--custom_schedule', type=int, default=None, nargs="+", help='LR schedule for multistep')
    parser.add_argument('--lr_decay_factor', type=float, default=0.2)

    parser.add_argument('--cert_reg', default=None, choices=[None, "bound_reg", "ibp_reg"], help='relu bounds regularization')
    parser.add_argument('--reg_lambda', type=float, default=0.5)
    parser.add_argument('--min_eps_reg', type=float, default=1e-6)

    parser.add_argument('--use_shrinking_box', default=False, action='store_true', help='apply shrink before ReLU layers')
    parser.add_argument('--shrinking_relu_state', default='all', type=str, choices=['all', 'cross', 'above', 'wide'], help='apply shrinking for which ReLUs')
    parser.add_argument('--shrinking_method', default='std_shrinking_box', type=str, choices=['std_shrinking_box', 'random_loc_shrinking_box', 'to_zero_shrinking_box', 'std_upper_shrinking_box'], help='shrink method')
    parser.add_argument('--shrinking_ratio', default=0.2, type=float)

    # parameters for warm up scheduler (kappa in loss function)
    parser.add_argument('--kappa_start', type=float, default=1.0, help='start value of kappa')
    parser.add_argument('--kappa_end', type=float, default=0.0, help='end value of kappa')
    parser.add_argument('--start_epoch_kappa', type=int, default=2, help='start epoch of increase of kappa')
    parser.add_argument('--end_epoch_kappa', type=int, default=2, help='end epoch of increase of kappa')  #MNIST: 15 # CIFAR 30?

    # parameter for epsilon annealing
    parser.add_argument('--eps_start', type=float, default=0.0, help='start value of epsilon')
    parser.add_argument('--eps_end', type=float, default=0.4, help='end value of epsilon')
    parser.add_argument('--start_epoch_eps', type=int, default=2, help='start epoch of increase of epsilon')
    parser.add_argument('--end_epoch_eps', type=int, default=100, help='end epoch of increase of epsilon')  #MNIST: 40 # CIFAR 60?
    parser.add_argument('--eps_scheduler_mode', type=str, default="exp", choices=["smooth", "exp", "linear", "step"], help='How to anneal epsilon')
    parser.add_argument('--eps_test', type=float, default=None, help='test value of epsilon')

    # parameters for small region propagation
    parser.add_argument('--lambda_ratio', type=float, default=0.1, help='ratio of small to large box')
    parser.add_argument('--dimwise_scaling', default=False, action='store_true', help='scale input region intersected with input domain')

    parser.add_argument('--start_anneal_lambda', type=int, default=None, help='start epoch of annealing lambda_ratio')
    parser.add_argument('--end_anneal_lambda', type=int, default=300, help='end epoch of annealing lambda_ratio')
    parser.add_argument('--end_lambda_ratio', type=float, default=0.0, help='end ratio of small to large box')
    parser.add_argument('--start_sound', default=False, action='store_true', help='train sound while eps<eps_unsound_max')
    parser.add_argument('--bn_mode_attack', default="eval", type=str, choices=["eval", "train"])
    parser.add_argument('--box_attack', default="pgd_concrete", type=str, choices=["pgd_concrete", "centre", "concrete_attack"], help='How to find box midpoints')
    parser.add_argument('--box_attack_loss_fn', default="CE", type=str, choices=["CE", "margin", "GAMA"], help='Loss function for attack')

    parser.add_argument('--adv_step_size', type=float, default=0.5)
    parser.add_argument('--adv_step_size_end', type=float, default=None)
    parser.add_argument('--adv_start_steps', type=int, default=8, help='number of steps used for adv box at start')
    parser.add_argument('--adv_end_steps', type=int, default=None, help='number of steps used for adv box at start')
    parser.add_argument('--start_epoch_adv', type=int, default=10, help='start epoch of increase of adv steps')
    parser.add_argument('--end_epoch_adv', type=int, default=250, help='end epoch of increase of adv steps')
    parser.add_argument('--start_adv_loss_2', type=int, default=250, help='start epoch to use 2nd adv loss')
    parser.add_argument('--adv_scale_eps', type=float, default=1.0)

    parser.add_argument('-D', '--dataset', default="mnist", help='pick dataset')
    parser.add_argument('--bs', type=int, default=128)
    parser.add_argument('--debug_ds', type=int, default=None)
    parser.add_argument('--data_augmentation', type=str, default="std", choices=["std", "no", "fast"])
    parser.add_argument('--clip_robust_gradient', default=False, action='store_true', help='Use gradient clipping for robust loss')
    # parameters for clip norm scheduler (gradient clipping)
    parser.add_argument('--clip_norm', type=float, default=1.0, help='value to clip l2 norm grad of robust loss')
    parser.add_argument('--end_clip_norm', type=float, default=None)
    parser.add_argument('--start_epoch_clip_norm', type=int, default=150)
    parser.add_argument('--end_epoch_clip_norm', type=int, default=250)
    parser.add_argument('--clip_combined_gradient', default=None, type=float)

    parser.add_argument('--loss_domain', type=str, choices=["box"], default="box")
    parser.add_argument('--loss_fn', type=str, choices=["CE", "PT1"], default="CE")
    parser.add_argument('--pt1_e', type=float, default=2.0)

    args = parser.parse_args()

    if args.eval_only:
        args.not_save = True

    if args.clip_combined_gradient is not None and args.clip_robust_gradient:
        args.clip_robust_gradient = False
        print(f"Warning: You choose combine_clip_norm f{args.clip_combined_gradient}. clip_robust_gradient (clipping for robust loss only) set to 'False'.")

    if args.adv_bn:
        assert args.bn

    args.save = not args.not_save

    return args
