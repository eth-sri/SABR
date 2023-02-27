import argparse
import random
import numpy as np
import torch
import re


RANDOM_SEED = 3
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)


def parse_conversion_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str, default="./models/SP1_B.pynet", help='model to test')
    parser.add_argument('--net', type=str, default='CNN7')
    parser.add_argument('--bn', default=False, action='store_true')
    parser.add_argument('--bn2', default=False, action='store_true')
    parser.add_argument('-D', '--dataset', type=str, default="mnist", help='pick dataset')
    parser.add_argument('--no_bn_merge', default=False, action='store_true', help='pick dataset')

    args = parser.parse_args()
    return args


def convert_state_dict(args):
    merge_bn = not args.no_bn_merge
    net_abs = torch.load(args.model_path)
    print(net_abs)
    if isinstance(net_abs,dict):
        if "state_dict" in net_abs:
            state_dict = net_abs["state_dict"]
        else:
            state_dict = net_abs
    else:
        state_dict = net_abs[0].state_dict()

    # offset = -len([k for k in state_dict.keys() if k.endswith(".sigma")])

    keys_to_be_deleted = [k for k in state_dict.keys() if k.endswith("deepz_lambda")]
    keys_to_be_deleted += [k for k in state_dict.keys() if k.endswith("num_batches_tracked")]
    keys_to_be_deleted += [k for k in state_dict.keys() if k.endswith(".mean")]
    keys_to_be_deleted += [k for k in state_dict.keys() if k.endswith(".sigma")]

    for k in keys_to_be_deleted:
        state_dict.pop(k)

    # layer_idxs = sorted(list({int(re.match("layers\.([0-9]+)\..*",k).group(1)) for k in state_dict.keys()}))
    layer_idxs = sorted(list({int(re.match("(blocks\.)?(layers\.)?([0-9]+)\..*", k).group(3)) for k in state_dict.keys()}))

    offset = -min(layer_idxs)
    conv = True
    last_lin = None

    new_state_dict = {}

    for layer_idx in layer_idxs:
        copy_params = True
        layer_keys = [k for k in state_dict.keys() if f".{layer_idx}." in k or k.startswith(f"{layer_idx}.")]
        if any(["running_mean" in k for k in layer_keys]):
            # batch norm layer
            if merge_bn and last_lin is not None:
                # merge BN with preceeding affine layer
                lin_layer_keys = [k for k in new_state_dict.keys() if f".{last_lin+offset}." in k]
                weight_key = [k for k in lin_layer_keys if "weight" in k][0]
                bias_key = [k for k in lin_layer_keys if "bias" in k][0]
                affine_weight = new_state_dict[weight_key]
                affine_bias = new_state_dict[bias_key]

                keys_to_be_deleted += layer_keys
                copy_params = False
                offset -= 1
                bn_weight = state_dict[[k for k in layer_keys if "weight" in k][0]]
                bn_bias = state_dict[[k for k in layer_keys if "bias" in k][0]]
                bn_var = state_dict[[k for k in layer_keys if "var" in k][0]]
                bn_mean = state_dict[[k for k in layer_keys if "mean" in k][0]]
                if isinstance(net_abs, dict):
                    bn_eps = 1e-5
                else:
                    bn_eps = net_abs[0][layer_idx].eps

                effective_bias = bn_bias + bn_weight * (affine_bias - bn_mean) / torch.sqrt(bn_var + bn_eps)
                effective_weight = affine_weight * (bn_weight/torch.sqrt(bn_var + bn_eps)).unsqueeze(1)
                new_state_dict[weight_key] = effective_weight
                new_state_dict[bias_key] = effective_bias

        else:
            weight_key = [k for k in layer_keys if "weight" in k][0]
            if state_dict[weight_key].dim() == 4:
                conv = conv
                last_lin = None
            elif state_dict[weight_key].dim() == 2:
                conv = False
                last_lin = layer_idx
            else:
                assert False, f"Unexpected weight dimensionality: {state_dict[weight_key].dim()}"

        if copy_params:
            for k in layer_keys:
                new_key = re.sub("(blocks\.)?(layers\.)?[0-9]+\.", f"blocks.{layer_idx + offset}.", k)
                new_key = re.sub("(conv\.)?(linear\.)?", f"", new_key)
                new_state_dict[new_key] = state_dict[k]

    file_name = ".".join(args.model_path.split(".")[0:-1])

    torch.save(new_state_dict, file_name + "_mnbab.pynet")


if __name__ == "__main__":
    args = parse_conversion_args()
    convert_state_dict(args)
