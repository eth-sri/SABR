SABR <img width="100" alt="portfolio_view" align="right" src="http://safeai.ethz.ch/img/sri-logo.svg">
======== 
[SABR](https://openreview.net/forum?id=7oFuxtJtUMH) (Small Adversarial Bounding Boxes), is a novel certified training method, 
based on reducing the regularization induced by training with imprecise bound propagation methods (IBP) by only propagating 
small but carefully selected sub-regions of the adversarial input specification. It is enabled by recent precise neural-
network verification methods such as [MN-BaB](https://www.sri.inf.ethz.ch/publications/ferrari2022complete).


### Setup
Create and activate a conda environment
```
conda create --name SABR python=3.10.4
conda activate SABR
```
Install the requirements
```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

Add the main directory to the PYTHONPATH (make sure you are in the top level directory)
```
export PYTHONPATH=$PWD:$PYTHONPATH
```

To download the TinyImageNet dataset navigate into the `data` directory and execute:
```
bash tinyimagenet_download.sh
```

Please use the following links to download our trained models:

[Mnist 0.1](https://files.sri.inf.ethz.ch/sabr/SABR_mnist_01_best.pynet)  
[Mnist 0.3](https://files.sri.inf.ethz.ch/sabr/SABR_mnist_01_best.pynet)  
[Cifar10 2/255](https://files.sri.inf.ethz.ch/sabr/SABR_cifar10_2_best.pynet)  
[Cifar10 8/255](https://files.sri.inf.ethz.ch/sabr/SABR_cifar10_8_best.pynet)  
[TinyImageNet 1/255](https://files.sri.inf.ethz.ch/sabr/SABR_TIN_1_best.pynet)  

### Training with SABR
To reproduce our main results (Table 1), train the models by executing the following commands from the top level directory.
For convenience's sake, we have included all the trained networks under `models`.
We have observed some hardware sensitivity and recommend reproducing our results on Nvidia RTX 2080 Tis.

MNIST 0.1
```
python src/main.py  --net CNN7 --bn --bn2 --lr 0.0005 --custom_schedule 50  60 --epochs 70 --eps_end 0.1 --dataset mnist --l1 1e-5 --end_epoch_eps 22 --cert_reg bound_reg --bs 256 --lambda_ratio 0.4 --eps_test 0.1
```

MNIST 0.3
```
python src/main.py  --net CNN7 --bn --bn2 --lr 0.0005 --custom_schedule 50  60 --epochs 70 --eps_end 0.3 --dataset mnist --l1 1e-6 --end_epoch_eps 22 --cert_reg bound_reg --bs 256 --lambda_ratio 0.6 --eps_test 0.3
```

CIFAR-10 2/255
```
python src/main.py  --net CNN7 --bn --bn2 --lr 0.0005 --custom_schedule 120 140 --epochs 160 --eps_end 0.00784313725 --dataset cifar10 --l1 1e-6 --end_epoch_eps 82 --cert_reg bound_reg --eps_test 0.00784313725 --data_augmentation fast --lambda_ratio 0.1 --shrinking_ratio 0.4 --use_shrinking_box --shrinking_relu_state cross --shrinking_method to_zero_shrinking_box
```

CIFAR-10 8/255
```
python src/main.py  --net CNN7 --bn --bn2 --lr 0.0005 --custom_schedule 140 160 --epochs 180 --eps_end 0.03137 --dataset cifar10 --end_epoch_eps 82 --cert_reg bound_reg --eps_test 0.03137 --data_augmentation fast --lambda_ratio 0.7
```

TinyImageNet 1/255
```
python src/main.py  --net CNN7 --bn --bn2 --lr 0.0005 --custom_schedule 140 160 --epochs 180 --eps_end 0.00392157 --eps_test 0.00392157 --dataset tinyimagenet --end_epoch_eps 82 --cert_reg bound_reg --lambda_ratio 0.4 --l1 1e-6
```


To train the models used in our ablation studies to compare IBP, SABR, and PGD use:

IBP
```
python src/main.py --box_attack centre --net CNN7 --bn --bn2 --lr 0.0005 --custom_schedule 120 140 --epochs 160 --eps_end 0.00784313725 --dataset cifar10 --end_epoch_eps 82 --cert_reg bound_reg --eps_test 0.00784313725 --data_augmentation fast --lambda_ratio 1.0
```

SABR
```
python src/main.py  --net CNN7 --bn --bn2 --lr 0.0005 --custom_schedule 120 140 --epochs 160 --eps_end 0.00784313725 --dataset cifar10 --l1 1e-5 --end_epoch_eps 82 --cert_reg bound_reg --eps_test 0.00784313725 --data_augmentation fast --lambda_ratio 0.05
```

PGD
```
python src/main.py --box_attack pgd_concrete  --net CNN7 --bn --bn2 --lr 0.0005 --custom_schedule 120 140 --epochs 160 --eps_end 0.00784313725 --dataset cifar10 --l1 1e-5 --end_epoch_eps 82 --cert_reg bound_reg --eps_test 0.00784313725 --data_augmentation fast --lambda_ratio 0.0
```

### Certification
We use MN-BaB for certification:  
Follow the the installation instructions here (use the `SABR_ready` branch):  
[github.com/eth-sri/mn-bab/tree/SABR_ready](https://github.com/eth-sri/mn-bab/tree/SABR_ready)

The final models are automatically converted to an MN-BaB compatible format. To convert intermediate models, use the following command:
```
python ./Experiments/convert_to_dict_mnbab.py --bn --bn2 --dataset cifar10 --model <path to model>
```

Copy the configs provided under `MNBAB_configs` to the folder `configs` in MN-BAB (adapt the model paths in the config if needed).  

To run the actual certification use the following command with the chosen config (e.g. here for CIFAR-10 2/255):
```
python src/verify.py -c configs/cifar10_small_eps.json
```

Citing This Work
----------------------

If you find this work useful for your research, please cite it as:

```
@inproceedings{
    mueller2023certified,
    title={Certified Training: Small Boxes are All You Need},
    author={Mark Niklas M{\"{u}}ller and Franziska Eckert and Marc Fischer and Martin Vechev},
    booktitle={International Conference on Learning Representations},
    year={2023},
    url={https://openreview.net/forum?id=7oFuxtJtUMH}
}
```

License and Copyright
---------------------

* Copyright (c) 2023 [Secure, Reliable, and Intelligent Systems Lab (SRI), Department of Computer Science ETH Zurich](https://www.sri.inf.ethz.ch/)
* Licensed under the [MIT License](LICENCE)
