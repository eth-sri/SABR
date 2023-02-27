import os
import torch
from torchvision import datasets, transforms


class IndexDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.dataset = data

    def __getitem__(self, index):
        data, target = self.dataset.__getitem__(index)
        return data, target, index

    def __len__(self):
        return len(self.dataset)


def get_data_loader(dataset, bs=128, num_workers=0, use_cuda=True, debug_ds=None, data_augmentation="std"):
    data_range = (0, 1)
    data_path = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data"))

    if dataset == "mnist":
        n_class = 10
        eps_test = 0.3
        train_data = datasets.MNIST(data_path, train=True, download=True, transform=transforms.ToTensor())
        test_data = datasets.MNIST(data_path, train=False, download=True, transform=transforms.ToTensor())
    elif dataset == "tinyimagenet":
        n_class = 200
        eps_test = 1./255.
        train_data = datasets.ImageFolder(os.path.join(data_path,"tiny-imagenet-200","train"),
                                          transform=transforms.Compose([
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomCrop(56, padding_mode='edge'),
                                              transforms.ToTensor(),
                                          ]))
        test_data = datasets.ImageFolder(os.path.join(data_path,"tiny-imagenet-200","val"),
                                         transform=transforms.Compose([
                                             # transforms.RandomResizedCrop(64, scale=(0.875, 0.875), ratio=(1., 1.)),
                                             transforms.CenterCrop(56),
                                             transforms.ToTensor(),
                                         ]))
    elif dataset == "cifar10":
        n_class = 10
        eps_test = 8./255.
        if data_augmentation == "std":
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        elif data_augmentation == "fast":
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=2, padding_mode='edge'),
                transforms.ToTensor(),
            ])
        elif data_augmentation == "pixmix":
            transform_train = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4)])
            mixing_set_transform = transforms.Compose([transforms.Resize(36), transforms.RandomCrop(32)])
        else:
            transform_train = transforms.ToTensor()

        train_data = datasets.CIFAR10(data_path, train=True, download=True, transform=transform_train)
        test_data = datasets.CIFAR10(data_path, train=False, download=True, transform=transforms.ToTensor())

    else:
        raise ValueError("dataset not available")

    if debug_ds is not None:
        train_data = cut_dataset(train_data, bs*debug_ds)
        test_data = cut_dataset(test_data, bs*debug_ds)

    train_data_w_indices = IndexDataset(train_data)
    test_data_w_indices = IndexDataset(test_data)

    train_loader = torch.utils.data.DataLoader(
        train_data_w_indices,
        batch_size=bs,
        shuffle=True,
        drop_last=True,
        **({'num_workers': num_workers, 'pin_memory': True} if use_cuda else {})
    )
    test_loader = torch.utils.data.DataLoader(
        test_data_w_indices,
        batch_size=bs,
        shuffle=False,
        drop_last=False,
        **({'num_workers': num_workers, 'pin_memory': True} if use_cuda else {})
    )
    return train_loader, test_loader, n_class, train_data[0][0].size(), data_range, eps_test


def cut_dataset(dataset, num_samples):
    tensors = ["data", "targets"]
    idxs = torch.randperm(len(dataset))[:num_samples]
    for tensor_name in tensors:
        if isinstance(getattr(dataset, tensor_name), list):
            setattr(dataset, tensor_name, torch.tensor(getattr(dataset, tensor_name))[idxs].tolist())
        else:
            setattr(dataset, tensor_name, getattr(dataset, tensor_name)[idxs])
    return dataset