import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


def repeat_func(x):
    return x.repeat(3, 1, 1)


def get_transforms(dataset_name):
    if dataset_name == 'cifar10':
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

        test_transforms = transforms.Compose([
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    elif dataset_name == 'mnist':
        train_transforms = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Lambda(repeat_func),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        test_transforms = train_transforms
    elif dataset_name == 'svhn':
        train_transforms = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        test_transforms = train_transforms
    else:
        train_transforms = None
        test_transforms = None

    return train_transforms, test_transforms


def get_datasets(dataset_name, train_transforms=None, test_transforms=None):
    if dataset_name == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10('data/', train=True, transform=train_transforms, download=True)
        test_dataset = torchvision.datasets.CIFAR10('data/', train=False, transform=test_transforms, download=True)
        num_outputs = 10
    elif dataset_name == 'mnist':
        train_dataset = torchvision.datasets.MNIST('data/', train=True, transform=train_transforms, download=True)
        test_dataset = torchvision.datasets.MNIST('data/', train=False, transform=test_transforms, download=True)
        num_outputs = 10
    elif dataset_name == 'svhn':
        train_dataset = torchvision.datasets.SVHN('data/', split='train', transform=train_transforms, download=True)
        test_dataset = torchvision.datasets.SVHN('data/', split='test', transform=test_transforms, download=True)
        num_outputs = 10
    else:
        raise NotImplementedError(f'No such data set {dataset_name}')

    return train_dataset, test_dataset, num_outputs


def get_dataloader(dataset_name, batch_size=128, pin_memory=False, shuffle=True, only_test_transforms=False, num_workers=0):
    train_transforms, test_transforms = get_transforms(dataset_name)
    if only_test_transforms:
        train_transforms = test_transforms

    train_dataset, test_dataset, num_outputs = get_datasets(dataset_name, train_transforms, test_transforms)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                      pin_memory=pin_memory)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                      pin_memory=pin_memory)

    return train_dataloader, test_dataloader, num_outputs


def get_train_test_val_dataloaders(dataset_name, batch_size=128, pin_memory=False, num_workers=0, valid_size=0.15,
                                   only_test_transforms=False):

    train_transforms, test_transforms = get_transforms(dataset_name)
    if only_test_transforms:
        train_transforms = test_transforms

    train_dataset, test_dataset, num_outputs = get_datasets(dataset_name, train_transforms, test_transforms)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.seed(1)
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, sampler=train_sampler)
    valid_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, sampler=valid_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, test_loader, valid_loader