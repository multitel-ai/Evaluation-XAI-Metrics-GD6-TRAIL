from os import path

import pandas as pd

import torch
from torch.utils.data import Dataset

from torchvision import transforms

from torchvision.datasets.imagenet import ImageNet
from torchvision.datasets.cifar import CIFAR10

# dict containing datasets information and parameters
datasets_dict = {
    'imagenet': {
        'class_fn': ImageNet,
        'n_output': 1000,
        'split': 'val',
        'indices_csv': '2000idx_ILSVRC2012.csv',
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize( (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    },
#     'mnist': {
#         'class_fn': MNIST,
#         'n_output': 10,
#         'train': False,
#         'transform': transforms.Compose([
#             transforms.ToTensor(),
#              transforms.Normalize((0.1307,), (0.3081,)),
#         ])
    
#     },
    'cifar10': {
        'class_fn': CIFAR10,
        'n_output': 10, 
        'train': False,
        'transform': transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    }
}


def get_dataset(name, root):
    """
    Return the Dataset by name
    :param name: name of the dataset to return
    :param root: path to the folder containing all the datasets
    :return: Dataset
    """
    cur_dict = datasets_dict[name]
    if name=='imagenet':
        dataset = cur_dict["class_fn"](path.join(root, name), split=cur_dict['split'], transform=cur_dict["transform"])
    elif name=='cifar10':
        dataset = cur_dict["class_fn"](path.join(root, name), train=cur_dict['train'], transform=cur_dict["transform"], download=True)

    try:
        print("Use preselected indexes")
        subset_indices = pd.read_csv(cur_dict['indices_csv'], header=None)[0].to_numpy()
        subset = torch.utils.data.Subset(dataset, subset_indices)
    except:
        subset = dataset

    return subset, cur_dict["n_output"]


class XAIDataset(Dataset):
    """
    Dataset combining the image Dataset with the saliency maps
    return a tuple of [image, map]
    """
    def __init__(self, dataset, xai):
        self.dataset = dataset
        self.xai = xai

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.xai[idx]