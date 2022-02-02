from os import path

import pandas as pd

import torch
from torch.utils.data import Dataset

from torchvision import transforms

from torchvision.datasets.imagenet import ImageNet

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
}


def get_dataset(name, root):
    cur_dict = datasets_dict[name]
    dataset = cur_dict["class_fn"](path.join(root, name), split=cur_dict['split'], transform=cur_dict["transform"])

    try:
        print("Use preselected indexes")
        subset_indices = pd.read_csv(cur_dict['indices_csv'], header=None)[0].to_numpy()
        subset = torch.utils.data.Subset(dataset, subset_indices)
    except:
        subset = dataset

    return subset, cur_dict["n_output"]


class XAIDataset(Dataset):
    def __init__(self, dataset, xai):
        self.dataset = dataset
        self.xai = xai

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.xai[idx]