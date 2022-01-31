from os import path

import pandas as pd

import torch

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


def get_dataset(name, root, conventional = True):
    cur_dict = datasets_dict[name]
    if not conventional:
      dataset = cur_dict["class_fn"](path.join(root, name), split=cur_dict['split'], transform=cur_dict["transform"])
    else:
      dataset = cur_dict["class_fn"](root, split=cur_dict['split'], transform=cur_dict["transform"])

    print(len(dataset))
    try:
        print("Use preselected indexes")
        subset_indices = pd.read_csv(cur_dict['indices_csv'], header=None)[0].to_numpy()
        subset = torch.utils.data.Subset(dataset, subset_indices)
    except:
        subset = dataset

    return subset, cur_dict["n_output"]