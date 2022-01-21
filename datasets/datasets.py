from os import path

import torchvision
from torchvision import transforms

from torchvision.datasets.imagenet import ImageNet

datasets_dict = {
    'imagenet': {
        'class_fn': ImageNet,
        'n_output': 1000,
        'split': 'val',
        'transform': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize( (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    },
}

def get_dataset(name, root):
    cur_dict = datasets_dict[name]
    return cur_dict["class_fn"](path.join(root, name), split=cur_dict['split'], transform=cur_dict["transform"]), cur_dict["n_output"]
