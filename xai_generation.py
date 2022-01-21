import random
import numpy as np
import torch

import argparse
from tqdm import tqdm

from datasets import get_dataset
from models import get_model
from methods import get_method

parser = argparse.ArgumentParser(description="Generate xai maps")

#########################
#### data parameters ####
#########################
parser.add_argument("--dataset_name", type=str, default='imagenet',
                    help="dataset name")
parser.add_argument("--dataset_root", type=str, default='../input/',
                    help="root folder for all datasets. Complete used path is `dataset_root/dataset_name`")

#########################
#### model parameters ###
#########################
parser.add_argument("--model", type=str, default='resnet50',
                    help="model architecture")

#########################
### method parameters ###
#########################
parser.add_argument("--method", type=str, default='gradcam',
                    help="xai method")

#########################
### other parameters ####
#########################
parser.add_argument("--gpu", dest="gpu", action='store_true',
                    help="use gpu (default)")
parser.add_argument("--cpu", dest="gpu", action='store_false',
                    help="use cpu instead of gpu")
parser.set_defaults(gpu=True)

parser.add_argument("--seed", type=int, default=123456,
                    help="Random seed")



#parser.add_argument("--batch_size", type=int, default=1,
#                    help="max batch size (when saliency method use it), default to 1")

def main():
    # Get arguements
    global args
    args = parser.parse_args()

    # Seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get dataset
    dataset, n_output = get_dataset(args.dataset_name, args.dataset_root)

    # Get model
    model = get_model(args.model, n_output)
    if args.gpu:
        model = model.cuda()

    # Get method
    method= get_method(args.method, model)

    # Loop over the dataset.
    # One image at a time since some methods process each image multiple times using internal batches
    for i in tqdm(range(len(dataset)), desc='generating xai maps'):
        sample, class_idx = dataset[i]
        sample = sample.unsqueeze(0)
        if args.gpu:
            sample = sample.cuda()
            class_idx = torch.tensor(class_idx, device='cuda')

        # First forward pass
        out = model(sample)

        # generate saliency map depending on the choosen method (sum over channels for gradient methods)
        saliency_map = method.attribute(sample, target=class_idx).sum(1)

        # TODO, compute metric

if __name__ == "__main__":
    main()