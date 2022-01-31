import random
import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F

from matplotlib import image as mpimg
from matplotlib import pyplot as plt


import sys


import argparse
from tqdm import tqdm

from datasets import get_dataset
from models import get_model
from methods import get_method
from metrics import get_results


sys.path.append("Quantus")

import quantus


parser = argparse.ArgumentParser(description="Generate xai maps")

#########################
#### data parameters ####
#########################
parser.add_argument("--dataset_name", type=str, default='imagenet',
                    help="dataset name")
parser.add_argument("--dataset_root", type=str, default='.',
                    help="root folder for all datasets. Complete used path is `dataset_root/dataset_name`")

#########################
#### model parameters ###
#########################
parser.add_argument("--model", type=str, default='resnet50',
                    help="model architecture")

#########################
### method parameters ###
#########################
parser.add_argument("--method", type=str, default='smoothgrad',
                    help = "xai method")

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

parser.add_argument("--val_size", type=int, default=200,
                    help="Validation size")

parser.add_argument("--batch_size", type=int, default=16,
                    help="Batch size")


parser.add_argument("--metrics", type=str, default="Faithfulness Correlation", help = "metrics used for benchmarking")


def main():
    # Get arguements
    global args
    args = parser.parse_args()
    batch_size = args.batch_size

    # Seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get dataset
    dataset, n_output = get_dataset(args.dataset_name, args.dataset_root)

    # Get model
    model = get_model(args.model, n_output)
    model = model.eval()

    if args.gpu:
        model = model.cuda()

    
    # Get method
    method= get_method(args.method, model)

    #Checking accuracy
    #acc_check = accuracy_checking(model, dataset)
    #print("[Check] Accuracy: ", acc_check)

    #Random sample
    subset_indices  = np.random.randint(0,high=len(dataset),size= args.val_size)
    subset = torch.utils.data.Subset(dataset, subset_indices)


    val_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle = False)

    scores = []

    for j, (X, y) in tqdm(enumerate(val_loader), total=len(val_loader), desc = "Processing validation subset"):
        saliencies_maps = []

        if args.gpu:
            X = X.cuda()
            y = y.cuda()
        
        # One image at a time since some methods process each image multiple times using internal batches
        for i in range(X.shape[0]):

            # First forward pass
            with torch.no_grad():
                out = model(X[i].unsqueeze(0))

            # generate saliency map depending on the choosen method (sum over channels for gradient methods)
            saliency_map = method.attribute(X[i].unsqueeze(0), target=y[i]).sum(1)

            saliencies_maps.append(saliency_map)

        saliencies_maps = torch.stack(saliencies_maps)

        # device
        if args.gpu:
            device = "cuda"
        else:
            device = "cpu" 

        """Compute metrics per batch
        x_batch: batch of images, y_batch: batch of labels, s_batch: batch of saliencies_maps
        s_batch: batch of masks for localisation metrics
        
        """
        scores_saliency = get_results(model, name = args.metrics,
            x_batch = X.cpu().detach().numpy(), y_batch = y.cpu().detach().numpy(),
             a_batch =saliencies_maps.cpu().detach().numpy(), s_batch = None, device = device)

        scores.append(scores_saliency)

    print(scores)


def accuracy_checking(model, dataset, nr_samples = 100):
    i = 0
    correct = 0
    for X, y in dataset:
        with torch.no_grad():
           output = model(X.unsqueeze(0))
           output = F.softmax(output, dim=1)
           correct += (output.argmax(axis  = 1) == y).float().sum()

        i+=1
        if i > nr_samples:
            break

    return correct.cpu().detach().numpy()*100/(nr_samples)


if __name__ == "__main__":
    main()
