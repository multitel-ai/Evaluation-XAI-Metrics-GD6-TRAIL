import random
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
import torch.nn.functional as F

from matplotlib import image as mpimg
from matplotlib import pyplot as plt

import sys
from os import path

import argparse
from tqdm import tqdm

from datasets import get_dataset, XAIDataset
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
### saving parameters ###
#########################
parser.add_argument("--csv_folder", type=str, default='csv',
                    help = "csv folder to save metrics")
parser.add_argument("--npz_folder", type=str, default='npz',
                    help = "npz folder to save or load xai maps id required")

parser.add_argument("--save_npz", dest='save_npz', action='store_true',
                    help = "save xai maps in a npz file")
parser.set_defaults(save_npz=False)

parser.add_argument("--npz_checkpoint", type=str, default='',
                    help = "use this option to load a checkpoint npz for metric computation, skip map computation if used")

parser.add_argument("--skip_metrics", dest='skip_metrics', action='store_true',
                    help = "skip metrics computation, useful to just produce the maps without metrics")
parser.set_defaults(skip_metrics=False)

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

parser.add_argument("--limit_val", type=int, default=0,
                    help="Limit validation size. Default to 0 => no limitation")

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

    
    # Get method if no checkpoint provided
    if args.npz_checkpoint:
        method = None
    else:
        method= get_method(args.method, model, batch_size=args.batch_size)

    #Checking accuracy
    #acc_check = accuracy_checking(model, dataset)
    #print("[Check] Accuracy: ", acc_check)

    # Limit val size
    if args.limit_val != 0:
        subset_indices  = np.random.randint(0, high=(len(dataset)-1), size=min(args.limit_val, len(dataset)))
        subset = torch.utils.data.Subset(dataset, subset_indices)
    else:
        subset = dataset

    val_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle = False)

    scores = []
    saliencies_maps = []

    if args.npz_checkpoint:
        print("loading saliencies maps from npz")
        try:
            saliencies_maps = torch.tensor(np.load(path.join(args.npz_folder, args.npz_checkpoint))['arr_0'])
        except:
            saliencies_maps = torch.tensor(np.load(args.npz_checkpoint)['arr_0'])
    else:
        for X, y in tqdm(val_loader, total=len(val_loader), desc = "Generating saliency maps"):

            if args.gpu:
                X = X.cuda()
                y = y.cuda()

            # One image at a time since some methods process each image multiple times using internal batches
            for i in range(X.shape[0]):
                # generate saliency map depending on the choosen method (sum over channels for gradient methods)
                saliency_map = method.attribute(X[i].unsqueeze(0), target=y[i]).sum(1)

                saliencies_maps.append(saliency_map)

        saliencies_maps = torch.stack(saliencies_maps)

        if args.save_npz:
            print("saving saliencies to npz")
            npz_name = args.method + "_" + args.model + "_" + args.dataset_name
            np.savez(path.join(args.npz_folder, npz_name), saliencies_maps.cpu().numpy())

    xai_dataset = XAIDataset(subset, saliencies_maps)
    xai_loader = torch.utils.data.DataLoader(xai_dataset, batch_size=batch_size, shuffle = False)

    if not args.skip_metrics:
        for (X, y), A in tqdm(xai_loader, desc="Computing metrics"):
            # device
            if args.gpu:
                device = "cuda"
            else:
                device = "cpu"

            """Compute metrics per batch
            x_batch: batch of images, y_batch: batch of labels, s_batch: batch of saliencies_maps
            s_batch: batch of masks for localisation metrics
            
            """
            scores_saliency = get_results(model,
                                          name = args.metrics,
                                          x_batch = X,
                                          y_batch = y,
                                          a_batch =A,
                                          s_batch = None,
                                          device = device)

            scores.append(scores_saliency)

        scores = np.concatenate(scores)

        # save metrics in csv files
        scores_df = pd.DataFrame(data=scores, index=None, columns=None)
        if args.npz_checkpoint:
            csv_name = args.npz_checkpoint.split('/')[-1].split('.')[0] + "_" + args.metrics + ".csv"
        else:
            csv_name = args.method + "_" + args.model + "_" + args.dataset_name + "_" + args.metrics + ".csv"
        scores_df.to_csv(path.join(args.csv_folder, csv_name), header=False, index=False)


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
