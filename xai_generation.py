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
### method parameters ###
#########################
parser.add_argument("--baseline", type=str, default='',
                    help = 'Indicates the type of baseline: "mean", "random", "uniform", "black" or "white", '
                           'use default by metric if not specified')

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
    # Get arguments
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

    # Use GPU
    if args.gpu:
        model = model.cuda()
    
    # Get method if no checkpoint provided
    if args.npz_checkpoint:
        method = None
    else:
        method= get_method(args.method, model, batch_size=args.batch_size)

    # Limit validation size if required in arguments (mostly for debugging purpose)
    if args.limit_val != 0:
        subset_indices  = np.random.randint(0, high=(len(dataset)-1), size=min(args.limit_val, len(dataset)))
        subset = torch.utils.data.Subset(dataset, subset_indices)
    else:
        subset = dataset

    # Get dataloader for generating the maps
    val_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle = False)

    scores = []
    saliencies_maps = []

    # Load precomputed maps if a checkpoint is specified, generate them otherwise
    if args.npz_checkpoint:
        print("loading saliencies maps from npz")
        try:
            saliencies_maps = torch.tensor(np.load(path.join(args.npz_folder, args.npz_checkpoint))['arr_0'])
        except:
            saliencies_maps = torch.tensor(np.load(args.npz_checkpoint)['arr_0'])
    else:
        for X, y in tqdm(val_loader, total=len(val_loader), desc = "Generating saliency maps"):

            # Store images and labels to GPU
            if args.gpu:
                X = X.cuda()
                y = y.cuda()

            # One image at a time since some methods process each image multiple times using internal batches
            for i in range(X.shape[0]):
                # generate saliency map depending on the choosen method (sum over channels for gradient methods)
                saliency_map = method.attribute(X[i].unsqueeze(0), target=y[i]).sum(1)

                saliencies_maps.append(saliency_map)

        # Convert the list of maps into one tensor
        saliencies_maps = torch.stack(saliencies_maps)

        # Save the maps into a npz file if required
        if args.save_npz:
            print("saving saliencies to npz")
            npz_name = args.method + "_" + args.model + "_" + args.dataset_name
            np.savez(path.join(args.npz_folder, npz_name), saliencies_maps.cpu().numpy())

    # Create a XAI dataset and loader. Useful to get the image with the corresponding map
    xai_dataset = XAIDataset(subset, saliencies_maps)
    xai_loader = torch.utils.data.DataLoader(xai_dataset, batch_size=batch_size, shuffle = False)



    # Compute metrics or skip it if required (in case of only generation)
    if not args.skip_metrics:
        # Perturbation baseline choose, this change the default baseline for metrics using perturb_baseline parameter
        if args.baseline == '':
            perturb_baseline = None
            csv_baseline_suffix = ""
        else:
            perturb_baseline = args.baseline
            csv_baseline_suffix= "_baseline_" + perturb_baseline

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

            #get image shape
            img_shape = list(X[0].shape)

            scores_saliency = get_results(model,
                                          name = args.metrics,
                                          x_batch = X,
                                          y_batch = y,
                                          a_batch =A,
                                          perturb_baseline = perturb_baseline,
                                          device = device,
                                          xai_method = lambda model, inputs, targets, batch_size = 1,
                                                           **kwargs: XAI_for_Quantus(args.method, model, inputs, targets, img_shape, device, batch_size)
                                        )

            scores.append(scores_saliency)

        # Stack results by batches if the results are dict, else concatenate them by images
        if isinstance(scores[0], dict):
            scores = np.stack(scores)
        else:
            scores = np.concatenate(scores)

        # save metrics in csv files
        scores_df = pd.DataFrame(data=scores, index=None, columns=None)
        if args.npz_checkpoint:
            csv_name = args.npz_checkpoint.split('/')[-1].split('.')[0] + "_" + args.metrics + csv_baseline_suffix + ".csv"
        else:
            csv_name = args.method + "_" + args.model + "_" + args.dataset_name + "_" + args.metrics + csv_baseline_suffix + ".csv"
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

def XAI_for_Quantus(method, model, inputs, targets, img_shape,  device, batch_size):
    XAI_method = get_method(method, model)
    list_maps = []

    for i in range(batch_size):
        maps = XAI_method.attribute(torch.Tensor(inputs.reshape([batch_size] + img_shape)[i]).unsqueeze(0).to(device),
                                target = torch.Tensor(targets)[i]).sum(1)
        list_maps.append(maps)

    list_maps = torch.stack(list_maps)

    #Upsample images if saliency's shape != image's shape
    if list_maps.shape[-2:] != img_shape[-2:]:
        list_maps = torch.nn.functional.interpolate(list_maps, img_shape[-2:], mode='bilinear')

    return list_maps.cpu().numpy()




if __name__ == "__main__":
    main()
