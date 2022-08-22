# ICIP2022-GD6-TRAIL

## Settings to run the code

* Install the requirements.txt file, for example: `pip install -r requirements.txt`, for the txt file in the main folder as well as the requirements.txt file in the quantus folder
* You should download the dataset files (ILSVRC2012_img_val.tar and ILSVRC2012_devkit_t12.tar.gz) inside a folder named imagenet (no need to extract, it's done automatically)
* Create a folder named npz to save npz saliency maps.
* Create a folder csv to save metric results.
* (Optional: If you want to use RISE, you should should also clone RISE repository, using `git clone https://github.com/eclique/RISE`)
* (Optional: If you want to use Polycam, you should should also clone Polycam repository, using `git clone https://github.com/andralex8/polycam`)
* (Optional: If you want to use CAMERAS, you should should also clone CAMERAS repository, using `git clone https://github.com/VisMIL/CAMERAS`)
* (Optional: If you want to use Extremal perturbation, you should should also install TorchRay, `pip install torchray`)

nb: It appears that "import quantus" does not work after a "pip install quantus". This is the reason why the Quantus' repository is cloned here.

## Settings to use the CIFAR10 dataset and pretrained models
* Clone the github repository with pretrained CIFAR10 models `git clone https://github.com/huyvnphan/PyTorch_CIFAR10.git`
* Download the weights of the models: `python weightsDownloadCIFAR.py`
* The dataset is downloaded automatically when using the script usage below.

## Script usage

### Introduction
The main script file to generate maps and compute metrics is `xai_generation.py`. Some common usages are shown below. You can also print help using `python xai_generation.py --help`.

### Generate npz
To avoid regenerating maps when not needed, they can be exported as .npz files
You can use `python xai_generation.py --skip_metrics --save_npz --method METHOD_NAME` where METHOD_NAME is the choosen method.

### Compute metrics from npz checkpoint
To compute the metrics using a previously generated npz checkpoint you can use
`python xai_generation.py --npz_checkpoint NPZ_NAME.npz --method METHOD_NAME --metrics METRIC_NAME` , where NPZ_NAME.zip is the name of the npz file, METHOD_NAME is the name of the method and METRIC_NAME is the name of the metric.

### Compute metrics from scratch
The command is the same as the above, except that you don't specify a checkpoint, then the maps will be generated before computing the metrics
`python xai_generation.py --method METHOD_NAME --metrics METRIC_NAME`

### Training on subpart of a dataset
This command splits the computation of the metric on subsets of the dataset by specifying the start index (`--start_idx`) and the end index(`--end_idx`). It is useful when computation time is very high. For example to compute on the 400 first images out of the 2000 validation images selected from ImageNet: 
`python xai_generation.py --dataset_root '' --npz_checkpoint rise_vgg16_imagenet.npz --model vgg16 --method rise --metrics 'Model Parameter Randomisation' --start_idx 0 --end_idx 400`

