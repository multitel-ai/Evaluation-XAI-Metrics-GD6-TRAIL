# ICIP2022-GD6-TRAIL

## Settings to run the code

* Install the requirements.txt file, for exemple: `pip install -r requirements.txt`
* You should download the dataset files (ILSVRC2012_img_val.tar and ILSVRC2012_devkit_t12.tar.gz) inside a folder named imagenet (no need to extract, it's done automatically)
* Create a folder named npz to save npz saliency maps.
* Create a folder csv to save metric results.
* (Optional: If you want to use RISE, you should should also clone RISE repository, using `git clone https://github.com/eclique/RISE`)
* (Optional: If you want to use Polycam, you should should also clone Polycam repository, using `git clone https://github.com/andralex8/polycam`)
* (Optional: If you want to use CAMERAS, you should should also clone CAMERAS repository, using `git clone https://github.com/VisMIL/CAMERAS`)

nb: It appears that "import quantus" does not work after a "pip install quantus". This is the reason why the Quantus' repository is cloned here.


## Script usage

### Introduction
The main script file to generate maps and compute metrics is `xai_generation.py`. Some common usages are shown bellow. You can also print help using `python xai_generation.py --help`.

### Generate npz
To avoid regenerating maps when not needed, they can be exported as .npz files
You can use `python xai_generation.py --skip_metrics --save_npz --method METHOD_NAME` where METHOD_NAME is the choosen method.

### Compute metrics from npz checkpoint
To compute the metrics using a previously generated npz checkpoint you can use
`python xai_generation.py --npz_checkpoint NPZ_NAME.npz --method METHOD_NAME --metrics METRIC_NAME` , where NPZ_NAME.zip is the name of the npz file, METHOD_NAME is the name of the method and METRIC_NAME is the name of the metric.

### Compute metrics from scratch
The command is the same as the above, except that you don't specify a checkpoint, then the maps will be generated before computing the metrics
`python xai_generation.py --method METHOD_NAME --metrics METRIC_NAME`
