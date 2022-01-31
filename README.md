# ICIP2022-GD6-TRAIL

# Settings to run the code

* Download the two files tar files (ILSVRC2012_img_val.tar and ILSVRC2012_devkit_t12.tar.gz)
* Move the two files in a folder name imagenet (no need to extract, it's done automatically)
* (Optionaly, you can use another root folder for the datasets, then use the --dataset_root option)
* It appears that "import quantus" does not work after a "pip install quantus". This is the reason why the Quantus' repository is cloned here such that we can easily insert the Quantus' path in the Python's path.
