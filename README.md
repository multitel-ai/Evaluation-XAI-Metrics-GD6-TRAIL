# ICIP2022-GD6-TRAIL

# Settings to run the code

* Download the two files tar files (ILSVRC2012_img_val.tar and ILSVRC2012_devkit_t12.tar.gz)
* Move the validation tar to val folder: mv ILSVRC2012_img_val.tar val/
* Extract the tar file in val/: cd val && tar -xvf ILSVRC2012_img_val.tar && sh valprep.sh;
* It appears that "import quantus" does not work after a "pip install quantus". This is the reason why the Quantus' repository is clone here such that we can easily insert the Quantus' path in the Python's path.