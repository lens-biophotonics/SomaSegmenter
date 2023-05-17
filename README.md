# Tensorflow-based Neuron Soma Segmentation
## The Project
This project allows for neuronal soma segmentation in fluorescence microscopy imaging datasets with the use for a parametrized family of deeplearning-based models based on the original *U-Net* model by Ronneberger et al. with some additional features such as *residual links* and tile-based frame reconstruction.

**keywords**: neuron segmentation, UNET, residual links, tiled inference, fluorescence imaging

## Requirements
The algorithm has a list of dependencies that need to be satisfied, it's advisable to create a dedicated **python 3** environment.

The required packages are
- `tensorflow=2.0.0`
- `numpy`
- `matplotlib`
- `scikit-image`
- `scikit-learn`
- `scipy`
- `pandas`
- `gitpython`
- `tifffile`
- `pyyaml`

If you're using Anaconda you can use the included configuration file to automatically generate a  conda environment with all the required dependencies

`conda env create -f neuron_segmentation_env.yml`

This will create a new env called `neuron_segmentation` in which you can execute the code in this repo.

## Basic Usage
The two main scripts that are needed for basic usage are
- `training.py`
- `predict.py`

that are used, respectively for the training and inference phase.

### Model training
 You can train a new model using the `training.py` script.
 The training algorithm accepts input data formatted as multipage `.tif`that need to be placed in the same directory and named as follows:
 - `train_frames.tif`
 - `train_masks.tif`
 - `val_frames.tif`
 - `val_masks.tif`
 - `test_frames.tif`
 - `test_masks.tif`
 
  This assumes you have already splitted your dataset into *train* , *validation* and *test* partitions and combined your examples into single multipage `.tif`files.
  Specifically, `*_frames.tif` files can contain multi-channel images while the labels `*_masks.tif` need to be 8-bit single-channel images ( if binary labels are used positive examples should be marked with 255 while background should be 0)

For a typical training session you need to specify 
- dataset's location with `-d` parameter
- output path with `-o` parameter 
- number of training epochs with `-e` parameter
- training batch size with `-b`

 The resulting training command will be
 
`python training.py -d /home/phil/dataset -o /home/phil/training_run -e 100 -b 50`

Writing long sequence of arguments in the CLI can be tedious and error-prone, you can avoid re-specifiyng arguments everytime by using a simple configuration file.
The configuration file is just a sequence of all the arguments you want to use, separated by a newline.

We can create a config file for the above run as simply as
```
-d /home/phil/dataset
-o /home/phil/training_run
-e 100
-b 50
```
and then load the arguments by using the `--file` special argument

`python training.py --file config_file`

You can override config file arguments by re-specifying them when calling the script, this can result handy when performing multiple training session while changing only some of the parameters :  

`python training.py --file config_file -e 200`
  
this command would run for 200 epochs instead of the 100 defined in the config file. 


Note that these are not the only usable parameters: you can find in-depth details on all the possible arguments by using the `--help` option  
`python training.py --help`

### Inference on new images
Model inference can be done using the `predict.py` script.
This algorithm incorporates a tiled prediction strategy (You can find details in`tp2d.py`) that allows for inference on arbitrary-sized input inference, regardless of the shape of the receptive field of the trained model.

For a typical prediction run you would need to specify
- the path to the image you want to make inference on, with the `-i` argument
- the output path, with the `-o` parameter
- which trained model you want to use, with `-m`

`python predict.py -i /home/phil/dataset/test.tif -o /home/phil/prediction_run -m /home/phil/training_run/weights.100.hdf5`

Again, you can use config files with the `--file` option and view all the possible parameters with the `--help` option.

Lastly, you can run trained models on a stack of images by using a multipage `.tif` file as input: the output prediction will be a same-sized `.tif` file as well.

## References

 - Ronneberger et al. - [*U-Net: Convolutional Networks for Biomedical Image Segmentation*](https://arxiv.org/abs/1505.04597)
 - He et al. - [*Deep Residual Learning for Image Recognition*](https://arxiv.org/abs/1512.03385)

## Planned Features
- *Advanced Usage* README section
- Instance segmentation integration
- `tensorflow 2.1` support
## Contacts

**Author:**

Filippo Maria Castelli  
castelli@lens.unifi.it  
LENS, European Laboratory for Non-linear Spectroscopy  
Via Nello Carrara 1  
50019 Sesto Fiorentino (FI), Italy
