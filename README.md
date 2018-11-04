# NoiseNet: Signal-dependent Noise Variance Estimation with Convolutional Neural Network #

Implements the model described in:

    [1] Uss M., Vozel B., Lukin V., Chehdi K. (2018) NoiseNet: Signal-Dependent Noise Variance Estimation with Convolutional Neural Network. In: Blanc-Talon J., Helbert D., Philips W., Popescu D., Scheunders P. (eds) Advanced Concepts for Intelligent Vision Systems. ACIVS 2018. Lecture Notes in Computer Science, vol 11182. Springer, Cham

The published version can be found [here](http://link-springer-com-443.webvpn.jxutcm.edu.cn/chapter/10.1007/978-3-030-01449-0_35).

If you use this code please cite [1].

Copyright (c) 2018 Mykhail Uss

## Basic Usage ##

### Setting the environment ###

Python Version : 3
Keras Version: 2.1.2
Tensorflow-gpu: 1.4.0.
matplotlib 2.1.1
rawpy 0.10.1
exifread 2.1.2
opencv-python 3.4.0.12
statsmodels 0.8.0
tqdm 4.23.4
xlwt 1.3.0

You'll need to install the dependencies, something like the following:

```
pip install numpy keras tensorflow-gpu scipy matplotlib
```

### Training ###

`main.py` is the entry point for both training and testing. Define the task you
want to do, output folder to where to save the results.

For example, to train the model

```
python main.py train NoiseNet.h5  --train_db path_to_train_db
```

To perform model testing on NED2012 database for each best checkpoint provide path to the database with --NED2012_path key

```
python main.py train NoiseNet.h5 --train_db path_to_train_db --NED2012_path path_to_NED_2012
```

Logs will be automatically saved to "Graph" folder. Notice, that this folder is automatically cleaned at each training run.
Training resuming is not supported in this release.

### Testing ###

For testing model on NED2012 database, specify eval key, and provide model location and path to the NED2012 database.

```
python main.py eval PretrainedModel/NoiseNet_v29.h5 --NED2012_path path_to_NED_2012
```

Note: If model path for evaluation is not provided, the pretrained model is used by default.

## Differences from the original version ##

The model version directly corresponds to the paper [1]

## Datasets and pretrained models ##

For model training we use 158 raw images taken by a Nikon D80 camera for which Noise Level Function was preliminary manually estimated using calibration images.
Estimated NLFs for ISO values 100, 200, 400, 800 and red, greed and blue channels are hardcoded in file utils.py as variables nlf_d80_iso100, nlf_d80_iso200,
nlf_d80_iso400, and nlf_d80_iso800. To train the model using raw images from some other camera, calibrated NLFs should be provided in similar manner.
Also notice, that bayer pattern for Nikon D80 camera is hardcoded as well.

The training dataset can be downloaded here

* [Train data for NoiseNet CNN (158 NikonD80 images)](https://data.mendeley.com/datasets/3tt2hkh5mr/1)

NoiseNet evaluation is based on NED2012 image database (M. L. Uss, B. Vozel, V. V. Lukin, and K. Chehdi, "Image informative maps for component-wise estimating parameters of signal-dependent noise," J. Electron. Imaging, vol. 22, pp. 013019-013019, 2013.) that is available at

* [NED2012 database for evaluation](https://www.researchgate.net/publication/280611662_NED2012_image_database_for_benchmarking_signal-dependent_noise_estimation_algorithms)


We provide model NoiseNet_v29.h5 trained on the `D80_158' dataset from the ACIVS paper in the "PretrainedModel" folder of this repository.


### constraints ###

NoiseNet CNN is designed to estimate spatially uncorrelated additive noise STD from local image patch of size 32 by 32 pixels
NI+NoiseNet BNPE method implementation is only tested for Nikon D80 images. For other raw images, especially with different Bayer pattern, code modifications are needed.
