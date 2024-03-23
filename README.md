# Semantic Segmentation for Subsaharan Satellite Images
This project aims to create a model which can accurately label pixels in satellite images by one of four class labels:
- 1: Human settlements without electricity
- 2: No human settlements without electricity
- 3: Human settlements with electricity
- 4: No human settlements with electricity
We created three models which succeed to varying degrees in labelling our data.
The models are:
- UNet
- SegmentationCNN
- FCNResnetTransfer

## Pipeline
![model pipeline](https://github.com/cs175cv-w2024/final-project-the-thunder-men/assets/116922352/b33daac7-c987-46e7-b37c-d574276759d1)

This project supports a full pipline from raw data to predicted outputs. Raw data is preprocessed and model predictions are saved automatically.

Raw `.tif` files are preprocessed to remove outliers and normalize data. Preprocessing consists of grouping files based on their satellite type, then applying band specific cleaning, such as quantile clipping and min-max scaling. Each satellite has its own processing function tailored to their band types.

Once preprocessed, files are grid sliced into subtiles, then converted into a `Dataset` and fed into a `Dataloader`. Augmentations can be applied at this time.

The data is split into training and validation sets, then fed into the model to be trained and validated on.

Models are evaluated on the validation data and visualizations of the predictions are generated and saved.


## Table of Contents
- [Installation](#installation)
- [Usage](#usage)

## Installation
1. Clone the repository
2. Install packages in `requirements.txt`
3. Download the IEEE GRSS 2021 dataset from: https://www.grss-ieee.org/community/technical-committees/2021-ieee-grss-data-fusion-contest-track-dse/
## Usage
### Data
To use this project, add the raw `.tif` files you wish to use to `data/raw/Train` giving each tile its own folder with the naming convention `TileX` where `X` is the number of the tile. Alternativly download the dataset from https://www.grss-ieee.org/community/technical-committees/2021-ieee-grss-data-fusion-contest-track-dse/ If the data has not been preprocessed, running `scripts/train.py` or `scripts/evaluate.py` will process the data. If you wish to add additional data for augmentation, you can run `src/esd_data/make_new_data.py` which will duplicate the dataset.
### Training the model
To train your own model, set up ESDConfig in `scripts/train.py` with the `model_type` you would like to use. The model types are:
- "UNet"
- "SegmentationCNN"
- "FCNResnetTransfer"
Adjust any hyperparameters you wish in ESDConfig, then execute `scripts/train.py`.
### Load from checkpoint
If you wish to load a model from a checkpoint, change the model_path in `scripts/train.py` to the path to the checkpoint you wish to use. Pretrained model checkpoints for each model type are available under `models`. Training starting with a checkpoint will pick up training where the checkpoint left off.
### Running the model
To run a model, pass the `model_path` of the model you wish to run to `EvalConfig` in `scripts/evaluate.py`. Pass the `results_dir` directory path of where you wish to save results to `EvalConfig` in `scripts/evaluate.py`. Then simply execute `scripts/evaluate.py`. This will run the model on the data in `data/processed/4x4/Val` and save .png and .tif files of the predictions to the directory `results_dir`.
### Further visualizations
If you wish to visualize other aspects of the data, there are a variety of helper plotting functions in `src/visualization/plot_utils.py`. These functions can plot specific bands from specific satellites and can be run on raw or processed data.
