# Measuring rotational (in)variance in Convolutional Neural Networks



This repository contains the code necessary to obtain the experimental results published in the article [Measuring rotational (in)variance in Convolutional Neural Networks]() (link and bib entry coming soon).

## Abstract

## What can you do with this code

You can train a model on the [MNIST](http://yann.lecun.com/exdb/mnist/) or [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) datasets. Two models will be generated for each training; a *rotated* model, for which the dataset's samples were randomly rotated before, and an *unrotated* model, for which they weren't modified at all.

The available models are [Resnet](), [VGG16](), [AllConvolutional network](https://arxiv.org/abs/1412.6806) and a [simple Convolutional Network](https://github.com/facundoq/rotational_invariance_data_augmentation/blob/master/pytorch/model/simple_conv.py)  

Afterwards, you can measure the (in)variance of each activation of the networks, and visualize them as heatmaps or plots. 

## How to run

These instructions have been tested on a modern ubuntu-based distro (>=18) with python version>=3.6.  

* Clone the repository and cd to it:
    * `git clone https://github.com/facundoq/rotational_variance.git`
    * `cd rotational_variance` 
* Create a virtual environment and activate it (requires python3 with the venv module and pip):
    * `python3 -m venv .env`
    * `source .env/bin/activate`
* Install libraries
    * `pip install -r requirements.txt`
    
* Run the experiments with `python experiment> <model> <dataset>`
    * `experiment_rotation.py` trains two models with the dataset: one with the vanilla version, the other with a data-augmented version via rotations.
    * `experiment_variance.py`  calculates the variance of the activations of the model for the rotated and unrotated model/dataset combinations. Results are saved by default to `~/variance_results/`
    * `plot_variances_models.py` generates plots of the variances for each/model dataset combination found in `~/variance_results/`. Both stratified/non-stratified versions of the measure are included in the plots. 
    
* The folder `plots` contains the results for any given model/dataset combination



