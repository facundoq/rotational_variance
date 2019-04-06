## Calculate the variance of each activation in a model.
## NOTE:
## You should run "experiment_rotation.py" before this script to generate and train the models for
## a given dataset/model combination

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'gray'
import logging

from pytorch import models
from pytorch import dataset as datasets
import torch
import pytorch.experiment.utils as utils

if __name__ == "__main__":
    model_name,dataset_name=utils.parse_model_and_dataset("Experiment: accuracy of model for rotated vs unrotated dataset.")
else:
    dataset_name="cifar10"
    model_name=models.AllConvolutional.__name__


print(f"### Loading dataset {dataset_name} and model {model_name}....")
verbose=False

use_cuda=torch.cuda.is_available()

dataset = datasets.get_dataset(dataset_name)
if verbose:
    print(dataset.summary())

from pytorch.experiment import rotation
model,rotated_model,scores,config=rotation.load_models(dataset,model_name,use_cuda)

if verbose:
    print("### ", model)
    print("### ", rotated_model)
    print("### Scores obtained:")
    rotation.print_scores(scores)

from pytorch.experiment import variance

n_rotations=16
conv_aggregation_function="sum"
logging.info(f"Calculating measure with aggregation function {conv_aggregation_function} for conv layers and {n_rotations} rotations...")
results=variance.run_all(model,rotated_model,dataset, config, n_rotations,conv_aggregation_function,batch_size=64)
logging.info("Saving results...")
variance.save_results(model.name,dataset.name,results,conv_aggregation_function)


# logging.info("Plotting results...")
# results=variance.load_results(model.name,dataset.name)
# variance.plot_all(model,rotated_model,dataset,results)