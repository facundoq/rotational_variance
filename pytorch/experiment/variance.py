
from pytorch.experiment.utils import RunningMeanAndVariance,RunningMean

import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
import os
from pytorch.dataset import ImageDataset



def run_all_dataset(model,dataset,config,rotations,conv_aggregation_function,batch_size=256):
    n=dataset.x_test.shape[0]
    effective_batch_size = min(n, batch_size)
    dataset= ImageDataset(dataset.x_test,dataset.y_test, rotation=True)
    layer_vars = eval_var(dataset, model, config, rotations, effective_batch_size,conv_aggregation_function)
    return [layer_vars],[0]

def run(model,dataset,config,rotations,conv_aggregation_function,batch_size=256):
    x = dataset.x_test
    y=dataset.y_test
    y_ids=y.argmax(axis=1)
    classes=np.unique(y_ids)
    classes.sort()

    class_layer_vars=[]
    # calculate the var measure for each class
    for i, c in enumerate(classes):
        # logging.debug(f"Evaluating vars for class {c}...")
        ids=np.where(y_ids==c)
        ids=ids[0]
        x_class,y_class=x[ids,:],y[ids]
        n=x_class.shape[0]
        class_dataset=ImageDataset(x_class,y_class,rotation=True)
        effective_batch_size=min(n,batch_size)
        layer_vars=eval_var(class_dataset, model, config, rotations, effective_batch_size,conv_aggregation_function)
        class_layer_vars.append(layer_vars)
    stratified_layer_vars=eval_stratified_var(class_layer_vars)

    return class_layer_vars,stratified_layer_vars,classes

# calculate the mean activation of each unit in each layer over the set of classes
def eval_stratified_var(class_layer_vars):

    layer_class_vars=[list(i) for i in zip(*class_layer_vars)]
    layer_vars=[ sum(layer_values)/len(layer_values) for layer_values in layer_class_vars]
    return [layer_vars]

# calculate the var measure for a given dataset and model
def eval_var(dataset,model,config,rotations,batch_size,conv_aggregation_function):
    #baseline vars
    layer_vars_baselines=get_baseline_variance_class(dataset,model,config,rotations,batch_size,conv_aggregation_function)
    layer_vars = eval_var_class(dataset, model, config, rotations,batch_size,conv_aggregation_function)
    normalized_layer_vars = calculate_var(layer_vars_baselines,layer_vars)

    # for i in range(len(layer_vars_baselines)):
    #     print(f"class {i}")
    #     n=len(layer_vars_baselines[i])
    #     print([ type(layer_vars_baselines[i][j]) for j in range(n)])
    #     print([type(layer_vars[i][j]) for j in range(n)])
    #     print([type(normalized_layer_vars[i][j]) for j in range(n)])


    return normalized_layer_vars

def calculate_var(layer_baselines, layer_measures):
    eps=0
    measures = []  # coefficient of variations

    for layer_baseline, layer_measure in zip(layer_baselines, layer_measures):
        #print(layer_baseline.shape, layer_measure.shape)
        normalized_measure= layer_measure.copy()
        normalized_measure[layer_baseline > eps] /= layer_baseline[layer_baseline > eps]
        both_below_eps=np.logical_and(layer_baseline <= eps,layer_measure <= eps )
        normalized_measure[both_below_eps] = 1
        only_baseline_below_eps=np.logical_and(layer_baseline <= eps,layer_measure > eps )
        normalized_measure[only_baseline_below_eps] = np.inf
        measures.append(normalized_measure)
    return measures

def get_baseline_variance_class(dataset,model,config,rotations,batch_size,conv_aggregation_function):
    n_intermediates = model.n_intermediates()
    baseline_variances = [RunningMeanAndVariance() for i in range(n_intermediates)]

    for i, r in enumerate(rotations):
        degrees = (r - 1, r + 1)
        # logging.debug(f"    Rotation {degrees}...")
        dataset.update_rotation_angle(degrees)
        dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=False, num_workers=0,drop_last=True)
        # calculate std for all examples and this rotation
        dataset_var=get_dataset_var(model,dataloader,config,conv_aggregation_function)
        #update the mean of the stds for every rotation
        # each std is intra rotation/class, so it measures the baseline
        # std for that activation
        for j,m in enumerate(dataset_var):
            baseline_variances[j].update(m.std())
    mean_baseline_variances=[b.mean() for b in baseline_variances]
    return mean_baseline_variances

def get_dataset_var(model,dataloader,config,conv_aggregation_function):
    n_intermediates = model.n_intermediates()
    var = [RunningMeanAndVariance() for i in range(n_intermediates)]
    for x,y_true in dataloader:
        if config.use_cuda:
            x = x.cuda()
        with torch.no_grad():
            y, intermediates = model.forward_intermediates(x)
            for j, intermediate in enumerate(intermediates):
                flat_activations=transform_activations(intermediate,conv_aggregation_function)
                for h in range(flat_activations.shape[0]):
                    var[j].update(flat_activations[h,:])
    return var





#returns a list of RunningMeanAndVariance objects,
# one for each intermediate output of the model.
#Each RunningMeanAndVariance contains the mean and std of each intermediate
# output over the set of rotations
def eval_var_class(dataset,model,config,rotations,batch_size,conv_aggregation_function):
    n_intermediates = model.n_intermediates()
    layer_vars= [RunningMeanAndVariance() for i in range(n_intermediates)]
    n = len(dataset)
    batch_ranges=[ range(i,i+batch_size) for i in range(n//batch_size)]

    for batch_range in batch_ranges:
        batch_var=BatchvarMeasure(batch_size,n_intermediates)
        for r in rotations:
            degrees = (r - 1, r + 1)
            dataset.update_rotation_angle(degrees)
            x,y_true=dataset.get_batch(batch_range)
            if config.use_cuda:
                x = x.cuda()
            with torch.no_grad():
                y, batch_activations= model.forward_intermediates(x)
                batch_activations=[transform_activations(a,conv_aggregation_function) for a in batch_activations]
                batch_var.update(batch_activations)
        batch_var.update_global_measures(layer_vars)

    mean_layer_vars = [b.mean() for b in layer_vars]
    return mean_layer_vars

class BatchvarMeasure:
    def __init__(self,batch_size,n_intermediates):
        self.batch_size=batch_size
        self.n_intermediates=n_intermediates
        self.batch_stats = [[RunningMeanAndVariance() for i in range(batch_size)] for j in range(n_intermediates)]
        self.batch_stats = np.array(self.batch_stats)

    def update(self,batch_activations):
        for i, layer_activations in enumerate(batch_activations):
            for j in range(layer_activations.shape[0]):
                self.batch_stats[i, j].update(layer_activations[j, :])

    def update_global_measures(self,dataset_stats):
        for i in range(self.n_intermediates):
            mean_var=dataset_stats[i]
            for j in range(self.batch_size):
                mean_var.update(self.batch_stats[i, j].std())


def transform_activations(activations_gpu,conv_aggregation_function):
    activations = activations_gpu.detach().cpu().numpy()

    # if conv average out spatial dims
    if len(activations.shape) == 4:
        n, c, w, h = activations.shape
        flat_activations = np.zeros((n, c))
        for i in range(n):
            if conv_aggregation_function=="mean":
                flat_activations[i, :] = np.nanmean(activations[i, :, :, :],axis=(1, 2))
            elif conv_aggregation_function=="max":
                flat_activations[i, :] = np.nanmax(activations[i, :, :, :],axis=(1, 2))
            elif conv_aggregation_function=="sum":
                flat_activations[i, :] = np.nansum(activations[i, :, :, :],axis=(1, 2))
            else:
                raise ValueError(f"Invalid aggregation function: {conv_aggregation_function}. Options: mean, max, sum")
        assert (len(flat_activations.shape) == 2)
    else:
        flat_activations = activations

    return flat_activations



def global_average_variance(result):
    rm=RunningMean()
    for layers in result:
        for layer in layers:
            for act in layer[:]:
                if np.isfinite(act):
                    rm.update(act)
    return rm.mean()

def run_all(model,rotated_model,dataset, config, n_rotations,conv_aggregation_function,batch_size=256):
    rotations = np.linspace(-180, 180, n_rotations, endpoint=False)

    print("Calculating variance for all samples by class...")
    rotated_var, rotated_stratified_layer_vars, classes = run(rotated_model, dataset, config, rotations,conv_aggregation_function,batch_size=batch_size,)
    var, stratified_layer_vars, classes = run(model, dataset, config, rotations,conv_aggregation_function,batch_size=batch_size)

    # Plot variance for all
    print("Calculating variance for all samples...")
    rotated_var_all_dataset, classes = run_all_dataset(rotated_model, dataset, config,
                                                       rotations,conv_aggregation_function,batch_size=batch_size)
    var_all_dataset, classes = run_all_dataset(model, dataset, config, rotations,conv_aggregation_function,batch_size=batch_size)
    return var,stratified_layer_vars,var_all_dataset,rotated_var,rotated_stratified_layer_vars,rotated_var_all_dataset
