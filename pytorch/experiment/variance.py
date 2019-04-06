import pickle
import matplotlib.pyplot as plt
from pytorch.experiment.utils import RunningMeanAndVariance,RunningMean
from collections import namedtuple
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
from torchvision import transforms
import os
import progressbar
import logging

from pytorch.dataset import get_data_generator

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
                flat_activations[i, :] = activations[i, :, :, :].nanmean(axis=(1, 2))
            elif conv_aggregation_function=="max":
                flat_activations[i, :] = activations[i, :, :, :].nanmax(axis=(1, 2))
            elif conv_aggregation_function=="sum":
                flat_activations[i, :] = activations[i, :, :, :].nansum(axis=(1, 2))
            else:
                raise ValueError(f"Invalid aggregation function: {conv_aggregation_function}. Options: mean, max, sum")
        assert (len(flat_activations.shape) == 2)
    else:
        flat_activations = activations

    return flat_activations




def plot_class_outputs(class_index,class_id, cvs,vmin,vmax, names,model_name,dataset_name,savefig,savefig_suffix):
    n = len(names)
    f, axes = plt.subplots(1, n, dpi=150)


    for i, (cv, name) in enumerate(zip(cvs, names)):
        ax = axes[i]
        ax.axis("off")
        cv = cv[:, np.newaxis]
        mappable=ax.imshow(cv,vmin=vmin,vmax=vmax,cmap='inferno',aspect="auto")
        #mappable = ax.imshow(cv, cmap='inferno')
        if n<40:
            if len(name)>6:
                name=name[:6]
            ax.set_title(name, fontsize=4)

        # logging.debug(f"plotting stats of layer {name} of class {class_id}, shape {stat.mean().shape}")
    f.suptitle(f"Variance of activations for class {class_index} ({class_id})")
    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar=f.colorbar(mappable, cax=cbar_ax, extend='max')
    cbar.cmap.set_over('green')
    cbar.cmap.set_bad(color='blue')
    if savefig:
        image_name=f"{savefig_suffix}_class{class_index:02}_{class_id}.png"
        path=os.path.join(savefig,image_name)
        plt.savefig(path)
    plt.show()
    plt.close()

def pearson_outlier_range(values,iqr_away):
    p50 = np.median(values)
    p75 = np.percentile(values, 75)
    p25 = np.percentile(values, 25)
    iqr = p75 - p25

    range = (p50 - iqr_away * iqr, p50 + iqr_away * iqr)
    return range

def outlier_range_all(std_list,iqr_away=5):
    var_values=[np.hstack([np.hstack(class_stds) for class_stds in stds]) for stds in std_list]
    var_values=np.hstack(var_values)
    return outlier_range_values(var_values,iqr_away)

    # minmaxs=[outlier_range(stds,iqr_away) for stds in std_list]
    # mins,maxs=zip(*minmaxs)
    # return max(mins),min(maxs)

def outlier_range_both(rotated_stds,unrotated_stds,iqr_away=5):
    rmin,rmax=outlier_range(rotated_stds,iqr_away)
    umin,umax= outlier_range(unrotated_stds,iqr_away)

    return (max(rmin,umin),min(rmax,umax))

def outlier_range_values(values,iqr_away):
    pmin, pmax = pearson_outlier_range(values, iqr_away)
    # if the pearson outlier range is away from the max and/or min, use max/or and min instead

    finite_values=values[np.isfinite(values)]
    # print(pmax, finite_values.max())
    return (max(pmin, finite_values.min()), min(pmax, finite_values.max()))

def outlier_range(stds,iqr_away):
    class_values=[np.hstack(class_stds) for class_stds in stds]
    values=np.hstack(class_values)

    return outlier_range_values(values,iqr_away)

def plot(all_stds,model,dataset_name,savefig=False,savefig_suffix="",class_names=None,vmax=None):
    vmin=0
    classes=len(all_stds)
    for i,c in enumerate(range(classes)):
        stds=all_stds[i]
        if class_names:
            name=class_names[c]
        else:
            name=str(c)
        plot_class_outputs(i,name, stds,vmin,vmax, model.intermediates_names(),model.name,
                           dataset_name,savefig,
                           savefig_suffix)

def plot_collapsing_layers(rotated_measures,measures,labels,savefig=None, savefig_suffix=""):
    rotated_measures_collapsed=collapse_measure_layers(rotated_measures)
    measures_collapsed=collapse_measure_layers(measures)
    n=len(rotated_measures)
    assert( n == len(measures))
    assert (n == len(labels))

    color = plt.cm.hsv(np.linspace(0.1, 0.9, n))


    f,ax=plt.subplots(dpi=min(350,max(150,n*15)))
    for rotated_measure,measure,label,i in zip(rotated_measures_collapsed,measures_collapsed,labels,range(n)):
        x_rotated=np.arange(rotated_measure.shape[0])
        x=np.arange(measure.shape[0])
        ax.plot(x_rotated,rotated_measure,label="rotated_"+label,linestyle="-",color=color[i,:])
        ax.plot(x,measure,label="unrotated_"+label,linestyle="--",color=color[i,:]*0.5)
        ax.set_ylabel("Variance")
        ax.set_xlabel("Layer")
        # ax.set_ylim(max_measure)
        ax.set_xticks(range(len(x)))
    handles, labels = ax.get_legend_handles_labels()

    # reverse the order
    lgd=ax.legend(handles[::-1], labels[::-1],bbox_to_anchor=(1.3, 1))
              # mode="expand", borderaxespad=0.3)
    # ax.autoscale_view()
    f.artists.append(lgd) # Here's the change
    # plt.tight_layout()
    if savefig:
        image_name=f"collapsed_{savefig_suffix}.png"
        path=os.path.join(savefig,image_name)
        plt.savefig(path,bbox_inches="tight")

    plt.show()
    plt.close()




def collapse_measure_layers(measures):
    return [np.array([np.nanmean(layer) for layer in measure]) for measure in measures]


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


def plots_base_folder():
    #return os.path.expanduser("~/variance_results/plots/")
    return os.path.join("plots")

def plots_folder(model,dataset,conv_aggregation):
    folderpath = os.path.join(plots_base_folder(), f"{model}_{dataset}_{conv_aggregation}")
    if not os.path.exists(folderpath):
        os.makedirs(folderpath,exist_ok=True)
    return folderpath

def plot_all(model,rotated_model,dataset,conv_aggregation,results):

    folderpath=plots_folder(model.name,dataset.name,conv_aggregation)
    var, stratified_layer_vars, var_all_dataset, rotated_var, rotated_stratified_layer_vars, rotated_var_all_dataset=results
    vmin, vmax = outlier_range_all(results,iqr_away=3)
    vmin = vmin_all = vmin_class = 0
    vmax_all = vmax_class = vmax
    # vmin_class, vmax_class = outlier_range_both(rotated_var, var)
    # vmin_class = 0
    # vmin_class, vmax_class = outlier_range_both(rotated_stratified_layer_vars, stratified_layer_vars)
    # vmin_class = 0
    # vmin_all, vmax_all = outlier_range_both(rotated_var, var)

    plot(rotated_var, model, dataset.name, savefig=folderpath,
         savefig_suffix="rotated", vmax=vmax_class,class_names=dataset.labels)
    plot(var, model, dataset.name, savefig=folderpath, savefig_suffix="unrotated",
         vmax=vmax_class,class_names=dataset.labels)
    plot(rotated_stratified_layer_vars, rotated_model, dataset.name,
         class_names=["all_stratified"], savefig=folderpath,
         savefig_suffix="rotated", vmax=vmax_class)
    plot(stratified_layer_vars, model, dataset.name, class_names=["all_stratified"],
         savefig=folderpath,savefig_suffix="unrotated",vmax=vmax_class)

    plot(rotated_var_all_dataset, rotated_model, dataset.name,
         savefig=folderpath, savefig_suffix="rotated", class_names=["all"], vmax=vmax_all)
    plot(var_all_dataset, model, dataset.name,
         savefig=folderpath, savefig_suffix="unrotated", class_names=["all"], vmax=vmax_all)


    plot_collapsing_layers(rotated_var, var, dataset.labels
                           , savefig=folderpath, savefig_suffix="classes")

    # max_rotated = max([m.max() for m in rotated_measures_collapsed])
    # max_unrotated = max([m.max() for m in measures_collapsed])
    # max_measure = max([max_rotated, max_unrotated])

    plot_collapsing_layers(rotated_stratified_layer_vars+rotated_var_all_dataset, stratified_layer_vars+var_all_dataset
                            , ["stratified","all"], savefig=folderpath, savefig_suffix="global")

    # plot_collapsing_layers(rotated_var_all_dataset, var_all_dataset,
    #                        , savefig=folderpath, savefig_suffix="all")


def run_and_plot_all(model,rotated_model,dataset, config, n_rotations = 16):
    results=run_all(model,rotated_model,dataset, config, n_rotations)
    plot_all(model,rotated_model,dataset,results)


def global_average_variance(result):
    rm=RunningMean()
    for layers in result:
        for layer in layers:
            for act in layer[:]:
                if np.isfinite(act):
                    rm.update(act)
    return rm.mean()

results_folder=os.path.expanduser("~/variance_results/values")

def get_path(model_name,dataset_name,conv_aggregation_function):
    return os.path.join(results_folder, f"{model_name}_{dataset_name}_{conv_aggregation_function}.pickle")

def get_model_and_dataset_from_path(path):
    filename_ext=os.path.basename(path)
    filename=os.path.splitext(filename_ext)[0]
    model,dataset,conv_aggregation_function=filename.split("_")
    return model, dataset,conv_aggregation_function

def save_results(model_name,dataset_name,results,conv_aggregation_function):
    path=get_path(model_name,dataset_name,conv_aggregation_function)
    basename=os.path.dirname(path)
    os.makedirs(basename,exist_ok=True)
    pickle.dump(results,open(path,"wb"))

def load_results(model_name,dataset_name,conv_aggregation_function):
    path = get_path(model_name, dataset_name,conv_aggregation_function)
    return pickle.load(open(path, "rb"))