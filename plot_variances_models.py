import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'gray'
import logging
import numpy as np
from pytorch.experiment import variance
import os

def load_results(folderpath):
    results = []
    for filename in os.listdir(folderpath):
        path = os.path.join(folderpath, filename)
        model, dataset = variance.get_model_and_dataset_from_path(path)

        result = variance.load_results(model, dataset)
        results.append((result, model, dataset))
    return results


def global_results(results):
    table={}
    for result, model, dataset in results:
        var,stratified_layer_vars,var_all_dataset,rotated_var,rotated_stratified_layer_vars,rotated_var_all_dataset=result
        table[f"{dataset}_rotated_{model}"]=variance.global_average_variance(rotated_var_all_dataset)
        table[f"{dataset}_unrotated_{model}"]=variance.global_average_variance(var_all_dataset)
        table[f"stratified_{dataset}_rotated_{model}"] = variance.global_average_variance(rotated_stratified_layer_vars)
        table[f"stratified_{dataset}_unrotated_{model}"] = variance.global_average_variance(stratified_layer_vars)
    return table

def plot_last_layers_per_class(results,folderpath):
    table={}
    for result, model, dataset in results:
        var,stratified_layer_vars,var_all_dataset,rotated_var,rotated_stratified_layer_vars,rotated_var_all_dataset=result
        combination_folderpath=os.path.join(folderpath,variance.plots_folder(model,dataset))
        os.makedirs(combination_folderpath,exist_ok=True)
        plot_last_layer(var,"unrotated",model,dataset,combination_folderpath)
        plot_last_layer(rotated_var,"rotated",model,dataset,combination_folderpath)

def plot_last_layer(class_measures,training,model,dataset,folderpath):
    classes=len(class_measures)
    f=plt.figure()
    variance_heatmap=np.zeros( (classes,classes) )

    for i,class_measure in enumerate(class_measures):
        variance_heatmap[:,i]=class_measure[-1]

    vmax=np.nanmax(variance_heatmap)
    mappable=plt.imshow(variance_heatmap,vmin=0,vmax=vmax,cmap='inferno',aspect="auto")

    plt.xticks(range(classes))
    plt.yticks(range(classes))

    cbar = f.colorbar(mappable, extend='max')
    cbar.cmap.set_over('green')
    cbar.cmap.set_bad(color='blue')
    plt.tight_layout()

    image_path= os.path.join(folderpath,f"last_layer_{training}.png")

    plt.savefig(image_path)
    plt.close()

results=load_results(variance.results_folder)

global_results_table=global_results(results)

table_str=""
for key in sorted(global_results_table.keys()):
    table_str += f"{key} => {global_results_table[key]}\n"

global_results_filepath=os.path.join(variance.plots_base_folder(),"global_invariance_comparison.txt")
with open(global_results_filepath, "w") as text_file:
    text_file.write(table_str)

print(table_str)

results_folderpath=os.path.expanduser("~/variance_results/")
plot_last_layers_per_class(results,results_folderpath)


import torch
from pytorch import dataset as datasets

use_cuda=torch.cuda.is_available()
from pytorch.experiment import rotation

for result, model_name, dataset_name in results:
    dataset = datasets.get_dataset(dataset_name)
    model,rotated_model,scores,config=rotation.load_models(dataset,model_name,use_cuda)
    variance.plot_all(model,rotated_model,dataset,result)