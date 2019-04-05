import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'gray'
import logging

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

results=load_results(variance.results_folder)

table={}
for result, model, dataset in results:
    #average_rotated_stratified=variance.global_average_variance(result)
    #average_unrotated_stratified=variance.global_average_variance(result)
    var,stratified_layer_vars,var_all_dataset,rotated_var,rotated_stratified_layer_vars,rotated_var_all_dataset=result
    table[f"{dataset}_rotated_{model}"]=variance.global_average_variance(rotated_var_all_dataset)
    table[f"{dataset}_unrotated_{model}"]=variance.global_average_variance(var_all_dataset)
    table[f"stratified_{dataset}_rotated_{model}"] = variance.global_average_variance(rotated_stratified_layer_vars)
    table[f"stratified_{dataset}_unrotated_{model}"] = variance.global_average_variance(stratified_layer_vars)

for key in sorted(table.keys()):
    print(key,table[key])