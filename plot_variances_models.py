import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'gray'
import numpy as np
from pytorch.experiment import variance,variancevisualization
import os

def load_results(folderpath):
    results = []
    for filename in os.listdir(folderpath):
        path = os.path.join(folderpath, filename)
        model, dataset, conv_aggregation = variancevisualization.get_model_and_dataset_from_path(path)
        result = variancevisualization.load_results(model, dataset,conv_aggregation)
        results.append((result, model, dataset, conv_aggregation))
    return results


def global_results(results):
    table={}
    for result, model, dataset, conv_aggregation  in results:
        var,stratified_layer_vars,var_all_dataset,rotated_var,rotated_stratified_layer_vars,rotated_var_all_dataset=result
        table[f"{conv_aggregation}_{dataset}_rotated_{model}"]=variance.global_average_variance(rotated_var_all_dataset)
        table[f"{conv_aggregation}_{dataset}_unrotated_{model}"]=variance.global_average_variance(var_all_dataset)
        table[f"{conv_aggregation}_stratified_{dataset}_rotated_{model}"] = variance.global_average_variance(rotated_stratified_layer_vars)
        table[f"{conv_aggregation}_stratified_{dataset}_unrotated_{model}"] = variance.global_average_variance(stratified_layer_vars)
    return table


def global_results_latex(results,stratified):
    table={}

    for result, model, dataset, conv_aggregation  in results:
        var,stratified_layer_vars,var_all_dataset,rotated_var,rotated_stratified_layer_vars,rotated_var_all_dataset=result
        conv_table=table.get(conv_aggregation,{})
        dataset_table=conv_table.get(dataset,{})

        rotated_table=dataset_table.get("rotated",{})
        unrotated_table = dataset_table.get("unrotated", {})
        if stratified:
            rotated_table[model] = variance.global_average_variance(rotated_stratified_layer_vars)
            unrotated_table[model] = variance.global_average_variance(stratified_layer_vars)
        else:
            rotated_table[model]=variance.global_average_variance(rotated_var_all_dataset)
            unrotated_table[model] = variance.global_average_variance(var_all_dataset)

        dataset_table["rotated"]=rotated_table
        dataset_table["unrotated"] = unrotated_table

        conv_table[dataset]=dataset_table
        table[conv_aggregation]=conv_table

    return table

def plot_last_layers_per_class(results,folderpath):

    for result, model, dataset, conv_aggregation in results:
        var,stratified_layer_vars,var_all_dataset,rotated_var,rotated_stratified_layer_vars,rotated_var_all_dataset=result
        combination_folderpath=os.path.join(folderpath,variancevisualization.plots_folder(model,dataset,conv_aggregation))
        os.makedirs(combination_folderpath,exist_ok=True)
        plot_last_layer(var,f"{conv_aggregation}_unrotated",model,dataset,combination_folderpath)
        plot_last_layer(rotated_var,f"{conv_aggregation}_rotated",model,dataset,combination_folderpath)

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




def format_results_latex(results_table):
    table_str = ""
    for conv,dataset_table in results_table.items():
        table_str += f"conv agg {conv}\n"
        for dataset,training_table in dataset_table.items():
            for training,models in training_table.items():
                table_str += f"{dataset.upper()} & {training} & "
                model_keys = sorted(models.keys())
                table_str += " & ".join([f"{models[key]:0.2}" for key in model_keys])
                table_str += " \\\\ \n"

    return table_str

def print_global_results(results):
    global_results_table_latex=global_results_latex(results,stratified=False)
    latex_table_str=format_results_latex(global_results_table_latex)
    print("Normal results:")
    print(latex_table_str)

    stratified_global_results_table_latex=global_results_latex(results,stratified=True)
    stratified_latex_table_str=format_results_latex(stratified_global_results_table_latex)
    print("Stratified results:")
    print(stratified_latex_table_str)

    # def format_results_latex(results_table):
    # table_str=""
    # for key in sorted(global_results_table.keys()):
    #     table_str += f"{key} => {global_results_table[key]}\n"
    #
    # global_results_filepath=os.path.join(variance.plots_base_folder(),"global_invariance_comparison.txt")
    # with open(global_results_filepath, "w") as text_file:
    #     text_file.write(table_str)
    #
    # print(table_str)
    #


def plot_all(results):

    import torch
    from pytorch import dataset as datasets

    use_cuda=torch.cuda.is_available()
    from pytorch.experiment import rotation

    for result, model_name, dataset_name, conv_aggregation in results:
        dataset = datasets.get_dataset(dataset_name)
        model,rotated_model,scores,config=rotation.load_models(dataset,model_name,use_cuda)
        variancevisualization.plot_all(model,rotated_model,dataset,conv_aggregation,result)
        del model
        del dataset

results_folderpath=os.path.expanduser("~/variance_results/")
results=load_results(variancevisualization.results_folder)
#plot_last_layers_per_class(results,results_folderpath)
print("Plotting heatmaps")
plot_all(results)
#print_global_results(results)