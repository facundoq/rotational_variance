from . import variance
import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import namedtuple
import os

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
    #plt.show()
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
    if n==2:
        color = np.array([[1,0,0],[0,0,1]])
    else:
        color = plt.cm.hsv(np.linspace(0.1, 0.9, n))


    f,ax=plt.subplots(dpi=min(350,max(150,n*15)))
    for rotated_measure,measure,label,i in zip(rotated_measures_collapsed,measures_collapsed,labels,range(n)):
        x_rotated=np.arange(rotated_measure.shape[0])
        x=np.arange(measure.shape[0])
        ax.plot(x_rotated,rotated_measure,label="rotated_"+label,linestyle="-",color=color[i,:])
        ax.plot(x,measure,label="unrotated_"+label,linestyle="--",color=color[i,:]*0.7)
        ax.set_ylabel("Variance")
        ax.set_xlabel("Layer")
        # ax.set_ylim(max_measure)
        n_layers=len(x)
        if n_layers<25:
            ax.set_xticks(range(n_layers))

    #handles, labels = ax.get_legend_handles_labels()

    # reverse the order
    # lgd=ax.legend(handles[::-1], labels[::-1],bbox_to_anchor=(1, 1))
    #           # mode="expand", borderaxespad=0.3)
    # # ax.autoscale_view()
    # f.artists.append(lgd) # Here's the change
    #plt.tight_layout()

    if savefig:
        image_name=f"collapsed_{savefig_suffix}.png"
        path=os.path.join(savefig,image_name)
        plt.savefig(path,bbox_inches="tight")

    #plt.show()
    plt.close()




def collapse_measure_layers(measures):

    return [np.array([np.mean(layer[np.isfinite(layer)]) for layer in measure]) for measure in measures]

def plots_base_folder():
    return os.path.expanduser("~/variance_results/plots/")
    #return os.path.join(results_folder,"plots/var")

def plots_folder(model,dataset,conv_aggregation):
    folderpath = os.path.join(plots_base_folder(), f"{model}_{dataset}_{conv_aggregation}")
    if not os.path.exists(folderpath):
        os.makedirs(folderpath,exist_ok=True)
    return folderpath


def plot_heatmaps(model,rotated_model,dataset,conv_aggregation,results,folderpath):
    var, stratified_layer_vars, var_all_dataset, rotated_var, rotated_stratified_layer_vars, rotated_var_all_dataset = results
    vmin, vmax = outlier_range_all(results, iqr_away=3)
    vmin = vmin_all = vmin_class = 0
    vmax_all = vmax_class = vmax
    # vmin_class, vmax_class = outlier_range_both(rotated_var, var)
    # vmin_class = 0
    # vmin_class, vmax_class = outlier_range_both(rotated_stratified_layer_vars, stratified_layer_vars)
    # vmin_class = 0
    # vmin_all, vmax_all = outlier_range_both(rotated_var, var)

    plot(rotated_var, model, dataset.name, savefig=folderpath,
         savefig_suffix="rotated", vmax=vmax_class, class_names=dataset.labels)
    plot(var, model, dataset.name, savefig=folderpath, savefig_suffix="unrotated",
         vmax=vmax_class, class_names=dataset.labels)
    plot(rotated_stratified_layer_vars, rotated_model, dataset.name,
         class_names=["all_stratified"], savefig=folderpath,
         savefig_suffix="rotated", vmax=vmax_class)
    plot(stratified_layer_vars, model, dataset.name, class_names=["all_stratified"],
         savefig=folderpath, savefig_suffix="unrotated", vmax=vmax_class)

    plot(rotated_var_all_dataset, rotated_model, dataset.name,
         savefig=folderpath, savefig_suffix="rotated", class_names=["all"], vmax=vmax_all)
    plot(var_all_dataset, model, dataset.name,
         savefig=folderpath, savefig_suffix="unrotated", class_names=["all"], vmax=vmax_all)


def plot_all(model,rotated_model,dataset,conv_aggregation,results):

    folderpath=plots_folder(model.name,dataset.name,conv_aggregation)
    var, stratified_layer_vars, var_all_dataset, rotated_var, rotated_stratified_layer_vars, rotated_var_all_dataset=results
    #plot_heatmaps(model, rotated_model, dataset, conv_aggregation, results,folderpath)

    # print("plotting layers invariance (by classes)")
    # plot_collapsing_layers(rotated_var, var, dataset.labels
    #                        , savefig=folderpath, savefig_suffix="classes")

    # max_rotated = max([m.max() for m in rotated_measures_collapsed])
    # max_unrotated = max([m.max() for m in measures_collapsed])
    # max_measure = max([max_rotated, max_unrotated])
    #print("plotting layers invariance (global)")
    plot_collapsing_layers(rotated_stratified_layer_vars+rotated_var_all_dataset, stratified_layer_vars+var_all_dataset
                            , ["stratified","all"], savefig=folderpath, savefig_suffix="global")

def run_and_plot_all(model,rotated_model,dataset, config, n_rotations = 16):
    results=variance.run_all(model,rotated_model,dataset, config, n_rotations)
    plot_all(model,rotated_model,dataset,results)



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