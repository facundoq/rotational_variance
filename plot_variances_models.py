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
        result = variance.load_results(path)
        model, dataset = variance.get_model_and_dataset_from_path(path)
        results.append((result, model, dataset))


results=load_results(variance.results_folder)
print(results)