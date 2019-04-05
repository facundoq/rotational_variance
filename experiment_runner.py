import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import os
import logging
import subprocess
import sys

if len(sys.argv)==1:
    venv_path = ".env"
    logging.info(f"No virtual environment path specified, defaulting to {venv_path}.")
elif len(sys.argv)==2:
    venv_path=sys.argv[1]
    logging.info(f"Using virtual environment {venv_path}.")
else:
    sys.exit("Wrong number of arguments")

def run_experiment(experiment, model_name, dataset_name,venv_path):
    python_command=f"{experiment}.py {model_name} {dataset_name}"
    # python_executable=sys.executable
    # python_executable=sys.argv[0]
    # python_executable="/home/facundo/faq/exp/.env/bin/python"
    python_executable="python3"
    venv_activate_path=os.path.join(venv_path,"bin","activate")
    #virtualenv=os.path.abspath("../.env/bin/activate")
    #virtualenv="/home/facundo/faq/exp/.env/bin/activate"
    command=f"source {venv_activate_path} && {python_executable} {python_command}"
    logging.info(f"Running {command}")
    print(f"Running {command}")
    subprocess.call(f'/bin/bash -c "{command}"', shell=True)
    # subprocess.call(f'{python_executable} {python_command}')

# DATASET
import datasets

from pytorch.experiment import model_loading

model_names=model_loading.get_model_names()
model_names=["AllConvolutional","SimpleConv","ResNet","VGGLike"]
dataset_names=datasets.names
dataset_names=["mnist","cifar10"]
train=False
experiments=["experiment_variance"]

message=f"""Running experiments, train={train}
Experiments: {", ".join(experiments)}
Models: {", ".join(model_names)}
Datasets: {", ".join(dataset_names)}
"""
logging.info("")

for model_name in model_names:
    for dataset_name in dataset_names:
        if train:
            run_experiment("experiment_rotation",model_name,dataset_name,venv_path)
        for experiment in experiments:
            run_experiment(experiment,model_name,dataset_name,venv_path)
