import sys

sys.path.append("../../NeuralNetworkClasses")
from extract_from_root import *
from NN_class import *
from hyperparameter_optimization import *

import numpy as np

cload = load_tree()
data_path="/lustre/alice/users/csonnab/TPC/NeuralNetworks/TrainingNetworks/LHC18b_FULL_GPU_SUPER_AGG_CUTS_3_smallNet/training_data.root"
data = cload.load(list_ignore=['fRelResoTPC','fNormMultTPC'], path=data_path)

labels = np.array(data[0]).astype('U32')
fit_data = np.array(data[1:]).astype(float)

fit_data = fit_data[fit_data.T[0]!=0]

np.random.shuffle(fit_data)
X = fit_data[:10000,2:-1]
y = fit_data[:10000,0]*fit_data[:10000,1]

class_inst = optuna_optimization(X, y)

class_inst.load_model(class_file_path="/lustre/alice/users/csonnab/TPC/NeuralNetworks/Neural-Networks/Notebooks/HyperparameterOptimization/network_class_test.py",
                      config_file_path="/lustre/alice/users/csonnab/TPC/NeuralNetworks/Neural-Networks/Notebooks/HyperparameterOptimization/configuration.py")

class_inst.call_objective()