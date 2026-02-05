import sys
from copy import copy
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from sklearn import preprocessing
from sklearn.pipeline import Pipeline

import timeit
import numpy as np
from NeuralNetworkClass.NeuralNetworkClasses.utils.functions import deep_update


class dataset(Dataset):

    def __init__(self, X, y):

        self.list = list(zip(X, y))
        self.length = len(self.list)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.list[idx]
    
    def mem_size(self):
        return sys.getsizeof(self.list)/8.

def custom_collate(batch):
    # For flexible data type. Expects something like this:
    # data = [([1.0, 2.0], 0, [3.0, 4.0]),
    #         ([5.0, 6.0, 7.0], 1, [8.0, 9.0, 10.0])]
    # dataset = CustomDataset(data)
    # dataloader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate)

    # Extract elements from the batch
    input_data, index_array, target = zip(*batch)

    # Pad sequences to the maximum length within the batch
    input_data_padded = pad_sequence(input_data, batch_first=True, padding_value=0)
    target_padded = pad_sequence(target, batch_first=True, padding_value=0)

    # Convert index_array to a tensor
    index_array_tensor = torch.stack(index_array)

    return input_data_padded, index_array_tensor, target_padded

class DataLoading(dataset):

    def __init__(self, training_data, validation_data, settings = {}):
        
        self.settings = {
            "GLOBAL" : {
                "transform_data" : True,
                "copy_to_device" : True,
            },
            "SCALERS" : {
                "X_data_scalers" : [('box-cox', preprocessing.PowerTransformer(method='box-cox', standardize=True))],
                "Y_data_scalers" : [('standard scaler', preprocessing.StandardScaler())],
            },
            "MACHINE_OPTIONS" : {
                "device" : None,
                "dtype_X" : torch.float,
                "dtype_Y" : torch.float,
                "amp": False,
                "flexible_data": False,
            },
            "OTHER" : {
                "verbose": True,
            }
        }
        self.settings = deep_update(self.settings, settings, name="Data settings")

        if self.settings["MACHINE_OPTIONS"]["device"] is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
            self.settings["MACHINE_OPTIONS"]["device"] = self.device
        else:
            self.device = self.settings["MACHINE_OPTIONS"]["device"]

        self.transform_data = self.settings["GLOBAL"]["transform_data"]
        self.loadTS = False
        self.loadVS = False

        if self.settings["OTHER"]["verbose"]:
            print("\n================ Data preparation ================\n")

        start_time_loader = timeit.default_timer()

        if ((not self.loadTS) and (not self.loadVS)):

            if not self.loadTS:
                if not self.transform_data:
                    self.transformers_X = []
                    self.transformers_Y = []
                    if self.settings["OTHER"]["verbose"]:
                        print("No transformation is performed on training data!")
                else:
                    self.transformers_X = self.settings["SCALERS"]["X_data_scalers"]
                    self.transformers_Y = self.settings["SCALERS"]["Y_data_scalers"]
                    if self.settings["OTHER"]["verbose"]:
                        print("Transforming training data...")

                self.scalingX = ScalingX(scalers = self.transformers_X, copy_to_dev = self.settings["GLOBAL"]["copy_to_device"], **self.settings["MACHINE_OPTIONS"])
                self.scalingY = ScalingY(scalers = self.transformers_Y, copy_to_dev = self.settings["GLOBAL"]["copy_to_device"], **self.settings["MACHINE_OPTIONS"])

                self.datasetTS = dataset(self.scalingX.scale(training_data[0]),
                                         self.scalingY.scale(training_data[1]))
                del training_data
                self.inverse_X = InverseScaling(scalers_fit = self.scalingX.fitted_scalers_X, copy_to_dev = self.settings["GLOBAL"]["copy_to_device"], device=self.settings["MACHINE_OPTIONS"]["device"], dtype=self.settings["MACHINE_OPTIONS"]["dtype_X"], amp=self.settings["MACHINE_OPTIONS"]["amp"], flexible_data=self.settings["MACHINE_OPTIONS"]["flexible_data"])
                self.inverse_Y = InverseScaling(scalers_fit = self.scalingY.pipe_y, copy_to_dev = self.settings["GLOBAL"]["copy_to_device"], device=self.settings["MACHINE_OPTIONS"]["device"], dtype=self.settings["MACHINE_OPTIONS"]["dtype_Y"], amp=self.settings["MACHINE_OPTIONS"]["amp"], flexible_data=self.settings["MACHINE_OPTIONS"]["flexible_data"])

                self.sizeTS = self.datasetTS.mem_size()
                self.loadTS = True
                if self.settings["OTHER"]["verbose"]:
                    print("Training data transformed.")
                    print("\nContinuing with validation data...\n")

            if not self.loadVS:

                if not self.transform_data:
                    if self.settings["OTHER"]["verbose"]:
                        print("\nNo transformation is performed on validation data!")
                else:
                    if self.settings["OTHER"]["verbose"]:
                        print("\nTransforming validation data...")
                self.datasetVS = dataset(self.scalingX.scale(validation_data[0]),
                                         self.scalingY.scale(validation_data[1]))
                del validation_data
                self.sizeVS = self.datasetVS.mem_size()
                self.loadVS = True

                if self.settings["OTHER"]["verbose"]:
                    print("Validation data transformed.\n")

        end_time_loader = timeit.default_timer()

        if self.settings["OTHER"]["verbose"]:
            print("Duration:", np.round(end_time_loader-start_time_loader, 3), "s")
            print("Training data:", len(self.datasetTS), "elements - Memory size: ", self.sizeTS/1e6, "MB")
            print("Validation data:", len(self.datasetVS), "elements - Memory size: ", self.sizeVS/1e6, "MB")
            print("Data is loaded, Training can begin!\n")
            print("================================================\n")


class ScalingX:
    
    def __init__(self, scalers=[], newscale=True, copy_to_dev=True, device='cpu', dtype_X=torch.float, dtype_Y=torch.float, amp=False, flexible_data=False):
        self.amp = amp
        self.scalers = scalers
        self.newscale = newscale
        self.device = device
        self.copy_to_dev = (copy_to_dev and self.device!='cpu')
        self.dtype_X, self.dtype_Y = dtype_X, dtype_Y
        self.flexible_data = flexible_data

    def scale(self, data):

        if self.newscale:
            if not self.scalers:
                self.pipe_X = None
                self.fitted_scalers_X = None
                if type(data) == torch.Tensor or self.flexible_data:
                    transformed_data = data
                else:
                    transformed_data = torch.tensor(data)
            else:
                self.pipe_X = Pipeline(self.scalers)
                self.fitted_scalers_X = self.pipe_X.fit(data)
                transformed_data = self.fitted_scalers_X.transform(data)

        else:
            if self.fitted_scalers_X:
                transformed_data = self.fitted_scalers_X.transform(data)
            else:
                if (type(data) == torch.Tensor) or self.flexible_data:
                    transformed_data = data
                else:
                    transformed_data = torch.tensor(data)

        if not self.amp and not self.flexible_data:
            transformed_data = transformed_data.to(dtype=self.dtype_X)
        if self.copy_to_dev and not self.flexible_data:
            transformed_data = transformed_data.to(device=self.device)
        
        self.newscale=False

        return transformed_data


class ScalingY:
    
    def __init__(self, scalers=[], newscale=True, copy_to_dev=True, device='cpu', dtype_X=torch.float, dtype_Y=torch.float, amp=False, flexible_data=False):
        self.amp = amp
        self.scalers = scalers
        self.newscale = newscale
        self.device = device
        self.copy_to_dev = (copy_to_dev and self.device!='cpu')
        self.dtype_X, self.dtype_Y = dtype_X, dtype_Y
        self.flexible_data = flexible_data

    def scale(self, data):

        if self.newscale:
            if not self.scalers:
                self.pipe_y = None
                self.fitted_scalers_y = None
                if type(data) == torch.Tensor or self.flexible_data:
                    transformed_data = data
                else:
                    transformed_data = torch.tensor(data)
            else:
                self.pipe_y = Pipeline(self.scalers)
                self.fitted_scalers_y = self.pipe_y.fit(data)
                transformed_data = self.fitted_scalers_y.transform(data)

        else:
            if self.fitted_scalers_y:
                transformed_data = self.fitted_scalers_y.transform(data)
            else:
                if (type(data) == torch.Tensor) or self.flexible_data:
                    transformed_data = data
                else:
                    transformed_data = torch.tensor(data)

        if not self.amp and not self.flexible_data:
            transformed_data = transformed_data.to(dtype=self.dtype_X)
        if self.copy_to_dev and not self.flexible_data:
            transformed_data = transformed_data.to(device=self.device)
        
        self.newscale=False

        return transformed_data


class InverseScaling:
    
    def __init__(self, scalers_fit, copy_to_dev=True, device='cpu', dtype=torch.float, amp=False, flexible_data=False):
        self.amp = amp
        self.scalers_fit = scalers_fit
        self.device = device
        self.copy_to_dev = (copy_to_dev and self.device!='cpu')
        self.dtype = dtype
        self.flexible_data = flexible_data

    def scale(self, data):
        output = self.scalers_fit.inverse_transform(data)
        
        if not self.amp and not self.flexible_data:
            output.to(dtype=self.dtype)
        if self.copy_to_dev and not flexible_data:
            output = output.to(self.device)

        return output
