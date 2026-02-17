import time
import copy
import numpy as np
import functools

import torch
import torch.onnx
import torch.profiler
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from onnxruntime.quantization import QuantFormat, QuantType, quantize_static, quantize_dynamic
from onnxconverter_common import float16
from onnxconverter_common import auto_mixed_precision

from NeuralNetworkClass.NeuralNetworkClasses.utils.functions import deep_update, print_dict
from  NeuralNetworkClass.NeuralNetworkClasses.utils.custom_loss_functions import *

from  NeuralNetworkClass.NeuralNetworkClasses.modules.linear import fc_layer
from  NeuralNetworkClass.NeuralNetworkClasses.modules.conv2d import conv2d_layer
from  NeuralNetworkClass.NeuralNetworkClasses.modules.conv1d import conv1d_layer
from  NeuralNetworkClass.NeuralNetworkClasses.modules.maxpool2d import maxpool2d_layer
from  NeuralNetworkClass.NeuralNetworkClasses.modules.batchnorm2d import batchnorm2d_layer
from  NeuralNetworkClass.NeuralNetworkClasses.modules.adaptiveAvgPool2d import adaptiveAvgPool2d_layer
from  NeuralNetworkClass.NeuralNetworkClasses.modules.flatten import flatten
from  NeuralNetworkClass.NeuralNetworkClasses.modules.resnet_basic import resnet_basic_layer
from  NeuralNetworkClass.NeuralNetworkClasses.modules.dropout import dropout_layer
from  NeuralNetworkClass.NeuralNetworkClasses.modules.RBF import gaussian_layer


import os
import timeit
import socket
import onnx

print_flush = functools.partial(print, flush=True)


### This layer dictionary will be used to assign the
### layers in General_NN accoriding to a list of strings

layer_dictionary = {
    "fc": "fc_layer",
    "conv2d": "conv2d_layer",
    "conv1d": "conv1d_layer",
    "maxpool": "maxpool2d_layer",
    "flatten": "flatten",
    "dropout": "dropout_layer",
    "adapool": "adaptiveAvgPool2d_layer",
    "downsample": "downsampling2d_channels_layer",
    "resnet_basic": "resnet_basic_layer",
    "rbf_gaussian": "gaussian_layer",
    "resnet_full": "ResNetImg",
}

### General_NN: A class which can define a Neural network according to strings given in layer_types,
### activation funcitons given in act_func and parameters given in params (typically dimensions of in, out and kernel)


class General_NN(nn.Module):

    def __init__(
        self,
        settings={
            "GLOBAL": {
                "parameters": [[1, 1, 3]],
                "layer_types": ["conv1d"],
                "activation_functions": [nn.ReLU],
                "weight_init": torch.nn.init.xavier_uniform_,
            },
            "DATA": {
                "scale_data": True,
            },
            "MACHINE_OPTIONS": {
                "device": None,
                "dtype": torch.float},
            "OTHER": {
                "verbose": True,
            },
        },
    ):

        super(General_NN, self).__init__()

        self.mode = "eval"

        self.parameters, self.layer_types, self.activation_functions = (
            settings["GLOBAL"]["parameters"],
            settings["GLOBAL"]["layer_types"],
            settings["GLOBAL"]["activation_functions"],
        )

        self.dtype = settings["MACHINE_OPTIONS"]["dtype"]
        if settings["MACHINE_OPTIONS"]["device"] is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = settings["MACHINE_OPTIONS"]["device"]

        self.scaling_X = []
        self.scaling_Y = []
        self.inverse_X = []
        self.inverse_Y = []
        self.scale = settings["DATA"]["scale_data"]

        if len(self.parameters) != len(self.activation_functions):

            raise ValueError(
                "len(layers_sizes): {val1} and len(act_func): {val2} have different length, but must be of same length!".format(
                    val1=len(self.parameters), val2=len(self.activation_functions)
                )
            )

        ########### Define the network ##############

        self.layers = nn.ModuleList()

        print_flush("\nThis is the network structure:\n")

        for i in range(len(self.params)):
            self.layers.append(
                eval(
                    layer_dictionary[self.layer_types[i]]
                    + '(params=self.params[i], activation=self.activation_functions[i], weight_init=settings["GLOBAL"]["weight_init"], verbose=settings["OTHER"]["verbose"], device=self.device, dtype=self.dtype)'
                )
            )

        self.layers_seq = nn.Sequential(*self.layers)

    @torch.jit.ignore
    def forward(self, X):

        if self.mode == "train":

            ### Data is expected to be scaled already

            output = self.layers_seq(self.dtype(X))

        elif self.mode == "eval":

            ### Check for device and datascaling

            if isinstance(X, np.ndarray):

                if self.scale and self.scaling_X:
                    scaled = self.scaling_X.scale(X)
                else:
                    scaled = torch.tensor(X)

                predict = self.layers_seq(self.dtype(scaled))

                if self.scale and self.inverse_Y:
                    output = self.inverse_Y.scale(predict.cpu().detach().numpy())
                else:
                    output = predict

            elif isinstance(X, torch.Tensor):

                if self.scale and self.scaling_X:
                    scaled = self.scaling_X.scale(X.cpu().detach().numpy())
                else:
                    scaled = X

                predict = self.layers_seq(self.dtype(scaled))

                if self.scale and self.inverse_Y:
                    output = self.inverse_Y.scale(predict.cpu().detach().numpy())
                else:
                    output = predict

            else:

                print_flush(
                    "Data was neither numpy.ndarray nor torch.Tensor... Evaluating by conversion..."
                )

                if self.scale and self.scaling_X:
                    scaled = self.scaling_X.scale(np.array(X))
                else:
                    scaled = torch.tensor(X)

                predict = self.layers_seq(self.dtype(scaled))

                if self.scale and self.inverse_Y:
                    output = self.inverse_Y.scale(predict.cpu().detach().numpy())
                else:
                    output = predict

        else:

            print_flush("Network must be in mode (eval) or (train). Please specify!")
            output = False

        return output


### NN: A class for training a Neural network and predicting output (so to say a wrapper class for a General_NN)


class NN:

    def __init__(self, neural_net, settings={}):

        self.network = neural_net
        self.settings = {
            "GLOBAL": {
                "epochs": 1,
                "loss_function": nn.MSELoss(),
                "optimizer": optim.Adam,
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau,
                "weight_init": torch.nn.init.xavier_uniform_,
                "amp": False, ### Mixed precision training
                "quantization": False, ### Quantize the model to e.g. Int8
                "profiler": False,
            },
            "DATA_OPTIONS": {
                "batchsize_schedule": [0],
                "batchsize_training": [None],
                "batchsize_validation": None,
                "shuffle_every_epoch": True,
                "num_workers": 0,
                "pin_memory": None,
                "copy_to_device": False,
            },
            "AMP_OPTIONS" : {
                "dtype": torch.float16,
            },
            "QUANTIZATION_OPTIONS" : {
                "ONNX": {
                    "weight_type": QuantType.QInt8,
                    "per_channel": False,
                },
                "PYTORCH": {
                    "dtype": torch.qint8,
                },
            },
            "LOSS_OPTIONS": {},
            "OPTIMIZER_OPTIONS": {
                "lr": 0.001,
                "weight_decay": 0
                },
            "SCHEDULER_OPTIONS": {
                "patience": 5,
                "factor": 0.5
                },
            "MACHINE_OPTIONS": {
                "device": None,
                "cpu_threads": None,
                "dtype": torch.float32,
                "multigpu": -1,
                "verbose": True,
            },
            "ONNX": {
                "export_params": True,
                "opset_version": 15,
                "do_constant_folding": True,
                "input_names": ["input"],
                "output_names": ["output"],
                "dynamic_axes": {
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"},
                },
            },
            # "PROFILER": {
            #     "schedule": torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            #     "on_trace_ready": torch.profiler.tensorboard_trace_handler('./profiler_log'),
            #     "record_shapes": True,
            #     "with_stack": True,
            # },
            "OTHER": {
                "verbose": True,
                "n_samples": np.inf,
                "save_at_epochs": None,
            },
        }

        self.settings = deep_update(self.settings, settings, name="Network settings")

        self.rank = 0
        self.worldsize = 1
        self.multigpu = self.settings["MACHINE_OPTIONS"]["multigpu"]
        self.verbose = self.settings["MACHINE_OPTIONS"]["verbose"] and (int(os.environ.get("SLURM_PROCID", 0)) == 0)

        if self.settings["GLOBAL"]["weight_init"] is not None:
            self.__weight_init__() ### Needs to be done before the DDP wrapping
        if self.multigpu:
            healthy = self.detect_healthy_gpus()
            self.rank, self.worldsize = self.multigpu_training_setup()

    def __call__(self, X):
        return self.network(X)

    def forward(self, X):
        if isinstance(self.network, torch.nn.parallel.DistributedDataParallel):
            model = self.network.module
        else:
            model = self.network
        return model(X)
    
    def __default_weight_init__(self, weight_init=nn.init.normal_):
        if weight_init is not None:
            for layer in self.network.modules():
                if hasattr(layer, 'weight'):
                    weight_init(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.1)

    def __weight_init__(self):
        try:
            self.network.__weight_init__(weight_init=self.settings["GLOBAL"]["weight_init"])
            if self.verbose:
                print_flush("Weight initialization successful with method:", self.settings["GLOBAL"]["weight_init"])
        except Exception as e1:
            print_flush("Weight initialization failed with error:", e1)
            try:
                self.network.weight_init(weight_init=self.settings["GLOBAL"]["weight_init"])
                if self.settings["OTHER"]["verbose"]:
                    print_flush("Weight initialization successful with method:", self.settings["GLOBAL"]["weight_init"])
            except Exception as e2:
                print_flush("Weight initialization failed with error:", e2)
                print_flush("Initializing weights according to own rule: {}".format(nn.init.normal_))
                self.__default_weight_init__(weight_init=nn.init.normal_)
                # print_flush("Continuing without weight initialization...")

    def training(self, data, settings={}):

        ### data = [DataLoader(training set), DataLoader(validation set), DataLoader(test set)]

        self.settings = deep_update(self.settings, settings, name="Network settings")
        if self.verbose:
            print_dict(self.settings, name="Training settings")

        ### Setting the device on which to run and data-type ###
        self.dtype = self.settings["MACHINE_OPTIONS"]["dtype"]
        self.multigpu = self.settings["MACHINE_OPTIONS"]["multigpu"]

        if not self.multigpu:
            if self.settings["MACHINE_OPTIONS"]["device"] is None:
                if torch.cuda.is_available():
                    self.device = "cuda:0"
                    # if self.settings["MACHINE_OPTIONS"]["device_id"] is None or "all":
                    #     self.device = "cuda"
                    # elif type(self.settings["MACHINE_OPTIONS"]["device_id"]) == type(list()):
                    #     self.device = "cuda:" + ",".join(list(map(str, self.settings["MACHINE_OPTIONS"]["device_id"])))
                elif torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            else:
                self.device = self.settings["MACHINE_OPTIONS"]["device"]

            self.network.to(device=self.device, dtype=self.dtype)

        ### print_flushing device information
        if self.multigpu:
            if self.verbose and self.rank == 0:
                print_flush("[Multi-GPU training]", self.worldsize, "participating GPUs.")
                for i in range(self.worldsize):
                    dev_id = torch.cuda.device_count()
                    dev_name = torch.cuda.get_device_name(i)
                    print_flush("Device ID: ", i, ", Device name: ", dev_name)
        else:
            if ("cuda" in self.device):
                dev_id = torch.cuda.current_device()  # returns you the ID of your current device
                dev_name = torch.cuda.get_device_name(dev_id)
                # dev_mem_use = torch.cuda.memory_allocated(dev_id)          #returns you the current GPU memory usage by tensors in bytes for a given device
                # dev_mem_man = torch.cuda.memory_reserved(dev_name)         #returns you the current GPU memory managed by caching allocator in bytes for a given device
                torch.cuda.empty_cache()  # clear variables in cache that are unused
                if self.verbose:
                    print_flush("\nRunning on GPU")
                    print_flush("Device ID: ", dev_id, ", Device name: ", dev_name, "\n")
            elif self.device == "mps":
                if self.verbose:
                    print_flush("\nRunning on MPS\n")
            else:
                if self.settings["MACHINE_OPTIONS"]["cpu_threads"] is not None:
                    torch.set_num_threads(int(self.settings["MACHINE_OPTIONS"]["cpu_threads"]))
                if self.verbose:
                    print_flush("\nRunning on CPU")
                    print_flush("{} CPU threads\n".format(torch.get_num_threads()))

        ### Setting some variables of the network ###

        self.epochs = self.settings["GLOBAL"]["epochs"]
        self.optimizer = self.settings["GLOBAL"]["optimizer"](self.network.parameters(), **self.settings["OPTIMIZER_OPTIONS"])
        self.scheduler = self.settings["GLOBAL"]["scheduler"](self.optimizer, **self.settings["SCHEDULER_OPTIONS"])
        self.loss_function = self.settings["GLOBAL"]["loss_function"]
        self.scaler = torch.amp.GradScaler(enabled=self.settings["GLOBAL"]["amp"])

        ### Checking if data was loaded properly and extracting the scalers ###

        self.len_TS = len(data.datasetTS)
        self.len_VS = len(data.datasetVS)

        if not data.loadTS or not data.loadVS:
            print_flush("The data has not been loaded. Shutting down.")
            exit()

        if data.transform_data:
            self.network.scaling_X = data.scaling_X
            self.network.scaling_Y = data.scaling_Y
            self.network.inverse_X = data.inverse_X
            self.network.inverse_Y = data.inverse_Y

        if self.settings["DATA_OPTIONS"]["batchsize_training"] in [-1, None, np.inf, [-1], [None], [np.inf]]:
            self.settings["DATA_OPTIONS"]["batchsize_training"] = [self.len_TS]
        self.batchsize_schedule, self.batchsizes = (
            self.settings["DATA_OPTIONS"]["batchsize_schedule"],
            self.settings["DATA_OPTIONS"]["batchsize_training"],
        )

        self.pin_memory = self.settings["DATA_OPTIONS"]["pin_memory"]
        if self.pin_memory is None:
            self.pin_memory = ("cuda" in self.device or "mps" in self.device) # and (self.dtype == torch.float)


        if self.verbose:
            print_flush("\n============ Neural Network training ============\n")

        ### Tensorboard profiler (https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html)
        #if self.settings["GLOBAL"]["profiler"]:
        #    prof = torch.profiler.profile(**self.settings["PROFILER"])
        #    prof.start()
        #    # torch.cuda.memory._record_memory_history(max_entries=100000)
        #    # self.snapshot_counter = 0

        ######################### Starting the training #########################

        self.network.mode = "train"
        training_loss = []
        validation_loss = []

        ### Creating the validation data once
        if self.settings["DATA_OPTIONS"]["batchsize_validation"] in [-1, None, np.inf]:
            self.settings["DATA_OPTIONS"]["batchsize_validation"] = self.len_VS

        validation_dataloader = DataLoader(
            dataset=data.datasetVS,
            batch_size=self.settings["DATA_OPTIONS"]["batchsize_validation"],
            num_workers=self.settings["DATA_OPTIONS"]["num_workers"],
            pin_memory=self.pin_memory,
            shuffle=False,
        )


        ### Training epochs ###
        for epoch in range(int(self.settings["GLOBAL"]["epochs"])):

            start_time = timeit.default_timer()  # Timer start

            if epoch in self.settings["DATA_OPTIONS"]["batchsize_schedule"]:
                idx_tr = self.settings["DATA_OPTIONS"]["batchsize_schedule"].index(epoch)
                if self.verbose:
                    print_flush("--- Batch size:", self.batchsizes[idx_tr], "---")

            ### Iterating through the training data ###

            if self.multigpu:
                dist.barrier()
                sampler = DistributedSampler(data.datasetTS, num_replicas=self.worldsize, rank=self.rank, shuffle=True)
                sampler.set_epoch(epoch)
            else:
                sampler = None

            train_dataloader = DataLoader(
                dataset=data.datasetTS,
                batch_size=self.batchsizes[idx_tr],
                num_workers=self.settings["DATA_OPTIONS"]["num_workers"],
                pin_memory=self.pin_memory,
                shuffle=(False if self.multigpu else self.settings["DATA_OPTIONS"]["shuffle_every_epoch"]),
                sampler=sampler
            )

            tr_loss = 0
            # av_tr_loss = 0

            for counter, entry_tr in enumerate(train_dataloader, 0):
                # if self.settings["GLOBAL"]["profiler"]:
                #     prof.step()
                #     # try:
                #     #     torch.cuda.memory._dump_snapshot("snapshot_{}.pickle".format(str(self.snaptshot_counter)))
                #     #     self.snapshot_counter += 1
                #     # except Exception as e:
                #     #     logger.error(f"Failed to capture memory snapshot {e}")

                BX, BY = entry_tr

                assert counter <= self.settings["OTHER"]["n_samples"]

                if (self.settings["DATA_OPTIONS"]["copy_to_device"] and self.device != "cpu"):
                    BX = BX.to(device=self.device)
                    BY = BY.to(device=self.device)


                if(self.settings["GLOBAL"]["amp"]):
                    BX = BX.to(dtype=torch.float32)
                    BY = BY.to(dtype=torch.float32)
                    self.network = self.network.to(dtype=torch.float32)
                    with torch.autocast(device_type=("cuda" if ("cuda" in self.device) else self.device)):
                        self.optimizer.zero_grad()
                        training_out = self.network(BX)
                        loss = self.loss_function(training_out, BY, **self.settings["LOSS_OPTIONS"])
                        self.scaler.scale(loss).backward()
                        # self.scaler.unscale_(self.optimizer) ### Unscales the gradients of optimizer's assigned parameters in-place
                        # nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.1) # Worth adding at some point?
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                else:
                    BX = BX.to(dtype=self.dtype)
                    BY = BY.to(dtype=self.dtype)

                    self.optimizer.zero_grad()
                    training_out = self.network(BX)
                    loss = self.loss_function(training_out, BY, **self.settings["LOSS_OPTIONS"])
                    self.scaler.scale(loss).backward()
                    # self.scaler.unscale_(self.optimizer) ### Unscales the gradients of optimizer's assigned parameters in-place
                    # nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.1) # Worth adding at some point?
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                tr_loss += loss

                # if self.settings["OTHER"]["verbose"]:
                #     av_tr_loss += loss.item()
                #     if counter % 1000 == 999:
                #         print_flush("{val1} minibatches. Average training loss: {val2}".format(val1=counter+1, val2 = np.round(av_tr_loss/1000.,6)))
                #         av_tr_loss = 0

                ### Free memory
                del BX, BY
                # if self.device == "cuda":
                #     torch.cuda.empty_cache()
                # elif self.device == "mps":
                #     torch.backends.mps.empty_cache()

                # torch.cuda.memory_allocated()/1e9 # in Gb
                # torch.cuda.memory_reserved()
            # ----------------------------

            if self.multigpu:
                dist.barrier()
    
            ### Iterating through the validation data ###
            val_loss = 0
            for counter, entry_val in enumerate(validation_dataloader, 0):
                val_obs, val_label = entry_val
                if (self.settings["DATA_OPTIONS"]["copy_to_device"] and self.device != "cpu"):
                    val_obs, val_label = val_obs.to(device=self.device, dtype=self.dtype), val_label.to(device=self.device, dtype=self.dtype)
                val_loss += self.loss_function(self.network(val_obs).float(), val_label.float(), **self.settings["LOSS_OPTIONS"]).cpu()
                del val_obs, val_label

            #    if verbose and counter % 1000 == 999:
            #        print_flush("Gone through {} samples in validation set".format(counter+1))

            # ----------------------------

            ### Prepare for output: Training loss and validation loss normalized to number of samples ###

            tr_loss = tr_loss.cpu() / len(train_dataloader)
            val_loss = val_loss.cpu() / len(validation_dataloader)
            training_loss.append(tr_loss.detach().numpy())
            validation_loss.append(val_loss.detach().numpy())
            self.scheduler.step(val_loss)

            # ----------------------------

            end_time = timeit.default_timer()  # Timer end

            ### Saving neural network at certain epochs

            if self.rank==0 and self.settings["OTHER"]["save_at_epochs"] is not None:

                # Create the output directory
                if not os.path.exists("./networks_epoch"):
                    os.makedirs("./networks_epoch")

                # Create some esample data for ONNX
                bx, by = next(iter(train_dataloader))
                example_data = bx[:1024].to(device="cpu", dtype=torch.float32)

                # Save model at given epochs
                if type(self.settings["OTHER"]["save_at_epochs"]) in [list, np.ndarray]:
                    if epoch in self.settings["OTHER"]["save_at_epochs"]:
                        print_flush("\n--- Saving intermediate network ---")
                        # self.save_net("./networks_epoch/net_" + str(epoch) + ".pt", avoid_q=True)
                        # self.save_jit_script(path="./networks_epoch/net_jit_" + str(epoch) + ".pt")
                        self.save_onnx(example_data, "./networks_epoch/net_" + str(epoch) + ".onnx")
                        # self.check_onnx("./networks_epoch/net_" + str(epoch) + ".onnx")
                        print_flush("--- Intermediate network saved ---\n")
                elif self.settings["OTHER"]["save_at_epochs"] in ["all", -1]:
                    print_flush("\n--- Saving intermediate network ---")
                    # self.save_net("./networks_epoch/net_" + str(epoch) + ".pt", avoid_q=True)
                    # self.save_jit_script(path="./networks_epoch/net_jit_" + str(epoch) + ".pt")
                    self.save_onnx(example_data, "./networks_epoch/net_" + str(epoch) + ".onnx")
                    # self.check_onnx("./networks_epoch/net_" + str(epoch) + ".onnx")
                    print_flush("--- Intermediate network saved ---\n")

                # Putting network back on the device and dtype that is needed in training
                self.network.to(device=self.device, dtype=self.dtype)


            ### print_flushing the stats ###

            if self.verbose:
                print_flush(
                    "Epoch ",
                    epoch + 1,
                    "/",
                    self.epochs,
                    "| batch size: ",
                    self.batchsizes[idx_tr],
                    ", validation loss: ",
                    np.round(val_loss.detach().numpy(), 6),
                    ", training loss: ",
                    np.round(tr_loss.detach().numpy(), 6),
                    ", delta_t : ",
                    np.round(end_time - start_time, 3),
                    "s",
                )

            # ----------------------------

        # print_flush(list(self.network.parameters()))
        if self.verbose:
            print_flush("\nTraining finished!\n")

        # if self.settings["GLOBAL"]["profiler"]:
        #     prof.stop()
        #     # torch.cuda.memory._record_memory_history(enabled=None)

        ###########################################################################

        # self.network.to('cpu')
        self.training_loss = training_loss
        self.validation_loss = validation_loss

        self.network.mode = "eval"

        if self.multigpu > 0:
            dist.destroy_process_group()
            self.delete_masteraddr_file()

    # def load_data(self, data):
    #
    #     train_dataloader = DataLoader(
    #         dataset=data.datasetTS,
    #         batch_size=self.batchsizes[idx_tr],
    #         num_workers=self.settings["DATA_OPTIONS"]["num_workers"],
    #         pin_memory=self.pin_memory,
    #         shuffle=(False if self.multigpu else self.settings["DATA_OPTIONS"]["shuffle_every_epoch"]),
    #         sampler=(DistributedSampler(data.datasetTS) if self.multigpu else None),
    #     )
    #
    #      validation_dataloader = DataLoader(
    #         dataset=data.datasetVS,
    #         batch_size=self.settings["DATA_OPTIONS"]["batchsize_validation"],
    #         num_workers=self.settings["DATA_OPTIONS"]["num_workers"],
    #         pin_memory=self.pin_memory,
    #         shuffle=False,
    #     )



    ### --- Multi-GPU training setup --- ###

    def multigpu_training_setup(self):
        """
        Setup PyTorch DistributedDataParallel using Slurm environment variables.
        Automatically handles single-node and multi-node jobs.

        Assumes:
            - Slurm launches the job with srun
            - self.multigpu contains the requested total number of GPUs (world size)
            - 8 GPUs per node
        """

        # ----------------------------------------------------------------------
        # 1. Read Slurm variables
        # ----------------------------------------------------------------------
        slurm_procid   = int(os.environ.get("SLURM_PROCID", 0))    # global rank
        slurm_localid  = int(os.environ.get("SLURM_LOCALID", 0))   # local rank on this node
        slurm_ntasks   = int(os.environ.get("SLURM_NTASKS", 1))    # total tasks (world size)
        slurm_nodeid   = int(os.environ.get("SLURM_NODEID", 0))    # node index (0 .. nnodes-1)
        slurm_nnodes   = int(os.environ.get("SLURM_NNODES", 1))    # number of nodes
        slurm_jobid    = int(os.environ.get("SLURM_JOBID", 0))
        user           = os.environ.get("USER", "unknown")

        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if slurm_ntasks > len(visible_devices):
            raise RuntimeError(
                f"Slurm assigned {slurm_ntasks} ranks but only {len(visible_devices)} GPUs are healthy."
            )
        local_rank = int(os.environ.get("SLURM_LOCALID", 0))

        # Let user-provided n_gpus override Slurm if desired
        worldsize = min(len(visible_devices), slurm_ntasks)

        # ----------------------------------------------------------------------
        # 2. Set PyTorch DDP expected environment variables
        # ----------------------------------------------------------------------
        os.environ["RANK"]       = str(slurm_procid)
        os.environ["WORLD_SIZE"] = str(worldsize)
        os.environ["LOCAL_RANK"] = str(slurm_localid)

        # ----------------------------------------------------------------------
        # 3. Determine MASTER_ADDR (multi-node safe)
        #    Only node 0 writes its hostname to a file shared across nodes.
        # ----------------------------------------------------------------------
        self.hostfile = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"master_addr_{user}_{slurm_jobid}.txt")

        if slurm_nodeid == 0:
            # Node 0 writes its hostname
            with open(self.hostfile, "w") as f:
                f.write(socket.gethostname())

        # Slurm ensures simultaneous startup → we simply read the file
        time_slept = 0
        while (not os.path.exists(self.hostfile)) and time_slept < 10:
            time.sleep(0.01)
            time_slept += 0.01
        if time_slept >= 10:
            raise TimeoutError("Timeout waiting for master address file.")
        with open(self.hostfile, "r") as f:
            master_addr = f.read().strip()

        os.environ["MASTER_ADDR"] = master_addr

        # ----------------------------------------------------------------------
        # 4. Safe, collision-free master port
        # ----------------------------------------------------------------------
        master_port = 20000 + (slurm_jobid % 20000)
        os.environ["MASTER_PORT"] = str(master_port)

        # ----------------------------------------------------------------------
        # 5. Initialize Process Group
        # ----------------------------------------------------------------------
        dist.init_process_group(
            backend="nccl",
            init_method="env://"
        )

        # ----------------------------------------------------------------------
        # 6. Bind process to its GPU
        # ----------------------------------------------------------------------
        # PyTorch will reindex CUDA_VISIBLE_DEVICES → 0..N-1
        ddp_device = local_rank

        torch.cuda.set_device(ddp_device)
        self.device = f"cuda:{ddp_device}"

        # ----------------------------------------------------------------------
        # 7. Wrap model in DDP
        # ----------------------------------------------------------------------
        self.network = DDP(self.network.to(self.device),
                        device_ids=[slurm_localid],
                        output_device=slurm_localid)

        return slurm_localid, worldsize

    def delete_masteraddr_file(self):
        """
        Delete the temporary master address file created during multi-node DDP setup.
        Only node 0 should perform the deletion.
        """

        if self.rank == 0:
            try:
                os.remove(self.hostfile)
            except OSError:
                pass

    def gpu_is_healthy(self, gpu_id):
        try:
            # Allocate a small tensor on the GPU
            t = torch.tensor([1.0], device=f"cuda:{gpu_id}")
            t = t * 2
            torch.cuda.synchronize(gpu_id)
            return True
        except Exception as e:
            print_flush(f"[GPU {gpu_id}] FAILED: {e}")
            return False


    def detect_healthy_gpus(self):
        slurm_procid = int(os.environ.get("SLURM_PROCID", -1))
        count = torch.cuda.device_count()

        if slurm_procid == -1:
            print_flush("Warning: SLURM_PROCID not set. Assuming single-process execution.")
            slurm_procid = 0
            healthy = list(range(count))
            print_flush("Healthy GPUs:", healthy)
            return healthy
        
        else:
            healthy = []

            for gpu in range(count):
                if self.gpu_is_healthy(gpu):
                    healthy.append(gpu)

            if slurm_procid == 0:
                print_flush("Healthy GPUs:", healthy)

            if not healthy:
                raise RuntimeError("No healthy GPUs available.")

            # Make only the healthy GPUs visible
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, healthy))

            return healthy

    def save_losses(self, path=["./training_loss.txt", "./validation_loss.txt"]):
        if self.rank == 0:
            np.savetxt(path[0], self.training_loss)
            np.savetxt(path[1], self.validation_loss)
            print_flush("Training and validation loss saved!")

    def eval(self):
        self.network.mode = "eval"
        self.network = self.network.cpu().eval()

    def save_net(self, path="./net.pt", avoid_q=False):
        if self.rank == 0:

            if isinstance(self.network, torch.nn.parallel.DistributedDataParallel):
                model = self.network.module
            else:
                model = self.network

            if not avoid_q:
                if os.path.isfile(path):
                    response = input("File exists. Do you want to overwrite it? [y/n] ")
                    if response in ["y", "yes", "Y", "Yes", "YES"]:
                        # torch.save(self.network.state_dict(), path)
                        torch.save(model.state_dict(), path)
                        print_flush("Network saved")
                    else:
                        print_flush("Network not saved!")

                else:
                    # torch.save(self.network.state_dict(), path)
                    torch.save(model.state_dict(), path)
                    print_flush("Network saved")

            else:
                torch.save(model.state_dict(), path)
                print_flush("Network saved")

            if self.settings["GLOBAL"]["quantization"]:
                model = self.network.module if isinstance(self.network, DDP) else self.network
                quantized_model = torch.ao.quantization.quantize_dynamic(
                    model.cpu(),
                    {torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv3d},  # a set of layers to dynamically quantize
                    **self.settings["QUANTIZATION_OPTIONS"]["PYTORCH"]
                )
                torch.save(quantized_model, path.split(".pt")[0] + "_quantized_" + str(self.settings["QUANTIZATION_OPTIONS"]["PYTORCH"]["dtype"]).split(".")[-1].lower() + ".pt")
                print_flush("Quantized PyTorch model saved.\n")

    def jit_script_model(self):
        self.jit_scripted_model = torch.jit.script(self.network.cpu())
        return self.jit_scripted_model

    def save_jit_script(self, path="./net_jit_script.pt"):
        if self.rank == 0:
            torch.jit.save(self.jit_script_model(), path)
            print_flush("Model saved!")

    def save_onnx(
        self,
        example_data=torch.tensor([[]], requires_grad=True),
        path="./net_onnx.onnx"
    ):
        if self.multigpu:
            dist.barrier()

        if self.rank == 0:
            # 1) unwrap DDP
            base_model = self.network.module if isinstance(self.network, DDP) else self.network

            # 2) make an explicit copy (so we don't mutate training model)
            model_copy = copy.deepcopy(base_model)

            # 3) export copy in a clean state
            model_copy.eval()
            model_copy.to(device="cpu", dtype=torch.float32)

            example_data = example_data.to(device="cpu", dtype=torch.float32)

            with torch.no_grad():
                torch.onnx.export(
                    model_copy,
                    example_data,
                    path,
                    **self.settings["ONNX"]
                )

            # optional: free RAM
            del model_copy

            if self.settings["GLOBAL"]["amp"]:
                model = onnx.load(path)
                model_fp16 = float16.convert_float_to_float16(
                    model,
                    min_positive_val=1e-7,
                    max_finite_val=1e4,
                    keep_io_types=False,
                    disable_shape_infer=False,
                    op_block_list=None,
                    node_block_list=None,
                )
                onnx.save(model_fp16, path.split(".onnx")[0] + "_fp16.onnx")

            if self.settings["GLOBAL"]["quantization"]:
                quantize_dynamic(
                    path,
                    path.split(".onnx")[0] + "_quantized_" +
                    str(self.settings["QUANTIZATION_OPTIONS"]["ONNX"]["weight_type"]).lower() + ".onnx",
                    **self.settings["QUANTIZATION_OPTIONS"]["ONNX"]
                )
                print_flush("Quantized ONNX model saved.")

        if self.multigpu:
            dist.barrier()

    def check_onnx(self, path="./net_onnx.onnx"):
        if self.multigpu:
            dist.barrier()
        if self.rank == 0:
            try:
                onnx_model = onnx.load(path)
                onnx.checker.check_model(onnx_model)
                print_flush("ONNX checker: Success!")
            except Exception as e:
                print_flush("Failure in ONNX checker!")
                print_flush(e)
        if self.multigpu:
            dist.barrier()

    def remove_cast_layer(onnx_fp16_path, onnx_fp16_fixed_path):
        """Removes the Cast layer at the beginning and updates all connected nodes."""
        model = onnx.load(onnx_fp16_path)
        graph = model.graph
        nodes = list(graph.node)

        # Find the Cast layer
        cast_node = None
        for node in nodes:
            if node.op_type == "Cast":
                cast_node = node
                break

        if not cast_node:
            print_flush("No Cast layer found. Model is already using FP16 input.")
            return

        cast_output = cast_node.output[0]  # The output of the Cast node
        cast_input = cast_node.input[0]  # The original input to the Cast node

        # Find all nodes that use the Cast output
        affected_nodes = [node for node in nodes if cast_output in node.input]

        # Redirect all affected nodes to use the original model input
        for node in affected_nodes:
            for i, inp in enumerate(node.input):
                if inp == cast_output:
                    node.input[i] = cast_input  # Replace with original model input

        # Remove the Cast node
        new_nodes = [node for node in nodes if node != cast_node]
        graph.ClearField("node")
        graph.node.extend(new_nodes)

        # Change model input type to FP16
        input_tensor = graph.input[0]
        input_tensor.type.tensor_type.elem_type = onnx.TensorProto.FLOAT16

        # Save modified model
        onnx.save(model, onnx_fp16_fixed_path)
        print_flush(f"Saved modified model without Cast layer to {onnx_fp16_fixed_path}")
