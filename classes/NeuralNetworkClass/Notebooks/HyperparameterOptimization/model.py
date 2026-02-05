import sys

sys.path.append("../NeuralNetworkClasses")
from NN_class import *
from custom_loss_functions import *
from dataset_loading import *

import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split

import itertools

class model():
    
    def __init__(self, X, y):
        super().__init__()
        TEST_SIZE = 0.2
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y,test_size=TEST_SIZE,shuffle=True)
        
    def score(self, num_layers, num_neurons_layer):

        X_SCALERS = []
        Y_SCALERS = []

        H_SIZES = list(itertools.chain(*[[[7,num_neurons_layer]],[[num_neurons_layer,num_neurons_layer]]*num_layers,[[num_neurons_layer,1]]]))
        LAYER_TYPES = list(itertools.chain(*[['fc'],['fc']*(len(H_SIZES)-2), ['fc']]))
        WEIGHT_INIT = nn.init.xavier_normal_
        GAIN = 5/3
        ACTIVATION =list(itertools.chain(*[[nn.ReLU()]*(len(H_SIZES)-1), [nn.Identity()]]))

        BATCH_SIZES = [200]
        N_EPOCHS = 3
        EPOCH_LS = [0]


        OPTIMIZER = optim.Adam
        WEIGHT_DECAY = 0
        SCHEDULER = optim.lr_scheduler.ReduceLROnPlateau 
        LOSS_FUNCTION = weighted_mse_loss
        LEARNING_RATE = 0.01

        NUM_THREADS = 0
        PATIENCE = 5
        FACTOR=0.5

        net = General_NN(params = H_SIZES, layer_types = LAYER_TYPES, act_func =ACTIVATION, w_init = WEIGHT_INIT, verbose=True)#, gain=GAIN)

        NeuralNet = NN(net)

        data = DataLoading([self.X_train, self.y_train], [self.X_test, self.y_test], 
                            X_data_scalers=X_SCALERS, y_data_scalers=Y_SCALERS,
                            batch_sizes=BATCH_SIZES, shuffle_every_epoch=True,
                            transformTS=False, transformVS=False)

        NeuralNet.training(data, epochs=N_EPOCHS, epochs_ls=EPOCH_LS, weights = False,
                           optimizer=OPTIMIZER,scheduler=SCHEDULER, loss_function=LOSS_FUNCTION, 
                           verbose=True, learning_rate=LEARNING_RATE, patience=PATIENCE, factor=FACTOR)

        return NeuralNet.validation_loss[-1]