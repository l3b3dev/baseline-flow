# -*- coding: utf-8 -*-
"""
Created on Tue May 10 11:08:04 2022

@author: USER
"""
import UQLibrary as uq

import os
import sys
import timeit
import torch
import numpy as np
from torch import nn
from torch.autograd import Variable, Function
from torch.utils.data import DataLoader, Dataset
import matplotlib as mpl

from read_dataset import data_from_name
from shallowdecoder_model import model_from_name, ShallowDecoder
from utils import sensor_from_name, error_summary, add_channels

import argparse

from BBB_network_sensing import main as network_fcn

mpl.style.available
mpl.style.use('seaborn-paper')

# ==============================================================================
# Analysis settings
# ==============================================================================

parser = argparse.ArgumentParser(description='PyTorch Example')
#
parser.add_argument('--name', type=str, default='shallow_decoder_drop', metavar='N', help='Model')
#
parser.add_argument('--data', type=str, default='cavflow30000.npy', metavar='N', help='dataset')
#
parser.add_argument('--sensor', type=str, default='leverage_score', metavar='N', help='chose between "wall" or "leverage_score"')
#
parser.add_argument('--n_sensors', type=int, default=25, metavar='N', help='number of sensors')
#
parser.add_argument('--epochs', type=int, default=4000, metavar='N', help='number of epochs to train (default: 10)')
#
parser.add_argument('--plotting', type=bool, default=False, metavar='N', help='number of epochs to train (default: 10)')
#
parser.add_argument('--network', type = str, default = '', metavar = 'N')
#
#
parser.add_argument('--sensor_pert', type = float, default = .1, metavar = 'N')
#
parser.add_argument('--out_dir', type = str, default = 'out', metavar = 'N')
#
parser.add_argument('--seed', type = int, default =10, metavar = 'N')
parser.add_argument('--logging', type = int, default =0, metavar = 'N')

parser.add_argument('--n_samp_morris', type = int, default = 40, metavar = 'N')
parser.add_argument('--l_morris', type = float, default =1/800, metavar = 'N')
parser.add_argument('--qoi_type', type = str, default = "error", metavar = 'N')
args = parser.parse_args()

np.random.seed(args.seed)
logging = args.logging


# ******************************************************************************
# Select dataset
# ******************************************************************************
dataset = args.data

# ******************************************************************************
# Paramteres
# ******************************************************************************
model_name = args.name

plotting = args.plotting
directory = f'data/cavflow/'


num_epochs = args.epochs
batch_size = 100
n_sensors = args.n_sensors
sensor_type = args.sensor
learning_rate = 1e-2
weight_decay = 1e-4
learning_rate_change = 0.9
weight_decay_change = 0.8
epoch_update = 100
alpha = 5e-8  # regularized pod for wall and sst

error_DD_train = []
error_dev_DD_train = []
error_DD_test = []
error_dev_DD_test = []

error_POD_train = []
error_dev_POD_train = []
error_POD_test = []
error_dev_POD_test = []

error_reg_POD_train = []
error_dev_reg_POD_train = []
error_reg_POD_test = []
error_dev_reg_POD_test = []

time_train = []

# ******************************************************************************
# read data and set sensor
# ******************************************************************************

X, X_test, m, n = data_from_name(directory + dataset)

X = np.asarray(X)
X_test = np.asarray(X_test)

# get size
outputlayer_size = X.shape[1]
n_snapshots_train = X.shape[0]
n_snapshots_test = X_test.shape[0]

# ******************************************************************************
# Rescale data between 0 and 1 for learning
# ******************************************************************************
Xmean = X.mean(axis=0)
X -= Xmean
X_test -= Xmean


# ******************************************************************************
# Get sensor locations
# ******************************************************************************
random_seed = np.random.choice(range(1000), 1)
if n_sensors == 5:
    sensor_locations = np.array([11228, 12957,  3018,  1065, 13747])
    sensors= X[:,sensor_locations]
    sensors_test = X_test[:,sensor_locations]
else:
    sensors, sensors_test, sensor_locations = sensor_from_name(sensor_type,
                                                                X, X_test,
                                                                sensor_num=n_sensors,
                                                                random_seed=random_seed, m=m, n=n)
    



#Convert sensor_locations to space definitions
x_center = -np.cos(np.pi*np.arange(0,m)/(n-1))
y_center = -np.cos(np.pi*np.arange(0,m)/(n-1))
yv, xv = np.meshgrid(x_center, y_center)
sensor_locations_2D = np.empty((n_sensors,2))
sensor_locations_2D[:,0] = xv.reshape(1, m*n)[:, sensor_locations]
sensor_locations_2D[:,1] = yv.reshape(1, m*n)[:, sensor_locations]

sensor_locations_1D = np.reshape(sensor_locations_2D, (2*n_sensors))

#Get ranges of sensor values
sensor_pert = args.sensor_pert
sensor_ranges= (sensor_locations_1D+np.tile(np.array([sensor_pert,-sensor_pert]),(2*n_sensors,1))\
                .transpose()).transpose()
#Bound sensor locations to [-1,1]^2 for the domain
sensor_ranges[sensor_ranges < -1] = -1
sensor_ranges[sensor_ranges > 1] = 1

sensor_ranges = sensor_ranges.transpose()


# ******************************************************************************
# Reshape data for pytorch into 4D tensor Samples x Channels x Width x Hight
# ******************************************************************************
X_tensored = add_channels(X)
X_test_tensored = add_channels(X_test)
sensors = add_channels(sensors)
sensors_test = add_channels(sensors_test)

# transfer to tensor
sensors = torch.from_numpy(sensors)
X_tensored = torch.from_numpy(X_tensored)

sensors_test = torch.from_numpy(sensors_test)
X_test_tensored = torch.from_numpy(X_test_tensored)


# ******************************************************************************
# Create Dataloader objects
# ******************************************************************************
if logging >= 1:
    print("Loading Dataloader")
print(sensors.shape)
print(X.shape)
train_data = torch.utils.data.TensorDataset(sensors, X_tensored)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

# ******************************************************************************
# Formulate Network
# ******************************************************************************
if logging >= 1:
    print("Loading Model")
model = model_from_name(model_name, outputlayer_size=outputlayer_size, n_sensors=n_sensors)
model = model.cuda()


# ******************************************************************************
# Train initial sensor settings
# ******************************************************************************
# ******************************************************************************
# Train: Initi model and set tuning parameters
# ******************************************************************************
rerror_train = []
rerror_test = []

# ******************************************************************************
# Optimizer and Loss Function
# ******************************************************************************
if logging >= 1:
    print("Defining optomizer")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# criterion = nn.L1Loss().cuda()
# criterion = nn.SmoothL1Loss().cuda()
criterion = nn.MSELoss().cuda()


def exp_lr_scheduler(optimizer, epoch, lr_decay_rate=0.8, weight_decay_rate=0.8, lr_decay_epoch=100):
    """Decay learning rate by a factor of lr_decay_rate every lr_decay_epoch epochs"""
    if epoch % lr_decay_epoch:
        return

        # if args.optimizer == 'sgd':
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay_rate
        param_group['weight_decay'] *= weight_decay_rate
    return

    # ******************************************************************************


# Start training
# ******************************************************************************
if logging >= 1:
    print("Starting Training")
t0 = timeit.default_timer()

for epoch in range(num_epochs):

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data).float(), Variable(target).float()

        # ===================forward=====================
        model.train()
        output = model(data)
        loss = criterion(output, target)

        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================adjusted lr========================
        exp_lr_scheduler(optimizer, epoch, lr_decay_rate=learning_rate_change,
                         weight_decay_rate=weight_decay_change,
                         lr_decay_epoch=epoch_update)

    # if epoch % 5 == 0:
    #     print('********** Epoche %s **********' % (epoch))
    #     rerror_train.append(error_summary(sensors, X, n_snapshots_train, model.eval(), Xmean, 'training'))
    #     rerror_test.append(
    #         error_summary(sensors_test, X_test, n_snapshots_test, model.eval(), Xmean, 'testing'))


error_summary(sensors_test, X_test_tensored, n_snapshots_test, model.eval(), Xmean, 'testing')
# ******************************************************************************
# Formulate model
# ******************************************************************************
network = {"model": model, "learning_rate": learning_rate, "optimizer": optimizer,\
           "criterion": criterion, "batch_size": batch_size, \
           "exp_lr_scheduler": exp_lr_scheduler, "learning_rate_change": learning_rate_change,\
           "weight_decay_change": weight_decay_change,"epoch_update": epoch_update,\
           "error_summary": error_summary}
    
if args.qoi_type.lower()== "error":
    qoi_names = np.array(["dummy","error"])
    qoi_selector = np.array(["dummy","error"])
    

    
eval_fcn = lambda sensor_locations_1D: network_fcn(network, sensor_locations_1D,\
                                                   X, X_test, \
                                                   Xmean, qoi_selector,num_epochs, logging = logging)
model = uq.Model(eval_fcn = eval_fcn,
                  base_poi = sensor_locations_1D,
                  dist_type = "uniform",
                  dist_param = sensor_ranges,
                  name_qoi = qoi_names
                  )
#Set options
uqOptions = uq.Options()
uqOptions.lsa.run=False
uqOptions.lsa.run_lsa = False
uqOptions.lsa.run_pss = False

uqOptions.gsa.run_sobol = False
uqOptions.gsa.run=True
uqOptions.gsa.run_morris = True
uqOptions.gsa.n_samp_morris = args.n_samp_morris
uqOptions.gsa.l_morris = args.l_morris

uqOptions.save = True
uqOptions.display = True
uqOptions.plot = True

results = uq.run_uq(model, uqOptions, logging =logging)

np.savez("results/sens_sensitivity_"+ args.qoi_type.lower() + "_s" + str(n_sensors) \
         + "e" + str(num_epochs) + "_samp" + str(args.n_samp_morris) + \
         "l" + str(int(1/args.l_morris)) + "_pert" + str(int(sensor_pert*100)),\
         base_sensors = sensor_locations_2D,
         base_train_error = rerror_train,
         base_test_error = rerror_test,
         morris_mean = results.gsa.morris_mean,
         morris_mean_abs = results.gsa.morris_mean_abs,
         morris_std = results.gsa.morris_std)
    
