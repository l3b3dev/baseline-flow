# -*- coding: utf-8 -*-
"""
Created on Tue May 24 20:58:00 2022

@author: USER
"""

import numpy as np
from cavity_spline import cavity_linear_spline as get_sensor_values
import sys


import torch
from torch import nn
from torch.autograd import Variable, Function
from torch.utils.data import DataLoader, Dataset
from jacobian import extend, JacobianMode

from utils import add_channels

def main(network, sensor_locations_1D, velocity_1D, velocity_1D_test, \
         velocity_mean_1D, qoi_selector,num_epochs, logging = 0):
    #Initalize arrays and parameters
    n_qoi = qoi_selector.size
    if sensor_locations_1D.ndim == 2:
        (n_samp,n_sensors) = sensor_locations_1D.shape
    elif sensor_locations_1D.ndim == 1:
        n_sensors = sensor_locations_1D.size
        sensor_locations_1D = sensor_locations_1D.reshape((1,n_sensors))
        n_samp = 1
    if logging > 1 :
        print("sensor_locations_1D: " + str(sensor_locations_1D))
        print("n_samp: " + str(n_samp))
        print("n_sensors: " + str(n_sensors))
        
    #Check even number of sensors
    if n_sensors/2 != int(n_sensors/2):
        raise Exception("odd number of sensor locations")
    n_sensors = int(n_sensors/2)
    [n_snapshots_test, n_cell] = velocity_1D_test.shape
    n_snapshots_train = velocity_1D.shape[0]
    
    #Unpack network
    model = network["model"]
    learning_rate = network["learning_rate"]
    optimizer = network["optimizer"]
    criterion = network["criterion"]
    batch_size = network["batch_size"]
    exp_lr_scheduler = network["exp_lr_scheduler"]
    learning_rate_change = network["learning_rate_change"]
    weight_decay_change = network["weight_decay_change"]
    epoch_update = network["epoch_update"]
    error_summary = network["error_summary"]
    #Convert data to sensor locations to 2D
    sensor_locations_2D = np.reshape(sensor_locations_1D,[n_samp,n_sensors,2])
    velocity_2D_test = velocity_1D_test.reshape(n_snapshots_test, int(np.sqrt(n_cell)), int(np.sqrt(n_cell)))
    velocity_2D = velocity_1D.reshape(n_snapshots_train, int(np.sqrt(n_cell)), int(np.sqrt(n_cell)))
    
    #Tensorfy velocity
    velocity_tensored = add_channels(velocity_1D)
    velocity_test_tensored  = add_channels(velocity_1D_test)
    velocity_tensored= torch.from_numpy(velocity_tensored)
    velocity_test_tensored= torch.from_numpy(velocity_test_tensored)
    #Loop through parameter samples
    for i_samp in range(n_samp):
        # Get sensor values
        # if logging > 1 :
         #    print("Sample: " + str(i_samp))
         #    print("Training sensor locations: " + str(sensor_locations_2D[i_samp:,]))
        sensors_2D = get_sensor_values(sensor_locations_2D[i_samp,:], velocity_2D)
        sensors_test_2D = get_sensor_values(sensor_locations_2D[i_samp,:], velocity_2D_test)
        #if logging > 1 : 
            # print("Training Sensor values: " + str(sensors_2D))
        sensors_1D = sensors_2D.reshape((n_snapshots_train, n_sensors))
        sensors_test_1D = sensors_test_2D.reshape((n_snapshots_test, n_sensors))
        
        # ******************************************************************************
        # Reshape data for pytorch into 4D tensor Samples x Channels x Width x Hight
        # ******************************************************************************
        sensors_tensored = add_channels(sensors_1D)
        sensors_test_tensored = add_channels(sensors_test_1D)

        # transfer to tensor
        sensors_tensored = torch.from_numpy(sensors_tensored)

        sensors_test_tensored = torch.from_numpy(sensors_test_tensored)
        
        
        train_data = torch.utils.data.TensorDataset(sensors_tensored, velocity_tensored)
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)


        # ******************************************************************************
        # Optimizer and Loss Function
        # ******************************************************************************
        # Train networks
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
        # Output qois
        qois_samp = np.empty((0,))
        for i_qoi in range(n_qoi):
            if qoi_selector[i_qoi].lower() == "dummy":
                qois_samp = np.append(qois_samp,0)
            # if qoi_selector[i_qoi].lower() == "energy":
            #     energy = 
            #     qois_samp = np.append(qois_samp, energy)
            # elif qoi_selector[i_qoi].lower() == "full":
            #     #Compute predictions for test data
            #     full_results
            #     #Concatenate results
            #     full_results_concatenated = 
            #     qois_samp = np.append(qois_samp, full_results_concatenated)
            elif qoi_selector[i_qoi].lower() == "error":
                dataloader_temp = iter(DataLoader(sensors_test_tensored, batch_size=n_snapshots_test))
                output_temp = model(Variable(dataloader_temp.next()).cuda().float())
                tt, _, mt = output_temp.shape

                redata = output_temp.cpu().data.numpy()
                redata = redata + velocity_mean_1D
                Xsmall_temp = velocity_test_tensored.data.numpy() + velocity_mean_1D
                error = np.linalg.norm(Xsmall_temp - redata) / np.linalg.norm(Xsmall_temp)
                
                qois_samp = np.append(qois_samp, error)
            elif qoi_selector[i_qoi].lower() == "jacobian":
                #run sensitivity
                extend(model, (1, n_sensors))
                #Compute Jacobian
                with JacobianMode(model):
                    out = model(input_vec)
                    out.sum().backward()
                    #jacobian tensor from which we calc. sensitivity
                    jac = model.jacobian()
            else :
                raise Exception("Unknown qoi: " + str(qoi_selector[i_qoi]))
        if i_samp == 0:
            qois = np.empty((n_samp, qois_samp.size))
        #print(qois_samp)
        qois[i_samp] = qois_samp
        del qois_samp

    return qois.squeeze()
    
if __name__ == "__main__":
    sys.exit(main())
