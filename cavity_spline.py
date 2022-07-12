# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 14:00:07 2022

@author: USER
"""
import numpy as np
import sys


def cavity_linear_spline(sensor_pos, velocity_2D):
    n_dim = 2
    if velocity_2D.ndim == 3:
        [n_snapshot, n_y, n_x] =velocity_2D.shape
    elif velocity_2D.ndim ==2:
        n_snapshot = 1
        [n_y, n_x] =velocity_2D.shape
    else:
        raise Exception("Only 2 or 3 dimensions allowed for velocity 2D")
    #Get x and y center vectors, Assume Chebyshev nodes
    x_center = -np.cos(np.pi*np.arange(0,n_x)/(n_x-1))
    y_center = -np.cos(np.pi*np.arange(0,n_y)/(n_y-1))
    if x_center[0] == x_center[1]:
        raise Exception("no stepping in x_centers")
    if y_center[0] == y_center[1]:
        raise Exception("no stepping in y_centers")
    # Check sensor_pos dimensions and intialize velocity data 
    if sensor_pos.ndim == 2:
        n_sensors = sensor_pos.shape[0]
        if sensor_pos.shape[1] !=2:
            raise Exception(str(sensor_pos.shape[1]) + " velocity dimensions detected. Only two implemented.")
    elif sensor_pos.ndim == 1:
        n_sensors = 1
        if sensor_pos.size !=2:
            raise Exception(str(sensor_pos.size) + " velocity dimensions detected. Only two implemented.")
    else:
        raise Exception("sensor_pos is more than 2D")
    spline_vel = np.empty((n_snapshot,n_sensors))
    # Loop Through sensor_pos sample points
    for i_sensor in range(n_sensors):
        (x_pos, y_pos) = (sensor_pos[i_sensor,0], sensor_pos[i_sensor,1])
        #Identify v00, v01, v10, v11
        x_upper_index = np.searchsorted(x_center, x_pos, side='left', sorter=None)
        y_upper_index = np.searchsorted(y_center, y_pos, side='left', sorter=None)
        x_upper = x_center[x_upper_index]
        y_upper = y_center[y_upper_index]
        x_lower = x_center[x_upper_index - 1]
        y_lower = y_center[y_upper_index - 1]
        #Compute a and b
        x_prop = (x_pos-x_lower)/(x_upper-x_lower)
        y_prop = (y_pos-y_lower)/(y_upper-y_lower)
        for i_snap in range(n_snapshot):
            if velocity_2D.ndim ==3:
                v11 = velocity_2D[i_snap,x_upper_index, y_upper_index]
                v00 = velocity_2D[i_snap,x_upper_index-1, y_upper_index-1]
                v01 = velocity_2D[i_snap,x_upper_index-1, y_upper_index]
                v10 = velocity_2D[i_snap,x_upper_index, y_upper_index-1]
            elif velocity_2D.ndim == 2:
                v11 = velocity_2D[x_upper_index, y_upper_index]
                v00 = velocity_2D[x_upper_index-1, y_upper_index-1]
                v01 = velocity_2D[x_upper_index-1, y_upper_index]
                v10 = velocity_2D[x_upper_index, y_upper_index-1]
        
            #compute velocity
            spline_vel[i_snap,i_sensor] = x_prop*y_prop*v11+x_prop*(1-y_prop)*v10 + \
                                  (1-x_prop)*y_prop*v01+(1-x_prop)*(1-y_prop)*v00
    return spline_vel


