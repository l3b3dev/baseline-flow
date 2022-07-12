# -*- coding: utf-8 -*-
"""
Created on Thu May 26 02:13:41 2022

@author: USER
"""

import numpy as np
import matplotlib.pyplot as plt


data_file = "results/"

results = np.load(data_file)

morris_mean_abs = results["morris_mean_abs"]
base_sensor_location = results["morris_mean_abs"]