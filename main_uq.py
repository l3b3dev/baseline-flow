import numpy as np
import torch
# from keras import Model
# from pytorch2keras import pytorch_to_keras
from torch.autograd import Variable
from torch.utils.data import DataLoader

#from UQ_methods import uncq
from read_dataset import data_from_name
from shallowdecoder_model import ShallowDecoder
from utils import plot_flow_cyliner, sensor_from_name, add_channels, plot_flow_cyliner_2, plot_spectrum

n_sensors = 5
sensor_type = 'wall'

Xsmall, Xsmall_test, m, n = data_from_name('flow_cylinder')
Xsmall = np.asarray(Xsmall)
Xsmall_test = np.asarray(Xsmall_test)

# get size
outputlayer_size = Xsmall.shape[1]
n_snapshots_train = Xsmall.shape[0]
n_snapshots_test = Xsmall_test.shape[0]

# ******************************************************************************
# Rescale data between 0 and 1 for learning
# ******************************************************************************
Xmean = Xsmall.mean(axis=0)
Xsmall -= Xmean
Xsmall_test -= Xmean

# ******************************************************************************
# Get sensor locations
# ******************************************************************************
# sensors = np.load("sensors.npy")
# sensors_test = np.load("sensors_test.npy")
# sensor_locations = np.load("sensor_locations.npy")

# TODO: Vary one of the sensors
random_seed = np.random.choice(range(1000), 1)
sensors, sensors_test, sensor_locations = sensor_from_name(sensor_type,
                                                           Xsmall, Xsmall_test,
                                                           sensor_num=n_sensors,
                                                           random_seed=random_seed)

plot_flow_cyliner(Xsmall_test, sensor_locations, m, n, Xmean)

# ******************************************************************************
# Reshape data for pytorch into 4D tensor Samples x Channels x Width x Hight
# ******************************************************************************
Xsmall = add_channels(Xsmall)
Xsmall_test = add_channels(Xsmall_test)
sensors = add_channels(sensors)
sensors_test = add_channels(sensors_test)

# transfer to tensor
sensors_np = torch.from_numpy(sensors)
Xsmall_np = torch.from_numpy(Xsmall)
sensors_test_np = torch.from_numpy(sensors_test)
Xsmall_test_np = torch.from_numpy(Xsmall_test)

model = ShallowDecoder(outputlayer_size=outputlayer_size, n_sensors=n_sensors)
model.load_state_dict(torch.load("./deepDecoder_flow_0049.pth"))

# Reconstruct
model.eval()
dataloader_temp = iter(DataLoader(sensors_test_np, batch_size=n_snapshots_train))
input_var = Variable(dataloader_temp.next()).float()
output_temp = model(input_var)

#plot_flow_cyliner(output_temp.cpu().data.numpy(), sensor_locations, m, n, Xmean)
plot_flow_cyliner(Xsmall_test-output_temp.cpu().data.numpy(), sensor_locations, m, n, Xmean)

#plot_flow_cyliner_2(output_temp.cpu().data.numpy()[0, :, :].reshape(m, n), m, n, Xmean)
del (dataloader_temp, output_temp)

# Keras model
# k_model = pytorch_to_keras(model, input_var, verbose=True)
#
# pmt = k_model.predict(sensors_test)
# plot_flow_cyliner_2(pmt[0, :, :].reshape(m, n), m, n, Xmean)
#
# ######################### QIPF PARAMETERS ################################
# orr = 14  #### ORDER = orr/ttt -2
# bwf = 200  #### Silverman kernel bandwidth multiplier
# ttt = 2  #### ttt = 2 (ever order moments)
#
# ######################## IMPLEMENT QIPF #############################
# import math
# import skimage.measure as sk
# from statsmodels.nonparametric.bandwidths import bw_silverman as bw
#
# ################## EXTRACT MODEL LAYERS IN fc #######################
# fc = []
# for i in range(len(k_model.layers)):
#     fc.append(k_model.layers[i])
#
# ################## EXTRACT LAYER WEIGHTS ############################
# # hm1 = sk.block_reduce(fc[0].get_weights()[0], (1, 1), np.mean).flatten()
# # hm2 = sk.block_reduce(fc[1].get_weights()[0], (1, 1), np.mean).flatten()
# # hm3 = sk.block_reduce(fc[2].get_weights()[0], (1, 1), np.mean).flatten()
# # hm4 = sk.block_reduce(fc[3].get_weights()[0], (1, 1), np.mean).flatten()
# # hm5 = sk.block_reduce(fc[7].get_weights()[0], (1, 5), np.mean).flatten()
#
# hm1 = sk.block_reduce(fc[1].get_weights()[0], (1, 1), np.mean).flatten()
# hm2 = sk.block_reduce(fc[6].get_weights()[0], (1, 1), np.mean).flatten()
# hm3 = sk.block_reduce(fc[11].get_weights()[0], (1, 1), np.mean).flatten()
#
# #################### CONCATENATE AND NORMALIZE WEIGHTS ####################
# hmt = np.concatenate((hm1, hm2, hm3))
# hmt = hmt.flatten()
# hmt = (hmt - np.mean(hmt)) / np.std(hmt)
#
# modelq = k_model  # Model(inputs=k_model.input, outputs=preds)
# xtest = sensors_test
# X_train = sensors
#
# pmt = modelq.predict(xtest)
#
# ########################## NORMALIZE WRT TO TRAIN OUTPUT ###########################
# rmt = modelq.predict(X_train)
# pmt1 = np.max(pmt, 1)
# rmt1 = np.max(rmt, 1)
# pmtn = (pmt1 - np.mean(rmt1)) / np.std(rmt1)
#
# ######################## START QIPF #################
# es = 0.2
# n = len(xtest)
# ct = 0
# print('start')
#
# s0m = []
# sqp = []
# sigg1 = bwf * np.average(bw(hmt))
#
# w1 = np.zeros((len(xtest), 3))
# N1 = len(xtest)
# N2 = len(hmt)
# import time
#
# start = time.time()
#
# for i in range(len(xtest)):
#
#     jh = pmtn[i]
#     jh = [jh - es, jh, jh + es]
#
#     ct += 1
#     if ct % 10 == 0:
#         print(" iter #: ", ct, ' / ', N2)
#
#     w1[i, 0] = (1 / N2) * np.sum(np.exp(-(np.power(jh[0] - hmt, 2)) / (2 * sigg1 ** 2)))
#     w1[i, 1] = (1 / N2) * np.sum(np.exp(-(np.power(jh[1] - hmt, 2)) / (2 * sigg1 ** 2)))
#     w1[i, 2] = (1 / N2) * np.sum(np.exp(-(np.power(jh[2] - hmt, 2)) / (2 * sigg1 ** 2)))
#
#     jh = [[w1[i, 0]], [w1[i, 1]], [w1[i, 2]]]
#     w0 = np.sqrt(jh)
#     x = w0
#     n = np.arange(1, orr + 1)
#     fn = np.floor(n / 2)
#     p = np.arange(0, orr + 1)
#     x = 2 * x
#     lex = len(x)
#     lenn = len(n)
#
#     if p[0] == 0:
#         xp = np.power(x, p[1::])
#         xp = np.concatenate([np.ones((lex, 1)), xp], axis=1)
#     else:
#         xp = np.power(x, p)
#
#     H = np.zeros((lex, lenn))
#     H = np.float64(H)
#     yy = np.zeros(lenn)
#     yy = np.float64(yy)
#
#     for k in range(lenn):
#         for m in range(int(fn[k]) + 1):
#             is_the_power = p == n[k] - (2 * m)
#             jj = (1 - 2 * np.mod(m, 2)) / math.factorial(m) / math.factorial(n[k] - (2 * m)) * xp[:,
#                                                                                                is_the_power]
#             H[:, k] += jj[:, 0]
#
#         ll = math.factorial(n[k])
#         H[:, k] = ll * H[:, k]
#     wy = H
#     sg = sigg1 ** 2
#     qe = np.gradient(np.gradient(np.abs(wy), axis=1), axis=1)
#     qe1 = np.abs(wy)
#     vc = np.multiply((sg / 2), np.divide(qe, qe1))
#     r = np.zeros((np.shape(vc)[0], int((np.shape(vc)[1] / ttt)) - 1))
#     for qk in range(1, int(orr / ttt)):
#         if len(wy) == 1:
#             r[:, qk - 1] = np.abs(vc[:, (ttt * qk) - 1])
#         else:
#             r[:, qk - 1] = vc[:, (ttt * qk) - 1] - np.min(vc[:, (ttt * qk) - 1])
#     r = r.T
#     qn0 = np.double(r[0:-1])
#     sk = qn0
#     qp = np.double(r[-1])
#
#     s0m.append(sk)
#     sqp.append(qp)
#
# stop = time.time()
# duration = stop - start
# print(duration)
#
# ####################### QIPF UNCERTAINTY MODES #########################################
# sm = s0m
#
# ######################### PROCESS AND NORMALIZE THE UNCERTAINTY MODES ##################
# mm = np.zeros((len(sm), int(orr / ttt) - 2))
# for i in range(len(sm)):
#     mm[i, :] = sm[i][:, 1]
#
# mmq = mm[:, 0:-1]
#
# # mmq = (mmq - np.min(mmq))/(np.max(mmq) - np.min(mmq))
# mm2 = np.copy(mmq)
#
# for i in range(len(mmq[0])):
#     mm2[:, i] = mmq[:, i] - np.mean(mmq, axis=1)
#
# ############################## NORMALIZE UNCERTAINTY MODES AROUND MODEL PREDICTIONS ####################
# qq = mm2 + pr
# a0 = x_test.ravel()
#
# a1 = pr.ravel() + np.max(qq, axis=1)
# a2 = pr.ravel() - np.max(qq, axis=1)
#
# a1m = pr.ravel() + np.mean(mmq, axis=1)
# a2m = pr.ravel() - np.mean(mmq, axis=1)
#
# a1s = pr.ravel() + np.std(qq, axis=1)
# a2s = pr.ravel() - np.std(qq, axis=1)
