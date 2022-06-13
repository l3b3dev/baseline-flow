import os
import sys
import timeit
import torch
from torch import nn
from torch.autograd import Variable, Function
from torch.utils.data import DataLoader, Dataset

from jacobian import extend, JacobianMode
from read_dataset import data_from_name
from shallowdecoder_model import model_from_name, ShallowDecoder
from utils import *

import argparse

mpl.style.available
mpl.style.use('seaborn-paper')
plt.rcParams.update({'font.size': 20})

# ==============================================================================
# Training settings
# ==============================================================================

parser = argparse.ArgumentParser(description='PyTorch Example')
#
parser.add_argument('--name', type=str, default='shallow_decoder_drop', metavar='N', help='Model')
#
parser.add_argument('--data', type=str, default='flow_cylinder', metavar='N', help='dataset')
#
parser.add_argument('--sensor', type=str, default='wall', metavar='N', help='chose between "wall" or "leverage_score"')
#
parser.add_argument('--n_sensors', type=int, default=5, metavar='N', help='number of sensors')
#
parser.add_argument('--epochs', type=int, default=4000, metavar='N', help='number of epochs to train (default: 10)')
#
parser.add_argument('--plotting', type=bool, default=False, metavar='N', help='number of epochs to train (default: 10)')
#
args = parser.parse_args()

# ******************************************************************************
# Select dataset
# ******************************************************************************
dataset = args.data

# ******************************************************************************
# Paramteres
# ******************************************************************************
model_name = args.name

plotting = args.plotting
directory = f'data/cavflow'
total_runs = 1

for file in os.listdir(directory):
    print(f'Processing {file}')
    filename = os.fsdecode(file)
    f = os.path.join(directory, file)
    re = ''.join(filter(str.isdigit, filename))

    out_dir = f'out/{re}'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    if dataset == 'flow_cylinder':
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

    #np.random.seed(1234)
    for runs in range(total_runs):

        # ******************************************************************************
        # read data and set sensor
        # ******************************************************************************
        if dataset == 'flow_cylinder':
            Xsmall, Xsmall_test, m, n = data_from_name(dataset)

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
        random_seed = np.random.choice(range(1000), 1)
        sensors, sensors_test, sensor_locations = sensor_from_name(sensor_type,
                                                                   Xsmall, Xsmall_test,
                                                                   sensor_num=n_sensors,
                                                                   random_seed=random_seed, m=m, n=n)

        #save the sensor location
        # np.save("sensors.npy", sensors)
        # np.save("sensors_test.npy", sensors_test)
        # np.save("sensor_locations.npy", sensor_locations)

        # ******************************************************************************
        # Plot first frame and overlay with sensor locations
        # ******************************************************************************
        if plotting:
            if dataset == 'flow_cylinder':
                plot_flow_cyliner(Xsmall, sensor_locations, m, n, Xmean, re)

        # ******************************************************************************
        # Reshape data for pytorch into 4D tensor Samples x Channels x Width x Hight
        # ******************************************************************************
        Xsmall = add_channels(Xsmall)
        Xsmall_test = add_channels(Xsmall_test)
        sensors = add_channels(sensors)
        sensors_test = add_channels(sensors_test)

        # transfer to tensor
        sensors = torch.from_numpy(sensors)
        Xsmall = torch.from_numpy(Xsmall)

        sensors_test = torch.from_numpy(sensors_test)
        Xsmall_test = torch.from_numpy(Xsmall_test)

        # ******************************************************************************
        # Create Dataloader objects
        # ******************************************************************************
        train_data = torch.utils.data.TensorDataset(sensors, Xsmall)
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

        # ******************************************************************************
        # Deep Decoder
        # ******************************************************************************
        model = model_from_name(model_name, outputlayer_size=outputlayer_size, n_sensors=n_sensors)
        model = model.cuda()

        # ******************************************************************************
        # Train: Initi model and set tuning parameters
        # ******************************************************************************
        rerror_train = []
        rerror_test = []

        # ******************************************************************************
        # Optimizer and Loss Function
        # ******************************************************************************
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

            if epoch % 5 == 0:
                print('********** Epoche %s **********' % (epoch))
                rerror_train.append(error_summary(sensors, Xsmall, n_snapshots_train, model.eval(), Xmean, 'training'))
                rerror_test.append(
                    error_summary(sensors_test, Xsmall_test, n_snapshots_test, model.eval(), Xmean, 'testing'))

        # model = ShallowDecoder(outputlayer_size=outputlayer_size, n_sensors=n_sensors)
        # model.load_state_dict(torch.load("./deepDecoder_flow_0049.pth", map_location="cpu"))

        # ******************************************************************************
        # Save model
        # ******************************************************************************
        #torch.save(model.state_dict(), './deepDecoder_flow_0049.pth')

        # ******************************************************************************
        # Reconstructed flow field
        # ******************************************************************************
        if plotting:
            model.eval()
            dataloader_temp = iter(DataLoader(sensors, batch_size=n_snapshots_train))
            input_vec = Variable(dataloader_temp.next()).float().cuda()
            output_temp = model(input_vec)

            #output_temp = model(Variable(dataloader_temp.next()).float().cpu())

            if dataset == 'flow_cylinder':
                plot_flow_cyliner_2(output_temp.cpu().data.numpy()[390, :, :].reshape(n, m), n, m, Xmean, re)
                #plot error
                plot_flow_cyliner_2(np.abs(Xsmall[390, :].cpu().data.numpy().reshape(n,m) - output_temp.cpu().data.numpy()[390, :, :].reshape(n, m)), n, m, Xmean, re,2)

            del (dataloader_temp, output_temp)


        # ******************************************************************************
        # Error plots
        # ******************************************************************************
        if plotting:
            fig = plt.figure()
            plt.plot(rerror_train, lw=2, label='Trainings error', color='#377eb8', )
            plt.plot(rerror_test, lw=2, label='Test error', color='#e41a1c', )

            plt.tick_params(axis='x', labelsize=14)
            plt.tick_params(axis='y', labelsize=14)
            plt.locator_params(axis='y', nbins=10)
            plt.locator_params(axis='x', nbins=10)

            plt.ylabel('Error', fontsize=14)
            plt.xlabel('Epochs', fontsize=14)
            plt.grid(False)
            plt.yscale("log")
            # ax[0].set_ylim([0.01,1])
            plt.legend(fontsize=14)
            fig.tight_layout()
            #plt.show()
            plt.savefig(f'out/{re}/shallow_decoder_convergence.png', dpi=300)
            plt.close()

            # ******************************************************************************
        # Model Summary
        # ******************************************************************************
        time_train.append(timeit.default_timer() - t0)

        print('********** Train time**********')
        print('Time: ', time_train[-1])

        print('********** ShallowDecoder Model Summary Training**********')
        out_a, out_b = final_summary(sensors, Xsmall, n_snapshots_train, model.eval(), Xmean, sensor_locations)

        error_DD_train.append(out_a)
        error_dev_DD_train.append(out_b)

        print('********** ShallowDecoder Model Summary Testing**********')
        out_a, out_b = final_summary(sensors_test, Xsmall_test, n_snapshots_test, model.eval(), Xmean, sensor_locations)

        error_DD_test.append(out_a)
        error_dev_DD_test.append(out_b)

        print('********** POD Training and Testing**********')

        out_a, out_b, out_c, out_d = summary_pod(Xsmall, Xsmall_test, sensors, sensors_test, Xmean, sensor_locations,
                                                 alpha=0, case='naive_pod')

        error_POD_train.append(out_a)
        error_dev_POD_train.append(out_b)
        error_POD_test.append(out_c)
        error_dev_POD_test.append(out_d)

        print('********** Regularized Plus POD Training and Testing**********')

        out_a, out_b, out_c, out_d = summary_pod(Xsmall, Xsmall_test, sensors, sensors_test, Xmean, sensor_locations,
                                                 alpha=alpha, case='plus_pod')

        error_reg_POD_train.append(out_a)
        error_dev_reg_POD_train.append(out_b)
        error_reg_POD_test.append(out_c)
        error_dev_reg_POD_test.append(out_d)

    original_stdout = sys.stdout # Save a reference to the original standard output
    with open(f'out/{re}/models_stats.txt', 'w') as f:
        sys.stdout = f # Change the standard output to the file we created.

        print('********** Final Summary  **********')
        print('********** ', model_name)
        print('********** *************  **********')

        print('Time : ', np.mean(time_train))

        print('Train - Relative error using ShallowDecoder: ', np.mean(error_DD_train), ',and stdev: ',
              np.var(error_DD_train) ** 0.5)
        print('Train - Relative error (deviation) using ShallowDecoder: ', np.mean(error_dev_DD_train), ',and stdev: ',
              np.var(error_dev_DD_train) ** 0.5)

        print('Test - Relative error using ShallowDecoder: ', np.mean(error_DD_test), ',and stdev: ',
              np.var(error_DD_test) ** 0.5)
        print('Test - Relative error (deviation) using ShallowDecoder: ', np.mean(error_dev_DD_test), ',and stdev: ',
              np.var(error_dev_DD_test) ** 0.5)

        print()
        print('Train - Relative error using POD: ', np.mean(error_POD_train), ',and stdev: ', np.var(error_POD_train) ** 0.5)
        print('Train - Relative error (deviation) using POD: ', np.mean(error_dev_POD_train), ',and stdev: ',
              np.var(error_dev_POD_train) ** 0.5)

        print('Test - Relative error using POD: ', np.mean(error_POD_test), ',and stdev: ', np.var(error_POD_test) ** 0.5)
        print('Test - Relative error (deviation) using POD: ', np.mean(error_dev_POD_test), ',and stdev: ',
              np.var(error_dev_POD_test) ** 0.5)

        print()
        print('Train - Relative error using Regularized POD: ', np.mean(error_reg_POD_train), ',and stdev: ',
              np.var(error_reg_POD_train) ** 0.5)
        print('Train - Relative error (deviation) using Regularized POD: ', np.mean(error_dev_reg_POD_train), ',and stdev: ',
              np.var(error_dev_reg_POD_train) ** 0.5)

        print('Test - Relative error using Regularized POD: ', np.mean(error_reg_POD_test), ',and stdev: ',
              np.var(error_reg_POD_test) ** 0.5)
        print('Test - Relative error (deviation) using Regularized POD: ', np.mean(error_dev_reg_POD_test), ',and stdev: ',
              np.var(error_dev_reg_POD_test) ** 0.5)

        sys.stdout = original_stdout  # Reset the standard output to its original value

    if plotting:
        plot_spectrum(sensors, Xsmall, sensors_test, Xsmall_test,
                      n_snapshots_train, n_snapshots_test,
                      model, Xmean, sensor_locations, plotting, train_or_test='training', re=re)



    #run sensitivity
    extend(model, (1, n_sensors))

    # Jacobian computed by the improved method
    with JacobianMode(model):
        out = model(input_vec)
        out.sum().backward()
        #jacobian tensor from which we calc. sensitivity
        jac = model.jacobian()

    # if plotting:
    #     plot_flow_cyliner_pod(Xsmall, sensors, Xmean, sensor_locations, m, n, re)
    #     plot_flow_cyliner_regularized_pod(Xsmall, sensors, Xmean, sensor_locations, m, n, alpha)
