import timeit
import torch
from torch import nn
from torch.autograd import Variable, Function
from torch.utils.data import DataLoader, Dataset

from sklearn.preprocessing import MinMaxScaler

from read_dataset import read_data
from shallowdecoder_model import model_from_name, ShallowDecoder, ShallowDecoderDrop
from utils import *

mpl.style.available
mpl.style.use('seaborn-paper')

total_runs = 1
num_epochs = 230
batch_size = 100
n_sensors = 49
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

np.random.seed(1234)
for runs in range(total_runs):

    # ******************************************************************************
    # read data and set sensor
    # ******************************************************************************
    X, X_test, Y, Y_test, m, n = read_data('data/dataset_1/marker_data')

    # get size
    outputlayer_size = Y.shape[1]
    n_snapshots_train = X.shape[0]
    n_snapshots_test = X_test.shape[0]

    # ******************************************************************************
    # Rescale data between 0 and 1 for learning
    # ******************************************************************************
    scaler = MinMaxScaler()

    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)

    # ******************************************************************************
    # Reshape data for pytorch into 4D tensor Samples x Channels x Data
    # ******************************************************************************
    X = add_channels(X)
    X_test = add_channels(X_test)
    Y = add_channels(Y)
    Y_test = add_channels(Y_test)

    # transfer to tensor
    Y = torch.from_numpy(Y)
    X = torch.from_numpy(X)

    Y_test = torch.from_numpy(Y_test)
    X_test = torch.from_numpy(X_test)

    # ******************************************************************************
    # Create Dataloader objects
    # ******************************************************************************
    train_data = torch.utils.data.TensorDataset(X, Y)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    # ******************************************************************************
    # Deep Decoder
    # ******************************************************************************
    model = ShallowDecoderDrop(outputlayer_size=outputlayer_size, n_sensors=n_sensors)
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
            rerror_train.append(error_summary(X, Y, n_snapshots_train, model.eval(), 'training'))
            rerror_test.append(
                error_summary(X_test, Y_test, n_snapshots_test, model.eval(), 'testing'))

    # ******************************************************************************
    # Save model
    # ******************************************************************************
    torch.save(model.state_dict(), './deepDecoder_flow_0049.pth')

    model = ShallowDecoderDrop(outputlayer_size=outputlayer_size, n_sensors=n_sensors)
    model.load_state_dict(torch.load("./deepDecoder_flow_0049.pth", map_location="cpu"))

    # ******************************************************************************
    # Error plots
    # ******************************************************************************
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
    plt.show()
    #plt.savefig('results/shallow_decoder_convergence.png', dpi=300)
    plt.close()

plot_dominant_modes(X, Y, X_test, Y_test,n_snapshots_train, n_snapshots_test, model, scaler, n,m)
