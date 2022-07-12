import numpy as np
from scipy.io import loadmat


# ******************************************************************************
# Read in data
# ******************************************************************************
def data_from_name(name):
    if name == 'flow_cylinder':
        return flow_cylinder()
    else:
        return flow_cavity(name)
    # else:
    #     raise ValueError('dataset {} not recognized'.format(name))


def rescale(Xsmall, Xsmall_test):
    # ******************************************************************************
    # Rescale data
    # ******************************************************************************
    Xmin = Xsmall.min()
    Xmax = Xsmall.max()

    Xsmall = ((Xsmall - Xmin) / (Xmax - Xmin))
    Xsmall_test = ((Xsmall_test - Xmin) / (Xmax - Xmin))

    return Xsmall, Xsmall_test


def flow_cavity(name):
    # X = np.load('data/flow_cylinder.npy')
    X = np.load(name)
    print(X.shape)

    # Split into train and test set
    # Xsmall = X[0:100, :, :]
    # t, m, n = Xsmall.shape
    # Xsmall = X[0:700, 0, 100:300, 100:700 ]
    n_snapshot = X.shape[0]
    n_train = int(n_snapshot*5/6)
    Xsmall = X[0:n_train, 0, :, :]
    t, n, m = Xsmall.shape

    # Xsmall = Xsmall.reshape(100, -1)
    # Xsmall = Xsmall.reshape(700, -1)
    Xsmall_1D = Xsmall.reshape(n_train, -1)

    # Xsmall_test = X[100:151, :, :].reshape(51, -1)
    Xsmall_1D_test = X[n_train:, 0, :, :].reshape(n_snapshot-n_train, -1)

    # ******************************************************************************
    # Return train and test set
    # ******************************************************************************
    return Xsmall_1D, Xsmall_1D_test, m, n


def flow_cylinder():
    # X = np.load('data/flow_cylinder.npy')
    X = np.load('data/cavflow/cavflow15500.npy')
    print(X.shape)

    # Split into train and test set
    # Xsmall = X[0:100, :, :]
    # t, m, n = Xsmall.shape
    # Xsmall = X[0:700, 0, 100:300, 100:700 ]
    Xsmall = X[0:5000, 0, :, :]
    t, n, m = Xsmall.shape

    # Xsmall = Xsmall.reshape(100, -1)
    # Xsmall = Xsmall.reshape(700, -1)
    Xsmall = Xsmall.reshape(5000, -1)

    # Xsmall_test = X[100:151, :, :].reshape(51, -1)
    Xsmall_test = X[5000:, 0, :, :].reshape(1001, -1)

    # ******************************************************************************
    # Return train and test set
    # ******************************************************************************
    return Xsmall, Xsmall_test, m, n

# def flow_cylinder():
#     X = np.load('data/flow_cylinder.npy')
#     #X = np.load('data/cyl2dflow_old.npy')
#     print(X.shape)
#
#     # Split into train and test set
#     Xsmall = X[0:100, :, :]
#     t, m, n = Xsmall.shape
#
#     Xsmall = Xsmall.reshape(100, -1)
#
#     Xsmall_test = X[100:151, :, :].reshape(51, -1)
#
#     # ******************************************************************************
#     # Return train and test set
#     # ******************************************************************************
#     return Xsmall, Xsmall_test, m, n
